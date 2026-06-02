"""
Orquestrador G3m — camada LangChain entre o API Gateway e os serviços distribuídos.

Fluxo (alinhado ao diagrama):
    query → [MCP Agent: Colormind + G.Trends + Weather]
                → query enriquecida → [Embedding OpenCLIP]
                → [Qdrant] → [PostgreSQL]
                → [LangChain LCEL: Prompt | Ollama LLM | Parser]
                → resposta enriquecida
"""
import asyncio
import os
from typing import Optional

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .ai import generate_text_embedding
from .vector_db import search_similar_images
from .db import get_image_metadata, get_brand_rules
from .mcp_client import call_mcp_tool
from ..config import settings


# ─── Prompt RAG usado pelo LangChain ──────────────────────────────────────────
EXPLANATION_PROMPT = PromptTemplate.from_template(
    """Você é o 'G3m AI', assistente inteligente de busca de ativos visuais corporativos.

Query original: "{query}"
Query enriquecida (MCP): "{enriched_query}"

Imagens recuperadas (metadados):
{image_summaries}

Contexto enriquecido pelas ferramentas MCP:
{mcp_context}

Diretrizes de Marca (RAG):
{brand_rules}

Com base nesses dados, gere uma justificativa em até 3 frases em português explicando \
por que as imagens recuperadas atendem a busca e se respeitam as diretrizes da marca. \
Seja profissional, conciso e direto."""
)


async def _skip_tool() -> dict:
    """Placeholder assíncrono quando a tool MCP não é aplicável."""
    return {"status": "skip"}


def _should_use_colormind(query: str) -> bool:
    terms = ["cor", "cores", "paleta", "color", "palette", "colorido", "colorida"]
    q = query.lower()
    return any(t in q for t in terms)


def _should_use_trends(query: str) -> bool:
    terms = ["tendencia", "tendencias", "trend", "trending", "editorial", "noticia", "viral", "campanha"]
    q = query.lower()
    return any(t in q for t in terms)


def _should_use_weather(query: str) -> bool:
    # CORRIGIDO: adicionado "estacao" à lista de gatilhos.
    # Sem ele, queries como "foto de estacao chuvosa" não acionavam o Weather MCP.
    terms = ["clima", "tempo", "chuva", "sol", "verao", "inverno", "outono", "primavera", "estacao"]
    q = query.lower()
    return any(t in q for t in terms)


async def orchestrate_search(query: str, category: str) -> dict:
    """
    Ponto de entrada do Orquestrador LangChain.
    Coordena todos os serviços distribuídos e retorna o pacote de resposta enriquecido.
    """

    # ── 1. MCP Agent — ferramentas em paralelo (condicional) ─────────────────
    use_palette = _should_use_colormind(query)
    use_trends = _should_use_trends(query)
    use_weather = _should_use_weather(query)

    print(
        "[Orchestrator] MCP decision -> Colormind=%s, GoogleTrends=%s, Weather=%s"
        % ("ON" if use_palette else "OFF", "ON" if use_trends else "OFF", "ON" if use_weather else "OFF")
    )

    palette_task = (
        call_mcp_tool("get_colormind_palette", {"category": category})
        if use_palette else _skip_tool()
    )
    trends_task = (
        call_mcp_tool("get_google_trends", {"keyword": query, "geo": "BR"})
        if use_trends else _skip_tool()
    )
    weather_task = (
        call_mcp_tool("get_weather", {"city": "São Paulo"})
        if use_weather else _skip_tool()
    )

    print("[Orchestrator] Chamando MCP Agent (Colormind + G.Trends + Weather) em paralelo...")
    palette_result, trends_result, weather_result = await asyncio.gather(
        palette_task, trends_task, weather_task
    )

    color_palette = (
        palette_result.get("result", {}).get("palette", [])
        if palette_result.get("status") == "success" else []
    )
    google_trends = (
        trends_result.get("result", {})
        if trends_result.get("status") == "success" else {}
    )
    weather_data = (
        weather_result.get("result", {})
        if weather_result.get("status") == "success" else {}
    )

    # ── 2. Query enriquecida ─────────────────────────────────────────────────
    enrichment_parts = [query]
    if google_trends.get("top_queries"):
        enrichment_parts.append(
            "Tendencias relacionadas: " + ", ".join(google_trends["top_queries"])
        )
    if color_palette:
        enrichment_parts.append("Paleta sugerida: " + ", ".join(color_palette))
    if weather_data:
        enrichment_parts.append(
            f"Clima em {weather_data.get('city', 'São Paulo')}: "
            f"{weather_data.get('condition', '')} {weather_data.get('temperature_c', '')}°C"
        )
    enriched_query = " | ".join(enrichment_parts)

    # ── 3. Embedding (OpenCLIP via SentenceTransformer) ───────────────────────
    try:
        query_vector = generate_text_embedding(query)
        print(f"[Orchestrator] Embedding gerado para query: '{query}'")
    except Exception as e:
        raise RuntimeError(f"[Orchestrator] Falha ao gerar embedding: {e}")

    # ── 4. Busca Vetorial (Qdrant) ────────────────────────────────────────────
    hits = search_similar_images(query_vector, limit=6)
    if not hits:
        return {
            "query": query,
            "enriched_query": enriched_query,
            "explanation": "Nenhuma imagem correspondente encontrada na base vetorial.",
            "mcp_info": {
                "color_palette": color_palette,
                "google_trends": google_trends,
                "weather": weather_data
            },
            "results": []
        }

    # ── 5. Metadados Estruturados (PostgreSQL) ────────────────────────────────
    results = []
    top_metadata: Optional[dict] = None

    metadata_tasks = [
        asyncio.to_thread(get_image_metadata, hit["image_id"])
        for hit in hits
    ]
    metadata_list = await asyncio.gather(*metadata_tasks)

    for hit, meta in zip(hits, metadata_list):
        if meta:
            filename = os.path.basename(meta["path"])
            
            # Normalizando o score bruto do CLIP (tipicamente máx 0.35 para clip-ViT-B-32)
            # para uma escala 0-1. Isso garante que o frontend exiba ~100% de similaridade
            # para matches excelentes, em vez de mostrar "< 35%".
            raw_score = float(hit["score"])
            normalized_score = min(raw_score / 0.35, 1.0) if raw_score > 0 else 0.0

            results.append({
                "id": meta["id"],
                "path": meta["path"],
                "filename": meta["filename"],
                "description": meta["description"],
                "author": meta["author"],
                "url": f"/images/{filename}",
                "score": normalized_score
            })
            if top_metadata is None:
                top_metadata = meta

    # ── 6. Contexto RAG — Regras de Marca (PostgreSQL) ───────────────────────
    brand_rules = get_brand_rules(category)
    rules_text = (
        "\n".join([f"- {r}" for r in brand_rules])
        if brand_rules else "- Nenhuma diretriz cadastrada."
    )

    # ── 7. Contexto MCP enriquecido ───────────────────────────────────────────
    mcp_parts = []
    if color_palette:
        mcp_parts.append(f"Paleta de cores dominantes: {', '.join(color_palette)}")
    if google_trends.get("top_queries"):
        mcp_parts.append(
            "Termos em alta (Google Trends): " + ", ".join(google_trends["top_queries"])
        )
    if weather_data:
        mcp_parts.append(
            f"Clima atual em {weather_data.get('city', 'São Paulo')}: "
            f"{weather_data.get('condition', '')} {weather_data.get('temperature_c', '')}°C, "
            f"umidade {weather_data.get('humidity_pct', '')}%"
        )

    mcp_context = "\n".join(mcp_parts) if mcp_parts else "Ferramentas MCP indisponíveis."

    # ── 8. LangChain LCEL: Prompt → Ollama (Llama 3) → Parser ────────────────
    image_summaries = "\n".join([
        f"- {img['filename']} | {img['description']} | Autor: {img['author']}"
        for img in results
    ]) or "- Nenhuma imagem encontrada"

    try:
        print(f"[Orchestrator] Chamando LLM (LangChain → Ollama/{settings.OLLAMA_MODEL})...")
        llm = Ollama(
            base_url=settings.OLLAMA_URL,
            model=settings.OLLAMA_MODEL,
            timeout=20
        )
        # LCEL chain: PromptTemplate | LLM | StrOutputParser
        chain = EXPLANATION_PROMPT | llm | StrOutputParser()

        explanation = await chain.ainvoke({
            "query": query,
            "enriched_query": enriched_query,
            "image_summaries": image_summaries,
            "mcp_context": mcp_context,
            "brand_rules": rules_text
        })
        print("[Orchestrator] Explicação LLM gerada com sucesso.")

    except Exception as e:
        import traceback

        print("========== OLLAMA ERROR ==========")
        traceback.print_exc()
        print("==================================")

        explanation = (
            f"Imagem selecionada por similaridade semântica (OpenCLIP). "
            f"Contexto MCP aplicado: {mcp_context[:120]}... "
            f"[Ollama offline em {settings.OLLAMA_URL}]"
        )

    # ── 9. Montar resposta final ───────────────────────────────────────────────
    return {
        "query": query,
        "enriched_query": enriched_query,
        "explanation": explanation,
        "mcp_info": {
            "color_palette": color_palette,
            "google_trends": google_trends,
            "weather": weather_data
        },
        "results": results
    }