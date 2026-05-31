"""
Orquestrador G3m — camada LangChain entre o API Gateway e os serviços distribuídos.

Fluxo:
  query → [Cache] → [Embedding OpenCLIP] → [Qdrant] → [PostgreSQL]
        → [MCP Agent (Color, Trends, Weather) — paralelo]
        → [LangChain LCEL: Prompt | Ollama LLM | Parser]
        → resposta enriquecida → [Cache]
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
from .cache import get_cached_search, set_cached_search
from .mcp_client import call_mcp_tool
from ..config import settings


# ─── Prompt RAG usado pelo LangChain ──────────────────────────────────────────
EXPLANATION_PROMPT = PromptTemplate.from_template(
    """Você é o 'G3m AI', assistente inteligente de busca de ativos visuais corporativos.

O usuário buscou por: "{query}"

Imagem mais relevante encontrada:
- Arquivo: {filename}
- Descrição: {description}
- Autor: {author}

Contexto enriquecido pelas ferramentas MCP:
{mcp_context}

Diretrizes de Marca (RAG):
{brand_rules}

Com base nesses dados, gere uma justificativa em até 3 frases em português explicando \
por que esta imagem atende à busca e se respeita as diretrizes da marca. \
Seja profissional, conciso e direto."""
)


async def _skip_tool() -> dict:
    """Placeholder assíncrono quando a tool MCP não é aplicável."""
    return {"status": "skip"}


async def orchestrate_search(query: str, category: str) -> dict:
    """
    Ponto de entrada do Orquestrador LangChain.
    Coordena todos os serviços distribuídos e retorna o pacote de resposta enriquecido.
    """

    # ── 1. Cache (Redis) ──────────────────────────────────────────────────────
    cached = get_cached_search(query)
    if cached:
        print(f"[Orchestrator] Cache HIT para: '{query}'")
        return cached

    # ── 2. Embedding (OpenCLIP via SentenceTransformer) ───────────────────────
    try:
        query_vector = generate_text_embedding(query)
        print(f"[Orchestrator] Embedding gerado para: '{query}'")
    except Exception as e:
        raise RuntimeError(f"[Orchestrator] Falha ao gerar embedding: {e}")

    # ── 3. Busca Vetorial (Qdrant) ────────────────────────────────────────────
    hits = search_similar_images(query_vector, limit=6)
    if not hits:
        return {
            "query": query,
            "explanation": "Nenhuma imagem correspondente encontrada na base vetorial.",
            "mcp_info": {},
            "results": []
        }

    # ── 4. Metadados Estruturados (PostgreSQL) ────────────────────────────────
    results = []
    top_metadata: Optional[dict] = None

    for hit in hits:
        meta = get_image_metadata(hit["image_id"])
        if meta:
            filename = os.path.basename(meta["path"])
            results.append({
                "id": meta["id"],
                "path": meta["path"],
                "filename": meta["filename"],
                "description": meta["description"],
                "author": meta["author"],
                "url": f"/images/{filename}",
                "score": float(hit["score"])
            })
            if top_metadata is None:
                top_metadata = meta

    # ── 5. MCP Agent — 3 ferramentas em paralelo ─────────────────────────────
    # (LangChain coordena via asyncio.gather; cada tool é um componente distribuído)
    color_task = (
        call_mcp_tool("get_color_palette", {"image_path": top_metadata["path"], "num_colors": 5})
        if top_metadata else _skip_tool()
    )
    trends_task = call_mcp_tool("get_design_trends", {"category": category})
    weather_task = call_mcp_tool("get_weather", {"city": "São Paulo"})

    print("[Orchestrator] Chamando MCP Agent (Color + Trends + Weather) em paralelo...")
    color_result, trends_result, weather_result = await asyncio.gather(
        color_task, trends_task, weather_task
    )

    color_palette = (
        color_result.get("result", {}).get("palette", [])
        if color_result.get("status") == "success" else []
    )
    design_trends = (
        trends_result.get("result", {})
        if trends_result.get("status") == "success" else {}
    )
    weather_data = (
        weather_result.get("result", {})
        if weather_result.get("status") == "success" else {}
    )

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
    if design_trends:
        concepts = ", ".join(design_trends.get("key_concepts", []))
        vibe = design_trends.get("general_vibe", "")
        mcp_parts.append(f"Tendências de design ({category}): {concepts} — Vibe: {vibe}")
    if weather_data:
        mcp_parts.append(
            f"Clima atual em {weather_data.get('city', 'São Paulo')}: "
            f"{weather_data.get('condition', '')} {weather_data.get('temperature_c', '')}°C, "
            f"umidade {weather_data.get('humidity_pct', '')}%"
        )

    mcp_context = "\n".join(mcp_parts) if mcp_parts else "Ferramentas MCP indisponíveis."

    # ── 8. LangChain LCEL: Prompt → Ollama (Llama 3) → Parser ────────────────
    top_desc = top_metadata["description"] if top_metadata else "Sem descrição"
    top_filename = top_metadata.get("filename", "") if top_metadata else ""
    top_author = top_metadata.get("author", "Desconhecido") if top_metadata else ""

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
            "filename": top_filename,
            "description": top_desc,
            "author": top_author,
            "mcp_context": mcp_context,
            "brand_rules": rules_text
        })
        print("[Orchestrator] Explicação LLM gerada com sucesso.")

    except Exception as e:
        print(f"[Orchestrator] LLM indisponível ({e}) — usando fallback.")
        explanation = (
            f"Imagem selecionada por similaridade semântica (OpenCLIP). "
            f"Contexto MCP aplicado: {mcp_context[:120]}... "
            f"[Ollama offline em {settings.OLLAMA_URL}]"
        )

    # ── 9. Montar resposta final ───────────────────────────────────────────────
    response = {
        "query": query,
        "explanation": explanation,
        "mcp_info": {
            "color_palette": color_palette,
            "design_trends": design_trends,
            "weather": weather_data
        },
        "results": results
    }

    # ── 10. Salvar no Cache (Redis) por 5 minutos ─────────────────────────────
    set_cached_search(query, response, ttl=300)
    print(f"[Orchestrator] Resposta cacheada para: '{query}'")

    return response
