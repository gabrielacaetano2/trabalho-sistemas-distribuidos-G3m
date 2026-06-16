"""
Orquestrador G3m — camada LangChain entre o API Gateway e os serviços distribuídos.

Fluxo:
query
    → MCP Agent
    → enriquecimento contextual
    → tradução PT→EN
    → embedding OpenCLIP
    → Qdrant
    → PostgreSQL
    → LangChain LCEL
    → Ollama
    → resposta final
"""

import asyncio
import os
import re
from typing import Optional

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from deep_translator import GoogleTranslator

from .ai import generate_text_embedding
from .vector_db import search_similar_images
from .db import get_image_metadata, get_brand_rules
from .mcp_client import call_mcp_tool
from ..config import settings


# ─────────────────────────────────────────────────────────────
# Prompt RAG
# ─────────────────────────────────────────────────────────────
EXPLANATION_PROMPT = PromptTemplate.from_template(
    """
Você é o 'G3m AI', assistente inteligente de busca de ativos visuais corporativos.

Query original:
"{query}"

Query enriquecida:
"{enriched_query}"

Imagens recuperadas:
{image_summaries}

Contexto MCP:
{mcp_context}

Diretrizes de marca:
{brand_rules}

Gere uma explicação curta em português explicando:
- por que as imagens foram selecionadas
- como elas se relacionam semanticamente com a busca
- se seguem as diretrizes da marca

Máximo: 3 frases.
"""
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
async def _skip_tool() -> dict:
    return {"status": "skip"}


def _clip_score_to_similarity(score: float) -> float:
    """
    Calibra score cosseno do CLIP para uma porcentagem de relevância.

    Scores text-image do clip-ViT-B-32 costumam parecer baixos em escala
    absoluta. Em bases pequenas, ~0.05 e abaixo é ruído; ~0.35 já é um match
    forte. Essa escala é só para exibição, mantendo o score bruto na resposta.
    """
    lower_bound = 0.05
    strong_match = 0.35

    normalized = (score - lower_bound) / (strong_match - lower_bound)

    return min(max(normalized, 0), 1) * 100


def _tokenize_search_text(text: str) -> set:
    normalized = text.lower()
    replacements = {
        "mãos": "maos",
        "mão": "mao",
        "humanos": "humana humano humanas humanos",
        "humano": "humana humano humanas humanos",
        "humanas": "humana humano humanas humanos",
        "pessoas": "pessoa pessoas",
        "pessoa": "pessoa pessoas",
        "braços": "bracos",
        "braço": "braco"
    }

    for source, target in replacements.items():
        normalized = normalized.replace(source, target)

    return set(re.findall(r"[a-z0-9]+", normalized))


def _description_match_score(query: str, description: str) -> float:
    query_tokens = _tokenize_search_text(query)

    if not query_tokens:
        return 0

    description_tokens = _tokenize_search_text(description)
    matches = query_tokens.intersection(description_tokens)

    return len(matches) / len(query_tokens)


def _should_use_colormind(query: str) -> bool:
    terms = [
        "cor",
        "cores",
        "paleta",
        "color",
        "palette",
        "colorido",
        "colorida"
    ]

    q = query.lower()

    return any(t in q for t in terms)


def _should_use_trends(query: str) -> bool:
    terms = [
        "tendencia",
        "tendencias",
        "trend",
        "trending",
        "editorial",
        "noticia",
        "viral",
        "campanha"
    ]

    q = query.lower()

    return any(t in q for t in terms)


def _should_use_weather(query: str) -> bool:
    terms = [
        "clima",
        "tempo",
        "chuva",
        "sol",
        "verao",
        "inverno",
        "outono",
        "primavera",
        "estacao"
    ]

    q = query.lower()

    return any(t in q for t in terms)


# ─────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────
async def orchestrate_search(query: str, category: str) -> dict:

    # ─────────────────────────────────────────────────────────
    # 1. MCP Agent
    # ─────────────────────────────────────────────────────────
    use_palette = True
    use_palette_in_query = _should_use_colormind(query)
    use_trends = True
    use_weather = _should_use_weather(query)

    print(
        "[Orchestrator] MCP decision -> "
        f"Colormind={'ON' if use_palette else 'OFF'}, "
        f"GoogleTrends={'ON' if use_trends else 'OFF'}, "
        f"Weather={'ON' if use_weather else 'OFF'}"
    )

    palette_task = (
        call_mcp_tool(
            "get_colormind_palette",
            {"category": category}
        )
        if use_palette else _skip_tool()
    )

    trends_task = (
        call_mcp_tool(
            "get_google_trends",
            {
                "keyword": query,
                "geo": "BR"
            }
        )
        if use_trends else _skip_tool()
    )

    weather_task = (
        call_mcp_tool(
            "get_weather",
            {
                "city": "São Paulo"
            }
        )
        if use_weather else _skip_tool()
    )

    palette_result, trends_result, weather_result = await asyncio.gather(
        palette_task,
        trends_task,
        weather_task
    )

    color_palette = (
        palette_result.get("result", {}).get("palette", [])
        if palette_result.get("status") == "success"
        else []
    )

    google_trends = (
        trends_result.get("result", {})
        if trends_result.get("status") == "success"
        else {}
    )

    weather_data = (
        weather_result.get("result", {})
        if weather_result.get("status") == "success"
        else {}
    )

    # ─────────────────────────────────────────────────────────
    # 2. Query Enrichment
    # ─────────────────────────────────────────────────────────
    enrichment_parts = [query]

    # Google Trends
    if google_trends.get("top_queries"):
        enrichment_parts.append(
            "Tendências relacionadas: "
            + ", ".join(google_trends["top_queries"])
        )

    # Paleta de cores
    if color_palette and use_palette_in_query:
        enrichment_parts.append(
            "Paleta visual: "
            + ", ".join(color_palette)
        )

    # Weather contextual
    if weather_data:

        condition = weather_data.get(
            "condition",
            ""
        ).lower()

        if "rain" in condition or "chuva" in condition:
            enrichment_parts.append(
                "ambiente chuvoso, iluminação fria, "
                "tons escuros, atmosfera urbana"
            )

        elif "sun" in condition or "sol" in condition:
            enrichment_parts.append(
                "ambiente iluminado, tons quentes, "
                "céu claro, atmosfera vibrante"
            )

        elif "cloud" in condition or "nublado" in condition:
            enrichment_parts.append(
                "céu nublado, iluminação suave, "
                "ambiente neutro"
            )

        else:
            enrichment_parts.append(
                f"Condição climática: {condition}"
            )

    enriched_query = " | ".join(enrichment_parts)

    print(f"[Orchestrator] Query enriquecida: {enriched_query}")

    # ─────────────────────────────────────────────────────────
    # 3. Tradução PT → EN
    # ─────────────────────────────────────────────────────────
    try:

        translated_query = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(enriched_query)

        print(
            f"[Orchestrator] Query traduzida: {translated_query}"
        )

    except Exception as e:
        raise RuntimeError(
            f"Erro ao traduzir query: {e}"
        )

    # ─────────────────────────────────────────────────────────
    # 4. Embedding OpenCLIP
    # ─────────────────────────────────────────────────────────
    try:

        query_vector = generate_text_embedding(
            translated_query
        )

        print(
            "[Orchestrator] Embedding gerado com sucesso."
        )

    except Exception as e:
        raise RuntimeError(
            f"Erro ao gerar embedding: {e}"
        )

    # ─────────────────────────────────────────────────────────
    # 5. Busca Vetorial Qdrant
    # ─────────────────────────────────────────────────────────
    hits = search_similar_images(
        query_vector,
        limit=6
    )

    if not hits:

        return {
            "query": query,
            "enriched_query": enriched_query,
            "results": [],
            "explanation": (
                "Nenhuma imagem semanticamente "
                "relevante encontrada."
            ),
            "mcp_info": {
                "color_palette": color_palette,
                "google_trends": google_trends,
                "weather": weather_data
            }
        }

    # ─────────────────────────────────────────────────────────
    # 6. Recuperar Metadados PostgreSQL
    # ─────────────────────────────────────────────────────────
    results = []

    metadata_tasks = [
        asyncio.to_thread(
            get_image_metadata,
            hit["image_id"]
        )
        for hit in hits
    ]

    metadata_list = await asyncio.gather(
        *metadata_tasks
    )

    for hit, meta in zip(hits, metadata_list):

        if not meta:
            continue

        filename = os.path.basename(
            meta["path"]
        )

        raw_score = round(float(hit["score"]), 4)
        text_score = _description_match_score(
            query,
            meta["description"] or ""
        )
        ranking_score = (raw_score * 0.75) + (text_score * 0.25)
        similarity = _clip_score_to_similarity(ranking_score)

        results.append({
            "id": meta["id"],
            "path": meta["path"],
            "filename": meta["filename"],
            "description": meta["description"],
            "author": meta["author"],
            "url": f"/images/{filename}",
            "score": raw_score,
            "text_score": round(text_score, 4),
            "ranking_score": round(ranking_score, 4),
            "similarity": round(
                similarity,
                2
            )
        })

    results.sort(
        key=lambda item: item["ranking_score"],
        reverse=True
    )

    # ─────────────────────────────────────────────────────────
    # 7. Brand Rules
    # ─────────────────────────────────────────────────────────
    brand_rules = get_brand_rules(category)

    rules_text = (
        "\n".join(
            [f"- {r}" for r in brand_rules]
        )
        if brand_rules
        else "- Nenhuma diretriz cadastrada."
    )

    # ─────────────────────────────────────────────────────────
    # 8. MCP Context
    # ─────────────────────────────────────────────────────────
    mcp_parts = []

    if color_palette:
        mcp_parts.append(
            "Paleta dominante: "
            + ", ".join(color_palette)
        )

    if google_trends.get("top_queries"):
        mcp_parts.append(
            "Tendências: "
            + ", ".join(google_trends["top_queries"])
        )

    if weather_data:
        mcp_parts.append(
            f"Clima atual: "
            f"{weather_data.get('condition', '')}, "
            f"{weather_data.get('temperature_c', '')}°C"
        )

    mcp_context = (
        "\n".join(mcp_parts)
        if mcp_parts
        else "Sem contexto MCP."
    )

    # ─────────────────────────────────────────────────────────
    # 9. LangChain + Ollama
    # ─────────────────────────────────────────────────────────
    image_summaries = "\n".join([
        f"- {img['description']} | Autor: {img['author']}"
        for img in results
    ])

    try:

        print(
            f"[Orchestrator] Chamando Ollama "
            f"({settings.OLLAMA_MODEL})..."
        )

        llm = OllamaLLM(
            base_url=settings.OLLAMA_URL,
            model=settings.OLLAMA_MODEL,
            timeout=30
        )

        chain = (
            EXPLANATION_PROMPT
            | llm
            | StrOutputParser()
        )

        explanation = await chain.ainvoke({
            "query": query,
            "enriched_query": enriched_query,
            "image_summaries": image_summaries,
            "mcp_context": mcp_context,
            "brand_rules": rules_text
        })

        print(
            "[Orchestrator] Explicação gerada."
        )

    except Exception as e:

        import traceback

        print("========== OLLAMA ERROR ==========")
        traceback.print_exc()
        print("==================================")

        explanation = (
            "Resultados encontrados por "
            "similaridade semântica OpenCLIP."
        )

    # ─────────────────────────────────────────────────────────
    # 10. Resposta Final
    # ─────────────────────────────────────────────────────────
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
