"""
G3m API Gateway — FastAPI
Recebe requisições do Frontend e delega ao Orquestrador LangChain.
Arquitetura:
  Frontend → Nginx → [FastAPI Gateway] → Orchestrator (LangChain)
                                       → Redis (cache)
                   → MCP Agent (SSE)
                   → Embedding + Qdrant + PostgreSQL + Ollama
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import os

# Config
from .config import settings

# Serviços auxiliares para indexação e consulta direta
from .services.db import init_db, add_image_metadata, get_image_metadata, get_all_images
from .services.vector_db import init_vector_db, upsert_image_vector
from .services.ai import generate_image_embedding
from .services.mcp_client import list_mcp_tools
from .services.cache import get_cached_search, set_cached_search

# ─────────────────────────────────────────────────────────────────────────────
# App FastAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="G3m API Gateway",
    description=(
        "Orquestrador distribuído de busca semântica de imagens usando "
        "RAG (Retrieval-Augmented Generation) e MCP (Model Context Protocol). "
        "Arquitetura: Frontend → Nginx → FastAPI → LangChain Orchestrator → "
        "Qdrant + PostgreSQL + Redis + MCP Agent + Ollama/Llama 3"
    ),
    version="2.0.0"
)

# CORS — permite chamadas do frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Imagens estáticas servidas diretamente pelo backend (Nginx faz o proxy em prod)
images_dir = "/data/images" if os.path.exists("/data/images") else "data/images"
os.makedirs(images_dir, exist_ok=True)
app.mount("/images", StaticFiles(directory=images_dir), name="images")


# ─────────────────────────────────────────────────────────────────────────────
# Modelos de Request
# ─────────────────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    category: str = "corporativo"

class IndexRequest(BaseModel):
    filename: str
    description: str
    author: str = "Desconhecido"


# ─────────────────────────────────────────────────────────────────────────────
# Startup — inicializa bancos de dados
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("[Gateway] Inicializando serviços distribuídos...")
    init_db()        # PostgreSQL (ou SQLite fallback)
    init_vector_db() # Qdrant (ou in-memory fallback)
    print("[Gateway] Todos os serviços inicializados com sucesso!")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    """Health check do Gateway."""
    return {
        "status": "healthy",
        "service": "G3m API Gateway",
        "version": "2.0.0",
        "orchestrator": "LangChain",
        "mcp_protocol": "SSE",
        "llm": "Ollama / Llama 3"
    }


@app.get("/tools")
async def list_available_mcp_tools():
    """Lista as ferramentas disponíveis no servidor MCP via protocolo MCP/SSE."""
    tools = await list_mcp_tools()
    return {"mcp_server_tools": tools}


@app.post("/search")
async def search_images(req: SearchRequest):
    """
    Endpoint principal de busca semântica.
    Delega ao Orquestrador LangChain que coordena:

    CORRIGIDO: fluxo atualizado para refletir o diagrama de arquitetura real.
    Cache → MCP Agent (enriquece query) → Embedding (OpenCLIP) → Qdrant → PostgreSQL → LLM → Resposta
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="A query não pode estar vazia.")

    cache_key = f"search-v6:{req.category}:{query}"
    cached = get_cached_search(cache_key)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.ORCHESTRATOR_URL}/orchestrate",
                json={"query": query, "category": req.category}
            )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )
        result = response.json()
        set_cached_search(cache_key, result, ttl=300)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao chamar orquestrador: {str(e)}")


@app.post("/index")
async def index_image(req: IndexRequest):
    """
    Indexa imagem utilizando o embedding visual do arquivo.
    """

    local_path = os.path.join(
        "data",
        "images",
        req.filename
    )

    docker_path = os.path.join(
        "/data",
        "images",
        req.filename
    )

    image_path = (
        local_path
        if os.path.exists(local_path)
        else docker_path
    )

    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail=f"Imagem '{req.filename}' não encontrada."
        )

    try:

        vector = generate_image_embedding(image_path)

        image_id = add_image_metadata(
            path=image_path,
            filename=req.filename,
            description=req.description,
            author=req.author
        )

        upsert_image_vector(
            image_id=image_id,
            vector=vector,
            filename=req.filename,
            description=req.description,
            author=req.author,
            path=image_path
        )

        return {
            "status": "success",
            "message": f"Imagem '{req.filename}' indexada com sucesso.",
            "id": image_id
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao indexar: {str(e)}"
        )
    
@app.get("/images")
def get_images():
    """Retorna metadados de todas as imagens indexadas."""
    try:
        db_images = get_all_images()
        formatted = []
        for img in db_images:
            filename = os.path.basename(img["path"])
            formatted.append({
                "id": img["id"],
                "filename": img["filename"],
                "description": img["description"],
                "author": img["author"],
                "url": f"/images/{filename}",
                "date_created": img.get("date_created")
            })
        return {"images": formatted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
