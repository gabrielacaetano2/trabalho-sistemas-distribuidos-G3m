"""
G3m Orchestrator Service — FastAPI
Expoe o fluxo de orquestracao LangChain como servico separado.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .services.orchestrator import orchestrate_search
from .services.db import init_db
from .services.vector_db import init_vector_db


app = FastAPI(
    title="G3m Orchestrator",
    description=(
        "Servico dedicado de orquestracao LangChain para busca semantica "
        "e geracao RAG do G3m."
    ),
    version="1.0.0"
)


class OrchestrateRequest(BaseModel):
    query: str
    category: str = "corporativo"


@app.on_event("startup")
async def startup_event():
    print("[Orchestrator Service] Inicializando servicos distribuidos...")
    init_db()
    init_vector_db()
    print("[Orchestrator Service] Servicos prontos.")


@app.post("/orchestrate")
async def orchestrate(req: OrchestrateRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="A query nao pode estar vazia.")

    try:
        return await orchestrate_search(query=query, category=req.category)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no orquestrador: {str(e)}")
