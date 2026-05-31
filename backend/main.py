from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import os

from services.db import init_db, add_image_metadata, get_image_metadata, get_brand_rules, get_all_images
from services.vector_db import init_vector_db, upsert_image_vector, search_similar_images
from services.cache import get_cached_search, set_cached_search
from services.ai import generate_text_embedding, generate_image_embedding, generate_explanation
from services.mcp_client import call_mcp_tool, list_mcp_tools

app = FastAPI(
    title="G3m API Gateway & Search Orchestrator",
    description="Orquestrador central de busca semantica distribuida utilizando RAG e MCP",
    version="1.0.0"
)

# Configurar CORS para permitir chamadas do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar pasta de imagens estaticas
images_dir = "data/images"
os.makedirs(images_dir, exist_ok=True)
app.mount("/images", StaticFiles(directory=images_dir), name="images")


class SearchRequest(BaseModel):
    query: str
    category: str = "corporativo"  # categoria para regras de design e mcp trends

class IndexRequest(BaseModel):
    filename: str
    description: str
    author: str = "Desconhecido"

@app.on_event("startup")
async def startup_event():
    """Inicializa os servicos distribuidos na subida do servidor"""
    print("Inicializando bancos de dados...")
    init_db()
    init_vector_db()
    print("Servicos distribuidos inicializados com sucesso!")

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "service": "G3m Search Orchestrator",
        "mcp_connection": "configured",
        "qdrant_connection": "configured"
    }

@app.get("/tools")
async def list_available_mcp_tools():
    """Endpoint para inspecionar ferramentas disponiveis no MCP Server"""
    tools = await list_mcp_tools()
    return {"mcp_server_tools": tools}

@app.post("/search")
async def search_images(req: SearchRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="A query de busca nao pode estar vazia.")
    
    # 1. Verificar Cache (Redis)
    cached_result = get_cached_search(query)
    if cached_result:
        return cached_result
    
    # 2. Gerar Embedding do Texto de Busca (OpenCLIP)
    try:
        query_vector = generate_text_embedding(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar embedding do texto: {str(e)}")
        
    # 3. Busca por Similaridade Vetorial (Qdrant)
    hits = search_similar_images(query_vector, limit=6)
    if not hits:
        # Se nao houver resultados vetoriais, retornar payload vazio
        response_data = {
            "query": query,
            "explanation": "Nenhuma imagem correspondente encontrada na base vetorial.",
            "mcp_info": {},
            "results": []
        }
        return response_data
        
    # 4. Obter Metadados Estruturados das Imagens (PostgreSQL)
    results = []
    top_image_metadata = None
    for hit in hits:
        img_id = hit["image_id"]
        meta = get_image_metadata(img_id)
        if meta:
            # Normalizar caminhos para exibicao no frontend
            # O frontend acessa via /images/<filename> que o Nginx serve
            filename = os.path.basename(meta["path"])
            web_url = f"/images/{filename}"
            
            results.append({
                "id": meta["id"],
                "path": meta["path"],
                "filename": meta["filename"],
                "description": meta["description"],
                "author": meta["author"],
                "url": web_url,
                "score": float(hit["score"])
            })
            
            if not top_image_metadata:
                top_image_metadata = meta

    # 5. Executar ferramentas MCP em paralelo para o resultado principal
    mcp_tasks = []
    
    # Ferramenta 1: Extrair paleta de cores dominante da imagem mais similar
    if top_image_metadata:
        mcp_tasks.append(
            call_mcp_tool("get_color_palette", {"image_path": top_image_metadata["path"], "num_colors": 5})
        )
    else:
        mcp_tasks.append(asyncio.sleep(0, result={"status": "skip"}))
        
    # Ferramenta 2: Consultar tendencias de design para a categoria
    mcp_tasks.append(
        call_mcp_tool("get_design_trends", {"category": req.category})
    )
    
    # Aguarda a execucao distribuida das ferramentas MCP
    mcp_results = await asyncio.gather(*mcp_tasks)
    
    color_palette = []
    design_trends = {}
    
    if mcp_results[0].get("status") == "success":
        color_palette = mcp_results[0]["result"]["palette"]
        
    if mcp_results[1].get("status") == "success":
        design_trends = mcp_results[1]["result"]

    # 6. Recuperar Normas e Diretrizes da Marca (PostgreSQL - RAG Contexto)
    brand_rules = get_brand_rules(req.category)
    
    # 7. Formar contexto para o Llama 3 explicar a busca (RAG)
    mcp_context_str = f"Paleta de cores extraida da imagem: {', '.join(color_palette)}\n"
    if design_trends:
        mcp_context_str += f"Tendencias de design recomendadas: {', '.join(design_trends.get('key_concepts', []))} (Vibe: {design_trends.get('general_vibe')})"
        
    top_image_desc = top_image_metadata["description"] if top_image_metadata else "Nenhuma descricao"
    
    # Chamar Ollama com Llama 3
    explanation = await generate_explanation(
        query=query,
        image_description=top_image_desc,
        brand_rules=brand_rules,
        metadata=top_image_metadata or {},
        mcp_context=mcp_context_str
    )
    
    # 8. Montar o pacote de retorno final
    response_data = {
        "query": query,
        "explanation": explanation,
        "mcp_info": {
            "color_palette": color_palette,
            "design_trends": design_trends
        },
        "results": results
    }
    
    # 9. Salvar no Cache (Redis) por 5 minutos (300 segundos)
    set_cached_search(query, response_data, ttl=300)
    
    return response_data

@app.post("/index")
async def index_image(req: IndexRequest):
    """Indexa uma nova imagem na base vetorial e relacional"""
    # Presume que a imagem ja foi copiada para /data/images/
    image_path = os.path.join("/data/images", req.filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Arquivo de imagem nao encontrado em {image_path}. Certifique-se de que a imagem esta na pasta mapeada."
        )
        
    try:
        # 1. Gerar Embedding Multimodal (OpenCLIP)
        vector = generate_image_embedding(image_path)
        
        # 2. Gravar Metadados no PostgreSQL
        image_id = add_image_metadata(
            path=image_path,
            filename=req.filename,
            description=req.description,
            author=req.author
        )
        
        # 3. Salvar Vetor no Qdrant
        upsert_image_vector(image_id=image_id, vector=vector)
        
        return {
            "status": "success",
            "message": f"Imagem '{req.filename}' indexada com sucesso!",
            "id": image_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao indexar imagem: {str(e)}")

@app.get("/images")
def get_images():
    """Retorna metadados de todas as imagens indexadas"""
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
                "date_created": img["date_created"]
            })
        return {"images": formatted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
