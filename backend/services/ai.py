from sentence_transformers import SentenceTransformer
from PIL import Image
import httpx
from ..config import settings

_clip_model = None

def get_clip_model():
    """Garante carregamento lazy (sob demanda) do modelo OpenCLIP"""
    global _clip_model
    if _clip_model is None:
        print("Carregando modelo OpenCLIP clip-ViT-B-32 na memoria...")
        _clip_model = SentenceTransformer('clip-ViT-B-32')
        print("Modelo OpenCLIP carregado!")
    return _clip_model

def generate_text_embedding(text: str) -> list:
    """Gera embedding vetorial a partir de um texto de busca"""
    model = get_clip_model()
    embedding = model.encode(text)
    return embedding.tolist()

def generate_image_embedding(image_path: str) -> list:
    """Gera embedding vetorial a partir do arquivo de imagem"""
    model = get_clip_model()
    img = Image.open(image_path)
    embedding = model.encode(img)
    return embedding.tolist()

async def generate_explanation(query: str, image_description: str, brand_rules: list, metadata: dict, mcp_context: str = None) -> str:
    """Invoca o Ollama local (Llama 3) para gerar uma explicacao RAG contextualizada"""
    url = f"{settings.OLLAMA_URL}/api/generate"
    
    rules_text = "\n".join([f"- {rule}" for rule in brand_rules]) if brand_rules else "- Nenhuma diretriz de marca cadastrada."
    
    mcp_section = ""
    if mcp_context:
        mcp_section = f"\nInformacoes adicionais extraidas por ferramentas (MCP):\n{mcp_context}\n"
        
    prompt = f"""
Voce e o 'G3m AI', um assistente inteligente de busca de ativos visuais corporativos.
O usuario buscou por: "{query}"

Encontramos uma imagem correspondente no banco vetorial. Aqui estao os dados dela:
- Nome do arquivo: {metadata.get('filename')}
- Descricao cadastrada: {image_description}
- Autor: {metadata.get('author', 'Desconhecido')}
{mcp_section}
Diretrizes e Regras de Estilo da Marca Cadastradas:
{rules_text}

Com base nesses dados, gere uma justificativa curta (no maximo 3 frases) em portugues explicando por que esta imagem atende a busca do usuario e se ela respeita as regras de estilo da marca. Seja profissional, conciso e direto.
"""

    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"Imagem selecionada via busca semantica. Nota de similaridade alta. (Servico Ollama retornou erro {response.status_code})"
    except Exception as e:
        print(f"Alerta: Nao foi possivel conectar ao Ollama em {url}. Erro: {e}")
        return f"Imagem selecionada por similaridade semantica (OpenCLIP). [Nota: O Llama 3 local esta offline ou nao foi inicializado no Ollama em {settings.OLLAMA_URL}]"
