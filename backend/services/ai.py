from sentence_transformers import SentenceTransformer
from PIL import Image
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

# CORRIGIDO: função generate_explanation() removida.
# Ela chamava o Ollama diretamente via HTTP (legado) e não era usada por ninguém.
# A geração de explicação via LangChain LCEL está corretamente implementada
# em backend/services/orchestrator.py, que usa:
#   chain = EXPLANATION_PROMPT | Ollama(...) | StrOutputParser()