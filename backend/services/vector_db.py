from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from backend.config import settings
import time
import os

_use_local_qdrant = False

def get_qdrant_client():
    """Conecta ao Qdrant remoto. Se falhar, utiliza armazenamento local no disco"""
    global _use_local_qdrant
    if _use_local_qdrant:
        return QdrantClient(path="data/qdrant_local")
        
    try:
        # Tenta conectar rápido
        client = QdrantClient(url=settings.QDRANT_URL, timeout=2.0)
        client.get_collections()
        return client
    except Exception:
        print("[FALLBACK] Qdrant server offline. Utilizando Qdrant local (em-disco) na pasta: data/qdrant_local")
        _use_local_qdrant = True
        os.makedirs("data/qdrant_local", exist_ok=True)
        return QdrantClient(path="data/qdrant_local")

def init_vector_db():
    """Garante que a colecao no Qdrant para embeddings de imagens exista"""
    client = get_qdrant_client()
    collection_name = settings.COLLECTION_NAME
    
    try:
        # Tenta buscar a colecao
        client.get_collection(collection_name=collection_name)
        print(f"Colecao '{collection_name}' no Qdrant ja existe.")
    except (UnexpectedResponse, Exception):
        # Cria se nao existir (OpenCLIP clip-ViT-B-32 gera embeddings de 512 dimensoes)
        print(f"Criando colecao '{collection_name}' no Qdrant (dimensao: 512)...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=512,
                distance=models.Distance.COSINE
            )
        )
        print(f"Colecao '{collection_name}' criada com sucesso.")

def upsert_image_vector(image_id: int, vector: list):
    """Insere ou atualiza o vetor de embedding de uma imagem no Qdrant"""
    client = get_qdrant_client()
    client.upsert(
        collection_name=settings.COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=image_id,
                vector=vector,
                payload={"image_id": image_id}
            )
        ]
    )

def search_similar_images(vector: list, limit: int = 6):
    """Realiza busca vetorial baseada na similaridade cosseno"""
    client = get_qdrant_client()
    try:
        search_result = client.search(
            collection_name=settings.COLLECTION_NAME,
            query_vector=vector,
            limit=limit
        )
        return [
            {
                "image_id": hit.payload["image_id"],
                "score": hit.score
            }
            for hit in search_result
        ]
    except Exception as e:
        print(f"Erro ao buscar no Qdrant: {e}")
        return []

