from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.config import settings

import os

_use_local_qdrant = False


def get_qdrant_client():
    """
    Conecta ao Qdrant.
    Fallback local caso o servidor esteja offline.
    """

    global _use_local_qdrant

    if _use_local_qdrant:
        return QdrantClient(path="data/qdrant_local")

    try:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            timeout=3.0
        )

        client.get_collections()

        return client

    except Exception:
        print(
            "[FALLBACK] Qdrant offline. Utilizando armazenamento local."
        )

        _use_local_qdrant = True

        os.makedirs("data/qdrant_local", exist_ok=True)

        return QdrantClient(path="data/qdrant_local")


def init_vector_db():
    """
    Cria coleção vetorial se não existir.
    """

    client = get_qdrant_client()

    collection_name = settings.COLLECTION_NAME

    try:
        client.get_collection(collection_name)

        print(f"Coleção '{collection_name}' já existe.")

    except (UnexpectedResponse, Exception):

        print(f"Criando coleção '{collection_name}'...")

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=512,
                distance=models.Distance.COSINE
            )
        )

        print("Coleção criada com sucesso.")


def upsert_image_vector(
    image_id: int,
    vector: list,
    filename: str,
    description: str,
    author: str,
    path: str
):
    """
    Salva vetor + payload completo.
    """

    client = get_qdrant_client()

    client.upsert(
        collection_name=settings.COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=image_id,
                vector=vector,
                payload={
                    "image_id": image_id,
                    "filename": filename,
                    "description": description,
                    "author": author,
                    "path": path
                }
            )
        ]
    )


def search_similar_images(
    vector: list,
    limit: int = 10,
    score_threshold: float = 0.25,
    fallback_threshold: float = 0.05
):
    """
    Busca vetorial semântica.
    """

    client = get_qdrant_client()

    try:
        result = _query_points(
            client=client,
            vector=vector,
            limit=limit,
            score_threshold=score_threshold
        )

        if not result.points and fallback_threshold < score_threshold:
            result = _query_points(
                client=client,
                vector=vector,
                limit=limit,
                score_threshold=fallback_threshold
            )

        return _format_points(result.points)

    except Exception as e:
        print(f"Erro Qdrant: {e}")

        return []


def _query_points(
    client,
    vector: list,
    limit: int,
    score_threshold: float
):
    return client.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=vector,
        limit=limit,
        score_threshold=score_threshold
    )


def _format_points(points):
    formatted = []

    for hit in points:
        formatted.append({
            "image_id": hit.payload.get("image_id"),
            "filename": hit.payload.get("filename"),
            "description": hit.payload.get("description"),
            "author": hit.payload.get("author"),
            "path": hit.payload.get("path"),
            "score": float(hit.score)
        })

    return formatted
