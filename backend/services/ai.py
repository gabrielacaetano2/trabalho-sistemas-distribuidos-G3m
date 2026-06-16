from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np

_clip_model = None


def get_clip_model():
    """
    Carrega o modelo OpenCLIP apenas uma vez.
    """
    global _clip_model

    if _clip_model is None:
        print("Carregando modelo OpenCLIP...")
        _clip_model = SentenceTransformer("clip-ViT-B-32")
        print("Modelo OpenCLIP carregado!")

    return _clip_model


def normalize_embedding(embedding):
    """
    Normaliza vetor para melhorar similaridade cosseno.
    """
    embedding = np.array(embedding, dtype=np.float32)

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding.tolist()

    return (embedding / norm).tolist()


def generate_text_embedding(text: str) -> list:
    """
    Gera embedding textual normalizado.
    """
    model = get_clip_model()

    embedding = model.encode(
        text,
        convert_to_numpy=True
    )

    return normalize_embedding(embedding)


def generate_image_embedding(image_path: str) -> list:
    """
    Gera embedding visual da imagem.
    """
    model = get_clip_model()

    image = Image.open(image_path).convert("RGB")

    embedding = model.encode(
        image,
        convert_to_numpy=True
    )

    return normalize_embedding(embedding)


def generate_hybrid_embedding(
    image_path: str,
    description: str
) -> list:
    """
    Combina embedding visual + textual.
    """

    image_vector = generate_image_embedding(image_path)

    text_vector = generate_text_embedding(description)

    combined = []

    for i, t in zip(image_vector, text_vector):
        combined.append((i + t) / 2)

    return normalize_embedding(combined)