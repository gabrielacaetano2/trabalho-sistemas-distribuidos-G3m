import sys
from sentence_transformers import SentenceTransformer

def download_models():
    print("Iniciando download do modelo clip-ViT-B-32...")
    # clip-ViT-B-32 é o modelo multimodal padrão para converter texto/imagem em embeddings
    try:
        model = SentenceTransformer('clip-ViT-B-32')
        print("Modelo clip-ViT-B-32 baixado e cacheado com sucesso!")
    except Exception as e:
        print(f"Erro ao baixar modelo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_models()
