import os
import sys
import urllib.request
import time

# CORRIGIDO: removido sys.path.append que adicionava /app/backend ao path.
# Isso fazia o Python importar services.ai como pacote top-level, quebrando
# os imports relativos (from ..config import settings) dentro de ai.py.
# PYTHONPATH=/app já está definido no Dockerfile, então basta usar
# os imports absolutos com o prefixo backend.* abaixo.

from backend.services.db import get_db_connection, add_image_metadata, add_brand_rule
from backend.services.vector_db import init_vector_db, upsert_image_vector
from backend.services.ai import generate_image_embedding

# Regras de design padrão para popular o banco relacional (RAG)
DEFAULT_RULES = [
    {
        "category": "corporativo",
        "content": "As imagens corporativas devem transmitir cooperacao, profissionalismo e sobriedade. Use paletas de cores frias com predominancia de azul escuro, cinza e branco. Evite filtros excessivamente quentes ou saturados."
    },
    {
        "category": "tecnologia",
        "content": "O estilo visual de tecnologia deve enfatizar inovacao e modernidade. Paletas escuras (Dark Mode) contrastando com neon (azul eletrico, verde ou violeta) sao altamente recomendadas. Elementos de telas, codigos e servidores sao bem-vindos."
    },
    {
        "category": "retro",
        "content": "O tema retro deve evocar nostalgia e criatividade. Recomendado o uso de tons terrosos, laranjas quentes, amarelos nostalgicos e texturas organicas ou granuladas que lembrem fotografia analoga."
    },
    {
        "category": "minimalista",
        "content": "O design minimalista preza pela simplicidade absoluta, limpeza visual e grandes espacos vazios. Use paletas neutras (bege, off-white, cinza claro) e foque em um unico assunto centralizado ou padroes geometricos simples."
    }
]

# Imagens de exemplo para download e indexação
SAMPLE_IMAGES = [
    {
        "filename": "office_collaboration.jpg",
        "url": "https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=800&q=80",
        "description": "Equipe de profissionais trabalhando juntos em volta de uma mesa de escritorio corporativo discutindo projetos em notebooks.",
        "author": "Annie Spratt"
    },
    {
        "filename": "tech_workspace.jpg",
        "url": "https://images.unsplash.com/photo-1517694712202-14dd9538aa97?w=800&q=80",
        "description": "Notebook aberto exibindo linhas de codigo de programacao em uma tela escura, em cima de uma mesa de madeira moderna.",
        "author": "Ilya Pavlov"
    },
    {
        "filename": "server_datacenter.jpg",
        "url": "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=800&q=80",
        "description": "Servidores de rede piscando luzes azuis e verdes em um rack de datacenter tecnologico moderno.",
        "author": "Ian Battaglia"
    },
    {
        "filename": "retro_camera.jpg",
        "url": "https://images.unsplash.com/photo-1516035069371-29a1b244cc32?w=800&q=80",
        "description": "Camera fotografica antiga analogica retro de filme deitada sobre um fundo de cores terrosas e vintage.",
        "author": "Alexander Andrews"
    },
    {
        "filename": "minimalist_interior.jpg",
        "url": "https://images.unsplash.com/photo-1499951360447-b19be8fe80f5?w=800&q=80",
        "description": "Ambiente de sala minimalista com uma mesa de trabalho limpa contendo apenas uma xicara de cafe e um computador em tons pasteis e neutros.",
        "author": "Domenico Loia"
    },
    {
        "filename": "creative_brainstorm.jpg",
        "url": "https://images.unsplash.com/photo-1531403009284-440f080d1e12?w=800&q=80",
        "description": "Designers criativos colando post-its coloridos em uma parede branca de vidro durante brainstorm de ideias.",
        "author": "Startups Stock Photos"
    }
]

def seed_database():
    print("=== INICIANDO SEMENTE DE DADOS (SEED) ===")
    
    # 1. Conectar e popular PostgreSQL Brand Rules
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Limpar regras anteriores para evitar duplicados no teste
        cur.execute("TRUNCATE TABLE brand_rules RESTART IDENTITY CASCADE;")
        conn.commit()
        print("Tabela brand_rules limpa.")
        
        for rule in DEFAULT_RULES:
            add_brand_rule(rule["category"], rule["content"])
        print(f"Inseridas {len(DEFAULT_RULES)} regras de marca no PostgreSQL.")
    except Exception as e:
        print(f"Erro ao inserir regras: {e}")
    finally:
        cur.close()
        conn.close()

    # 2. Inicializar colecao no Qdrant
    init_vector_db()

    # 3. Criar pasta de imagens se nao existir
    images_dir = "/data/images"
    os.makedirs(images_dir, exist_ok=True)

    # 4. Baixar e Indexar Imagens
    for idx, img_info in enumerate(SAMPLE_IMAGES):
        dest_path = os.path.join(images_dir, img_info["filename"])
        
        # Download da imagem caso nao exista no disco
        if not os.path.exists(dest_path):
            print(f"Baixando imagem [{idx+1}/{len(SAMPLE_IMAGES)}]: {img_info['filename']}...")
            try:
                # Customizar User-Agent para evitar bloqueio do servidor Unsplash
                req = urllib.request.Request(
                    img_info["url"], 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"Download concluído: {img_info['filename']}")
            except Exception as e:
                print(f"Erro ao baixar {img_info['filename']}: {e}")
                continue
        else:
            print(f"Imagem {img_info['filename']} ja existe no disco. Pulando download.")

        # Indexacao
        try:
            print(f"Gerando embeddings (OpenCLIP) para {img_info['filename']}...")
            vector = generate_image_embedding(dest_path)
            
            print(f"Salvando metadados no Postgres...")
            img_id = add_image_metadata(
                path=dest_path,
                filename=img_info["filename"],
                description=img_info["description"],
                author=img_info["author"]
            )
            
            print(f"Inserindo vetor no Qdrant (ID: {img_id})...")
            upsert_image_vector(image_id=img_id, vector=vector)
            
            print(f"Indexacao bem sucedida para a imagem ID {img_id}!\n")
        except Exception as e:
            print(f"Falha ao indexar {img_info['filename']}: {e}\n")
            
    print("=== PROCESSO DE SEED CONCLUÍDO COM SUCESSO! ===")

if __name__ == "__main__":
    # Aguarda 5 segundos para garantir que os outros containers estejam totalmente prontos
    print("Aguardando inicializacao dos bancos...")
    time.sleep(5)
    seed_database()