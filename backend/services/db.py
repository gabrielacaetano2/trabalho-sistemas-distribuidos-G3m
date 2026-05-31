import psycopg2
from psycopg2.extras import RealDictCursor
import time
from config import settings

def get_db_connection():
    """Tenta conectar ao PostgreSQL com retry para aguardar o banco iniciar"""
    retries = 5
    while retries > 0:
        try:
            conn = psycopg2.connect(settings.DATABASE_URL, cursor_factory=RealDictCursor)
            return conn
        except psycopg2.OperationalError as e:
            print(f"Postgres nao esta pronto ainda. Tentando novamente em 3 segundos... ({retries} tentativas restantes)")
            time.sleep(3)
            retries -= 1
    raise Exception("Nao foi possivel conectar ao PostgreSQL.")

def init_db():
    """Cria as tabelas necessarias no banco de dados se nao existirem"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Tabela de Imagens
    cur.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id SERIAL PRIMARY KEY,
            path VARCHAR(500) NOT NULL UNIQUE,
            filename VARCHAR(255) NOT NULL,
            description TEXT,
            author VARCHAR(100),
            date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Tabela de Regras de Design/Marca (para o RAG Contextual)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS brand_rules (
            id SERIAL PRIMARY KEY,
            category VARCHAR(100) NOT NULL,
            content TEXT NOT NULL
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("PostgreSQL tabelas inicializadas com sucesso.")

def add_image_metadata(path: str, filename: str, description: str, author: str):
    """Insere ou atualiza os metadados de uma imagem"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO images (path, filename, description, author)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (path) DO UPDATE 
            SET filename = EXCLUDED.filename,
                description = EXCLUDED.description,
                author = EXCLUDED.author
            RETURNING id;
        """, (path, filename, description, author))
        row = cur.fetchone()
        conn.commit()
        return row['id']
    finally:
        cur.close()
        conn.close()

def get_image_metadata(image_id: int):
    """Busca os metadados de uma imagem por ID"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM images WHERE id = %s;", (image_id,))
        return cur.fetchone()
    finally:
        cur.close()
        conn.close()

def get_all_images():
    """Busca todas as imagens cadastradas"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM images ORDER BY id DESC;")
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()

def get_brand_rules(category: str = None):
    """Recupera regras de design filtrando opcionalmente por categoria"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        if category:
            cur.execute("SELECT content FROM brand_rules WHERE category = %s;", (category,))
        else:
            cur.execute("SELECT content FROM brand_rules;")
        rows = cur.fetchall()
        return [row['content'] for row in rows]
    finally:
        cur.close()
        conn.close()

def add_brand_rule(category: str, content: str):
    """Adiciona uma nova regra de marca para RAG"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO brand_rules (category, content)
            VALUES (%s, %s)
            RETURNING id;
        """, (category, content))
        row = cur.fetchone()
        conn.commit()
        return row['id']
    finally:
        cur.close()
        conn.close()
