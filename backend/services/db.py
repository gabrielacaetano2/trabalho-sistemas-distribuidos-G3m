import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
import time
import os
from config import settings

USE_SQLITE = False
SQLITE_DB_PATH = "data/g3m.db"

def get_db_connection():
    """Tenta conectar ao PostgreSQL. Se falhar, usa SQLite local como fallback"""
    global USE_SQLITE
    if USE_SQLITE:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
        
    try:
        conn = psycopg2.connect(settings.DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception:
        print(f"[FALLBACK] PostgreSQL offline. Utilizando SQLite local em: {SQLITE_DB_PATH}")
        USE_SQLITE = True
        # Garantir diretório criado
        os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

def init_db():
    """Cria as tabelas necessarias no banco de dados se nao existirem"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        if USE_SQLITE:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    description TEXT,
                    author TEXT,
                    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS brand_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL
                );
            """)
        else:
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS brand_rules (
                    id SERIAL PRIMARY KEY,
                    category VARCHAR(100) NOT NULL,
                    content TEXT NOT NULL
                );
            """)
        conn.commit()
        print("Tabelas SQL inicializadas.")
    finally:
        cur.close()
        conn.close()

def add_image_metadata(path: str, filename: str, description: str, author: str):
    """Insere ou atualiza os metadados de uma imagem"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        if USE_SQLITE:
            # SQLite UPSERT sintaxe compatível
            cur.execute("""
                INSERT INTO images (path, filename, description, author)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (path) DO UPDATE 
                SET filename = EXCLUDED.filename,
                    description = EXCLUDED.description,
                    author = EXCLUDED.author
                RETURNING id;
            """, (path, filename, description, author))
            row = cur.fetchone()
            conn.commit()
            if row:
                return row[0]
            # Se retornar vazio, buscar id pelo path
            cur.execute("SELECT id FROM images WHERE path = ?;", (path,))
            return cur.fetchone()[0]
        else:
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
        if USE_SQLITE:
            cur.execute("SELECT * FROM images WHERE id = ?;", (image_id,))
            row = cur.fetchone()
            if row:
                return {k: row[k] for k in row.keys()}
            return None
        else:
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
        if USE_SQLITE:
            cur.execute("SELECT * FROM images ORDER BY id DESC;")
            rows = cur.fetchall()
            return [{k: row[k] for k in row.keys()} for row in rows]
        else:
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
        if USE_SQLITE:
            if category:
                cur.execute("SELECT content FROM brand_rules WHERE category = ?;", (category,))
            else:
                cur.execute("SELECT content FROM brand_rules;")
            rows = cur.fetchall()
            return [row[0] for row in rows]
        else:
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
        if USE_SQLITE:
            cur.execute("""
                INSERT INTO brand_rules (category, content)
                VALUES (?, ?)
                RETURNING id;
            """, (category, content))
            row = cur.fetchone()
            conn.commit()
            if row:
                return row[0]
            cur.execute("SELECT MAX(id) FROM brand_rules;")
            return cur.fetchone()[0]
        else:
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

