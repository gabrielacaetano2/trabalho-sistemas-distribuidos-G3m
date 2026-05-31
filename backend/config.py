import os

class Settings:
    # Banco Relacional (Postgres)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://admin:admin_password@postgres:5432/g3m")
    
    # Banco Vetorial (Qdrant)
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    COLLECTION_NAME: str = "g3m_images"
    
    # Cache (Redis)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    
    # Integrações
    MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://mcp-server:8001")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

settings = Settings()
