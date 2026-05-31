import os

class Settings:
    # Banco Relacional (Postgres)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://admin:admin_password@localhost:5432/g3m")

    # Banco Vetorial (Qdrant)
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME: str = "g3m_images"

    # Cache (Redis)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # MCP Server — endpoint SSE do protocolo MCP
    # Em Docker: http://mcp-server:8001/sse  |  Dev local: http://localhost:8001/sse
    MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://localhost:8001/sse")

    # LLM (Ollama local)
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

settings = Settings()

