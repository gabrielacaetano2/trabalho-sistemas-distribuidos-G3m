FROM python:3.10-slim

WORKDIR /app

# Instalar dependências de sistema necessárias para psycopg2 e pillow
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copiar scripts e rodar pre-download do modelo CLIP
COPY backend/scripts/ /app/backend/scripts/
RUN python /app/backend/scripts/download_models.py

# Copiar o restante da aplicação
COPY backend/ /app/backend/

# Garantir import do pacote backend
ENV PYTHONPATH=/app

# Criar os diretórios de dados locais caso não existam
RUN mkdir -p /data/images /data/documents

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
