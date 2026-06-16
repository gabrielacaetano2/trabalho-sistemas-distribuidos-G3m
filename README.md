# G3m — Motor de Busca de Imagens Semântico & Distribuído

<div align="center">

![G3m Banner](https://img.shields.io/badge/G3m-Smart%20Image%20Search-7c3aed?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0yMSAxNS41NVY0YTIgMiAwIDAgMC0yLTJINWEyIDIgMCAwIDAtMiAydjEyYTIgMiAwIDAgMCAyIDJoOC41NW0tLjU1IDEuNDVMMjEgMjFtLTQuNDUtNC40NUExNi41NSAxNi41NSAwIDAgMSAxNS41NSAyMUEzIDMgMCAxIDAgMTYuNSAxNi41eiIvPjwvc3ZnPg==)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com/)

[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-FF6B6B?style=flat-square&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Ollama](https://img.shields.io/badge/Ollama-LLM%20Local-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.com/)
[![OpenCLIP](https://img.shields.io/badge/OpenCLIP-ViT--B--32-FF9A3C?style=flat-square)](https://github.com/mlfoundations/open_clip)
[![MCP](https://img.shields.io/badge/MCP-SSE%20Protocol-7c3aed?style=flat-square)](https://modelcontextprotocol.io/)
[![Nginx](https://img.shields.io/badge/Nginx-Reverse%20Proxy-009639?style=flat-square&logo=nginx&logoColor=white)](https://nginx.org/)

**Trabalho Final — Sistemas Distribuídos · UFLA · 2026**

*Sistema distribuído de busca semântica de imagens corporativas com RAG, embeddings multimodais, orquestração LangChain e agentes externos via MCP.*

</div>

---

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura do Sistema](#arquitetura-do-sistema)
- [Tecnologias e Justificativas](#tecnologias-e-justificativas)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Fluxo de Orquestração](#fluxo-de-orquestração)
- [Serviços e Microsserviços](#serviços-e-microsserviços)
- [API Reference](#api-reference)
- [Pré-requisitos](#pré-requisitos)
- [Como Executar](#como-executar)
- [Indexando Imagens](#indexando-imagens)
- [Variáveis de Ambiente](#variáveis-de-ambiente)

---

## Visão Geral

O **G3m** é um sistema distribuído de busca semântica de imagens corporativas desenvolvido como protótipo RAG/MCP para a disciplina de Sistemas Distribuídos. A proposta central é ir além de uma busca textual simples: o sistema combina **embeddings multimodais (OpenCLIP)**, **banco vetorial (Qdrant)**, **metadados relacionais (PostgreSQL)**, **cache distribuído (Redis)**, **LLM local (Ollama/Qwen)** e **agentes externos via MCP (Model Context Protocol)** para produzir resultados de busca semanticamente enriquecidos, contextualizados por dados em tempo real (tendências, clima, paletas de cores) e validados por diretrizes de marca da empresa.

### O que o sistema faz, do ponto de vista do usuário

1. O usuário digita uma query em linguagem natural — por exemplo: *"equipe reunida no escritório moderno"*
2. O sistema consulta o cache Redis; se não houver resposta, aciona o **Orquestrador LangChain**
3. O orquestrador invoca paralelamente três **ferramentas MCP externas**: paleta de cores Colormind, termos em alta no Google Trends e condição climática atual (Open-Meteo)
4. Os dados retornados enriquecem a query original, que é traduzida PT→EN via `deep-translator`
5. O modelo OpenCLIP (`clip-ViT-B-32`) gera um embedding vetorial normalizado da query enriquecida
6. O Qdrant busca as imagens mais similares por similaridade de cosseno
7. Os metadados das imagens são recuperados do PostgreSQL em paralelo via `asyncio`
8. Um score híbrido (75% CLIP + 25% correspondência textual na descrição) reordena os resultados
9. Um prompt RAG é enviado ao LLM local (Ollama/Qwen) para gerar uma justificativa contextualizada
10. A resposta final é cacheada no Redis e retornada ao frontend React

---

## Arquitetura do Sistema

```
                         ┌─────────────────────────────────────────────────────────┐
                         │                   Docker Network: g3m-network            │
                         │                                                           │
  Usuário ──► Nginx :80  │  ┌──────────────┐         ┌──────────────────────────┐  │
               │         │  │   Frontend   │         │    API Gateway (FastAPI)  │  │
               ├──────►  │  │  React/Vite  │ ──────► │     backend :8000        │  │
               │         │  │   :5173      │         │                          │  │
               │         │  └──────────────┘         └──────────┬───────────────┘  │
               │         │                                       │                  │
               │         │                                       │ HTTP POST        │
               │         │                                       ▼                  │
               │         │                         ┌────────────────────────────┐  │
               │         │                         │  Orchestrator (LangChain)  │  │
               │         │                         │     backend :8002          │  │
               │         │                         └──┬─────┬──────┬────────────┘  │
               │         │                            │     │      │               │
               │         │              ┌─────────────┘     │      └──────────────┐│
               │         │              │                    │                     ││
               │         │              ▼                    ▼                     ▼│
               │         │  ┌─────────────────┐  ┌──────────────────┐  ┌──────────┐│
               │         │  │   MCP Server    │  │  Qdrant :6333    │  │PostgreSQL││
               │         │  │   FastMCP/SSE   │  │  (Vector DB)     │  │  :5433   ││
               │         │  │   :8001         │  └──────────────────┘  └──────────┘│
               │         │  │                 │                                     │
               │         │  │ ┌─────────────┐ │  ┌──────────────────┐  ┌──────────┐│
               │         │  │ │  Colormind  │ │  │   Ollama :11434  │  │ Redis    ││
               │         │  │ │  Trends     │ │  │  (Qwen / Llama3) │  │  :6379   ││
               │         │  │ │  Open-Meteo │ │  └──────────────────┘  └──────────┘│
               │         │  │ └─────────────┘ │                                     │
               │         │  └─────────────────┘                                     │
               │         └─────────────────────────────────────────────────────────┘
               │
          [/images/*] ──► Nginx serve arquivos estáticos do volume data/images
```

### Decisões de Design

| Decisão | Alternativas Consideradas | Justificativa |
|---|---|---|
| **Dois serviços FastAPI** (Gateway + Orchestrator) | Monólito único | Separação de responsabilidades; o Orchestrator pode escalar independentemente |
| **Qdrant** para busca vetorial | FAISS (in-memory), Chroma | Suporte nativo a Docker, API REST robusta, modo persistente sem configuração extra |
| **Redis** para cache | Memcached, cache em memória | Compartilhado entre instâncias; TTL configurável por chave; zero serialização extra |
| **OpenCLIP ViT-B-32** | CLIP da OpenAI, ResNet | Modelo multimodal que alinha texto e imagem no mesmo espaço vetorial; open-source |
| **MCP via SSE** | REST puro, gRPC | Protocolo padronizado para agentes LLM; permite trocar ferramentas sem mudar o orquestrador |
| **Ollama local** | OpenAI API, Hugging Face Inference | Custo zero; privacidade; funciona offline; modelo trocável sem deploy |
| **Nginx** como reverse proxy | Traefik, Caddy | Configuração declarativa simples; zero overhead para arquivos estáticos |

---

## Tecnologias e Justificativas

### Backend

| Tecnologia | Versão | Papel |
|---|---|---|
| **FastAPI** | ≥ 0.100 | Framework assíncrono para os dois microsserviços HTTP |
| **LangChain** | ≥ 0.2 | Orquestração de cadeia LLM (LCEL: Prompt → LLM → Parser) |
| **langchain-ollama** | latest | Integração do LangChain com o runtime Ollama local |
| **sentence-transformers** | ≥ 2.2 | Carrega o modelo `clip-ViT-B-32` e gera embeddings texto/imagem |
| **Qdrant Client** | ≥ 1.3 | SDK Python para operações no banco vetorial |
| **psycopg2-binary** | ≥ 2.9 | Driver PostgreSQL; armazena metadados das imagens |
| **redis-py** | ≥ 4.6 | Cache de respostas de busca (TTL = 300s) |
| **httpx** | ≥ 0.24 | Cliente HTTP assíncrono para chamadas ao Orchestrator e APIs externas |
| **deep-translator** | latest | Tradução PT→EN da query enriquecida antes do embedding |
| **Pillow** | ≥ 9.5 | Abertura e conversão RGB de imagens para embedding visual |

### MCP Server

| Tecnologia | Versão | Papel |
|---|---|---|
| **FastMCP** | via `mcp[cli]` | Servidor MCP com transport SSE; expõe ferramentas para o Orchestrator |
| **pytrends** | ≥ 4.9 | Consulta unofficial Google Trends API |
| **httpx** | ≥ 0.24 | Chamadas async a Colormind e Open-Meteo |

### Frontend

| Tecnologia | Papel |
|---|---|
| **React 18** | UI reativa com hooks (useState, useEffect) |
| **Vite** | Bundler + dev server com HMR |
| **lucide-react** | Ícones SVG leves (Search, Sparkles, Palette, etc.) |
| **CSS Custom Properties** | Tema dark com variáveis (`--panel-border`, `--text-muted`, etc.) |

### Infraestrutura

| Serviço | Imagem Docker | Porta |
|---|---|---|
| **Nginx** | `nginx:alpine` | 80 |
| **PostgreSQL** | `postgres:15-alpine` | 5433 |
| **Redis** | `redis:7-alpine` | 6379 |
| **Qdrant** | `qdrant/qdrant:latest` | 6333, 6334 |
| **Ollama** | `ollama/ollama` | 11434 |

---

## Estrutura do Projeto

```
trabalho-sistemas-distribuidos-G3m/
│
├── docker-compose.yml              # Orquestra os 8 contêineres
│
├── backend/                        # Código Python compartilhado entre Gateway e Orchestrator
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── __init__.py
│   ├── config.py                   # Settings via variáveis de ambiente
│   ├── main.py                     # API Gateway (FastAPI, porta 8000)
│   ├── orchestrator_app.py         # Orchestrator Service (FastAPI, porta 8002)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ai.py                   # OpenCLIP: embeddings de texto e imagem
│   │   ├── cache.py                # Redis: get/set/clear de respostas de busca
│   │   ├── db.py                   # PostgreSQL: metadados de imagens e brand rules
│   │   ├── mcp_client.py           # Cliente SSE para chamar ferramentas no MCP Server
│   │   ├── orchestrator.py         # Pipeline principal de orquestração (10 etapas)
│   │   └── vector_db.py            # Qdrant: upsert e busca vetorial
│   │
│   └── scripts/
│       └── download_models.py      # Pre-download do modelo CLIP no build
│
├── mcp_server/                     # Servidor MCP independente (porta 8001)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── server.py                   # Ferramentas: Colormind, Google Trends, Open-Meteo
│
├── frontend/                       # React + Vite (porta 5173)
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── App.jsx                 # Componente único: busca, galeria, indexação, sidebar
│       └── index.css               # Tema dark com glass morphism
│
├── nginx/
│   └── nginx.conf                  # Proxy: /api/* → backend, /* → frontend, /images/* → volume
│
└── data/
    ├── images/                     # Imagens indexáveis (volume montado nos contêineres)
    ├── documents/
    └── qdrant_local/               # Dados persistentes do Qdrant em modo local
```

---

## Fluxo de Orquestração

O pipeline dentro do `orchestrate_search()` percorre 10 etapas sequenciais e paralelas:

```
query (PT)
│
├─── [Etapa 1] MCP Agent (paralelo, asyncio.gather)
│         ├── get_colormind_palette → paleta hex da categoria
│         ├── get_google_trends    → top 5 queries relacionadas no Brasil
│         └── get_weather          → condição climática de São Paulo
│
├─── [Etapa 2] Query Enrichment
│         └── query + tendências + paleta (se query mencionar cores) + contexto climático
│
├─── [Etapa 3] Tradução PT→EN
│         └── deep-translator (Google Translate, sem API key)
│
├─── [Etapa 4] Embedding OpenCLIP
│         └── clip-ViT-B-32 → vetor normalizado de 512 dimensões
│
├─── [Etapa 5] Busca Vetorial Qdrant
│         └── cosine similarity → top 6 hits
│
├─── [Etapa 6] Recuperação de Metadados PostgreSQL (paralelo, asyncio)
│         └── get_image_metadata para cada hit
│
├─── [Etapa 7] Score Híbrido + Ranking
│         ├── raw_score × 0.75   (CLIP cosine)
│         └── text_score × 0.25  (match de tokens na descrição)
│
├─── [Etapa 8] Brand Rules
│         └── get_brand_rules(category) → diretrizes da categoria no PostgreSQL
│
├─── [Etapa 9] LangChain LCEL + Ollama
│         └── EXPLANATION_PROMPT | OllamaLLM | StrOutputParser → justificativa em PT
│
└─── [Etapa 10] Resposta Final
          └── { query, enriched_query, explanation, mcp_info, results[] }
```

### Score de Similaridade

O score exibido ao usuário não é o cosine bruto do CLIP (que costuma ser baixo em bases pequenas), mas um valor calibrado:

```python
# Score híbrido
ranking_score = (raw_clip_score × 0.75) + (text_match_score × 0.25)

# Calibração para exibição (lower_bound=0.05, strong_match=0.35)
similarity_pct = clamp((ranking_score - 0.05) / (0.35 - 0.05), 0, 1) × 100
```

---

## Serviços e Microsserviços

### API Gateway (`backend/main.py` — porta 8000)

Ponto de entrada HTTP do sistema. Responsável por:
- Receber requisições do frontend (via Nginx)
- Verificar o cache Redis antes de chamar o Orchestrator
- Fazer proxy da requisição de busca para o Orchestrator em `/orchestrate`
- Indexar novas imagens diretamente (embedding + PostgreSQL + Qdrant)
- Servir metadados de imagens indexadas

### Orchestrator Service (`backend/orchestrator_app.py` — porta 8002)

Serviço dedicado ao pipeline de orquestração LangChain. Separado do Gateway para:
- Escalar independentemente (o processo de embedding e LLM é CPU-intensivo)
- Isolar falhas: se o Orchestrator cair, o Gateway continua respondendo do cache
- Permitir múltiplas instâncias atrás de um load balancer futuramente

### MCP Server (`mcp_server/server.py` — porta 8001)

Servidor de ferramentas externas usando o protocolo MCP com transport SSE. Expõe três ferramentas:

| Ferramenta | API Externa | Dados Retornados |
|---|---|---|
| `get_colormind_palette` | Colormind.io | Paleta de 5 cores hex gerada por modelo generativo |
| `get_google_trends` | Google Trends (pytrends) | Top 5 queries relacionadas no Brasil (últimas 7d) |
| `get_weather` | Open-Meteo (gratuito, sem API key) | Temperatura, umidade, vento e condição WMO |

O Orchestrator invoca essas ferramentas via `call_mcp_tool()`, que se comunica com o servidor MCP usando o protocolo SSE, conforme especificado pelo Model Context Protocol.

---

## API Reference

### `GET /`
Health check do Gateway.

```json
{
  "status": "healthy",
  "service": "G3m API Gateway",
  "version": "2.0.0",
  "orchestrator": "LangChain",
  "mcp_protocol": "SSE",
  "llm": "Ollama / Llama 3"
}
```

---

### `POST /search`
Busca semântica principal. Aciona o pipeline completo de orquestração.

**Request:**
```json
{
  "query": "equipe reunida no escritório moderno",
  "category": "corporativo"
}
```

**Response:**
```json
{
  "query": "equipe reunida no escritório moderno",
  "enriched_query": "equipe reunida no escritório moderno | Tendências: ...",
  "explanation": "As imagens foram selecionadas por...",
  "mcp_info": {
    "color_palette": ["#3a2c5f", "#7a5c9f", ...],
    "google_trends": { "top_queries": ["team building", ...] },
    "weather": { "condition": "Parcialmente nublado", "temperature_c": 22 }
  },
  "results": [
    {
      "id": 1,
      "filename": "reuniao.jpg",
      "description": "Grupo de pessoas...",
      "author": "Autor",
      "url": "/images/reuniao.jpg",
      "score": 0.2841,
      "text_score": 0.6,
      "ranking_score": 0.3631,
      "similarity": 83.74
    }
  ]
}
```

**Categorias disponíveis:** `corporativo` | `tecnologia` | `retro` | `minimalista`

---

### `POST /index`
Indexa uma nova imagem. O arquivo deve existir em `data/images/`.

**Request:**
```json
{
  "filename": "foto_reuniao.jpg",
  "description": "Grupo de desenvolvedores bebendo café no escritório azul",
  "author": "Gabriela Caetano"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Imagem 'foto_reuniao.jpg' indexada com sucesso.",
  "id": 7
}
```

---

### `GET /images`
Retorna metadados de todas as imagens indexadas no sistema.

---

### `GET /tools`
Lista as ferramentas disponíveis no MCP Server via protocolo MCP/SSE.

---

### `POST /cache/clear`
Remove todas as entradas de busca do cache Redis.

```json
{
  "status": "success",
  "deleted_keys": 12
}
```

---

## Pré-requisitos

- **Docker** ≥ 24 e **Docker Compose** ≥ 2.20
- Mínimo **8 GB de RAM** disponível (Ollama + CLIP consomem bastante na inicialização)
- Mínimo **10 GB de espaço em disco** (modelos Ollama + imagens Docker)
- Acesso à internet no primeiro build (download do modelo CLIP e imagens Docker)

---

## Como Executar

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd trabalho-sistemas-distribuidos-G3m
```

### 2. Criar a pasta de imagens

```bash
mkdir -p data/images
```

Copie para `data/images/` as imagens que deseja indexar (`.jpg`, `.png`).

### 3. Subir todos os contêineres

```bash
docker compose up --build
```

> O primeiro build leva alguns minutos: o modelo `clip-ViT-B-32` é baixado durante o build do backend.

### 4. Baixar o modelo LLM no Ollama

Após os contêineres subirem, execute em outro terminal:

```bash
docker exec -it g3m-ollama ollama pull qwen3.5:latest
```

> Alternativamente, use `llama3:latest` ou qualquer modelo disponível no [Ollama Hub](https://ollama.com/library).
> Se usar outro modelo, atualize `OLLAMA_MODEL` no `docker-compose.yml`.

### 5. Acessar o sistema

| Serviço | URL |
|---|---|
| **Frontend** | http://localhost |
| **API Gateway (Swagger)** | http://localhost:8000/docs |
| **Orchestrator (Swagger)** | http://localhost:8002/docs |
| **MCP Server** | http://localhost:8001/sse |
| **Qdrant Dashboard** | http://localhost:6333/dashboard |

### 6. Parar o ambiente

```bash
docker compose down
```

Para remover também os volumes (banco de dados e modelos Ollama):

```bash
docker compose down -v
```

---

## Indexando Imagens

Antes de buscar, o sistema precisa de imagens indexadas. Há duas formas:

### Via Interface Web

1. Acesse http://localhost
2. Role até a seção **"Indexador de Novos Ativos de Imagem"**
3. Preencha o nome exato do arquivo (que deve estar em `data/images/`), a descrição semântica e o autor
4. Clique em **"Indexar Ativo Visual"**

> **Dica:** A qualidade da descrição semântica é crucial para a busca. Descreva em detalhes o que está na imagem, incluindo ambiente, pessoas, cores dominantes, mood e elementos visuais.

### Via API (curl)

```bash
# Certifique-se de que o arquivo existe em data/images/foto.jpg
curl -X POST http://localhost/api/index \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "foto.jpg",
    "description": "Grupo de quatro profissionais reunidos em volta de uma mesa de vidro, com laptops abertos, em ambiente corporativo moderno com iluminação natural",
    "author": "Departamento de Marketing"
  }'
```

---

## Variáveis de Ambiente

Todas as variáveis têm valores padrão funcionais para desenvolvimento local. Para produção, sobrescreva no `docker-compose.yml` ou em um arquivo `.env`.

| Variável | Padrão | Descrição |
|---|---|---|
| `DATABASE_URL` | `postgresql://admin:admin_password@postgres:5432/g3m` | Conexão PostgreSQL |
| `REDIS_URL` | `redis://redis:6379/0` | Conexão Redis |
| `QDRANT_URL` | `http://qdrant:6333` | Endpoint Qdrant |
| `MCP_SERVER_URL` | `http://mcp-server:8001/sse` | Endpoint SSE do MCP Server |
| `ORCHESTRATOR_URL` | `http://orchestrator:8002` | URL interna do Orchestrator |
| `OLLAMA_URL` | `http://ollama:11434` | Endpoint Ollama |
| `OLLAMA_MODEL` | `qwen3.5:latest` | Modelo LLM para geração de explicações |

---

## Considerações de Sistemas Distribuídos

Este projeto aplica na prática os seguintes conceitos da disciplina:

**Comunicação entre processos distribuídos**
- REST/HTTP síncrono entre os microsserviços internos (Gateway → Orchestrator)
- SSE (Server-Sent Events) para comunicação com o MCP Server, seguindo o protocolo MCP

**Concorrência e paralelismo**
- `asyncio.gather()` para invocar as três ferramentas MCP em paralelo
- `asyncio.to_thread()` para operações de banco de dados bloqueantes no loop assíncrono

**Tolerância a falhas e fallbacks**
- Se o Qdrant não estiver disponível, o `vector_db.py` cai para modo in-memory
- Se o PostgreSQL não estiver disponível, o `db.py` cai para SQLite local
- Se o Ollama falhar na geração, a explicação é substituída por uma mensagem padrão
- Respostas de busca são cacheadas no Redis por 5 minutos

**Separação de responsabilidades**
- Gateway: roteamento, cache, indexação
- Orchestrator: inteligência, orquestração, LLM
- MCP Server: ferramentas externas isoladas e intercambiáveis

**Persistência distribuída**
- PostgreSQL: dados relacionais (metadados, brand rules)
- Qdrant: dados vetoriais (embeddings de imagens)
- Redis: cache de respostas
- Volume Docker: arquivos de imagem

---

<div align="center">

**G3m Smart Search Engine** &copy; 2026 · Trabalho de Sistemas Distribuídos (Fase 2 — Protótipo RAG/MCP)

*UFLA — Universidade Federal de Lavras*

</div>
