import redis
import json
import hashlib
from backend.config import settings

_redis_client = None

def get_redis_client():
    """Retorna cliente Redis singleton com tratamento de conexao"""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            _redis_client.ping()
        except Exception as e:
            print(f"Erro ao conectar ao Redis: {e}")
            return None
    return _redis_client

def _get_query_hash(query: str) -> str:
    """Gera hash MD5 para a chave do cache para normalizar strings de busca"""
    return hashlib.md5(query.strip().lower().encode('utf-8')).hexdigest()
    """Gera hash MD5 para a chave do cache para normalizar strings de busca"""
    return hashlib.md5(query.strip().lower().encode('utf-8')).hexdigest()

def get_cached_search(query: str):
    """Busca o resultado cacheado de uma busca no Redis"""
    client = get_redis_client()
    if not client:
        return None
    
    cache_key = f"search:{_get_query_hash(query)}"
    try:
        data = client.get(cache_key)
        if data:
            print(f"Redis Cache HIT para busca: '{query}'")
            return json.loads(data)
        print(f"Redis Cache MISS para busca: '{query}'")
        return None
    except Exception as e:
        print(f"Falha ao ler cache do Redis: {e}")
        return None

def set_cached_search(query: str, results: dict, ttl: int = 300):
    """Grava o resultado da busca no Redis com tempo de vida definido (TTL)"""
    client = get_redis_client()
    if not client:
        return
    
    cache_key = f"search:{_get_query_hash(query)}"
    try:
        client.setex(
            cache_key,
            ttl,
            json.dumps(results)
        )
        print(f"Resultados para a busca '{query}' persistidos no Redis Cache (TTL: {ttl}s).")
    except Exception as e:
        print(f"Falha ao salvar cache no Redis: {e}")
