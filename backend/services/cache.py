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


def _get_query_hash(cache_key: str) -> str:
    """Gera hash MD5 para normalizar a chave do cache"""
    return hashlib.md5(cache_key.strip().lower().encode('utf-8')).hexdigest()


# CORRIGIDO: parâmetro renomeado de `query` para `cache_key` nas duas funções.
# O parâmetro recebia strings compostas como "corporativo:mulher segurando caneca"
# (categoria + query), não apenas a query. O nome `query` era enganoso.
# Renomear para `cache_key` deixa claro que pode ser qualquer string identificadora.

def get_cached_search(cache_key: str):
    """Busca o resultado cacheado de uma busca no Redis"""
    client = get_redis_client()
    if not client:
        return None

    redis_key = f"search:{_get_query_hash(cache_key)}"
    try:
        data = client.get(redis_key)
        if data:
            print(f"Redis Cache HIT para: '{cache_key}'")
            return json.loads(data)
        print(f"Redis Cache MISS para: '{cache_key}'")
        return None
    except Exception as e:
        print(f"Falha ao ler cache do Redis: {e}")
        return None


def set_cached_search(cache_key: str, results: dict, ttl: int = 300):
    """Grava o resultado da busca no Redis com tempo de vida definido (TTL)"""
    client = get_redis_client()
    if not client:
        return

    redis_key = f"search:{_get_query_hash(cache_key)}"
    try:
        client.setex(
            redis_key,
            ttl,
            json.dumps(results)
        )
        print(f"Cache salvo para '{cache_key}' (TTL: {ttl}s).")
    except Exception as e:
        print(f"Falha ao salvar cache no Redis: {e}")


def clear_search_cache() -> int:
    """Remove apenas entradas de cache de busca."""
    client = get_redis_client()
    if not client:
        raise RuntimeError("Redis indisponível.")

    deleted = 0

    for key in client.scan_iter(match="search:*", count=100):
        deleted += client.delete(key)

    print(f"Cache de busca limpo. Chaves removidas: {deleted}.")

    return deleted
