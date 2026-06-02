"""
G3m MCP Server — usa FastMCP (protocolo MCP via SSE)
Ferramentas disponíveis:
  - get_colormind_palette : Gera paletas via API Colormind
  - get_google_trends     : Retorna termos em alta via Google Trends
  - get_weather           : Clima atual de uma cidade (Open-Meteo, sem API key)
"""
import httpx
from pytrends.request import TrendReq
from mcp.server.fastmcp import FastMCP

# Instancia o servidor MCP com SSE
mcp = FastMCP("G3m MCP Server", host="0.0.0.0", port=8001, sse_path="/sse")


# ─────────────────────────────────────────────────────────────────────────────
# Ferramenta 1: Colormind Palette
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
async def get_colormind_palette(category: str = "corporativo") -> dict:
    """Gera paleta de cores via API Colormind para enriquecer contexto de busca."""
    model_by_category = {
        "corporativo": "ui",
        "tecnologia": "default",
        "retro": "default",
        "minimalista": "ui"
    }
    model = model_by_category.get(category.lower(), "default")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "http://colormind.io/api/",
                json={"model": model}
            )
        if response.status_code != 200:
            return {"status": "error", "error": f"Colormind HTTP {response.status_code}"}

        data = response.json()
        palette_rgb = data.get("result", [])
        palette_hex = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in palette_rgb if len(c) == 3]

        return {
            "status": "success",
            "tool": "get_colormind_palette",
            "result": {
                "palette": palette_hex,
                "model": model
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Ferramenta 2: Google Trends
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_google_trends(keyword: str, geo: str = "BR") -> dict:
    """Retorna termos relacionados em alta via Google Trends (pytrends)."""
    try:
        pytrends = TrendReq(hl="pt-BR", tz=180)
        pytrends.build_payload([keyword], geo=geo)
        related = pytrends.related_queries()
        related_for_kw = related.get(keyword, {})
        top_df = related_for_kw.get("top")
        top_queries = []
        if top_df is not None and not top_df.empty:
            top_queries = top_df["query"].head(5).tolist()

        return {
            "status": "success",
            "tool": "get_google_trends",
            "result": {
                "keyword": keyword,
                "geo": geo,
                "top_queries": top_queries
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Ferramenta 3: Clima Atual via Open-Meteo (sem API key)
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
async def get_weather(city: str = "São Paulo") -> dict:
    """Obtém condições climáticas atuais de uma cidade usando a API Open-Meteo (gratuita, sem key)."""
    WMO_CODES = {
        0: "Céu limpo", 1: "Principalmente limpo", 2: "Parcialmente nublado",
        3: "Nublado", 45: "Névoa", 48: "Geada com névoa",
        51: "Garoa leve", 53: "Garoa moderada", 55: "Garoa intensa",
        61: "Chuva leve", 63: "Chuva moderada", 65: "Chuva forte",
        71: "Neve leve", 73: "Neve moderada", 75: "Neve forte",
        80: "Pancadas de chuva leves", 81: "Pancadas moderadas", 82: "Pancadas fortes",
        95: "Trovoada", 96: "Trovoada com granizo leve", 99: "Trovoada com granizo"
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Passo 1 — geocoding
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "pt", "format": "json"}
            )
            geo_data = geo.json()

            if not geo_data.get("results"):
                return {"status": "error", "error": f"Cidade '{city}' não encontrada."}

            loc = geo_data["results"][0]
            lat, lon = loc["latitude"], loc["longitude"]

            # Passo 2 — condições climáticas atuais
            weather = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                    "timezone": "auto"
                }
            )
            current = weather.json().get("current", {})
            code = current.get("weather_code", 0)

        return {
            "status": "success",
            "tool": "get_weather",
            "result": {
                "city": loc.get("name", city),
                "country": loc.get("country", ""),
                "temperature_c": current.get("temperature_2m"),
                "humidity_pct": current.get("relative_humidity_2m"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "condition": WMO_CODES.get(code, f"Código {code}")
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Ponto de entrada — serve via SSE (protocolo MCP)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # CORRIGIDO: host/port agora ficam no construtor do FastMCP.
    mcp.run(transport="sse")