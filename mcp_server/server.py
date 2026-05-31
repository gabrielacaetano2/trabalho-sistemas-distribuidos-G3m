"""
G3m MCP Server — usa FastMCP (protocolo MCP via SSE)
Ferramentas disponíveis:
  - get_color_palette  : Extrai paleta dominante de uma imagem (K-Means)
  - get_design_trends  : Retorna tendências de design por categoria
  - get_weather        : Clima atual de uma cidade (Open-Meteo, sem API key)
"""
import os
import httpx
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from mcp.server.fastmcp import FastMCP

# Instancia o servidor MCP com SSE
mcp = FastMCP("G3m MCP Server")


# ─────────────────────────────────────────────────────────────────────────────
# Ferramenta 1: Paleta de Cores (K-Means sobre pixels da imagem)
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_color_palette(image_path: str, num_colors: int = 5) -> dict:
    """Extrai as cores dominantes de uma imagem física usando clustering K-Means."""
    resolved = image_path
    if not os.path.exists(resolved):
        resolved = os.path.join("/data/images", os.path.basename(image_path))
        if not os.path.exists(resolved):
            return {
                "status": "error",
                "error": f"Imagem não encontrada: {image_path}"
            }

    try:
        img = Image.open(resolved).convert("RGB").resize((150, 150))
        pixels = np.array(img).reshape(-1, 3)

        n = min(num_colors, len(pixels))
        kmeans = KMeans(n_clusters=n, n_init=10, random_state=42)
        kmeans.fit(pixels)
        centers = kmeans.cluster_centers_.astype(int)

        palette = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in centers]

        return {
            "status": "success",
            "tool": "get_color_palette",
            "result": {
                "palette": palette,
                "image_analyzed": os.path.basename(resolved)
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Ferramenta 2: Tendências de Design por Categoria
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_design_trends(category: str) -> dict:
    """Retorna tendências visuais e conceitos de design com base na categoria informada."""
    TRENDS = {
        "corporativo": {
            "palettes": [["#1a365d", "#2b6cb0", "#4299e1", "#edf2f7"],
                         ["#2d3748", "#4a5568", "#a0aec0", "#e2e8f0"]],
            "concepts": ["Minimalismo corporativo", "Tons sóbrios de azul e cinza",
                         "Iluminação natural suave", "Espaços abertos de trabalho"],
            "vibe": "Profissional, moderno e confiável"
        },
        "tecnologia": {
            "palettes": [["#0f172a", "#3b82f6", "#10b981", "#f43f5e"],
                         ["#1e1b4b", "#6366f1", "#a855f7", "#ec4899"]],
            "concepts": ["Estilo Cyberpunk/Neon", "Dark Mode predominante",
                         "Linhas abstratas e elementos digitais", "Foco em hardware futurista"],
            "vibe": "Inovador, dinâmico e de alta performance"
        },
        "retro": {
            "palettes": [["#f59e0b", "#d97706", "#b45309", "#78350f"],
                         ["#e11d48", "#be123c", "#4c0519", "#fef2f2"]],
            "concepts": ["Cores quentes dos anos 70/80", "Granulado de filme fotográfico",
                         "Contrastes fortes", "Vibe vintage e nostálgica"],
            "vibe": "Nostálgico, caloroso e artístico"
        },
        "minimalista": {
            "palettes": [["#fafaf9", "#f5f5f4", "#e7e5e4", "#78716c"],
                         ["#fcfcfc", "#f3f4f6", "#e5e7eb", "#4b5563"]],
            "concepts": ["Espaços vazios amplos", "Móveis geométricos",
                         "Cores neutras e tons pastel", "Foco no essencial"],
            "vibe": "Limpo, pacífico e sofisticado"
        },
        "editorial": {
            "palettes": [["#1a1a2e", "#16213e", "#0f3460", "#e94560"]],
            "concepts": ["Tipografia expressiva", "Fotografia de alto contraste",
                         "Layout assimétrico", "Cores ousadas e inesperadas"],
            "vibe": "Criativo, impactante e autoral"
        }
    }

    selected = TRENDS.get(category.lower(), TRENDS["corporativo"])

    return {
        "status": "success",
        "tool": "get_design_trends",
        "result": {
            "category": category,
            "recommended_palettes": selected["palettes"],
            "key_concepts": selected["concepts"],
            "general_vibe": selected["vibe"]
        }
    }


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
    # Exposição via SSE para que o cliente MCP conecte em http://localhost:8001/sse
    mcp.run(transport="sse", host="127.0.0.1", port=8001)
