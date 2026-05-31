import os
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

app = FastAPI(
    title="G3m MCP Server",
    description="Servidor MCP para análise de imagens e tendências visuais",
    version="1.0.0"
)

# Definição das ferramentas suportadas (padrão MCP)
TOOLS = [
    {
        "name": "get_color_palette",
        "description": "Extrai as cores dominantes de uma imagem física usando clustering K-Means.",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Caminho da imagem no disco (relativo a /data/images ou absoluto)"
                },
                "num_colors": {
                    "type": "integer",
                    "description": "Número de cores para extrair da imagem",
                    "default": 5
                }
            },
            "required": ["image_path"]
        }
    },
    {
        "name": "get_design_trends",
        "description": "Busca tendências atuais de paleta de cores e conceitos visuais baseados na categoria.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Categoria das tendências (ex: corporativo, tecnologia, editorial, retro)"
                }
            },
            "required": ["category"]
        }
    }
]

class ColorPaletteRequest(BaseModel):
    image_path: str
    num_colors: int = 5

class DesignTrendsRequest(BaseModel):
    category: str

@app.get("/tools")
def list_tools():
    """Retorna a lista de ferramentas disponíveis (MCP List Tools)"""
    return {"tools": TOOLS}

@app.post("/tools/call/get_color_palette")
def call_get_color_palette(req: ColorPaletteRequest):
    """Executa a ferramenta get_color_palette"""
    path = req.image_path
    
    # Resolver caminho
    resolved_path = path
    if not os.path.exists(resolved_path):
        # Tentar caminho relativo ao volume compartilhado /data/images
        resolved_path = os.path.join("/data/images", os.path.basename(path))
        if not os.path.exists(resolved_path):
            # Tentar relativo a /data
            resolved_path = os.path.join("/data", path.lstrip("/"))
            if not os.path.exists(resolved_path):
                raise HTTPException(
                    status_code=404, 
                    detail=f"Arquivo de imagem nao encontrado no MCP Server: {path}. Caminhos testados: {req.image_path}, {resolved_path}"
                )

    try:
        # Abrir imagem e redimensionar para acelerar processamento
        img = Image.open(resolved_path)
        img = img.resize((150, 150))
        img_arr = np.array(img)

        # Tratar dimensões e canais de cores
        if len(img_arr.shape) == 3:
            pixels = img_arr.reshape(-1, img_arr.shape[-1])
        else:
            pixels = img_arr.reshape(-1, 1)

        # Descartar canal alpha se houver
        if pixels.shape[1] == 4:
            pixels = pixels[:, :3]

        # Aplicar K-Means para agrupar cores dominantes
        kmeans = KMeans(n_clusters=req.num_colors, n_init=10, random_state=42)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)

        # Converter para formato Hexadecimal
        hex_palette = []
        for color in colors:
            if len(color) == 1:
                hex_str = f"#{color[0]:02x}{color[0]:02x}{color[0]:02x}"
            else:
                hex_str = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            hex_palette.append(hex_str)

        return {
            "status": "success",
            "tool": "get_color_palette",
            "result": {
                "palette": hex_palette,
                "image_analyzed": os.path.basename(resolved_path)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento da imagem: {str(e)}")

@app.post("/tools/call/get_design_trends")
def call_get_design_trends(req: DesignTrendsRequest):
    """Executa a ferramenta get_design_trends (Simulando consulta externa a API)"""
    category = req.category.lower()
    
    trends = {
        "corporativo": {
            "palettes": [["#1a365d", "#2b6cb0", "#4299e1", "#edf2f7"], ["#2d3748", "#4a5568", "#a0aec0", "#e2e8f0"]],
            "concepts": ["Minimalismo corporativo", "Tons sóbrios de azul e cinza", "Iluminação natural suave", "Espaços abertos de trabalho"],
            "vibe": "Profissional, moderno e confiável"
        },
        "tecnologia": {
            "palettes": [["#0f172a", "#3b82f6", "#10b981", "#f43f5e"], ["#1e1b4b", "#6366f1", "#a855f7", "#ec4899"]],
            "concepts": ["Estilo Cyberpunk/Neon", "Dark Mode predominante", "Linhas abstratas e elementos digitais", "Foco em hardware futurista e telas"],
            "vibe": "Inovador, dinâmico e de alta performance"
        },
        "retro": {
            "palettes": [["#f59e0b", "#d97706", "#b45309", "#78350f"], ["#e11d48", "#be123c", "#4c0519", "#fef2f2"]],
            "concepts": ["Cores quentes e saturadas dos anos 70/80", "Granulado de filme fotográfico", "Contrastes fortes", "Vibe vintage e nostálgica"],
            "vibe": "Nostálgico, caloroso e artístico"
        },
        "minimalista": {
            "palettes": [["#fafaf9", "#f5f5f4", "#e7e5e4", "#78716c"], ["#fcfcfc", "#f3f4f6", "#e5e7eb", "#4b5563"]],
            "concepts": ["Espaços vazios amplos", "Móveis geométricos", "Cores neutras e tons pastel", "Foco no essencial"],
            "vibe": "Limpo, pacífico e sofisticado"
        }
    }

    # Categoria padrão se não mapeada
    selected_trend = trends.get(category, trends["corporativo"])
    
    return {
        "status": "success",
        "tool": "get_design_trends",
        "result": {
            "category": category,
            "recommended_palettes": selected_trend["palettes"],
            "key_concepts": selected_trend["concepts"],
            "general_vibe": selected_trend["vibe"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
