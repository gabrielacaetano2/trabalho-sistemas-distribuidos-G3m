import httpx
from ..config import settings

async def list_mcp_tools() -> list:
    """Consulta as ferramentas expostas pelo servidor MCP"""
    url = f"{settings.MCP_SERVER_URL}/tools"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get("tools", [])
            return []
    except Exception as e:
        print(f"Nao foi possivel listar ferramentas do servidor MCP em {url}: {e}")
        return []

async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """Executa de forma distribuida uma ferramenta no servidor MCP"""
    url = f"{settings.MCP_SERVER_URL}/tools/call/{tool_name}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=arguments)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Erro ao invocar ferramenta MCP '{tool_name}': {response.status_code} - {response.text}")
                return {"status": "error", "error": f"Erro HTTP {response.status_code}"}
    except Exception as e:
        print(f"Falha de conexao com MCP Server '{tool_name}': {e}")
        return {"status": "error", "error": str(e)}
