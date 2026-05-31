"""
Cliente MCP para o backend G3m.
Usa o SDK oficial MCP (mcp.client.sse) para comunicação via protocolo MCP/SSE,
conforme exigido pela arquitetura do projeto.
"""
import json
from mcp.client.sse import sse_client
from mcp import ClientSession
from ..config import settings


async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """
    Executa uma ferramenta no servidor MCP via protocolo MCP/SSE (não HTTP REST simples).
    Conecta ao endpoint SSE, inicializa sessão MCP, chama a tool e retorna o resultado.
    """
    try:
        async with sse_client(url=settings.MCP_SERVER_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name=tool_name, arguments=arguments)

                if result.content:
                    content_item = result.content[0]
                    # ToolResult pode ser TextContent ou ImageContent
                    text = getattr(content_item, "text", None)
                    if text:
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            return {"status": "success", "result": text}

                return {"status": "error", "error": "Resultado vazio do servidor MCP"}

    except Exception as e:
        print(f"[MCP Client] Erro ao chamar '{tool_name}' em {settings.MCP_SERVER_URL}: {e}")
        return {"status": "error", "error": str(e)}


async def list_mcp_tools() -> list:
    """
    Lista as ferramentas disponíveis no servidor MCP via protocolo MCP/SSE.
    """
    try:
        async with sse_client(url=settings.MCP_SERVER_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                return [
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.inputSchema
                    }
                    for t in tools_result.tools
                ]
    except Exception as e:
        print(f"[MCP Client] Erro ao listar ferramentas em {settings.MCP_SERVER_URL}: {e}")
        return []
