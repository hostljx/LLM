from asyncio import transports
from mcp.server.mcpserver import MCPServer


mcp = MCPServer()


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b



if __name__ == "__main__":
    mcp.run(transport="sse",host="0.0.0.0",port=19530)
