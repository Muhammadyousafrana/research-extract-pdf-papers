import sys, os, asyncio
sys.path.insert(0, '.')
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    stack = AsyncExitStack()
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["research_server.py"],
        env={**os.environ},
    )
    read, write = await stack.enter_async_context(stdio_client(server_params))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    print("INIT OK")
    result = await session.call_tool("search_papers", {"topic": "test", "max_results": 1})
    print("CALL OK:")
    for item in result.content:
        if hasattr(item, "text"):
            print(item.text[:200])
    await stack.aclose()

asyncio.run(main())
