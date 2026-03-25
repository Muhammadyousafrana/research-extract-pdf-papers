import asyncio
import os
import json
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://default:QeH1b3zT3ZEpEbv11YdN0ac6mqozwpzq@redis-18243.c90.us-east-1-3.ec2.cloud.redislabs.com:18243",
)

async def main():
    server_params = StdioServerParameters(
        command="uvx",
        args=[
            "--from", "redis-mcp-server@latest",
            "redis-mcp-server",
            "--url", REDIS_URL
        ]
    )
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        
        # We know paper_id is 2512.19700v1 from the user's message
        redis_key = "doc:paper:2512.19700v1:chunk:0"
        
        # Call hgetall
        print(f"Calling hgetall for {redis_key}...")
        res = await session.call_tool(
            "hgetall",
            {
                "name": redis_key,
            }
        )
        with open("chunk_data.json", "w", encoding="utf-8") as f:
            if getattr(res, "content", None):
                for i, item in enumerate(res.content):
                    f.write(item.text)
                    
        print("Calling get_vector_from_hash...")
        res = await session.call_tool(
            "get_vector_from_hash",
            {
                "name": redis_key,
                "vector_field": "vector"
            }
        )
        print("Got vector?", len(res.content[0].text) if res.content else "No content")

if __name__ == "__main__":
    asyncio.run(main())
