import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import List

import arxiv
import fitz
import httpx
import pymupdf4llm
from dotenv import load_dotenv
from litellm import embedding
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP

load_dotenv()

PAPER_DIR = "papers"
REDIS_PREFIX = "doc:"

# Embedding config
EMBEDDING_MODEL = "gemini/gemini-embedding-001"
EMBEDDING_DIMS = 3072

# Redis MCP server config — reads URL from env for security
REDIS_URL = os.getenv("REDIS_URL")
REDIS_MCP_CONFIG = {
    "command": "uvx",
    "args": [
        "--from",
        "redis-mcp-server@latest",
        "redis-mcp-server",
        "--url",
        REDIS_URL,
    ],
}

mcp = FastMCP("research")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def _get_embedding(text: str) -> List[float]:
    """Generate an embedding vector for a single text string."""
    response = embedding(model=EMBEDDING_MODEL, input=[text])
    return response.data[0]["embedding"]


def _get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embedding vectors for a batch of texts in one API call."""
    response = embedding(model=EMBEDDING_MODEL, input=texts)
    return [item["embedding"] for item in response.data]


def _update_indexed_flag(paper_id: str):
    """Mark a paper as indexed in local metadata files."""
    if not os.path.exists(PAPER_DIR):
        return
    for item in os.listdir(PAPER_DIR):
        file_path = os.path.join(PAPER_DIR, item, "papers_info.json")
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                if paper_id in data:
                    data[paper_id]["indexed"] = True
                    with open(file_path, "w") as f:
                        json.dump(data, f, indent=2)
                    return
            except (FileNotFoundError, json.JSONDecodeError):
                continue


def _extract_text_chunks(pdf_bytes: bytes) -> List[str]:
    """Extract markdown text from PDF bytes and split into chunks."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    md_text = pymupdf4llm.to_markdown(doc)
    doc.close()

    if not md_text.strip():
        return []

    return [c.strip() for c in md_text.split("\n## ") if c.strip()]


async def _get_redis_session(exit_stack: AsyncExitStack) -> ClientSession:
    """Create and initialize an MCP client session to the Redis MCP server."""
    server_params = StdioServerParameters(**REDIS_MCP_CONFIG)
    read, write = await exit_stack.enter_async_context(stdio_client(server_params))
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


# ─────────────────────────────────────────────
# MCP TOOLS
# ─────────────────────────────────────────────


@mcp.tool()
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.

    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)

    Returns:
        List of paper IDs found in the search
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)

    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "papers_info.json")

    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        papers_info[paper.get_short_id()] = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "pdf_url": str(paper.pdf_url),
            "published": str(paper.published.date()),
            "indexed": False,
        }

    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)

    print(f"Results saved in: {file_path}")
    return paper_ids


@mcp.tool()
def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.

    Args:
        paper_id: The ID of the paper to look for

    Returns:
        JSON string with paper information if found, error message if not found
    """
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue

    return f"There's no saved information related to paper {paper_id}."


@mcp.tool()
async def index_paper(paper_id: str) -> str:
    """
    Download, chunk, embed, and store a paper in Redis — all in one step.
    This is the single tool to call when the user wants to index a paper.

    The tool will:
    1. Download the PDF from arXiv using the paper_id
    2. Extract text and split into chunks
    3. Generate embeddings for all chunks (batched)
    4. Push all chunk data + embeddings directly to Redis
    5. Store paper metadata in Redis
    6. Mark the paper as indexed locally

    Args:
        paper_id: arXiv paper ID (e.g. "1706.03762" or "1706.03762v2")

    Returns:
        JSON with status, paper_id, total_chunks stored, and any errors
    """
    pdf_url = f"https://arxiv.org/pdf/{paper_id}"

    try:
        # ── Step 1: Download PDF ──────────────────────────────────────
        response = httpx.get(pdf_url, timeout=60, follow_redirects=True)
        response.raise_for_status()

        # ── Step 2: Extract text chunks ───────────────────────────────
        chunks = _extract_text_chunks(response.content)
        if not chunks:
            return json.dumps(
                {"status": "error", "error": "No text extracted from PDF"}
            )

        # ── Step 3: Batch embed all chunks ────────────────────────────
        vectors = _get_embeddings_batch(chunks)

        # ── Step 4: Look up paper metadata from local files ───────────
        paper_meta = None
        info_str = extract_info(paper_id)
        if not info_str.startswith("There's no saved"):
            paper_meta = json.loads(info_str)

        # ── Step 5: Push everything to Redis via MCP sub-session ──────
        async with AsyncExitStack() as stack:
            redis_session = await _get_redis_session(stack)

            # Create vector index (safely ignore if it already exists)
            try:
                await redis_session.call_tool(
                    "create_vector_index_hash",
                    {
                        "index_name": "idx:chunks",
                        "prefix": "doc:paper:",
                        "dim": EMBEDDING_DIMS,
                    },
                )
            except Exception as e:
                # Often throws if it already exists, which is fine
                print(f"Vector index creation note: {e}")

            # Store paper metadata
            meta_to_store = paper_meta or {
                "paper_id": paper_id,
                "pdf_url": pdf_url,
            }
            await redis_session.call_tool(
                "json_set",
                {
                    "name": f"paper:{paper_id}",
                    "path": "$",
                    "value": json.dumps(meta_to_store),
                },
            )

            # Store each chunk: text metadata + embedding vector
            for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
                redis_key = f"{REDIS_PREFIX}paper:{paper_id}:chunk:{i}"

                # Store text fields in hash
                await redis_session.call_tool(
                    "hset", {"name": redis_key, "key": "text", "value": chunk_text}
                )
                await redis_session.call_tool(
                    "hset", {"name": redis_key, "key": "paper_id", "value": paper_id}
                )
                await redis_session.call_tool(
                    "hset", {"name": redis_key, "key": "chunk_index", "value": str(i)}
                )

                # Store embedding vector
                await redis_session.call_tool(
                    "set_vector_in_hash",
                    {
                        "name": redis_key,
                        "vector": vector,
                    },
                )

        # ── Step 6: Mark as indexed locally ───────────────────────────
        _update_indexed_flag(paper_id)

        return json.dumps(
            {
                "status": "ok",
                "paper_id": paper_id,
                "title": (paper_meta or {}).get("title", "Unknown"),
                "total_chunks": len(chunks),
                "message": f"Successfully indexed {len(chunks)} chunks into Redis",
            }
        )

    except Exception as e:
        return json.dumps({"status": "error", "paper_id": paper_id, "error": str(e)})


@mcp.tool()
async def query_paper(question: str, paper_id: str = "") -> str:
    """
    Answer a question about an indexed paper by searching Redis embeddings.
    Embeds the question, performs vector similarity search in Redis,
    and returns the most relevant chunks.

    Args:
        question: The question to answer about the paper
        paper_id: Optional arXiv paper ID to scope the search.
                  If empty, searches across all indexed papers.

    Returns:
        JSON with the top matching chunks and their text content
    """
    try:
        # ── Step 1: Embed the question ────────────────────────────────
        query_vector = _get_embedding(question)

        # ── Step 2: Search Redis via MCP sub-session ──────────────────
        async with AsyncExitStack() as stack:
            redis_session = await _get_redis_session(stack)

            # Vector similarity search
            search_result = await redis_session.call_tool(
                "vector_search_hash",
                {
                    "index_name": "idx:chunks",
                    "query_vector": query_vector,
                    "k": 5,
                    "return_fields": ["text", "chunk_index", "paper_id"]
                },
            )

            # Parse results and fetch chunk texts
            results = []
            if search_result and search_result.content:
                for item in search_result.content:
                    if hasattr(item, "text"):
                        try:
                            parsed = json.loads(item.text)
                            if isinstance(parsed, list):
                                for doc in parsed:
                                    # Extract fields which are inside doc["payload"]
                                    payload = doc.get("payload") or {}
                                    if paper_id and paper_id != "all":
                                        if payload.get("paper_id") == paper_id:
                                            results.append(doc)
                                    else:
                                        results.append(doc)
                            else:
                                results.append(parsed)
                        except (json.JSONDecodeError, TypeError):
                            results.append({"raw": item.text})

        return json.dumps(
            {
                "status": "ok",
                "question": question,
                "paper_id": paper_id or "all",
                "results": results,
            },
            ensure_ascii=False,
        )

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


if __name__ == "__main__":
    mcp.run(transport="stdio")
