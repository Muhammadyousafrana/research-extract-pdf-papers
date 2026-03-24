import pymupdf4llm
import fitz
import httpx
import arxiv
import json
import os
from typing import List
from mcp.server.fastmcp import FastMCP
from litellm import embedding

PAPER_DIR = "papers"
REDIS_PREFIX = "doc:"

# Gemini: "gemini/gemini-embedding-001" (GEMINI_API_KEY, 3072 dims)
# Voyage: "voyage/voyage-3"             (VOYAGE_API_KEY, 1024 dims)
# OpenAI: "openai/text-embedding-3-small"(OPENAI_API_KEY, 1536 dims)
EMBEDDING_MODEL = "gemini/gemini-embedding-001"
EMBEDDING_DIMS = 3072

mcp = FastMCP("research")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def _get_embedding(text: str) -> List[float]:
    response = embedding(model=EMBEDDING_MODEL, input=[text])
    return response.data[0]["embedding"]


def _update_indexed_flag(paper_id: str):
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
            except FileNotFoundError, json.JSONDecodeError:
                continue


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
    except FileNotFoundError, json.JSONDecodeError:
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
def extract_chunks(paper_id: str, pdf_url: str) -> str:
    """
    Step 1 of indexing: Fetch PDF, extract markdown, return text chunks.
    No embeddings yet — just clean text chunks ready for embedding.

    After calling this tool, for each chunk you should:
      1. Call embed_chunk(chunk_text) to get the vector
      2. Call RedisMCPServer:hset to store text fields
      3. Call RedisMCPServer:set_vector_in_hash to store the vector

    Redis key format: doc:paper:{paper_id}:chunk:{index}

    Args:
        paper_id: arXiv paper ID
        pdf_url:  Direct PDF URL

    Returns:
        JSON with paper_id, total_chunks, and list of {index, redis_key, text}
    """
    try:
        response = httpx.get(pdf_url, timeout=30, follow_redirects=True)
        response.raise_for_status()

        doc = fitz.open(stream=response.content, filetype="pdf")
        md_text = pymupdf4llm.to_markdown(doc)
        doc.close()

        if not md_text.strip():
            return json.dumps(
                {"status": "error", "error": "No text extracted from PDF"}
            )

        raw_chunks = [c.strip() for c in md_text.split("\n## ") if c.strip()]

        chunks = [
            {
                "index": i,
                "redis_key": f"{REDIS_PREFIX}paper:{paper_id}:chunk:{i}",
                "text": chunk,
            }
            for i, chunk in enumerate(raw_chunks)
        ]

        return json.dumps(
            {
                "status": "ok",
                "paper_id": paper_id,
                "total_chunks": len(chunks),
                "chunks": chunks,
            },
            ensure_ascii=False,
        )

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
def embed_chunk(text: str) -> str:
    """
    Step 2 of indexing: Generate embedding vector for a single chunk.
    Call this once per chunk, then store the result with RedisMCPServer tools.

    After calling this tool:
      - Call RedisMCPServer:set_vector_in_hash(name=redis_key, vector=<returned vector>)

    Args:
        text: The chunk text to embed

    Returns:
        JSON with the embedding vector as a list of floats
    """
    try:
        vector = _get_embedding(text)
        return json.dumps({"status": "ok", "vector": vector})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
def mark_paper_indexed(paper_id: str) -> str:
    """
    Step 3 of indexing: Mark a paper as fully indexed in local metadata.
    Call this after all chunks have been embedded and stored in Redis.

    Args:
        paper_id: arXiv paper ID

    Returns:
        Status message
    """
    _update_indexed_flag(paper_id)
    return f"Paper '{paper_id}' marked as indexed."


if __name__ == "__main__":
    mcp.run(transport="stdio")
