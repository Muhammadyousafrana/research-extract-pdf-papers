# ── MUST be first: redirect all print() to stderr before any library imports ──
# Any library that prints during import or runtime corrupts the MCP stdio protocol
import builtins as _builtins
import sys as _sys
_orig_print = _builtins.print
_builtins.print = lambda *a, **kw: _orig_print(*a, **kw) if 'file' in kw else _orig_print(*a, file=_sys.stderr, **kw)

import asyncio
import json
import logging
import os
from typing import List

import arxiv
import fitz
import httpx
import numpy as np
import redis.asyncio as aioredis
from dotenv import load_dotenv
from litellm import embedding, completion
import litellm
litellm.suppress_debug_info = True
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("research")
logging.basicConfig(level=logging.INFO, stream=__import__("sys").stderr)
_log_file = __import__("sys").stderr
try:
    import tempfile, atexit
    _fh = __import__("logging").FileHandler(__import__("os").path.join(tempfile.gettempdir(), "research_server_crash.log"), mode="a")
    _fh.setLevel(__import__("logging").INFO)
    _fh.setFormatter(__import__("logging").Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)
    import builtins as _b
    _b._logfile = _fh
    import atexit as _atexit
    _atexit.register(lambda: logger.info("research_server.py EXIT"))
except Exception:
    pass

# Log any unhandled exception before the process dies
def _fatal_hook(exc_type, exc_value, exc_tb):
    import traceback
    logger.critical("Unhandled exception: %s", "".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
    __import__("sys").stderr.flush()
__import__("sys").excepthook = _fatal_hook

load_dotenv()

PAPER_DIR = "papers"
REDIS_PREFIX = "doc:"

# Embedding config
EMBEDDING_MODEL = "gemini/gemini-embedding-001"
EMBEDDING_DIMS = 3072

# LLM config for answer generation
LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemma-4-31b-it")

# Direct Redis connection (no MCP subprocess — eliminates subprocess crash issues)
REDIS_URL = os.getenv("REDIS_URL")
_redis: aioredis.Redis | None = None


async def _ensure_redis() -> aioredis.Redis:
    """Return a direct Redis connection, creating one lazily."""
    global _redis
    if _redis is not None:
        try:
            await _redis.ping()
            return _redis
        except Exception:
            logger.info("Redis connection lost, reconnecting...")
            _redis = None
    r = aioredis.Redis.from_url(
        REDIS_URL,
        protocol=2,
        socket_connect_timeout=15,
        socket_timeout=30,
        retry_on_timeout=True,
        decode_responses=True,
    )
    await r.ping()
    logger.info("Direct Redis connection ready")
    _redis = r
    return _redis


async def _redis_init_index():
    """Create the vector search index if it doesn't already exist."""
    try:
        r = await _ensure_redis()
        from redis.commands.search.field import TextField, VectorField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        schema = (
            TextField("text"),
            TextField("paper_id"),
            TextField("chunk_index"),
            VectorField("vector", "FLAT", {"TYPE": "FLOAT32", "DIM": EMBEDDING_DIMS, "DISTANCE_METRIC": "COSINE"}),
        )
        definition = IndexDefinition(prefix=["doc:paper:"], index_type=IndexType.HASH)
        await r.ft("idx:chunks").create_index(schema, definition=definition)
        logger.info("Vector index 'idx:chunks' created")
    except Exception as e:
        logger.info("Vector index creation note (may already exist): %s", e)


def _pack_vector(vector: list) -> bytes:
    """Pack a float list into float32 bytes for Redis vector storage."""
    return np.array(vector, dtype=np.float32).tobytes()

mcp = FastMCP("research")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def _get_embedding(text: str) -> List[float]:
    """Generate an embedding vector for a single text string."""
    response = embedding(model=EMBEDDING_MODEL, input=[text])
    return response.data[0]["embedding"]


def _get_embeddings_batch(texts: List[str], batch_size: int = 10, max_retries: int = 3) -> List[List[float]]:
    """Generate embedding vectors in small batches with retry on 429."""
    results: List[List[float]] = []
    import time as _time
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        for attempt in range(max_retries):
            try:
                response = embedding(model=EMBEDDING_MODEL, input=batch)
                results.extend(item["embedding"] for item in response.data)
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning("Rate limited on batch %d, retrying in %ds...", start // batch_size, wait)
                    _sys.stderr.flush()
                    _time.sleep(wait)
                else:
                    raise
    return results


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


def _extract_text_chunks(pdf_bytes: bytes) -> tuple[list[str], str]:
    """Extract text from PDF bytes using PyMuPDF directly and split into chunks.
    Returns (chunks, error_msg)."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            all_text += page.get_text("text") + "\n"
        doc.close()
    except Exception as e:
        logger.warning("PDF extraction failed: %s", e)
        return [], f"PDF extraction error: {e}"

    if not all_text.strip():
        return [], "PDF produced no text content"

    paras = [p.strip() for p in all_text.split("\n\n") if p.strip()]

    MIN_CHUNK = 400
    MAX_CHUNK = 2000
    chunks = []
    buf = ""
    for para in paras:
        if len(buf) + len(para) < MAX_CHUNK and (not buf or len(buf) < MIN_CHUNK):
            buf = (buf + "\n\n" + para).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)

    if not chunks:
        return [], "Splitting produced no chunks"
    return chunks, ""



def _generate_answer(question: str, chunks: list) -> str:
    """Generate an LLM answer grounded on the retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        pid = chunk.get("paper_id", "unknown") if isinstance(chunk, dict) else "unknown"
        cidx = chunk.get("chunk_index", i - 1) if isinstance(chunk, dict) else i - 1
        context_parts.append(f"[Source: paper {pid}, chunk #{cidx}]\n{text}")

    context = "\n\n".join(context_parts)

    response = completion(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a research assistant. Answer the user's question concisely based on the provided paper excerpts. Cite sources in brackets like [paper ID, chunk #N]. If the context lacks enough information, say so rather than guessing.",
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
            },
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


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
        List of JSON strings, each containing paper_id and title
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

    results = []
    for paper in papers:
        pid = paper.get_short_id()
        papers_info[pid] = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "pdf_url": str(paper.pdf_url),
            "published": str(paper.published.date()),
            "indexed": False,
        }
        results.append(json.dumps({"paper_id": pid, "title": paper.title}))

    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)

    logger.info("Results saved in: %s", file_path)
    return results


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
                    logger.warning("Error reading %s: %s", file_path, e)
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
        logger.info("index_paper[%s]: downloading PDF...", paper_id)
        _sys.stderr.flush()
        response = httpx.get(pdf_url, timeout=60, follow_redirects=True)
        response.raise_for_status()
        logger.info("index_paper[%s]: PDF downloaded (%.1f KB)", paper_id, len(response.content) / 1024)
        _sys.stderr.flush()

        # ── Step 2: Extract text chunks ───────────────────────────────
        logger.info("index_paper[%s]: extracting text chunks...", paper_id)
        _sys.stderr.flush()
        chunks, err = _extract_text_chunks(response.content)
        if err or not chunks:
            return json.dumps(
                {"status": "error", "error": f"No text extracted from PDF: {err}"}
            )
        logger.info("index_paper[%s]: extracted %d chunks", paper_id, len(chunks))
        _sys.stderr.flush()

        # ── Step 3: Batch embed all chunks ────────────────────────────
        logger.info("index_paper[%s]: embedding %d chunks via %s...", paper_id, len(chunks), EMBEDDING_MODEL)
        _sys.stderr.flush()
        vectors = _get_embeddings_batch(chunks)
        logger.info("index_paper[%s]: embedding done (%d vectors)", paper_id, len(vectors))
        _sys.stderr.flush()

        # ── Step 4: Look up paper metadata from local files ───────────
        paper_meta = None
        info_str = extract_info(paper_id)
        if not info_str.startswith("There's no saved"):
            paper_meta = json.loads(info_str)

        # ── Step 5: Push everything to Redis directly ─────────────────
        try:
            r = await _ensure_redis()
        except Exception as e:
            logger.error("Failed to connect to Redis: %s", e)
            return json.dumps({"status": "error", "paper_id": paper_id, "error": f"Redis connection failed: {e}"})

        # Create vector index (safely ignore if it already exists)
        await _redis_init_index()

        # Store paper metadata as JSON
        meta_to_store = paper_meta or {
            "paper_id": paper_id,
            "pdf_url": pdf_url,
        }
        try:
            await r.json().set(f"paper:{paper_id}", "$", meta_to_store)
        except Exception as e:
            logger.error("Failed to store paper metadata in Redis: %s", e)

        # Batch-store all chunks in Redis via one pipeline
        logger.info("index_paper[%s]: pushing %d chunks to Redis via pipeline...", paper_id, len(chunks))
        _sys.stderr.flush()
        pipe = r.pipeline(transaction=False)
        for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
            redis_key = f"{REDIS_PREFIX}paper:{paper_id}:chunk:{i}"
            pipe.hset(redis_key, mapping={
                "text": chunk_text,
                "paper_id": paper_id,
                "chunk_index": str(i),
                "vector": _pack_vector(vector),
            })
        stored = len(chunks)
        try:
            await pipe.execute()
        except Exception as e:
            logger.error("Redis pipeline failed after %d chunks: %s", stored, e)
            return json.dumps({"status": "error", "paper_id": paper_id, "error": f"Redis write failed: {e}"})

        if stored:
            _update_indexed_flag(paper_id)

        return json.dumps(
            {
                "status": "ok" if stored else "error",
                "paper_id": paper_id,
                "title": (paper_meta or {}).get("title", "Unknown"),
                "total_chunks": len(chunks),
                "stored_chunks": stored,
                "message": f"Indexed {stored}/{len(chunks)} chunks into Redis",
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
        logger.info("query_paper: embedding question (len=%d) via LLM API...", len(question))
        import sys as _sys
        _sys.stderr.flush()
        try:
            query_vector = _get_embedding(question)
        except Exception as e:
            logger.error("query_paper: embedding failed: %s", e)
            return json.dumps({"status": "error", "error": f"Embedding failed: {e}"})

        # ── Step 2: Search Redis directly with vector similarity ──────
        logger.info("query_paper: performing vector search...")
        _sys.stderr.flush()

        try:
            r = await _ensure_redis()
            from redis.commands.search.query import Query
            query_vector_bytes = _pack_vector(query_vector)
            q = Query("*=>[KNN 15 @vector $vec AS score]") \
                .return_fields("text", "paper_id", "chunk_index", "score") \
                .sort_by("score") \
                .dialect(2)
            search_result = await r.ft("idx:chunks").search(
                q, query_params={"vec": query_vector_bytes}
            )
        except Exception as e:
            logger.error("query_paper: vector search failed: %s", e)
            _sys.stderr.flush()
            return json.dumps({"status": "error", "error": f"vector search failed: {e}"})

        # Parse results
        results = []
        if hasattr(search_result, "docs"):
            for doc in search_result.docs:
                payload = {"text": doc.text, "paper_id": doc.paper_id, "chunk_index": doc.chunk_index, "score": doc.score}
                if paper_id and paper_id != "all":
                    if payload.get("paper_id") == paper_id:
                        results.append(payload)
                else:
                    results.append(payload)

        # ── Step 3: Generate LLM answer from chunks ────────────────────
        answer = _generate_answer(question, results) if results else ""

        return json.dumps(
            {
                "status": "ok",
                "question": question,
                "paper_id": paper_id or "all",
                "answer": answer,
                "results": results,
            },
            ensure_ascii=False,
        )

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
async def admin_list_papers() -> str:
    """List all indexed paper IDs stored in Redis.

    Returns:
        JSON array of paper IDs that have been indexed.
    """
    try:
        r = await _ensure_redis()
        paper_ids = set()
        cursor = 0
        while True:
            cursor, keys = await r.scan(cursor=cursor, match="doc:paper:*:chunk:0", count=100)
            for k in keys:
                parts = k.split(":")
                if len(parts) >= 3:
                    paper_ids.add(parts[2])
            if cursor == 0:
                break
        return json.dumps(sorted(paper_ids))
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def admin_delete_paper(paper_id: str) -> str:
    """Delete all indexed chunks and metadata for a paper from Redis.

    Args:
        paper_id: arXiv paper ID to remove

    Returns:
        JSON with status and count of deleted keys
    """
    try:
        r = await _ensure_redis()
        keys_to_delete = []
        cursor = 0
        while True:
            cursor, keys = await r.scan(cursor=cursor, match=f"doc:paper:{paper_id}:*", count=100)
            keys_to_delete.extend(keys)
            if cursor == 0:
                break
        keys_to_delete.append(f"paper:{paper_id}")
        if keys_to_delete:
            await r.delete(*keys_to_delete)
        return json.dumps({"status": "ok", "paper_id": paper_id, "deleted_keys": len(keys_to_delete)})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
async def admin_redis_stats() -> str:
    """Get Redis stats: total indexed papers and chunks.

    Returns:
        JSON with paper_count and chunk_count
    """
    try:
        r = await _ensure_redis()
        papers = set()
        cursor = 0
        while True:
            cursor, keys = await r.scan(cursor=cursor, match="doc:paper:*:chunk:0", count=100)
            for k in keys:
                parts = k.split(":")
                if len(parts) >= 3:
                    papers.add(parts[2])
            if cursor == 0:
                break
        chunk_count = 0
        cursor = 0
        while True:
            cursor, keys = await r.scan(cursor=cursor, match="doc:paper:*:chunk:*", count=100)
            chunk_count += len(keys)
            if cursor == 0:
                break
        return json.dumps({"paper_count": len(papers), "chunk_count": chunk_count})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def admin_redis_clear() -> str:
    """Delete ALL indexed paper chunks and metadata from Redis.

    Returns:
        JSON with status and count of deleted keys.
    """
    try:
        r = await _ensure_redis()
        all_keys: list[str] = []
        for pattern in ["doc:paper:*", "paper:*"]:
            cursor = 0
            while True:
                cursor, keys = await r.scan(cursor=cursor, match=pattern, count=100)
                all_keys.extend(keys)
                if cursor == 0:
                    break
        if all_keys:
            await r.delete(*all_keys)
        return json.dumps({"status": "ok", "deleted_keys": len(all_keys)})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


if __name__ == "__main__":
    mcp.run(transport="stdio")
