# research-extract-pdf-papers

> **MCP server that searches arXiv for research papers, downloads their PDFs, extracts structured content, and indexes it with semantic embeddings — all through a Model Context Protocol (MCP) interface.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Tech Stack & Language Composition](#3-tech-stack--language-composition)
4. [Repository Structure](#4-repository-structure)
5. [Installation & Setup](#5-installation--setup)
6. [Usage](#6-usage)
7. [Configuration](#7-configuration)
8. [Testing](#8-testing)
9. [Troubleshooting](#9-troubleshooting)
10. [Contributing](#10-contributing)
11. [License](#11-license)
12. [Security & Privacy Notes](#12-security--privacy-notes)

---

## 1. Project Overview

**research-extract-pdf-papers** is a Python-based [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that enables AI assistants and LLM pipelines to:

- **Search** arXiv for research papers on any topic.
- **Download and parse** paper PDFs into clean Markdown text.
- **Chunk** the extracted text into sections ready for embedding.
- **Embed** chunks using configurable embedding models (Google Gemini, Voyage AI, or OpenAI).
- **Store** embeddings in Redis for later semantic retrieval.
- **Retrieve** saved paper metadata by paper ID.

### Primary Use Cases

| Use case | Description |
|---|---|
| Research assistant | Feed an LLM the full text of relevant papers retrieved by topic. |
| Semantic paper search | Build a vector store of paper chunks for RAG (Retrieval-Augmented Generation). |
| Literature review automation | Automate discovery, download, and indexing of papers on any topic. |
| Citation and metadata lookup | Quickly look up author, summary, and publication date for a paper ID. |

---

## 2. Features

- 🔍 **arXiv search** — query arXiv by topic with configurable result count; results are saved locally as JSON.
- 📄 **PDF extraction** — fetches PDFs over HTTP and converts them to Markdown using PyMuPDF/pymupdf4llm.
- ✂️ **Section-aware chunking** — splits papers at `##`-level headings to produce meaningful text chunks.
- 🧮 **Pluggable embedding models** — supports Gemini (`gemini-embedding-001`), Voyage (`voyage-3`), and OpenAI (`text-embedding-3-small`) via LiteLLM.
- 🗄️ **Redis-compatible key scheme** — chunks are keyed as `doc:paper:<paper_id>:chunk:<index>` for easy retrieval.
- ✅ **Indexed-state tracking** — local JSON metadata tracks whether a paper has been fully embedded.
- 🐳 **Docker support** — includes a `Dockerfile` for containerised deployment.
- 🧪 **Test suite** — pytest-based tests with mocks for all network/file I/O.
- 🛠️ **Makefile automation** — one-command format, lint, test, and coverage workflows.

---

## 3. Tech Stack & Language Composition

| Layer | Technology |
|---|---|
| Language | Python 3.14 |
| MCP framework | [`mcp[fastmcp]`](https://github.com/modelcontextprotocol/python-sdk) |
| arXiv API client | [`arxiv`](https://pypi.org/project/arxiv/) |
| PDF parsing | [`PyMuPDF` (`fitz`)](https://pymupdf.readthedocs.io/) + [`pymupdf4llm`](https://pypi.org/project/pymupdf4llm/) |
| HTTP client | [`httpx`](https://www.python-httpx.org/) |
| Embedding models | [`litellm`](https://github.com/BerriAI/litellm) (Gemini / Voyage / OpenAI) |
| Package manager | [`uv`](https://github.com/astral-sh/uv) |
| Code formatting | [`black`](https://black.readthedocs.io/) |
| Linting | [`pylint`](https://pylint.readthedocs.io/) |
| Testing | [`pytest`](https://docs.pytest.org/) + [`pytest-cov`](https://pytest-cov.readthedocs.io/) |
| Containerisation | Docker (python:3.12-slim base — see note below) |

> **⚠️ Note:** `pyproject.toml` declares `requires-python = ">=3.14"` and `.python-version` pins `3.14`, but the `Dockerfile` currently uses `python:3.12-slim`. For a fully consistent container build, update the `FROM` line in `Dockerfile` to `python:3.14-slim` once a stable image is available.
| Environment | [`python-dotenv`](https://pypi.org/project/python-dotenv/) |

---

## 4. Repository Structure

```
research-extract-pdf-papers/
├── research_server.py   # Main MCP server — all tool definitions live here
├── main.py              # Minimal standalone entrypoint (prints hello message)
├── test_server.py       # pytest test suite for research_server.py
├── pyproject.toml       # Project metadata and dependency declarations (uv/pip)
├── uv.lock              # Locked dependency versions for reproducible installs
├── Makefile             # Dev automation: format / lint / test / coverage / clean
├── Dockerfile           # Container build definition
├── .dockerignore        # Files excluded from Docker build context
├── .gitignore           # Git-ignored paths (envs, caches, PDFs, logs)
├── .python-version      # Pinned Python version (3.14, used by pyenv/uv)
├── LICENSE              # MIT License
└── papers/              # Auto-created at runtime; stores per-topic sub-directories
    └── <topic>/
        └── papers_info.json  # Metadata for papers found under that topic
```

### Key file: `research_server.py`

Contains five MCP tools exposed to any connected MCP client:

| Tool | Purpose |
|---|---|
| `search_papers(topic, max_results)` | Search arXiv and save metadata to `papers/<topic>/papers_info.json` |
| `extract_info(paper_id)` | Return saved metadata JSON for a specific paper ID |
| `extract_chunks(paper_id, pdf_url)` | Download PDF, convert to Markdown, split into numbered chunks |
| `embed_chunk(text)` | Generate an embedding vector for a single chunk of text |
| `mark_paper_indexed(paper_id)` | Set `indexed: true` in local metadata once a paper is fully stored |

---

## 5. Installation & Setup

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.14 |
| [`uv`](https://github.com/astral-sh/uv) | latest recommended |
| Redis | any recent version (needed at runtime for vector storage) |
| An embedding API key | Gemini, Voyage AI, or OpenAI |

### Step-by-step local setup

```bash
# 1. Clone the repository
git clone https://github.com/Muhammadyousafrana/research-extract-pdf-papers.git
cd research-extract-pdf-papers

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install project dependencies
uv pip install .

# 4. Copy and configure environment variables
cp .env.example .env          # create .env from the example (see Configuration section)
# Edit .env and fill in at least one embedding API key

# 5. Run the MCP server (stdio transport — suitable for MCP clients)
python research_server.py
```

### Docker setup

```bash
# Build the image
docker build -t research-mcp .

# Run (pass env vars at runtime)
docker run --rm -e GEMINI_API_KEY=your_key research-mcp
```

---

## 6. Usage

### Running the MCP server

The server uses **stdio transport** and is designed to be connected to an MCP-compatible client (e.g. Claude Desktop, an LLM pipeline, or any MCP host).

```bash
python research_server.py
```

Once connected, an MCP client can call the tools directly. Below are example interactions:

### Example: search for papers

```python
# Via MCP client call
search_papers(topic="retrieval augmented generation", max_results=5)
# Returns: ["2301.07041", "2305.14283", ...]
# Side effect: saves papers/retrieval_augmented_generation/papers_info.json
```

### Example: retrieve paper metadata

```python
extract_info(paper_id="2301.07041")
# Returns JSON:
# {
#   "title": "Precise Zero-Shot Dense Retrieval without Relevance Labels",
#   "authors": ["Luyu Gao", "Xueguang Ma", ...],
#   "summary": "...",
#   "pdf_url": "https://arxiv.org/pdf/2301.07041",
#   "published": "2023-01-17",
#   "indexed": false
# }
```

### Example: extract PDF chunks (Step 1 of indexing)

```python
extract_chunks(
    paper_id="2301.07041",
    pdf_url="https://arxiv.org/pdf/2301.07041"
)
# Returns JSON:
# {
#   "status": "ok",
#   "paper_id": "2301.07041",
#   "total_chunks": 12,
#   "chunks": [
#     {"index": 0, "redis_key": "doc:paper:2301.07041:chunk:0", "text": "..."},
#     ...
#   ]
# }
```

### Example: embed a chunk (Step 2 of indexing)

```python
embed_chunk(text="Retrieval-augmented generation combines parametric...")
# Returns JSON:
# {"status": "ok", "vector": [0.012, -0.034, ..., 0.091]}   # 3072 floats (Gemini)
```

### Example: mark paper as indexed (Step 3 of indexing)

```python
mark_paper_indexed(paper_id="2301.07041")
# Returns: "Paper '2301.07041' marked as indexed."
```

### Full indexing workflow

```
1. search_papers(topic)          → get list of paper IDs
2. extract_info(paper_id)        → get pdf_url from metadata
3. extract_chunks(paper_id, url) → get text chunks + redis_key for each
4. For each chunk:
     embed_chunk(chunk.text)     → get vector
     RedisMCPServer:hset(...)    → store text fields
     RedisMCPServer:set_vector_in_hash(...) → store vector
5. mark_paper_indexed(paper_id)  → flag as done
```

---

## 7. Configuration

Configuration is provided via environment variables (loaded automatically with `python-dotenv` if a `.env` file is present).

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | If using Gemini | API key for Google Gemini embeddings |
| `VOYAGE_API_KEY` | If using Voyage | API key for Voyage AI embeddings |
| `OPENAI_API_KEY` | If using OpenAI | API key for OpenAI embeddings |

Create a `.env` file in the project root:

```dotenv
# .env — fill in whichever embedding provider you use
GEMINI_API_KEY=your_gemini_key_here
# VOYAGE_API_KEY=your_voyage_key_here
# OPENAI_API_KEY=your_openai_key_here
```

### Embedding model selection

The active model is controlled by two constants in `research_server.py`:

```python
# Gemini (default) — 3072 dimensions
EMBEDDING_MODEL = "gemini/gemini-embedding-001"
EMBEDDING_DIMS  = 3072

# Voyage — uncomment to switch
# EMBEDDING_MODEL = "voyage/voyage-3"
# EMBEDDING_DIMS  = 1024

# OpenAI — uncomment to switch
# EMBEDDING_MODEL = "openai/text-embedding-3-small"
# EMBEDDING_DIMS  = 1536
```

### Redis key prefix

Chunk keys follow the pattern `doc:paper:<paper_id>:chunk:<index>`. The prefix `doc:` is configurable via the `REDIS_PREFIX` constant in `research_server.py`.

### Paper storage directory

Downloaded paper metadata is stored under `./papers/` by default. This is controlled by the `PAPER_DIR` constant in `research_server.py`.

---

## 8. Testing

The project uses **pytest** with unit tests in `test_server.py`. All network and file I/O is mocked so tests run entirely offline.

### Run the test suite

```bash
# Using make (recommended)
make test

# Or directly with pytest
python -m pytest test_server.py -v --tb=short
```

### Run with coverage

```bash
make coverage
# Opens a full HTML report at htmlcov/index.html
```

### Run formatting and linting checks

```bash
make format        # auto-format with black
make format-check  # check formatting without changes (suitable for CI)
make lint          # run pylint (fails if score < 7.0)
```

### Run everything (format + lint + test)

```bash
make
```

### Test organisation

Tests are grouped by component in `test_server.py`:

| Test class | What it covers |
|---|---|
| `TestUpdateIndexedFlag` | `_update_indexed_flag` helper — file writes, missing dir, corrupt JSON |
| `TestExtractInfo` | `extract_info` tool — found, not found, corrupt JSON |
| `TestMarkPaperIndexed` | `mark_paper_indexed` tool — return message, file update |
| `TestEmbedChunk` | `embed_chunk` tool — success vector, API failure |
| `TestExtractChunks` | `extract_chunks` tool — success, empty PDF, HTTP failure, key format |

---

## 9. Troubleshooting

### PDF parsing produces no text

**Symptom:** `extract_chunks` returns `{"status": "error", "error": "No text extracted from PDF"}`

**Causes & fixes:**
- The PDF may be image-only (scanned). Install `tesseract-ocr` (already included in the Docker image) and confirm `pymupdf4llm` is using OCR.
- The `pdf_url` may redirect to a login page. Verify the URL manually with `curl -L <url>` and check for HTTP 200.
- Some arXiv papers have restricted access. Use the `/pdf/` variant of the URL, not the abstract page.

### HTTP errors when downloading PDFs

**Symptom:** `{"status": "error", "error": "connection refused"}` or `httpx.HTTPStatusError`

**Fixes:**
- Check your internet connection / corporate proxy settings.
- arXiv sometimes rate-limits rapid downloads. Add a short delay between calls.
- The `httpx.get` call uses a 30-second timeout. Increase it in `research_server.py` if needed: `timeout=60`.

### Embedding API key errors

**Symptom:** `litellm.AuthenticationError` or similar

**Fixes:**
- Ensure the correct environment variable is set (`GEMINI_API_KEY`, `VOYAGE_API_KEY`, or `OPENAI_API_KEY`).
- Confirm the key is loaded — add `from dotenv import load_dotenv; load_dotenv()` at the top of `research_server.py` if it is not already.
- Verify the key has sufficient quota/credits.

### `papers_info.json` not found

**Symptom:** `extract_info` always returns `"There's no saved information related to paper <id>."`

**Fix:** Run `search_papers` first for the relevant topic. The JSON file is created on the first successful search call.

### Python version mismatch

**Symptom:** Syntax errors or import failures on Python < 3.14

**Fix:** The project requires Python 3.14+. Use `pyenv install 3.14` or switch to the Docker image which uses Python 3.12 (adjust `pyproject.toml` `requires-python` if you need to target 3.12).

### `uv` command not found

**Fix:** Install `uv` with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 10. Contributing

Contributions are welcome! Please follow this workflow:

1. **Fork** the repository on GitHub.
2. **Create a branch** for your change:
   ```bash
   git checkout -b feat/my-new-feature
   ```
3. **Make your changes**, then run the full quality suite:
   ```bash
   make          # format + lint + test
   ```
4. **Commit** with a clear, descriptive message:
   ```bash
   git commit -m "feat: add support for Semantic Scholar API"
   ```
5. **Push** your branch and open a **Pull Request** against `main`.

### Guidelines

- Keep functions focused and add/update docstrings to match.
- New tools should follow the existing MCP `@mcp.tool()` decorator pattern.
- All new code must be covered by tests in `test_server.py`.
- Run `make format-check` and `make lint` before submitting — CI will enforce both.

---

## 11. License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for the full text.

---

## 12. Security & Privacy Notes

### Handling PDFs safely

- PDFs are fetched over HTTPS and parsed entirely in memory using PyMuPDF — no temporary PDF files are written to disk.
- Malicious PDFs can exploit PDF parser vulnerabilities. Keep `pymupdf` / `fitz` updated to the latest version.
- Do not pass untrusted, user-supplied URLs directly to `extract_chunks` without validation; an attacker could cause the server to make requests to internal network resources (SSRF). Add URL allowlisting (e.g. restrict to `arxiv.org`) if the server is exposed to untrusted inputs.

### API keys & secrets

- Never commit `.env` files or API keys to version control. `.env` is already listed in `.gitignore`.
- Rotate any key that is accidentally exposed.
- Use environment-specific secrets management (e.g. Docker secrets, GitHub Actions secrets) rather than plain `.env` files in production.

### Sensitive data in papers

- Downloaded paper text may include names, institutional affiliations, and other personally identifiable information. Handle extracted content in accordance with applicable data-protection regulations (GDPR, etc.) if your use case re-exposes this data.
- Embedding vectors stored in Redis should be protected with authentication and TLS in production deployments.

### Dependency security

- Dependencies are locked in `uv.lock` for reproducibility.
- Regularly audit dependencies with `uv pip check` or a tool such as `pip-audit`.

