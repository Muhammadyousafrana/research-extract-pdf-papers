import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import sys
import time
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from supabase import create_client, Client
import resend

load_dotenv()

logger = logging.getLogger("web_app")
logging.basicConfig(level=logging.INFO, stream=__import__("sys").stderr)

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8080")

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html") if os.path.isdir(STATIC_DIR) else None
SCHEMA_SQL = os.path.join(HERE, "supabase_schema.sql")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

_TABLES_OK = False

session: ClientSession | None = None
exit_stack: AsyncExitStack | None = None

# Simple in-memory cache with TTL for admin Redis endpoints
_cache: dict[str, tuple[float, Any]] = {}
CACHE_TTL_REDIS = 15  # seconds — Redis data changes infrequently

def _cache_get(key: str):
    entry = _cache.get(key)
    if entry:
        expire, value = entry
        if expire > time.time():
            return value
        del _cache[key]
    return None

def _cache_set(key: str, value: Any, ttl: float = CACHE_TTL_REDIS):
    _cache[key] = (time.time() + ttl, value)

# Tool log infrastructure
_log_buffers: dict[str, list[dict]] = {}
_log_subscribers: dict[str, list[asyncio.Queue]] = {}


async def _ensure_users_rls() -> bool:
    """Try to create RLS policy for users table via Supabase pg_ddl endpoint."""
    try:
        import httpx
        sql = """
DO $$ BEGIN
  CREATE POLICY "anon_all_users" ON users FOR ALL TO anon USING (true) WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
"""
        ddl_url = SUPABASE_URL.rstrip("/") + "/rest/v1/pg_ddl"
        headers = {"apikey": SUPABASE_KEY, "Authorization": "Bearer " + SUPABASE_KEY, "Content-Type": "application/json"}
        res = httpx.post(ddl_url, json={"query": sql}, headers=headers, timeout=15)
        return res.status_code in (200, 201)
    except Exception:
        return False


async def _ensure_admin_table() -> bool:
    """Try to create the admins table via Supabase pg_ddl endpoint."""
    try:
        import httpx
        sql = """
CREATE TABLE IF NOT EXISTS admins (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  username TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
DO $$ BEGIN
  CREATE POLICY "anon_all_admins" ON admins FOR ALL TO anon USING (true) WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
"""
        ddl_url = SUPABASE_URL.rstrip("/") + "/rest/v1/pg_ddl"
        headers = {"apikey": SUPABASE_KEY, "Authorization": "Bearer " + SUPABASE_KEY, "Content-Type": "application/json"}
        res = httpx.post(ddl_url, json={"query": sql}, headers=headers, timeout=15)
        return res.status_code in (200, 201)
    except Exception:
        return False

# Admin auth
_admin_tokens: dict[str, str] = {}  # token -> username

# User auth
_user_sessions: dict[str, dict] = {}  # session_token -> {user_id, email, username}

# Active paper per conversation (conv_id -> {paper_id, title})
_active_papers: dict[str, dict] = {}


def send_verification_email(user_email: str, verify_token: str):
    """Send verification email via Resend."""
    if not RESEND_API_KEY:
        return False
    verify_link = f"{BASE_URL}/api/auth/verify?token={verify_token}"
    try:
        params: resend.Emails.SendParams = {
            "from": "Research Chat <noreply@yousafrana.me>",
            "to": [user_email],
            "subject": "Verify your email address",
            "html": f"""
                <div style="font-family: sans-serif; padding: 20px; max-width: 500px;">
                    <h2>Verify your account</h2>
                    <p>Thank you for signing up! Click the button below to verify your email address:</p>
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{verify_link}" style="background: #6366f1; color: #fff; padding: 12px 28px; text-decoration: none; border-radius: 6px; font-weight: 600; display: inline-block;">Verify Email</a>
                    </div>
                    <p style="color: #666; font-size: 13px;">Or copy this link into your browser:<br><span style="color: #6366f1;">{verify_link}</span></p>
                    <p style="color: #999; font-size: 12px; margin-top: 20px;">If you did not create this account, please ignore this email.</p>
                </div>
            """,
        }
        resend.Emails.send(params)
        return True
    except Exception as e:
        print(f"Failed to send verification email: {e}")
        return False


ADMIN_HTML = os.path.join(STATIC_DIR, "admin.html") if os.path.isdir(STATIC_DIR) else None


def _hash_password(password: str) -> str:
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
    return base64.b64encode(salt + key).decode()


def _verify_password(password: str, stored: str) -> bool:
    try:
        data = base64.b64decode(stored.encode())
        salt, key = data[:32], data[32:]
        return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000) == key
    except Exception:
        return False


def _require_admin(request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, detail="Unauthorized")
    token = auth[7:]
    if token not in _admin_tokens:
        raise HTTPException(401, detail="Invalid or expired token")
    return _admin_tokens[token]


def _publish_log(conv_id: str, tool: str, status: str, message: str, detail: dict = None):
    entry = {
        "tool": tool,
        "status": status,
        "message": message,
        "detail": detail or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if conv_id not in _log_buffers:
        _log_buffers[conv_id] = []
    _log_buffers[conv_id].append(entry)
    if conv_id in _log_subscribers:
        dead = []
        for q in _log_subscribers[conv_id]:
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            _log_subscribers[conv_id].remove(q)


def _get_user_from_request(request: Request) -> dict | None:
    """Extract authenticated user from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:]
    return _user_sessions.get(token)


def _verify_conv_owner(conv_id: str, user: dict | None):
    """Verify the authenticated user owns this conversation (no-op if no auth)."""
    if user is None:
        return
    check = supabase.table("conversations").select("id").eq("id", conv_id).eq("user_id", user["user_id"]).execute()
    if not check.data:
        raise HTTPException(404, "Conversation not found")


def _verify_tables():
    global _TABLES_OK
    if _TABLES_OK or supabase is None:
        return True
    try:
        supabase.table("conversations").select("id").limit(1).execute()
        _TABLES_OK = True
        return True
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, exit_stack
    exit_stack = AsyncExitStack()
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["research_server.py"],
        env={**os.environ},
    )
    try:
        read, write = await exit_stack.enter_async_context(stdio_client(server_params))
        session = await exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
    except Exception as e:
        import traceback
        msg = f"MCP session init failed (will retry on first request): {e}"
        print(msg, file=__import__("sys").stderr)
        traceback.print_exc()
        logger.warning(msg)
        session = None
        try:
            await exit_stack.aclose()
        except Exception:
            pass
        exit_stack = AsyncExitStack()
    _verify_tables()
    yield
    try:
        if exit_stack:
            await exit_stack.aclose()
    except Exception:
        pass  # ignore shutdown ordering issues with stdio_client cancel scopes


app = FastAPI(title="Research Paper Chat", lifespan=lifespan)

# ── Security middleware ──

app.add_middleware(
    CORSMiddleware,
    allow_origins=[BASE_URL],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response

# Simple in-memory rate limiter (per-IP, sliding window)
_rate_limit: dict[str, list[float]] = {}
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 20     # requests per window

def _check_rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = now - RATE_LIMIT_WINDOW
    timestamps = _rate_limit.get(client_ip, [])
    timestamps = [t for t in timestamps if t > window]
    if len(timestamps) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    timestamps.append(now)
    _rate_limit[client_ip] = timestamps


@app.get("/")
async def serve_index():
    if INDEX_HTML and os.path.isfile(INDEX_HTML):
        return FileResponse(INDEX_HTML)
    return {"error": "index.html not found"}


# ── Helpers ──

def _ensure_supabase():
    if supabase is None:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    if not _verify_tables():
        raise HTTPException(status_code=503, detail="Database not set up. Go to Supabase SQL Editor and run supabase_schema.sql")


async def _restart_session():
    """Try to restart a dead MCP session."""
    global session, exit_stack
    try:
        if exit_stack:
            await exit_stack.aclose()
    except Exception:
        pass
    exit_stack = AsyncExitStack()
    try:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["research_server.py"],
            env={**os.environ},
        )
        read, write = await exit_stack.enter_async_context(stdio_client(server_params))
        session = await exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
    except Exception:
        session = None
        raise


async def _ensure_session():
    if session is not None:
        return
    try:
        await _restart_session()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="MCP server not connected")




def _extract_texts(result: Any) -> list[str]:
    if not result or not result.content:
        return []
    return [item.text for item in result.content if hasattr(item, "text")]


def _parse_first(result: Any) -> Any:
    texts = _extract_texts(result)
    if not texts:
        return None
    try:
        return json.loads(texts[0])
    except (json.JSONDecodeError, TypeError):
        return texts[0]

def _parse_paper_results(result: Any) -> list[dict]:
    """Parse search_papers results (each TextContent is a JSON string with paper_id, title)."""
    texts = _extract_texts(result)
    papers = []
    for t in texts:
        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and "paper_id" in obj:
                papers.append(obj)
        except (json.JSONDecodeError, TypeError):
            pass
    return papers


_APOLOGY_PATTERNS = ["i am sorry", "i'm sorry", "sorry,", "does not contain", "no information", "no relevant", "cannot answer", "i cannot"]


def _is_apology(text: str | None) -> bool:
    if not text:
        return False
    lower = text.lower().strip()
    return any(p in lower for p in _APOLOGY_PATTERNS)


_SEARCH_PATTERNS = [
    re.compile(r"search\s+(?:for\s+)?(.+)", re.I),
    re.compile(r"find\s+papers?\s+(?:about|on|related\s+to)\s+(.+)", re.I),
    re.compile(r"find\s+(.+)", re.I),
    re.compile(r"(?:i(?:'m| am)\s+)?looking\s+for\s+(?:papers?\s+)?(?:about\s+)?(.+)", re.I),
    re.compile(r"look\s+(?:for\s+)?(.+)", re.I),
    re.compile(r"(?:i\s+)?want\s+(?:to\s+)?(?:find|search\s+for)\s+(.+)", re.I),
]


_TOPIC_CLEANUP = re.compile(r"\s+(and\s+(summarize|tell|give|show|list)\s.*|please\s*)$", re.I)

_QUESTION_PREFIXES = re.compile(
    r"^(what\s+(?:are|is)\s+(?:the\s+)?|how\s+(?:to|do|does|can|would)\s+|"
    r"tell\s+me\s+(?:about|regarding)\s+|can\s+(?:you\s+)?(?:tell|explain|describe)\s+(?:me\s+)?)"
    r"(.+)$", re.I
)


def _detect_intent(text: str) -> tuple[str, dict]:
    trimmed = text.strip()
    if trimmed.startswith("/"):
        parts = trimmed[1:].split(" ", 1)
        cmd = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""
        if cmd == "search":
            return "search", {"topic": rest, "max_results": 5}
        if cmd == "query":
            return "query", {"question": rest}
        return "query", {"question": trimmed}

    for pat in _SEARCH_PATTERNS:
        m = pat.search(trimmed)
        if m:
            topic = m.group(1).strip().strip('"').strip("'")
            topic = _TOPIC_CLEANUP.sub("", topic).strip()
            if topic:
                return "search", {"topic": topic, "max_results": 5}

    return "query", {"question": trimmed}


def _save_msg(conv_id: str, role: str, content: str, metadata: dict) -> dict:
    _ensure_supabase()
    data = {"conversation_id": conv_id, "role": role, "content": content, "metadata": metadata}
    res = supabase.table("messages").insert(data).execute()
    return res.data[0]


# ── Setup endpoint ──

@app.get("/api/setup")
async def setup_db():
    if supabase is None:
        return {"status": "error", "message": "Supabase not configured"}
    if _verify_tables():
        return {"status": "ok", "message": "Tables already exist"}
    try:
        import httpx
        if not os.path.isfile(SCHEMA_SQL):
            return {"status": "error", "message": "supabase_schema.sql not found"}
        sql = open(SCHEMA_SQL, encoding="utf-8").read()
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": "Bearer " + SUPABASE_KEY,
            "Content-Type": "application/json",
        }
        url = SUPABASE_URL.rstrip("/") + "/rest/v1/rpc/"
        res = httpx.post(url, json={"query": sql}, headers=headers, timeout=15)
        if res.status_code == 200:
            _TABLES_OK = True
            return {"status": "ok", "message": "Tables created"}
        ddl_url = SUPABASE_URL.rstrip("/") + "/rest/v1/pg_ddl"
        res2 = httpx.post(ddl_url, json={"query": sql}, headers=headers, timeout=15)
        if res2.status_code in (200, 201):
            _TABLES_OK = True
            return {"status": "ok", "message": "Tables created"}
        return {"status": "error", "message": "Cannot auto-create tables. Run supabase_schema.sql in Supabase SQL Editor.", "detail": res.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Conversation endpoints ──

@app.get("/api/conversations")
async def list_conversations(request: Request):
    _ensure_supabase()
    user = _get_user_from_request(request)
    query = supabase.table("conversations").select("*").order("updated_at", desc=True)
    if user:
        query = query.eq("user_id", user["user_id"])
    else:
        query = query.is_("user_id", "null")
    res = query.execute()
    return {"conversations": res.data}


@app.post("/api/conversations")
async def create_conversation(request: Request):
    _ensure_supabase()
    user = _get_user_from_request(request)
    data = {"title": "New Chat", "updated_at": datetime.now(timezone.utc).isoformat()}
    if user:
        data["user_id"] = user["user_id"]
    res = supabase.table("conversations").insert(data).execute()
    return res.data[0]


@app.patch("/api/conversations/{conv_id}")
async def update_conversation(conv_id: str, body: dict, request: Request):
    _ensure_supabase()
    updates = {"updated_at": datetime.now(timezone.utc).isoformat()}
    if "title" in body:
        updates["title"] = body["title"]
    query = supabase.table("conversations").update(updates).eq("id", conv_id)
    user = _get_user_from_request(request)
    if user:
        query = query.eq("user_id", user["user_id"])
    res = query.execute()
    if not res.data:
        raise HTTPException(404, "Conversation not found")
    return res.data[0]


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str, request: Request):
    _ensure_supabase()
    user = _get_user_from_request(request)
    query = supabase.table("conversations").select("id").eq("id", conv_id)
    if user:
        query = query.eq("user_id", user["user_id"])
    check = query.execute()
    if not check.data:
        raise HTTPException(404, "Conversation not found")
    supabase.table("conversation_papers").delete().eq("conversation_id", conv_id).execute()
    supabase.table("messages").delete().eq("conversation_id", conv_id).execute()
    supabase.table("conversations").delete().eq("id", conv_id).execute()
    if conv_id in _log_buffers:
        del _log_buffers[conv_id]
    return {"status": "ok"}


# ── Log streaming endpoint ──

@app.get("/api/conversations/{conv_id}/logs")
async def stream_logs(conv_id: str, request: Request):
    user = _get_user_from_request(request)
    _verify_conv_owner(conv_id, user)
    async def event_gen():
        q = asyncio.Queue(maxsize=200)
        if conv_id not in _log_subscribers:
            _log_subscribers[conv_id] = []
        _log_subscribers[conv_id].append(q)
        try:
            for entry in _log_buffers.get(conv_id, []):
                yield f"data: {json.dumps(entry)}\n\n"
            while True:
                entry = await q.get()
                yield f"data: {json.dumps(entry)}\n\n"
        except asyncio.CancelledError:
            if conv_id in _log_subscribers and q in _log_subscribers[conv_id]:
                _log_subscribers[conv_id].remove(q)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ── Message endpoints ──

@app.get("/api/conversations/{conv_id}/messages")
async def list_messages(conv_id: str, request: Request):
    _ensure_supabase()
    _verify_conv_owner(conv_id, _get_user_from_request(request))
    res = supabase.table("messages").select("*").eq("conversation_id", conv_id).order("created_at").execute()
    for m in res.data:
        if isinstance(m.get("metadata"), str):
            try:
                m["metadata"] = json.loads(m["metadata"])
            except (json.JSONDecodeError, TypeError):
                m["metadata"] = {}
    return {"messages": res.data}


@app.post("/api/conversations/{conv_id}/messages")
async def send_message(conv_id: str, body: dict, request: Request):
    global session
    _ensure_supabase()
    await _ensure_session()
    _verify_conv_owner(conv_id, _get_user_from_request(request))

    content = body.get("content", "").strip()
    if not content:
        raise HTTPException(400, "Message content is required")

    user_msg = _save_msg(conv_id, "user", content, {})
    intent, params = _detect_intent(content)
    _publish_log(conv_id, intent, "running", f"Detected intent: {intent}", {"params": params})

    try:
        if intent == "search":
            _publish_log(conv_id, "search_papers", "running", f"Searching for papers on: {params.get('topic')}")
            try:
                result = await session.call_tool("search_papers", params)
            except Exception as search_err:
                _publish_log(conv_id, "search_papers", "error", str(search_err))
                assistant_msg = _save_msg(conv_id, "assistant", f"Search failed: {search_err}", {"type": "error"})
                return {"assistant_message": assistant_msg}
            papers = _parse_paper_results(result)
            paper_ids = [p["paper_id"] for p in papers]
            _publish_log(conv_id, "search_papers", "done", f"Found {len(paper_ids)} paper(s)", {"paper_ids": paper_ids})
            for p in papers:
                try:
                    supabase.table("conversation_papers").insert({
                        "conversation_id": conv_id, "paper_id": p["paper_id"], "title": p.get("title", ""),
                    }).execute()
                except Exception:
                    pass
            if paper_ids:
                paper_lines = "\n".join([f"- **{p.get('title', 'Untitled')}** (`{p['paper_id']}`)" for p in papers])
                msg = f"Found {len(paper_ids)} paper(s):\n\n{paper_lines}\n\nSelect a paper from the list above to start chatting about it."
            else:
                msg = "No papers found."
            assistant_msg = _save_msg(conv_id, "assistant", msg, {
                "type": "search", "paper_ids": paper_ids, "topic": params.get("topic"),
                "papers": [{"paper_id": p["paper_id"], "title": p.get("title", "")} for p in papers],
            })

        elif intent == "query":
            query_params = {"question": params.get("question", content)}
            active = _active_papers.get(conv_id)
            if active:
                query_params["paper_id"] = active["paper_id"]
                scope = f"paper: {active.get('title', active['paper_id'])}"
            else:
                scope = "all indexed papers"
            _publish_log(conv_id, "query_paper", "running", f"Querying {scope}: {params.get('question')[:60]}...", {"scope": scope})
            result = await session.call_tool("query_paper", query_params)
            parsed = _parse_first(result)
            answer = (parsed.get("answer") or "") if isinstance(parsed, dict) else ""
            results = parsed.get("results", []) if isinstance(parsed, dict) else []

            if not results:
                if active:
                    # Strict scoping — no arXiv fallback when a paper is selected
                    _publish_log(conv_id, "query_paper", "done", "No relevant info found for this question in the selected paper.")
                    msg = f"I can only answer questions related to **{active.get('title', active['paper_id'])}**. Your question doesn't seem to be covered by this paper. Please ask something about the paper."
                    metadata_type = "query_no_answer"
                else:
                    # No active paper — fall back to arXiv search
                    fallback_topic = params.get("question", content)
                    m = _QUESTION_PREFIXES.match(fallback_topic)
                    if m:
                        fallback_topic = m.group(2).strip().rstrip("?.")
                    _publish_log(conv_id, "query_paper", "done", "No relevant indexed chunks. Falling back to arXiv search...")
                    _publish_log(conv_id, "search_papers", "running", f"Searching arXiv for: {fallback_topic[:80]}...")
                    try:
                        search_result = await session.call_tool("search_papers", {"topic": fallback_topic, "max_results": 5})
                        papers = _parse_paper_results(search_result)
                        paper_ids = [p["paper_id"] for p in papers]
                        for p in papers:
                            try:
                                supabase.table("conversation_papers").insert({
                                    "conversation_id": conv_id, "paper_id": p["paper_id"], "title": p.get("title", ""),
                                }).execute()
                            except Exception:
                                pass
                        _publish_log(conv_id, "search_papers", "done", f"Found {len(paper_ids)} paper(s) on arXiv.")
                        if paper_ids:
                            paper_lines = "\n".join([f"- **{p.get('title', 'Untitled')}** (`{p['paper_id']}`)" for p in papers])
                            msg = f"I searched arXiv and found the following papers:\n\n{paper_lines}\n\nSelect a paper from the list above to start chatting about it."
                            metadata_type = "search"
                        else:
                            msg = "I couldn't find any papers on arXiv related to your question. Try rephrasing or using `/search <topic>`."
                            metadata_type = "query_no_results"
                    except Exception as search_err:
                        _publish_log(conv_id, "search_papers", "error", f"arXiv search failed: {search_err}")
                        msg = "I couldn't find relevant content in indexed papers or on arXiv. Try searching with `/search <topic>`."
                        metadata_type = "query_no_results"
                assistant_msg = _save_msg(conv_id, "assistant", msg, {
                    "type": metadata_type, "answer": answer, "results": results, "question": content,
                    "paper_ids": paper_ids if metadata_type == "search" else [],
                    "papers": [{"paper_id": p["paper_id"], "title": p.get("title", "")} for p in papers] if metadata_type == "search" and papers else [],
                })
            else:
                _publish_log(conv_id, "query_paper", "done", f"Generated answer ({len(answer)} chars) from {len(results)} chunk(s).")
                msg = answer
                assistant_msg = _save_msg(conv_id, "assistant", msg, {
                    "type": "query", "answer": answer, "results": results, "question": content,
                })

        else:
            query_params = {"question": content}
            active = _active_papers.get(conv_id)
            if active:
                query_params["paper_id"] = active["paper_id"]
                _publish_log(conv_id, "query_paper", "running", f"Querying paper: {active.get('title', active['paper_id'])}")
            result = await session.call_tool("query_paper", query_params)
            parsed = _parse_first(result)
            answer = (parsed.get("answer") or "") if isinstance(parsed, dict) else ""
            results = parsed.get("results", []) if isinstance(parsed, dict) else []
            if active and not results:
                msg = f"I can only answer questions related to **{active.get('title', active['paper_id'])}**. Your question doesn't seem to be covered by this paper. Please ask something about the paper."
                assistant_msg = _save_msg(conv_id, "assistant", msg, {"type": "query_no_answer", "answer": answer, "results": results, "question": content, "active_paper": active})
            else:
                msg = answer or "I'm not sure how to help with that."
                assistant_msg = _save_msg(conv_id, "assistant", msg, {"type": "chat", "answer": answer, "results": results, "question": content})

        now = datetime.now(timezone.utc).isoformat()
        supabase.table("conversations").update({"updated_at": now}).eq("id", conv_id).execute()

        return {"user_message": user_msg, "assistant_message": assistant_msg}

    except Exception as e:
        _publish_log(conv_id, intent, "error", str(e))
        err_str = str(e).lower()
        if "connection closed" in err_str or "connection refused" in err_str or "eof" in err_str:
            session = None
            _publish_log(conv_id, intent, "recovery", "Session marked dead, will restart on next request")
        err_msg = _save_msg(conv_id, "assistant", f"Error: {str(e)}", {"type": "error", "error": str(e)})
        return {"user_message": user_msg, "assistant_message": err_msg}


# ── Paper endpoints ──

@app.get("/api/conversations/{conv_id}/papers")
async def list_papers(conv_id: str, request: Request):
    _ensure_supabase()
    _verify_conv_owner(conv_id, _get_user_from_request(request))
    res = supabase.table("conversation_papers").select("*").eq("conversation_id", conv_id).execute()
    return {"papers": res.data}


@app.post("/api/conversations/{conv_id}/papers")
async def add_paper(conv_id: str, body: dict, request: Request):
    _ensure_supabase()
    _verify_conv_owner(conv_id, _get_user_from_request(request))
    paper_id = body.get("paper_id", "").strip()
    if not paper_id:
        raise HTTPException(400, "paper_id required")
    try:
        res = supabase.table("conversation_papers").insert({
            "conversation_id": conv_id, "paper_id": paper_id, "title": body.get("title", ""),
        }).execute()
        return res.data[0]
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/conversations/{conv_id}/select-paper")
async def select_paper(conv_id: str, body: dict, request: Request):
    """Select a paper for a conversation: index it, get info, set as active paper."""
    _ensure_supabase()
    await _ensure_session()
    _verify_conv_owner(conv_id, _get_user_from_request(request))
    paper_id = body.get("paper_id", "").strip()
    title = body.get("title", "")
    if not paper_id:
        raise HTTPException(400, "paper_id required")
    try:
        supabase.table("conversation_papers").insert({
            "conversation_id": conv_id, "paper_id": paper_id, "title": title,
        }).execute()
    except Exception:
        pass
    # Check if already indexed (local flag + actual Redis data)
    info_result = await session.call_tool("extract_info", {"paper_id": paper_id})
    info_parsed = _parse_first(info_result)
    info = info_parsed if isinstance(info_parsed, dict) else {}
    already_indexed = isinstance(info, dict) and info.get("indexed") is True
    if already_indexed:
        import redis.asyncio as aioredis
        try:
            rr = aioredis.Redis.from_url(os.environ.get("REDIS_URL", ""), protocol=2, socket_connect_timeout=5, socket_timeout=5, decode_responses=True)
            cursor = 0
            found = 0
            while True:
                cursor, keys = await rr.scan(cursor=cursor, match=f"doc:paper:{paper_id}:chunk:*", count=10)
                found += len(keys)
                if cursor == 0:
                    break
            await rr.aclose()
            if found == 0:
                already_indexed = False
        except Exception:
            pass
    if already_indexed:
        total_chunks = info.get("total_chunks", info.get("chunk_count", 0)) or 0
    else:
        idx_result = await session.call_tool("index_paper", {"paper_id": paper_id})
        idx_parsed = _parse_first(idx_result)
        if not (isinstance(idx_parsed, dict) and idx_parsed.get("status") == "ok"):
            err = idx_parsed.get("error", "Indexing failed") if isinstance(idx_parsed, dict) else "Indexing failed"
            raise HTTPException(500, err)
        total_chunks = idx_parsed.get("total_chunks", 0)
        # Re-fetch info after indexing
        info_result = await session.call_tool("extract_info", {"paper_id": paper_id})
        info_parsed = _parse_first(info_result)
        info = info_parsed if isinstance(info_parsed, dict) else {}
    _active_papers[conv_id] = {"paper_id": paper_id, "title": title or paper_id}
    return {
        "status": "ok",
        "paper_id": paper_id,
        "title": title or info.get("title", paper_id),
        "total_chunks": total_chunks,
        "info": info,
        "already_indexed": already_indexed,
    }


@app.get("/api/conversations/{conv_id}/active-paper")
async def get_active_paper(conv_id: str, request: Request):
    _verify_conv_owner(conv_id, _get_user_from_request(request))
    paper = _active_papers.get(conv_id)
    return {"active_paper": paper}


# ── Legacy API endpoints ──

@app.post("/api/search")
async def api_search(req_body: dict):
    await _ensure_session()
    try:
        result = await session.call_tool("search_papers", {"topic": req_body.get("topic", ""), "max_results": req_body.get("max_results", 5)})
        parsed = _parse_first(result)
        if isinstance(parsed, list):
            return {"paper_ids": parsed}
        if isinstance(parsed, str):
            if "error" in parsed.lower():
                return {"paper_ids": [], "error": parsed}
            return {"paper_ids": [parsed]}
        return {"paper_ids": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/query")
async def api_query(req_body: dict):
    await _ensure_session()
    try:
        params = {"question": req_body.get("question", "")}
        if req_body.get("paper_id", "").strip():
            params["paper_id"] = req_body["paper_id"].strip()
        result = await session.call_tool("query_paper", params)
        parsed = _parse_first(result)
        if parsed is None:
            return {"status": "error", "error": "No response from server"}
        return parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    await _ensure_session()
    return {"status": "ok"}


# ── User auth endpoints ──

@app.post("/api/auth/register")
async def register(body: dict, request: Request):
    _check_rate_limit(request)
    _ensure_supabase()
    email = body.get("email", "").strip().lower()
    username = body.get("username", "").strip()
    password = body.get("password", "").strip()
    if not email or "@" not in email:
        raise HTTPException(400, detail="Valid email required")
    if not username or len(username) < 3:
        raise HTTPException(400, detail="Username must be at least 3 characters")
    if not password or len(password) < 8:
        raise HTTPException(400, detail="Password must be at least 8 characters")
    try:
        existing = supabase.table("users").select("id").eq("email", email).execute()
        if existing.data:
            raise HTTPException(409, detail="Email already registered")
        existing_u = supabase.table("users").select("id").eq("username", username).execute()
        if existing_u.data:
            raise HTTPException(409, detail="Username already taken")
    except HTTPException:
        raise
    except Exception as e:
        err = str(e).lower()
        if "relation" in err and "does not exist" in err:
            raise HTTPException(503, detail="Users table not found. Please run supabase_schema.sql in Supabase SQL Editor first.")
        raise HTTPException(500, detail=f"Database error: {e}")
    pw_hash = _hash_password(password)
    verify_token = secrets.token_urlsafe(32)
    try:
        supabase.table("users").insert({
            "email": email, "username": username,
            "password_hash": pw_hash, "verification_token": verify_token,
        }).execute()
    except Exception as e:
        err_str = str(e)
        if "row-level security" in err_str.lower() or "42501" in err_str:
            created = await _ensure_users_rls()
            if created:
                supabase.table("users").insert({
                    "email": email, "username": username,
                    "password_hash": pw_hash, "verification_token": verify_token,
                }).execute()
            else:
                raise HTTPException(503, detail="Cannot create user due to RLS policy. Run supabase_schema.sql in Supabase SQL Editor.")
        else:
            raise HTTPException(500, detail=f"Failed to create user: {e}")
    email_sent = send_verification_email(email, verify_token)
    if not email_sent:
        verify_link = f"{BASE_URL}/api/auth/verify?token={verify_token}"
        return {
            "status": "ok",
            "message": "Account created. Could not send email. Use this link to verify:",
            "verification_link": verify_link,
            "verify_token": verify_token,
        }
    return {
        "status": "ok",
        "message": "Account is created. Verify your email by checking the mail inbox.",
    }


@app.get("/api/auth/verify")
async def verify_email(token: str):
    _ensure_supabase()
    success = False
    try:
        res = supabase.table("users").select("*").eq("verification_token", token).eq("email_verified", False).execute()
        if res.data:
            supabase.table("users").update({"email_verified": True, "verification_token": None}).eq("id", res.data[0]["id"]).execute()
            success = True
    except Exception:
        pass
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Email Verification</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f0f1a;color:#e2e8f0;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}}
.card{{background:#1a1b23;border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:40px;max-width:420px;text-align:center}}
.icon{{font-size:48px;margin-bottom:16px}}
h2{{margin:0 0 8px;font-size:20px}}
p{{color:#94a3b8;font-size:14px;line-height:1.6;margin:0 0 20px}}
.btn{{display:inline-block;background:#6366f1;color:#fff;padding:10px 24px;border-radius:8px;text-decoration:none;font-size:14px;font-weight:600}}
.btn:hover{{background:#818cf8}}
</style></head><body>
<div class="card">
<div class="icon">{'&#9989;' if success else '&#10060;'}</div>
<h2>{'Email Verified!' if success else 'Verification Failed'}</h2>
<p>{'Your email has been verified. You can now close this tab and sign in.' if success else 'Invalid or expired verification link.'}</p>
{f'<a href="{BASE_URL}/" class="btn">Sign In</a>' if success else ''}
</div></body></html>"""
    return HTMLResponse(content=html, status_code=200)


def send_password_reset_email(user_email: str, reset_token: str):
    """Send password reset email via Resend."""
    if not RESEND_API_KEY:
        return False
    reset_link = f"{BASE_URL}/api/auth/reset-password?token={reset_token}"
    try:
        params: resend.Emails.SendParams = {
            "from": "Research Chat <noreply@yousafrana.me>",
            "to": [user_email],
            "subject": "Reset your password",
            "html": f"""
                <div style="font-family: sans-serif; padding: 20px; max-width: 500px;">
                    <h2>Reset your password</h2>
                    <p>Click the button below to reset your password:</p>
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{reset_link}" style="background: #6366f1; color: #fff; padding: 12px 28px; text-decoration: none; border-radius: 6px; font-weight: 600; display: inline-block;">Reset Password</a>
                    </div>
                    <p style="color: #666; font-size: 13px;">Or copy this link into your browser:<br><span style="color: #6366f1;">{reset_link}</span></p>
                    <p style="color: #999; font-size: 12px; margin-top: 20px;">If you did not request this, please ignore this email.</p>
                </div>
            """,
        }
        resend.Emails.send(params)
        return True
    except Exception as e:
        print(f"Failed to send reset email: {e}")
        return False


@app.post("/api/auth/forgot-password")
async def forgot_password(body: dict, request: Request):
    _check_rate_limit(request)
    _ensure_supabase()
    email = body.get("email", "").strip().lower()
    if not email:
        raise HTTPException(400, detail="Email required")
    res = supabase.table("users").select("id,email_verified").eq("email", email).execute()
    if not res.data:
        # Don't reveal whether email exists
        return {"status": "ok", "message": "If that email is registered, a reset link has been sent."}
    user = res.data[0]
    if not user.get("email_verified"):
        return {"status": "ok", "message": "If that email is registered, a reset link has been sent."}
    reset_token = secrets.token_urlsafe(32)
    supabase.table("users").update({"verification_token": reset_token}).eq("id", user["id"]).execute()
    send_password_reset_email(email, reset_token)
    return {"status": "ok", "message": "If that email is registered, a reset link has been sent."}


@app.get("/api/auth/reset-password")
async def reset_password_page(token: str):
    _ensure_supabase()
    res = supabase.table("users").select("id").eq("verification_token", token).eq("email_verified", True).execute()
    valid = len(res.data) > 0
    esc_token = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Reset Password</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f0f1a;color:#e2e8f0;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}}
.card{{background:#1a1b23;border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:40px;max-width:420px;text-align:center;width:100%}}
h2{{margin:0 0 8px;font-size:20px}}
p{{color:#94a3b8;font-size:14px;line-height:1.6;margin:0 0 20px}}
.field{{margin:20px 0;text-align:left}}
.field label{{display:block;font-size:13px;color:#94a3b8;margin-bottom:6px}}
.field .wrap{{position:relative}}
.field input{{width:100%;padding:12px 40px 12px 12px;background:#0f0f1a;border:1px solid rgba(255,255,255,0.08);border-radius:8px;color:#e2e8f0;font-size:14px;outline:none;font-family:inherit}}
.field input:focus{{border-color:#6366f1}}
.eye{{position:absolute;right:10px;top:50%;transform:translateY(-50%);cursor:pointer;color:#94a3b8;font-size:16px;user-select:none;background:none;border:none;padding:0;font-family:inherit}}
.btn{{background:#6366f1;color:#fff;padding:12px 24px;border-radius:8px;font-size:14px;font-weight:600;border:none;cursor:pointer;width:100%;font-family:inherit}}
.btn:hover{{background:#818cf8}}
.error{{color:#ef4444;font-size:13px;margin-top:8px;display:none;text-align:left}}
.success{{color:#22c55e;font-size:14px;margin-top:12px;display:none}}
</style></head><body>
<div class="card">
<h2>Reset Password</h2>"""
    if valid:
        html += f"""<p>Enter your new password below.</p>
<div id="reset-form-wrap">
<form id="reset-form">
<div class="field"><label>New Password</label><div class="wrap"><input type="password" id="new-password" placeholder="Min 6 characters" minlength="6" required><span class="eye" onclick="togglePw('new-password',this)">&#128065;</span></div></div>
<div class="field"><label>Confirm Password</label><div class="wrap"><input type="password" id="confirm-password" placeholder="Re-enter password" required><span class="eye" onclick="togglePw('confirm-password',this)">&#128065;</span></div></div>
<div class="error" id="reset-error"></div>
<button type="submit" class="btn">Update Password</button>
</form>
</div>
<div class="success" id="reset-success" style="display:none"><p style="margin-bottom:16px">Password updated!</p><a href="{BASE_URL}/" class="btn">Sign In</a></div>
<script>
function togglePw(id,el){{var inp=document.getElementById(id);if(inp.type==='password'){{inp.type='text';el.innerHTML='&#128064;'}}else{{inp.type='password';el.innerHTML='&#128065;'}}}}
document.getElementById("reset-form").addEventListener("submit",async function(e){{e.preventDefault();var p=document.getElementById("new-password").value;var cp=document.getElementById("confirm-password").value;var err=document.getElementById("reset-error");err.style.display="none";if(p!==cp){{err.textContent="Passwords do not match";err.style.display="block";return}}try{{var r=await fetch("/api/auth/reset-password",{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify({{token:"{esc_token}",password:p}})}});var d=await r.json();if(!r.ok) throw new Error(d.detail||"Failed");document.getElementById("reset-form-wrap").style.display="none";document.getElementById("reset-success").style.display="block"}}catch(e){{err.textContent=e.message;err.style.display="block"}}}})</script>"""
    else:
        html += "<p>Invalid or expired reset link.</p>"
    html += "</div></body></html>"
    return HTMLResponse(content=html, status_code=200)


@app.post("/api/auth/reset-password")
async def reset_password_execute(body: dict):
    _ensure_supabase()
    token = body.get("token", "")
    password = body.get("password", "").strip()
    if not password or len(password) < 8:
        raise HTTPException(400, detail="Password must be at least 8 characters")
    res = supabase.table("users").select("id", "password_hash").eq("verification_token", token).eq("email_verified", True).execute()
    if not res.data:
        raise HTTPException(400, detail="Invalid or expired reset token")
    user = res.data[0]
    if _verify_password(password, user["password_hash"]):
        raise HTTPException(400, detail="This is your old password. Please create a different one.")
    pw_hash = _hash_password(password)
    supabase.table("users").update({"password_hash": pw_hash, "verification_token": None}).eq("id", user["id"]).execute()
    return {"status": "ok", "message": "Password updated successfully"}


@app.post("/api/auth/login")
async def login(body: dict, request: Request):
    _check_rate_limit(request)
    _ensure_supabase()
    email = body.get("email", "").strip().lower()
    password = body.get("password", "").strip()
    res = supabase.table("users").select("*").eq("email", email).execute()
    if not res.data:
        raise HTTPException(401, detail="Invalid email or password")
    user = res.data[0]
    if not _verify_password(password, user["password_hash"]):
        raise HTTPException(401, detail="Invalid email or password")
    if not user.get("email_verified"):
        raise HTTPException(403, detail="Email not verified. Check your inbox for the verification email.")
    # Update last_login
    supabase.table("users").update({"last_login": datetime.now(timezone.utc).isoformat()}).eq("id", user["id"]).execute()
    # Create session
    session_token = secrets.token_urlsafe(48)
    _user_sessions[session_token] = {
        "user_id": user["id"],
        "email": user["email"],
        "username": user["username"],
    }
    return {"status": "ok", "token": session_token, "user": {"id": user["id"], "email": user["email"], "username": user["username"]}}


@app.post("/api/auth/me")
async def auth_me(body: dict):
    token = body.get("token", "")
    if token in _user_sessions:
        user_data = _user_sessions[token]
        try:
            supabase.table("users").update({"last_login": datetime.now(timezone.utc).isoformat()}).eq("id", user_data["user_id"]).execute()
        except Exception:
            pass
        return {"authenticated": True, "user": user_data}
    return {"authenticated": False}


@app.post("/api/auth/logout")
async def logout(body: dict):
    token = body.get("token", "")
    _user_sessions.pop(token, None)
    return {"status": "ok"}


# ── Admin endpoints ──

@app.get("/admin")
async def serve_admin():
    if ADMIN_HTML and os.path.isfile(ADMIN_HTML):
        return FileResponse(ADMIN_HTML)
    return {"error": "admin.html not found"}


@app.post("/api/admin/setup")
async def admin_setup(body: dict, request: Request):
    _check_rate_limit(request)
    """Create the first admin account (auto-creates admins table if needed)."""
    try:
        _ensure_supabase()
        admins_ok = False
        try:
            supabase.table("admins").select("id").limit(1).execute()
            admins_ok = True
        except Exception:
            pass
        if not admins_ok:
            created = await _ensure_admin_table()
            if not created:
                raise HTTPException(503, detail="Admins table not found in Supabase. Open the SQL Editor at https://supabase.com/dashboard/project/" + SUPABASE_URL.split("//")[1].split(".")[0] + "/sql/new and run the full supabase_schema.sql")
        existing = supabase.table("admins").select("id").limit(1).execute()
        if existing.data:
            raise HTTPException(400, detail="Admin already exists.")
        username = body.get("username", "").strip()
        password = body.get("password", "").strip()
        if not username or len(username) < 3:
            raise HTTPException(400, detail="Username must be at least 3 characters")
        if not password or len(password) < 8:
            raise HTTPException(400, detail="Password must be at least 8 characters")
        pw_hash = _hash_password(password)
        supabase.table("admins").insert({"username": username, "password_hash": pw_hash}).execute()
        token = secrets.token_urlsafe(48)
        _admin_tokens[token] = username
        return {"status": "ok", "token": token, "username": username}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Setup error: {type(e).__name__}: {e}")


@app.post("/api/admin/login")
async def admin_login(body: dict, request: Request):
    _check_rate_limit(request)
    _ensure_supabase()
    username = body.get("username", "").strip()
    password = body.get("password", "").strip()
    try:
        res = supabase.table("admins").select("*").eq("username", username).execute()
    except Exception as e:
        if "relation" in str(e).lower() and "does not exist" in str(e).lower():
            raise HTTPException(503, detail="Admins table not found. Create an admin account first via /api/admin/setup")
        raise HTTPException(503, detail=f"Database error: {e}")
    if not res.data:
        raise HTTPException(401, detail="Invalid credentials")
    admin = res.data[0]
    if not _verify_password(password, admin["password_hash"]):
        raise HTTPException(401, detail="Invalid credentials")
    token = secrets.token_urlsafe(48)
    _admin_tokens[token] = username
    return {"status": "ok", "token": token, "username": username}


@app.post("/api/admin/verify")
async def admin_verify(body: dict):
    token = body.get("token", "")
    if token in _admin_tokens:
        return {"valid": True, "username": _admin_tokens[token]}
    return {"valid": False}


@app.get("/api/admin/stats")
async def admin_stats(request: Request):
    _require_admin(request)
    _ensure_supabase()
    await _ensure_session()
    conv_res = supabase.table("conversations").select("id", count="exact").execute()
    msg_res = supabase.table("messages").select("id", count="exact").execute()
    paper_res = supabase.table("conversation_papers").select("id", count="exact").execute()
    admin_res = supabase.table("admins").select("username").execute()
    cached = _cache_get("admin_redis_stats")
    if cached is not None:
        redis_stats = cached
    else:
        redis_result = await session.call_tool("admin_redis_stats", {})
        redis_stats = _parse_first(redis_result) or {}
        _cache_set("admin_redis_stats", redis_stats)
    return {
        "conversations": conv_res.count or 0,
        "messages": msg_res.count or 0,
        "linked_papers": paper_res.count or 0,
        "redis": redis_stats,
        "admins": [a["username"] for a in admin_res.data],
    }


@app.get("/api/admin/conversations")
async def admin_list_conversations(request: Request):
    _require_admin(request)
    _ensure_supabase()
    res = supabase.table("conversations").select("*").order("updated_at", desc=True).execute()
    convs = res.data
    for c in convs:
        msg_res = supabase.table("messages").select("id", count="exact").eq("conversation_id", c["id"]).execute()
        c["message_count"] = msg_res.count or 0
        paper_res = supabase.table("conversation_papers").select("paper_id").eq("conversation_id", c["id"]).execute()
        c["paper_ids"] = [p["paper_id"] for p in paper_res.data]
    return {"conversations": convs}


@app.delete("/api/admin/conversations/{conv_id}")
async def admin_delete_conversation(conv_id: str, request: Request):
    _require_admin(request)
    _ensure_supabase()
    supabase.table("conversation_papers").delete().eq("conversation_id", conv_id).execute()
    supabase.table("messages").delete().eq("conversation_id", conv_id).execute()
    supabase.table("conversations").delete().eq("id", conv_id).execute()
    if conv_id in _log_buffers:
        del _log_buffers[conv_id]
    return {"status": "ok", "deleted": conv_id}


@app.get("/api/admin/check-setup")
async def admin_check_setup():
    """Check if admin table exists — unauthenticated, returns minimal info."""
    if supabase is None:
        return {"ready": False}
    try:
        supabase.table("admins").select("id").limit(1).execute()
        return {"ready": True}
    except Exception:
        return {"ready": False}


@app.get("/api/admin/supabase-project")
async def admin_supabase_project():
    """Get Supabase project reference for SQL Editor link."""
    if not SUPABASE_URL:
        return {"url": None}
    ref = SUPABASE_URL.split("//")[1].split(".")[0] if "//" in SUPABASE_URL else None
    return {"project_ref": ref, "sql_editor_url": f"https://supabase.com/dashboard/project/{ref}/sql/new" if ref else None}


LOCAL_PAPERS_DIR = os.path.join(HERE, "papers")


@app.get("/api/admin/local-papers")
async def admin_local_papers(request: Request):
    _require_admin(request)
    if not os.path.isdir(LOCAL_PAPERS_DIR):
        return {"topics": []}
    topics = []
    for entry in sorted(os.listdir(LOCAL_PAPERS_DIR)):
        json_path = os.path.join(LOCAL_PAPERS_DIR, entry, "papers_info.json")
        if os.path.isfile(json_path):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            papers = []
            for pid, info in data.items():
                papers.append({
                    "paper_id": pid,
                    "title": info.get("title", "Untitled"),
                    "indexed": info.get("indexed", False),
                })
            if papers:
                topics.append({"topic": entry, "papers": papers})
    return {"topics": topics}


@app.delete("/api/admin/local-papers/{paper_id}")
async def admin_delete_local_paper(paper_id: str, request: Request):
    _require_admin(request)
    if not os.path.isdir(LOCAL_PAPERS_DIR):
        raise HTTPException(404, detail="Papers directory not found")
    for entry in os.listdir(LOCAL_PAPERS_DIR):
        json_path = os.path.join(LOCAL_PAPERS_DIR, entry, "papers_info.json")
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if paper_id in data:
            del data[paper_id]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            # Remove empty topic folder
            if not data:
                import shutil
                shutil.rmtree(os.path.join(LOCAL_PAPERS_DIR, entry), ignore_errors=True)
            return {"status": "ok", "paper_id": paper_id, "topic": entry}
    raise HTTPException(404, detail=f"Paper {paper_id} not found")


@app.get("/api/admin/activity")
async def admin_activity(request: Request):
    """Daily conversation and message counts for the last 30 days."""
    _require_admin(request)
    _ensure_supabase()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=30)
    conv_res = supabase.table("conversations").select("created_at").gte("created_at", cutoff.isoformat()).execute()
    msg_res = supabase.table("messages").select("created_at").gte("created_at", cutoff.isoformat()).execute()
    daily = {}
    for i in range(30, -1, -1):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        daily[day] = {"conversations": 0, "messages": 0}
    for c in conv_res.data:
        day = c["created_at"][:10] if c["created_at"] else None
        if day and day in daily:
            daily[day]["conversations"] += 1
    for m in msg_res.data:
        day = m["created_at"][:10] if m["created_at"] else None
        if day and day in daily:
            daily[day]["messages"] += 1
    labels = list(daily.keys())
    convs = [daily[d]["conversations"] for d in labels]
    msgs = [daily[d]["messages"] for d in labels]
    return {"labels": labels, "conversations": convs, "messages": msgs}


@app.post("/api/admin/redis/clear")
async def admin_redis_clear(request: Request):
    _require_admin(request)
    await _ensure_session()
    result = await session.call_tool("admin_redis_clear", {})
    parsed = _parse_first(result) or {}
    _cache.clear()
    return parsed


def send_inactive_warning_email(user_email: str, username: str):
    """Send inactivity warning email via Resend."""
    if not RESEND_API_KEY:
        return False
    try:
        login_link = f"{BASE_URL}/"
        params: resend.Emails.SendParams = {
            "from": "Research Chat <noreply@yousafrana.me>",
            "to": [user_email],
            "subject": "Your account is inactive — we'll miss you!",
            "html": f"""
                <div style="font-family: sans-serif; padding: 20px; max-width: 500px;">
                    <h2>We miss you, {username}!</h2>
                    <p>Your Research Chat account has been inactive for a while. If you don't sign in within the next 2 weeks, your account and all associated data will be deleted.</p>
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{login_link}" style="background: #6366f1; color: #fff; padding: 12px 28px; text-decoration: none; border-radius: 6px; font-weight: 600; display: inline-block;">Sign In Now</a>
                    </div>
                    <p style="color: #999; font-size: 12px; margin-top: 20px;">If you no longer wish to use this service, no action is needed.</p>
                </div>
            """,
        }
        resend.Emails.send(params)
        return True
    except Exception as e:
        print(f"Failed to send warning email: {e}")
        return False


def _parse_iso_dt(s):
    """Parse ISO 8601 string to datetime, handling 'Z' suffix."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _latest_activity(last_login, conv_updated):
    """Return the latest of last_login and last conversation update as ISO string."""
    login_dt = _parse_iso_dt(last_login)
    conv_dt = _parse_iso_dt(conv_updated)
    latest = conv_dt if conv_dt and (not login_dt or conv_dt > login_dt) else login_dt
    return latest.isoformat() if latest else None


@app.get("/api/admin/users")
async def admin_list_users(request: Request):
    _require_admin(request)
    _ensure_supabase()
    users_res = supabase.table("users").select("*").order("created_at", desc=True).execute()
    users = users_res.data
    now = datetime.now(timezone.utc)
    cutoff_warn = now - timedelta(days=14)
    cutoff_delete = now - timedelta(days=28)
    for u in users:
        conv_res = supabase.table("conversations").select("updated_at").eq("user_id", u["id"]).order("updated_at", desc=True).limit(1).execute()
        last_conv = conv_res.data[0]["updated_at"] if conv_res.data else None
        last_login = u.get("last_login")
        u["last_active"] = _latest_activity(last_login, last_conv)
        last_dt = _parse_iso_dt(u["last_active"])
        warned_at = u.get("warned_at")
        warned_dt = _parse_iso_dt(warned_at)
        if not last_dt or last_dt > cutoff_warn:
            u["status"] = "active"
        elif warned_dt:
            if last_dt < cutoff_delete and warned_dt < cutoff_warn:
                u["status"] = "expired"
            else:
                u["status"] = "warned"
        else:
            u["status"] = "inactive"
        u["warned_at"] = warned_at
    return {"users": users}


@app.post("/api/admin/users/warn-inactive")
async def admin_warn_inactive(request: Request):
    _require_admin(request)
    _ensure_supabase()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=14)
    warned = 0
    failed = 0
    users_res = supabase.table("users").select("*").execute()
    for u in users_res.data:
        if u.get("warned_at"):
            continue
        conv_res = supabase.table("conversations").select("updated_at").eq("user_id", u["id"]).order("updated_at", desc=True).limit(1).execute()
        last_conv = conv_res.data[0]["updated_at"] if conv_res.data else None
        last_dt = _parse_iso_dt(_latest_activity(u.get("last_login"), last_conv))
        if last_dt and last_dt > cutoff:
            continue
        if not last_dt:
            continue
        ok = send_inactive_warning_email(u["email"], u.get("username", "User"))
        if ok:
            supabase.table("users").update({"warned_at": now.isoformat()}).eq("id", u["id"]).execute()
            warned += 1
        else:
            failed += 1
    return {"status": "ok", "warned": warned, "failed": failed}


@app.post("/api/admin/users/cleanup-expired")
async def admin_cleanup_expired(request: Request):
    _require_admin(request)
    _ensure_supabase()
    now = datetime.now(timezone.utc)
    cutoff_delete = now - timedelta(days=28)
    cutoff_warn = now - timedelta(days=14)
    deleted = 0
    failed = 0
    users_res = supabase.table("users").select("*").execute()
    for u in users_res.data:
        warned_at = u.get("warned_at")
        warned_dt = _parse_iso_dt(warned_at)
        if not warned_dt:
            continue
        conv_res = supabase.table("conversations").select("updated_at").eq("user_id", u["id"]).order("updated_at", desc=True).limit(1).execute()
        last_conv = conv_res.data[0]["updated_at"] if conv_res.data else None
        last_dt = _parse_iso_dt(_latest_activity(u.get("last_login"), last_conv))
        if not last_dt:
            continue
        if last_dt < cutoff_delete and warned_dt < cutoff_warn:
            try:
                supabase.table("conversations").delete().eq("user_id", u["id"]).execute()
                supabase.table("users").delete().eq("id", u["id"]).execute()
                deleted += 1
            except Exception:
                failed += 1
    return {"status": "ok", "deleted": deleted, "failed": failed}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
