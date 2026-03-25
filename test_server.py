"""
Tests for research_server.py
Run via: make test
"""

import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# ── make research_server importable without triggering mcp.run() ─────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import research_server as server

# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

SAMPLE_PAPER_ID = "2301.07041"
SAMPLE_PDF_URL = "https://arxiv.org/pdf/2301.07041"
SAMPLE_METADATA = {
    SAMPLE_PAPER_ID: {
        "title": "Test Paper",
        "authors": ["Alice", "Bob"],
        "summary": "A test summary.",
        "pdf_url": SAMPLE_PDF_URL,
        "published": "2023-01-17",
        "indexed": False,
    }
}


# ─────────────────────────────────────────────
# HELPER: _update_indexed_flag
# ─────────────────────────────────────────────


class TestUpdateIndexedFlag:

    def test_marks_paper_as_indexed(self, tmp_path):
        topic_dir = tmp_path / "rag"
        topic_dir.mkdir()
        meta_file = topic_dir / "papers_info.json"
        meta_file.write_text(json.dumps(SAMPLE_METADATA), encoding="utf-8")

        with patch.object(server, "PAPER_DIR", str(tmp_path)):
            server._update_indexed_flag(SAMPLE_PAPER_ID)

        updated = json.loads(meta_file.read_text(encoding="utf-8"))
        assert updated[SAMPLE_PAPER_ID]["indexed"] is True

    def test_nonexistent_paper_id_does_nothing(self, tmp_path):
        topic_dir = tmp_path / "rag"
        topic_dir.mkdir()
        meta_file = topic_dir / "papers_info.json"
        meta_file.write_text(json.dumps(SAMPLE_METADATA), encoding="utf-8")

        with patch.object(server, "PAPER_DIR", str(tmp_path)):
            server._update_indexed_flag("nonexistent_id")

        updated = json.loads(meta_file.read_text(encoding="utf-8"))
        assert updated[SAMPLE_PAPER_ID]["indexed"] is False  # unchanged

    def test_missing_paper_dir_does_nothing(self, tmp_path):
        with patch.object(server, "PAPER_DIR", str(tmp_path / "missing")):
            server._update_indexed_flag(SAMPLE_PAPER_ID)  # should not raise

    def test_corrupt_json_skipped(self, tmp_path):
        topic_dir = tmp_path / "rag"
        topic_dir.mkdir()
        (topic_dir / "papers_info.json").write_text("NOT JSON", encoding="utf-8")

        with patch.object(server, "PAPER_DIR", str(tmp_path)):
            server._update_indexed_flag(SAMPLE_PAPER_ID)  # should not raise


# ─────────────────────────────────────────────
# HELPER: _extract_text_chunks
# ─────────────────────────────────────────────


class TestExtractTextChunks:

    def test_returns_chunks_from_markdown(self):
        fake_md = "# Title\n## Introduction\nHello world.\n## Conclusion\nThe end."
        with (
            patch("research_server.fitz.open", return_value=MagicMock()),
            patch("research_server.pymupdf4llm.to_markdown", return_value=fake_md),
        ):
            chunks = server._extract_text_chunks(b"%PDF-fake")

        assert len(chunks) >= 1
        assert any("Introduction" in c for c in chunks)

    def test_returns_empty_for_blank_pdf(self):
        with (
            patch("research_server.fitz.open", return_value=MagicMock()),
            patch("research_server.pymupdf4llm.to_markdown", return_value="   "),
        ):
            chunks = server._extract_text_chunks(b"%PDF-fake")

        assert chunks == []


# ─────────────────────────────────────────────
# TOOL: extract_info
# ─────────────────────────────────────────────


class TestExtractInfo:

    def test_returns_metadata_for_known_paper(self, tmp_path):
        topic_dir = tmp_path / "rag"
        topic_dir.mkdir()
        (topic_dir / "papers_info.json").write_text(
            json.dumps(SAMPLE_METADATA), encoding="utf-8"
        )

        with patch.object(server, "PAPER_DIR", str(tmp_path)):
            result = server.extract_info(SAMPLE_PAPER_ID)

        data = json.loads(result)
        assert data["title"] == "Test Paper"
        assert data["authors"] == ["Alice", "Bob"]

    def test_returns_error_message_for_unknown_paper(self, tmp_path):
        topic_dir = tmp_path / "rag"
        topic_dir.mkdir()
        (topic_dir / "papers_info.json").write_text(
            json.dumps(SAMPLE_METADATA), encoding="utf-8"
        )

        with patch.object(server, "PAPER_DIR", str(tmp_path)):
            result = server.extract_info("9999.99999")

        assert "no saved information" in result.lower()

    def test_skips_corrupt_json_file(self, tmp_path):
        topic_dir = tmp_path / "rag"
        topic_dir.mkdir()
        (topic_dir / "papers_info.json").write_text("CORRUPT", encoding="utf-8")

        with patch.object(server, "PAPER_DIR", str(tmp_path)):
            result = server.extract_info(SAMPLE_PAPER_ID)

        assert "no saved information" in result.lower()


# ─────────────────────────────────────────────
# TOOL: index_paper
# ─────────────────────────────────────────────


class TestIndexPaper:

    def _make_mock_response(self):
        mock_resp = MagicMock()
        mock_resp.content = b"%PDF-fake"
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def _make_mock_redis_session(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=MagicMock(content=[]))
        return session

    @pytest.mark.asyncio
    async def test_indexes_paper_successfully(self, tmp_path):
        fake_md = "## Section A\nContent A.\n## Section B\nContent B."
        mock_vectors = [[0.1] * 10, [0.2] * 10]
        mock_session = self._make_mock_redis_session()

        topic_dir = tmp_path / "test"
        topic_dir.mkdir()
        (topic_dir / "papers_info.json").write_text(
            json.dumps(SAMPLE_METADATA), encoding="utf-8"
        )

        with (
            patch("research_server.httpx.get", return_value=self._make_mock_response()),
            patch("research_server.fitz.open", return_value=MagicMock()),
            patch("research_server.pymupdf4llm.to_markdown", return_value=fake_md),
            patch.object(server, "_get_embeddings_batch", return_value=mock_vectors),
            patch.object(server, "_get_redis_session", return_value=mock_session),
            patch.object(server, "PAPER_DIR", str(tmp_path)),
        ):
            result = json.loads(await server.index_paper(SAMPLE_PAPER_ID))

        assert result["status"] == "ok"
        assert result["paper_id"] == SAMPLE_PAPER_ID
        assert result["total_chunks"] == 2

        # Verify Redis calls were made
        call_tool_calls = mock_session.call_tool.call_args_list
        # Should have: 1 vector index create + 1 json_set + 6 hset (3 keys/chunk * 2 chunks) + 2 set_vector = 10 calls
        assert len(call_tool_calls) == 10

    @pytest.mark.asyncio
    async def test_returns_error_on_empty_pdf(self):
        with (
            patch("research_server.httpx.get", return_value=self._make_mock_response()),
            patch("research_server.fitz.open", return_value=MagicMock()),
            patch("research_server.pymupdf4llm.to_markdown", return_value="   "),
        ):
            result = json.loads(await server.index_paper(SAMPLE_PAPER_ID))

        assert result["status"] == "error"
        assert "no text" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_returns_error_on_http_failure(self):
        with patch(
            "research_server.httpx.get", side_effect=Exception("connection refused")
        ):
            result = json.loads(await server.index_paper(SAMPLE_PAPER_ID))

        assert result["status"] == "error"
        assert "connection refused" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_marks_paper_as_indexed_locally(self, tmp_path):
        fake_md = "# Title\n## Section\nContent."
        mock_vectors = [[0.1] * 10]
        mock_session = self._make_mock_redis_session()

        topic_dir = tmp_path / "test"
        topic_dir.mkdir()
        meta_file = topic_dir / "papers_info.json"
        meta_file.write_text(json.dumps(SAMPLE_METADATA), encoding="utf-8")

        with (
            patch("research_server.httpx.get", return_value=self._make_mock_response()),
            patch("research_server.fitz.open", return_value=MagicMock()),
            patch("research_server.pymupdf4llm.to_markdown", return_value=fake_md),
            patch.object(server, "_get_embeddings_batch", return_value=mock_vectors),
            patch.object(server, "_get_redis_session", return_value=mock_session),
            patch.object(server, "PAPER_DIR", str(tmp_path)),
        ):
            await server.index_paper(SAMPLE_PAPER_ID)

        updated = json.loads(meta_file.read_text(encoding="utf-8"))
        assert updated[SAMPLE_PAPER_ID]["indexed"] is True

    @pytest.mark.asyncio
    async def test_redis_stores_correct_keys(self, tmp_path):
        fake_md = "# Title\n## Intro\nSome content."
        mock_vectors = [[0.5] * 10]
        mock_session = self._make_mock_redis_session()

        with (
            patch("research_server.httpx.get", return_value=self._make_mock_response()),
            patch("research_server.fitz.open", return_value=MagicMock()),
            patch("research_server.pymupdf4llm.to_markdown", return_value=fake_md),
            patch.object(server, "_get_embeddings_batch", return_value=mock_vectors),
            patch.object(server, "_get_redis_session", return_value=mock_session),
            patch.object(server, "PAPER_DIR", str(tmp_path)),
        ):
            await server.index_paper("1234.5678")

        # Check that hset was called with correct redis key format
        hset_calls = [
            c for c in mock_session.call_tool.call_args_list
            if c[0][0] == "hset"
        ]
        for call in hset_calls:
            args = call[0][1]
            assert args["name"].startswith("doc:paper:1234.5678:chunk:")


# ─────────────────────────────────────────────
# TOOL: query_paper
# ─────────────────────────────────────────────


class TestQueryPaper:

    def _make_mock_redis_session(self):
        session = AsyncMock()
        session.initialize = AsyncMock()
        # Simulate vector search returning a result
        mock_content = MagicMock()
        mock_content.text = json.dumps({"text": "relevant chunk", "score": 0.95})
        mock_result = MagicMock()
        mock_result.content = [mock_content]
        session.call_tool = AsyncMock(return_value=mock_result)
        return session

    @pytest.mark.asyncio
    async def test_returns_results_on_success(self):
        mock_vector = [0.1] * 10
        mock_session = self._make_mock_redis_session()

        with (
            patch.object(server, "_get_embedding", return_value=mock_vector),
            patch.object(server, "_get_redis_session", return_value=mock_session),
        ):
            result = json.loads(
                await server.query_paper("What is attention?", "1706.03762")
            )

        assert result["status"] == "ok"
        assert len(result["results"]) > 0

    @pytest.mark.asyncio
    async def test_returns_error_on_embedding_failure(self):
        with patch.object(
            server, "_get_embedding", side_effect=Exception("API down")
        ):
            result = json.loads(
                await server.query_paper("test question", "1234.5678")
            )

        assert result["status"] == "error"
        assert "API down" in result["error"]

    @pytest.mark.asyncio
    async def test_searches_all_papers_without_id(self):
        mock_vector = [0.1] * 10
        mock_session = self._make_mock_redis_session()

        with (
            patch.object(server, "_get_embedding", return_value=mock_vector),
            patch.object(server, "_get_redis_session", return_value=mock_session),
        ):
            result = json.loads(await server.query_paper("What is attention?"))

        assert result["paper_id"] == "all"
