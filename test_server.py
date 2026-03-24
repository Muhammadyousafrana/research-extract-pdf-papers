"""
Tests for research_server.py
Run via: make test
"""

import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

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
# TOOL: mark_paper_indexed
# ─────────────────────────────────────────────


class TestMarkPaperIndexed:

    def test_returns_confirmation_message(self, tmp_path):
        topic_dir = tmp_path / "rag"
        topic_dir.mkdir()
        (topic_dir / "papers_info.json").write_text(
            json.dumps(SAMPLE_METADATA), encoding="utf-8"
        )

        with patch.object(server, "PAPER_DIR", str(tmp_path)):
            result = server.mark_paper_indexed(SAMPLE_PAPER_ID)

        assert SAMPLE_PAPER_ID in result
        assert "indexed" in result.lower()

    def test_updates_flag_in_file(self, tmp_path):
        topic_dir = tmp_path / "rag"
        topic_dir.mkdir()
        meta_file = topic_dir / "papers_info.json"
        meta_file.write_text(json.dumps(SAMPLE_METADATA), encoding="utf-8")

        with patch.object(server, "PAPER_DIR", str(tmp_path)):
            server.mark_paper_indexed(SAMPLE_PAPER_ID)

        updated = json.loads(meta_file.read_text(encoding="utf-8"))
        assert updated[SAMPLE_PAPER_ID]["indexed"] is True


# ─────────────────────────────────────────────
# TOOL: embed_chunk
# ─────────────────────────────────────────────


class TestEmbedChunk:

    def test_returns_vector_on_success(self):
        mock_vector = [0.1] * 1024

        with patch.object(server, "_get_embedding", return_value=mock_vector):
            result = json.loads(server.embed_chunk("some chunk text"))

        assert result["status"] == "ok"
        assert len(result["vector"]) == 1024
        assert result["vector"][0] == pytest.approx(0.1)

    def test_returns_error_on_api_failure(self):
        with patch.object(server, "_get_embedding", side_effect=Exception("API error")):
            result = json.loads(server.embed_chunk("some text"))

        assert result["status"] == "error"
        assert "API error" in result["error"]


# ─────────────────────────────────────────────
# TOOL: extract_chunks
# ─────────────────────────────────────────────


class TestExtractChunks:

    def _make_mock_response(self):
        mock_resp = MagicMock()
        mock_resp.content = b"%PDF-fake"
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_returns_chunks_on_success(self):
        fake_md = "# Title\n## Introduction\nHello world.\n## Conclusion\nThe end."

        with (
            patch("research_server.httpx.get", return_value=self._make_mock_response()),
            patch("research_server.fitz.open", return_value=MagicMock()),
            patch("research_server.pymupdf4llm.to_markdown", return_value=fake_md),
        ):

            result = json.loads(server.extract_chunks(SAMPLE_PAPER_ID, SAMPLE_PDF_URL))

        assert result["status"] == "ok"
        assert result["paper_id"] == SAMPLE_PAPER_ID
        assert result["total_chunks"] >= 1
        assert "redis_key" in result["chunks"][0]
        assert result["chunks"][0]["redis_key"].startswith("doc:paper:")

    def test_returns_error_on_empty_pdf(self):
        with (
            patch("research_server.httpx.get", return_value=self._make_mock_response()),
            patch("research_server.fitz.open", return_value=MagicMock()),
            patch("research_server.pymupdf4llm.to_markdown", return_value="   "),
        ):

            result = json.loads(server.extract_chunks(SAMPLE_PAPER_ID, SAMPLE_PDF_URL))

        assert result["status"] == "error"
        assert "no text" in result["error"].lower()

    def test_returns_error_on_http_failure(self):
        with patch(
            "research_server.httpx.get", side_effect=Exception("connection refused")
        ):
            result = json.loads(server.extract_chunks(SAMPLE_PAPER_ID, SAMPLE_PDF_URL))

        assert result["status"] == "error"
        assert "connection refused" in result["error"].lower()

    def test_chunk_redis_key_format(self):
        fake_md = "# Title\n## Section A\nContent A.\n## Section B\nContent B."

        with (
            patch("research_server.httpx.get", return_value=self._make_mock_response()),
            patch("research_server.fitz.open", return_value=MagicMock()),
            patch("research_server.pymupdf4llm.to_markdown", return_value=fake_md),
        ):

            result = json.loads(server.extract_chunks("1234.5678", SAMPLE_PDF_URL))

        for chunk in result["chunks"]:
            assert chunk["redis_key"] == f"doc:paper:1234.5678:chunk:{chunk['index']}"
