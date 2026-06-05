from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.api.routes as routes


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    return TestClient(app)


def test_pdf_upload_uses_docling_parser(monkeypatch, tmp_path):
    client = make_client()
    monkeypatch.setattr(routes.settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))

    fake_rag = Mock()
    fake_rag.ainsert = AsyncMock()
    fake_parser = AsyncMock(return_value="# Điều 1\n\nNội dung")

    monkeypatch.setattr(routes.RAGEngine, "get_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "parse_pdf_to_markdown", fake_parser)

    response = client.post(
        "/api/upload",
        files={"file": ("law.pdf", b"%PDF-1.4\n", "application/pdf")},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["filename"] == "law.pdf"
    assert body["status"] == "success"
    assert "Docling" in body["message"]
    fake_parser.assert_awaited_once_with(str(tmp_path / "law.pdf"))
    fake_rag.ainsert.assert_awaited_once_with("# Điều 1\n\nNội dung", file_paths=["law.pdf"])


def test_pdf_upload_returns_error_when_docling_parse_fails(monkeypatch, tmp_path):
    client = make_client()
    monkeypatch.setattr(routes.settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))

    fake_rag = Mock()
    fake_rag.ainsert = AsyncMock()
    fake_parser = AsyncMock(side_effect=RuntimeError("Docling failed to parse PDF: boom"))

    monkeypatch.setattr(routes.RAGEngine, "get_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "parse_pdf_to_markdown", fake_parser)

    response = client.post(
        "/api/upload",
        files={"file": ("law.pdf", b"%PDF-1.4\n", "application/pdf")},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["filename"] == "law.pdf"
    assert body["status"] == "error"
    assert "Docling failed to parse PDF" in body["message"]
    fake_rag.ainsert.assert_not_awaited()


def test_txt_upload_still_reads_local_text_file(monkeypatch, tmp_path):
    client = make_client()
    monkeypatch.setattr(routes.settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))

    fake_rag = Mock()
    fake_rag.ainsert = AsyncMock()
    fake_parser = AsyncMock()

    monkeypatch.setattr(routes.RAGEngine, "get_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "parse_pdf_to_markdown", fake_parser)

    response = client.post(
        "/api/upload",
        files={"file": ("law.txt", "Điều 1: tốc độ tối đa".encode("utf-8"), "text/plain")},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["filename"] == "law.txt"
    assert body["status"] == "success"
    fake_parser.assert_not_called()
    fake_rag.ainsert.assert_awaited_once_with("Điều 1: tốc độ tối đa", file_paths=["law.txt"])
