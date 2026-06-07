from fastapi import FastAPI
from fastapi.testclient import TestClient
from types import SimpleNamespace
from unittest.mock import AsyncMock

import backend.api.routes as routes


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    return TestClient(app)


def test_chat_uses_query_rag_for_non_streaming_requests(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="Câu trả lời"))

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)

    response = client.post(
        "/api/chat",
        json={"message": "Tốc độ tối đa là bao nhiêu?", "stream": False, "comparison_mode": False},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "Câu trả lời"
    fake_rag.aquery.assert_awaited_once()


def test_documents_uses_query_rag_doc_status(monkeypatch):
    client = make_client()
    fake_status = SimpleNamespace(status=SimpleNamespace(value="processed"), file_path="law.pdf", content_summary="abc")
    fake_doc_status = SimpleNamespace(get_docs_paginated=AsyncMock(return_value=((("doc-1", fake_status),), None)))
    fake_rag = SimpleNamespace(doc_status=fake_doc_status)

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)

    response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json()[0]["id"] == "doc-1"


def test_chat_streaming_keeps_sse_event_shape(monkeypatch):
    client = make_client()

    async def fake_stream():
        yield "chunk-1"
        yield "chunk-2"

    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value=fake_stream()))
    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)

    response = client.post(
        "/api/chat",
        json={"message": "Cho tôi câu trả lời", "stream": True, "comparison_mode": False},
    )

    assert response.status_code == 200
    assert 'data: {"type": "chunk", "mode": "hybrid", "content": "chunk-1"}' in response.text
    assert 'data: {"type": "done"}' in response.text
