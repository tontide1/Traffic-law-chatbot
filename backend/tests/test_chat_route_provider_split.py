from fastapi import FastAPI
from fastapi.testclient import TestClient
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import backend.api.routes as routes


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    return TestClient(app)


def test_chat_uses_query_rag_for_non_streaming_requests(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="unused"))
    controlled = AsyncMock(return_value="Câu trả lời")

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Tốc độ tối đa là bao nhiêu?", "stream": False, "comparison_mode": False},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "Câu trả lời"
    controlled.assert_awaited_once_with(
        rag=fake_rag,
        message="Tốc độ tối đa là bao nhiêu?",
        stream=False,
    )
    fake_rag.aquery.assert_not_awaited()



def test_documents_returns_indexed_provider(monkeypatch):
    client = make_client()
    fake_status = SimpleNamespace(
        status=SimpleNamespace(value="processed"),
        file_path="law.pdf",
        content_summary="abc",
    )
    fake_doc_status = SimpleNamespace(get_docs_paginated=AsyncMock(return_value=((("doc-1", fake_status),), None)))
    fake_rag = SimpleNamespace(doc_status=fake_doc_status)
    fake_store = Mock()
    fake_store.get_all_providers = AsyncMock(return_value={"law.pdf": "google_studio"})

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "IndexingProviderStore", lambda working_dir: fake_store)

    response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json()[0]["id"] == "doc-1"
    assert response.json()[0]["indexed_provider"] == "google_studio"


def test_chat_streaming_keeps_sse_event_shape(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="unused"))

    async def controlled_stream(**kwargs):
        async def gen():
            yield "chunk-1"
            yield "chunk-2"
        return gen()

    controlled = AsyncMock(side_effect=controlled_stream)

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Cho tôi câu trả lời", "stream": True, "comparison_mode": False},
    )

    assert response.status_code == 200
    assert 'data: {"type": "chunk", "mode": "hybrid", "content": "chunk-1"}' in response.text
    assert 'data: {"type": "done"}' in response.text
    controlled.assert_awaited_once_with(
        rag=fake_rag,
        message="Cho tôi câu trả lời",
        stream=True,
    )



def test_chat_streaming_uses_controlled_pipeline_and_keeps_sse_shape(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="unused"))

    async def controlled_stream(**kwargs):
        async def gen():
            yield "controlled-1"
        return gen()

    controlled = AsyncMock(side_effect=controlled_stream)

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Cho tôi câu trả lời", "stream": True, "comparison_mode": False},
    )

    assert response.status_code == 200
    assert 'data: {"type": "chunk", "mode": "hybrid", "content": "controlled-1"}' in response.text
    assert 'data: {"type": "done"}' in response.text
    controlled.assert_awaited_once()

