from fastapi import FastAPI
from fastapi.testclient import TestClient
from types import SimpleNamespace
from unittest.mock import AsyncMock

import backend.api.routes as routes
from backend.core.legal_prompts import build_legal_system_prompt


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    return TestClient(app)


def _call_by_mode(mock: AsyncMock, mode: str):
    for awaited_call in mock.await_args_list:
        param = awaited_call.kwargs["param"]
        if param.mode == mode:
            return awaited_call
    raise AssertionError(f"Missing awaited call for mode={mode!r}")


def _assert_legal_query_call(awaited_call, message: str, mode: str, system_prompt: str, stream: bool = False):
    assert awaited_call.args == (message,)
    assert awaited_call.kwargs["system_prompt"] == system_prompt
    assert awaited_call.kwargs["param"].mode == mode
    assert awaited_call.kwargs["param"].stream == stream


def test_chat_passes_hybrid_legal_prompt_for_non_streaming_requests(monkeypatch):
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


def test_chat_passes_hybrid_legal_prompt_for_streaming_requests(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="unused"))

    async def controlled_stream(**kwargs):
        async def gen():
            yield "chunk-1"
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


def test_chat_passes_mode_specific_legal_prompts_for_comparison_requests(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="Naive answer"))
    controlled = AsyncMock(return_value="Hybrid answer")

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Quy định nồng độ cồn thế nào?", "stream": False, "comparison_mode": True},
    )

    assert response.status_code == 200
    assert response.json()["naive"]["response"] == "Naive answer"
    assert response.json()["hybrid"]["response"] == "Hybrid answer"
    _assert_legal_query_call(
        _call_by_mode(fake_rag.aquery, "naive"),
        "Quy định nồng độ cồn thế nào?",
        "naive",
        build_legal_system_prompt("naive"),
    )
    controlled.assert_awaited_once_with(
        rag=fake_rag,
        message="Quy định nồng độ cồn thế nào?",
        stream=False,
    )


def test_chat_passes_mode_specific_legal_prompts_for_streaming_comparison_requests(monkeypatch):
    client = make_client()

    async def naive_stream():
        yield "naive-1"

    async def hybrid_stream():
        yield "hybrid-1"

    async def fake_aquery(query, param=None, system_prompt=None):
        if param.mode == "naive":
            return naive_stream()
        raise AssertionError(f"Unexpected direct rag mode: {param.mode!r}")

    controlled = AsyncMock(return_value=hybrid_stream())
    fake_rag = SimpleNamespace(aquery=AsyncMock(side_effect=fake_aquery))
    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Quy định nồng độ cồn thế nào?", "stream": True, "comparison_mode": True},
    )

    assert response.status_code == 200
    assert 'data: {"type": "chunk", "mode": "naive", "content": "naive-1"}' in response.text
    assert 'data: {"type": "chunk", "mode": "hybrid", "content": "hybrid-1"}' in response.text
    assert 'data: {"type": "done"}' in response.text
    _assert_legal_query_call(
        _call_by_mode(fake_rag.aquery, "naive"),
        "Quy định nồng độ cồn thế nào?",
        "naive",
        build_legal_system_prompt("naive"),
        stream=True,
    )
    controlled.assert_awaited_once_with(
        rag=fake_rag,
        message="Quy định nồng độ cồn thế nào?",
        stream=True,
    )


def test_single_hybrid_chat_uses_controlled_pipeline(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="unused direct rag answer"))
    controlled = AsyncMock(return_value="Controlled answer")

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Hành lang an toàn đường bộ được xác định từ đâu?", "stream": False, "comparison_mode": False},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "Controlled answer"
    controlled.assert_awaited_once_with(
        rag=fake_rag,
        message="Hành lang an toàn đường bộ được xác định từ đâu?",
        stream=False,
    )
    fake_rag.aquery.assert_not_awaited()


def test_comparison_uses_naive_direct_call_and_controlled_hybrid(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="Naive answer"))
    controlled = AsyncMock(return_value="Controlled hybrid answer")

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Hành lang an toàn đường bộ được xác định từ đâu?", "stream": False, "comparison_mode": True},
    )

    assert response.status_code == 200
    assert response.json()["naive"]["response"] == "Naive answer"
    assert response.json()["hybrid"]["response"] == "Controlled hybrid answer"
    _assert_legal_query_call(
        _call_by_mode(fake_rag.aquery, "naive"),
        "Hành lang an toàn đường bộ được xác định từ đâu?",
        "naive",
        build_legal_system_prompt("naive"),
    )
    controlled.assert_awaited_once_with(
        rag=fake_rag,
        message="Hành lang an toàn đường bộ được xác định từ đâu?",
        stream=False,
    )


