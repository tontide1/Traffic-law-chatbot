import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import AsyncMock

import numpy as np

import backend.core.llm_services as llm_services


class FakeEmbeddingResponse:
    def __init__(self, vector):
        self.data = [SimpleNamespace(embedding=vector)]


class FakeChatResponse:
    def __init__(self, content):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


def setup_function():
    llm_services.reset_client_caches()


def test_get_indexing_llm_client_uses_deepseek_settings(monkeypatch):
    created = []

    class DummyClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(llm_services.openai, "AsyncOpenAI", DummyClient)
    monkeypatch.setattr(llm_services.settings, "INDEXING_LLM_API_KEY", "deepseek-key")
    monkeypatch.setattr(llm_services.settings, "INDEXING_LLM_BASE_URL", "https://api.deepseek.com")

    llm_services.get_indexing_llm_client()

    assert created == [{"api_key": "deepseek-key", "base_url": "https://api.deepseek.com"}]


def test_get_answer_llm_client_falls_back_to_openrouter_key(monkeypatch):
    created = []

    class DummyClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(llm_services.openai, "AsyncOpenAI", DummyClient)
    monkeypatch.setattr(llm_services.settings, "ANSWER_LLM_API_KEY", None)
    monkeypatch.setattr(llm_services.settings, "OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setattr(llm_services.settings, "ANSWER_LLM_BASE_URL", "https://openrouter.ai/api/v1")

    llm_services.get_answer_llm_client()

    assert created == [{"api_key": "openrouter-key", "base_url": "https://openrouter.ai/api/v1"}]


def test_indexing_llm_func_disables_thinking(monkeypatch):
    create = AsyncMock(return_value=FakeChatResponse("indexed"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(llm_services, "get_indexing_llm_client", lambda: client)
    monkeypatch.setattr(llm_services.settings, "INDEXING_LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(llm_services.settings, "INDEXING_LLM_THINKING_MODE", "disabled")

    result = asyncio.run(llm_services.indexing_llm_func("build graph"))

    assert result == "indexed"
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "deepseek-v4-flash"
    assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}


def test_vllm_embedding_func_uses_local_endpoint_and_prefix_without_forcing_dimensions(monkeypatch):
    create = AsyncMock(return_value=FakeEmbeddingResponse([0.1, 0.2, 0.3]))
    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))

    monkeypatch.setattr(llm_services, "get_embedding_client", lambda: client)
    monkeypatch.setattr(llm_services.settings, "EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding_v2")
    monkeypatch.setattr(llm_services.settings, "EMBEDDING_DIM", 1024)

    embedding_func = llm_services.VLLMEmbeddingFunc(prefix="search_query: ")
    result = asyncio.run(embedding_func(["tốc độ tối đa là bao nhiêu"]))

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3)
    create.assert_awaited_once_with(
        model="AITeamVN/Vietnamese_Embedding_v2",
        input="search_query: tốc độ tối đa là bao nhiêu",
    )


async def _collect_async_gen(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


def test_indexing_client_factory_memoizes_until_reset(monkeypatch):
    created = []

    class DummyClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(llm_services.openai, "AsyncOpenAI", DummyClient)
    monkeypatch.setattr(llm_services.settings, "INDEXING_LLM_API_KEY", "deepseek-key")
    monkeypatch.setattr(llm_services.settings, "INDEXING_LLM_BASE_URL", "https://api.deepseek.com")

    first = llm_services.get_indexing_llm_client()
    second = llm_services.get_indexing_llm_client()

    assert first is second
    assert len(created) == 1

    llm_services.reset_client_caches()
    third = llm_services.get_indexing_llm_client()

    assert third is not first
    assert len(created) == 2


def test_answer_client_factory_memoizes_until_reset(monkeypatch):
    created = []

    class DummyClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(llm_services.openai, "AsyncOpenAI", DummyClient)
    monkeypatch.setattr(llm_services.settings, "ANSWER_LLM_API_KEY", "answer-key")
    monkeypatch.setattr(llm_services.settings, "ANSWER_LLM_BASE_URL", "https://openrouter.ai/api/v1")

    first = llm_services.get_answer_llm_client()
    second = llm_services.get_answer_llm_client()

    assert first is second
    assert len(created) == 1

    llm_services.reset_client_caches()
    third = llm_services.get_answer_llm_client()

    assert third is not first
    assert len(created) == 2


def test_answer_llm_func_does_not_inject_thinking(monkeypatch):
    create = AsyncMock(return_value=FakeChatResponse("answer"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(llm_services, "get_answer_llm_client", lambda: client)
    monkeypatch.setattr(llm_services.settings, "ANSWER_LLM_MODEL", "openai/gpt-oss-120b")

    result = asyncio.run(llm_services.answer_llm_func("legal question"))

    assert result == "answer"
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "openai/gpt-oss-120b"
    assert "extra_body" not in kwargs or "thinking" not in (kwargs.get("extra_body") or {})


def test_streaming_response_yields_delta_contents():
    class FakeDelta:
        def __init__(self, content):
            self.content = content

    class FakeChunk:
        def __init__(self, content):
            self.choices = [SimpleNamespace(delta=FakeDelta(content))]

    async def fake_stream():
        yield FakeChunk("hello ")
        yield FakeChunk("world")

    result = llm_services._stream_or_content(fake_stream(), stream=True)

    chunks = asyncio.run(_collect_async_gen(result))
    assert chunks == ["hello ", "world"]


def test_indexing_llm_func_annotation_covers_streaming_output():
    return_hint = get_type_hints(llm_services.indexing_llm_func)["return"]

    assert return_hint == str | AsyncIterator[str]


def test_answer_llm_func_annotation_covers_streaming_output():
    return_hint = get_type_hints(llm_services.answer_llm_func)["return"]

    assert return_hint == str | AsyncIterator[str]


def test_get_google_studio_llm_client_uses_google_settings(monkeypatch):
    created = []

    class DummyClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(llm_services.openai, "AsyncOpenAI", DummyClient)
    monkeypatch.setattr(llm_services.settings, "GOOGLE_STUDIO_API_KEY", "google-key")
    monkeypatch.setattr(
        llm_services.settings,
        "GOOGLE_STUDIO_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    llm_services.get_google_studio_llm_client()

    assert created == [
        {
            "api_key": "google-key",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        }
    ]


def test_google_studio_indexing_llm_func_sends_correct_payload(monkeypatch):
    create = AsyncMock(return_value=FakeChatResponse("indexed with google"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(llm_services, "get_google_studio_llm_client", lambda: client)
    monkeypatch.setattr(llm_services.settings, "GOOGLE_STUDIO_MODEL", "gemini-2.5-flash")

    result = asyncio.run(llm_services.google_studio_indexing_llm_func("build graph"))

    assert result == "indexed with google"
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "gemini-2.5-flash"
    assert kwargs["messages"][-1]["content"] == "build graph"
