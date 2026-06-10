import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import backend.core.rag_engine as rag_engine


def setup_function():
    rag_engine.RAGEngine._deepseek_indexing_instance = None
    rag_engine.RAGEngine._google_studio_indexing_instance = None
    rag_engine.RAGEngine._query_instance = None


def test_getters_raise_before_initialize():
    with pytest.raises(RuntimeError, match="DeepSeek indexing RAG engine not initialized"):
        asyncio.run(rag_engine.RAGEngine.get_indexing_instance("deepseek"))

    with pytest.raises(RuntimeError, match="Query RAG engine not initialized"):
        rag_engine.RAGEngine.get_query_instance()


def test_initialize_validates_answer_key_before_boot(monkeypatch):
    events = []

    class FakeLightRAG:
        def __init__(self, **kwargs):
            events.append("light_rag")

        async def initialize_storages(self):
            events.append("initialize_storages")
            return None

    monkeypatch.setattr(rag_engine, "LightRAG", FakeLightRAG)
    monkeypatch.setattr(rag_engine, "EmbeddingFunc", lambda **kwargs: kwargs)
    monkeypatch.setattr(rag_engine, "VLLMEmbeddingFunc", lambda prefix="": f"embed:{prefix}")
    monkeypatch.setattr(rag_engine.RAGEngine, "_apply_postgres_environment", lambda: events.append("apply_postgres_environment"))
    monkeypatch.setattr(type(rag_engine.settings), "get_answer_llm_api_key", lambda self: events.append("validate_answer_key") or (_ for _ in ()).throw(ValueError("ANSWER_LLM_API_KEY is required")))
    monkeypatch.setattr(rag_engine.settings, "EMBEDDING_DIM", 1024)
    monkeypatch.setattr(rag_engine.settings, "EMBEDDING_MAX_TOKEN_SIZE", 512)
    monkeypatch.setattr(rag_engine.settings, "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    monkeypatch.setattr(rag_engine.settings, "EMBEDDING_DOCUMENT_PREFIX", "")
    monkeypatch.setattr(rag_engine.settings, "EMBEDDING_QUERY_PREFIX", "search_query: ")

    with pytest.raises(ValueError, match="ANSWER_LLM_API_KEY is required"):
        asyncio.run(rag_engine.RAGEngine.initialize())

    assert events == ["apply_postgres_environment", "validate_answer_key"]


def test_initialize_builds_two_indexing_engines_and_one_query_engine(monkeypatch):
    created = []

    class FakeLightRAG:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.doc_status = SimpleNamespace(get_docs_paginated=AsyncMock(return_value=(([], 0), None)))

        async def initialize_storages(self):
            return None

    monkeypatch.setattr(rag_engine, "LightRAG", FakeLightRAG)
    monkeypatch.setattr(rag_engine, "EmbeddingFunc", lambda **kwargs: kwargs)
    monkeypatch.setattr(rag_engine, "VLLMEmbeddingFunc", lambda prefix="": f"embed:{prefix}")
    monkeypatch.setattr(rag_engine.settings, "EMBEDDING_DIM", 1024)
    monkeypatch.setattr(rag_engine.settings, "EMBEDDING_MAX_TOKEN_SIZE", 512)
    monkeypatch.setattr(rag_engine.settings, "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    monkeypatch.setattr(rag_engine.settings, "EMBEDDING_DOCUMENT_PREFIX", "")
    monkeypatch.setattr(rag_engine.settings, "EMBEDDING_QUERY_PREFIX", "search_query: ")
    monkeypatch.setattr(type(rag_engine.settings), "get_answer_llm_api_key", lambda self: "answer-key")

    asyncio.run(rag_engine.RAGEngine.initialize())

    assert len(created) == 2
    assert created[0]["llm_model_func"] is rag_engine.indexing_llm_func
    assert created[1]["llm_model_func"] is rag_engine.answer_llm_func
    assert created[0]["embedding_func"]["func"] == "embed:"
    assert created[1]["embedding_func"]["func"] == "embed:search_query: "


def test_get_indexing_instance_selects_provider_after_initialize(monkeypatch):
    rag_engine.RAGEngine._deepseek_indexing_instance = "deepseek-engine"
    rag_engine.RAGEngine._google_studio_indexing_instance = "google-engine"

    assert asyncio.run(rag_engine.RAGEngine.get_indexing_instance("deepseek")) == "deepseek-engine"
    assert asyncio.run(rag_engine.RAGEngine.get_indexing_instance("google_studio")) == "google-engine"
