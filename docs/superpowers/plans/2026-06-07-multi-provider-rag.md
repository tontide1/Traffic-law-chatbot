# Multi-Provider RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the backend into three explicit inference roles by using DeepSeek direct API for KG indexing, OpenRouter `openai/gpt-oss-120b` for answer generation, and local `Qwen/Qwen3-Embedding-0.6B` via vLLM for embeddings without breaking the current upload, chat, or SSE contracts.

**Architecture:** Introduce role-specific settings and client factories, replace the single `RAGEngine` singleton with two coordinated `LightRAG` instances that share storage, and wire routes so `/api/upload` uses the indexing engine while `/api/chat` uses the query engine. Keep embeddings local and shared by both engines, migrate the embedding dimension from `1536` to `1024`, and update runtime/docs so Dockerized backend services can reach a host-served vLLM embedding endpoint.

**Tech Stack:** Python 3.11, FastAPI, LightRAG core, OpenAI-compatible SDKs, DeepSeek API, OpenRouter, vLLM, PostgreSQL/pgvector/Apache AGE, pytest, Docker Compose

---

## File Map

- Modify: `backend/config.py`  
  Replace single-model settings with role-specific settings and explicit helper methods for provider keys and URLs.

- Modify: `backend/core/llm_services.py`  
  Add dedicated client factories for indexing, answers, and embeddings; add explicit DeepSeek non-thinking mode handling; add deterministic embedding prefix handling.

- Modify: `backend/core/rag_engine.py`  
  Replace the single `LightRAG` singleton with an engine manager that initializes and serves separate indexing and query instances.

- Modify: `backend/api/routes.py`  
  Route upload to the indexing engine and chat/documents to the query engine while preserving the existing API contract.

- Modify: `backend/tests/test_upload_route.py`  
  Update upload route tests to assert the indexing engine accessor is used.

- Create: `backend/tests/test_multi_provider_config.py`  
  Config regression tests for new settings and key fallback behavior.

- Create: `backend/tests/test_multi_provider_llm_services.py`  
  Unit tests for role-specific client construction, DeepSeek non-thinking requests, and local embeddings.

- Create: `backend/tests/test_rag_engine_manager.py`  
  Tests that two `LightRAG` instances are initialized with shared storage and role-specific functions.

- Create: `backend/tests/test_chat_route_provider_split.py`  
  Route tests that `/api/chat` and `/api/documents` use the query engine and preserve SSE semantics.

- Create: `backend/tests/test_multi_provider_docs.py`  
  Regression tests for `.env.example`, `README.md`, and `docker-compose.yml`.

- Modify: `.env.example`  
  Document role-specific settings for DeepSeek, OpenRouter, and local vLLM embeddings.

- Modify: `docker-compose.yml`  
  Add `host.docker.internal` routing for host-served vLLM and align backend and `rag-ui` with the new embedding dimension and role-specific providers.

- Modify: `README.md`  
  Describe the new multi-provider architecture, setup, and local embedding service requirement.

## Notes Before Execution

- This plan assumes the backend continues to run in Docker while vLLM runs on the host machine at port `8002`.
- The implementation should not add a new generic provider framework. Keep the system limited to the three explicit roles in the approved spec.
- The implementation should preserve `comparison_mode` as `naive vs hybrid`.
- Re-indexing is required after the embedding dimension changes to `1024`. This plan documents and verifies that requirement but does not implement bulk re-index tooling.

### Task 1: Add split provider settings and configuration tests

**Files:**
- Create: `backend/tests/test_multi_provider_config.py`
- Modify: `backend/config.py`

- [ ] **Step 1: Write the failing configuration tests**

Create `backend/tests/test_multi_provider_config.py` with this exact content:

```python
from backend.config import Settings


def test_multi_provider_defaults_match_the_new_architecture():
    settings = Settings(_env_file=None)

    assert settings.INDEXING_LLM_BASE_URL == "https://api.deepseek.com"
    assert settings.INDEXING_LLM_MODEL == "deepseek-v4-flash"
    assert settings.INDEXING_LLM_THINKING_MODE == "disabled"
    assert settings.ANSWER_LLM_BASE_URL == "https://openrouter.ai/api/v1"
    assert settings.ANSWER_LLM_MODEL == "openai/gpt-oss-120b"
    assert settings.EMBEDDING_BASE_URL == "http://host.docker.internal:8002/v1"
    assert settings.EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.EMBEDDING_DIM == 1024
    assert settings.EMBEDDING_DOCUMENT_PREFIX == ""
    assert "Given a legal query" in settings.EMBEDDING_QUERY_PREFIX


def test_answer_key_falls_back_to_openrouter_key():
    settings = Settings(
        _env_file=None,
        OPENROUTER_API_KEY="openrouter-key",
        ANSWER_LLM_API_KEY=None,
    )

    assert settings.get_answer_llm_api_key() == "openrouter-key"


def test_missing_required_provider_keys_raise_clear_errors():
    settings = Settings(_env_file=None)

    try:
        settings.get_indexing_llm_api_key()
    except ValueError as exc:
        assert str(exc) == "INDEXING_LLM_API_KEY is required"
    else:
        raise AssertionError("Expected indexing API key lookup to raise")

    try:
        settings.get_answer_llm_api_key()
    except ValueError as exc:
        assert str(exc) == "ANSWER_LLM_API_KEY or OPENROUTER_API_KEY is required"
    else:
        raise AssertionError("Expected answer API key lookup to raise")


def test_entity_types_string_parsing_still_works():
    settings = Settings(
        _env_file=None,
        ENTITY_TYPES='["Điều khoản", "Văn bản pháp luật"]',
    )

    assert settings.ENTITY_TYPES == ["Điều khoản", "Văn bản pháp luật"]
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run:

```bash
pytest backend/tests/test_multi_provider_config.py -q
```

Expected: FAIL because the new split-provider settings and helper methods do not exist in `backend/config.py`.

- [ ] **Step 3: Implement the role-specific settings in `backend/config.py`**

Replace `backend/config.py` with this exact content:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import json
from pydantic import field_validator


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/law_assistant"

    # Postgres individual components for LightRAG
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DATABASE: str = "law_assistant"

    REDIS_URL: Optional[str] = None

    OPENROUTER_API_KEY: Optional[str] = None

    INDEXING_LLM_BASE_URL: str = "https://api.deepseek.com"
    INDEXING_LLM_API_KEY: Optional[str] = None
    INDEXING_LLM_MODEL: str = "deepseek-v4-flash"
    INDEXING_LLM_THINKING_MODE: str = "disabled"

    ANSWER_LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    ANSWER_LLM_API_KEY: Optional[str] = None
    ANSWER_LLM_MODEL: str = "openai/gpt-oss-120b"

    EMBEDDING_BASE_URL: str = "http://host.docker.internal:8002/v1"
    EMBEDDING_API_KEY: str = "EMPTY"
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_MAX_TOKEN_SIZE: int = 512
    EMBEDDING_QUERY_PREFIX: str = (
        "Instruct: Given a legal query, retrieve relevant statutes and legal passages.\nQuery: "
    )
    EMBEDDING_DOCUMENT_PREFIX: str = ""

    SUMMARY_LANGUAGE: str = "Vietnamese"
    ENTITY_TYPES: list[str] = [
        "Văn bản pháp luật",
        "Điều khoản",
        "Cơ quan ban hành",
        "Đối tượng áp dụng",
        "Hành vi vi phạm",
        "Hình thức xử phạt",
        "Thời hạn",
        "Khái niệm pháp lý",
    ]

    LIGHTRAG_WORKING_DIR: str = "./backend/data"

    @field_validator("ENTITY_TYPES", mode="before")
    @classmethod
    def parse_entity_types(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in value.split(",")]
        return value

    def get_indexing_llm_api_key(self) -> str:
        if not self.INDEXING_LLM_API_KEY:
            raise ValueError("INDEXING_LLM_API_KEY is required")
        return self.INDEXING_LLM_API_KEY

    def get_answer_llm_api_key(self) -> str:
        key = self.ANSWER_LLM_API_KEY or self.OPENROUTER_API_KEY
        if not key:
            raise ValueError("ANSWER_LLM_API_KEY or OPENROUTER_API_KEY is required")
        return key

    def get_embedding_api_key(self) -> str:
        return self.EMBEDDING_API_KEY or "EMPTY"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
```

- [ ] **Step 4: Run the config tests to verify they pass**

Run:

```bash
pytest backend/tests/test_multi_provider_config.py -q
```

Expected: PASS with `4 passed`.

- [ ] **Step 5: Commit the configuration split**

Run:

```bash
git add backend/config.py backend/tests/test_multi_provider_config.py
git commit -m "feat: add multi-provider rag settings"
```

### Task 2: Split LLM and embedding clients behind focused unit tests

**Files:**
- Create: `backend/tests/test_multi_provider_llm_services.py`
- Modify: `backend/core/llm_services.py`

- [ ] **Step 1: Write the failing llm-service tests**

Create `backend/tests/test_multi_provider_llm_services.py` with this exact content:

```python
import asyncio
from types import SimpleNamespace
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


def test_vllm_embedding_func_uses_local_endpoint_and_prefix(monkeypatch):
    create = AsyncMock(return_value=FakeEmbeddingResponse([0.1, 0.2, 0.3]))
    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))

    monkeypatch.setattr(llm_services, "get_embedding_client", lambda: client)
    monkeypatch.setattr(llm_services.settings, "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    monkeypatch.setattr(llm_services.settings, "EMBEDDING_DIM", 1024)

    embedding_func = llm_services.VLLMEmbeddingFunc(prefix="search_query: ")
    result = asyncio.run(embedding_func(["tốc độ tối đa là bao nhiêu"]))

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3)
    create.assert_awaited_once_with(
        model="Qwen/Qwen3-Embedding-0.6B",
        input="search_query: tốc độ tối đa là bao nhiêu",
        dimensions=1024,
    )
```

- [ ] **Step 2: Run the llm-service tests to verify they fail**

Run:

```bash
pytest backend/tests/test_multi_provider_llm_services.py -q
```

Expected: FAIL because `reset_client_caches`, `get_indexing_llm_client`, `get_answer_llm_client`, `indexing_llm_func`, and `VLLMEmbeddingFunc` do not exist yet.

- [ ] **Step 3: Replace `backend/core/llm_services.py` with role-specific clients**

Replace `backend/core/llm_services.py` with this exact content:

```python
import openai
from typing import List, Optional

import numpy as np

from backend.config import settings

_indexing_client: Optional[openai.AsyncOpenAI] = None
_answer_client: Optional[openai.AsyncOpenAI] = None
_embedding_client: Optional[openai.AsyncOpenAI] = None


def reset_client_caches():
    global _indexing_client, _answer_client, _embedding_client
    _indexing_client = None
    _answer_client = None
    _embedding_client = None


def _build_async_client(api_key: str, base_url: str) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)


def get_indexing_llm_client():
    global _indexing_client
    if _indexing_client is None:
        _indexing_client = _build_async_client(
            api_key=settings.get_indexing_llm_api_key(),
            base_url=settings.INDEXING_LLM_BASE_URL,
        )
    return _indexing_client


def get_answer_llm_client():
    global _answer_client
    if _answer_client is None:
        _answer_client = _build_async_client(
            api_key=settings.get_answer_llm_api_key(),
            base_url=settings.ANSWER_LLM_BASE_URL,
        )
    return _answer_client


def get_embedding_client():
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = _build_async_client(
            api_key=settings.get_embedding_api_key(),
            base_url=settings.EMBEDDING_BASE_URL,
        )
    return _embedding_client


def _base_messages(prompt: str, system_prompt: str = None, history: List[dict] = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


def _stream_or_content(response, stream: bool):
    if stream:
        async def stream_generator():
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return stream_generator()
    return response.choices[0].message.content


class VLLMEmbeddingFunc:
    def __init__(self, prefix: str = ""):
        self.model_name = settings.EMBEDDING_MODEL
        self.prefix = prefix

    async def __call__(self, texts: List[str]):
        client = get_embedding_client()
        vectors = []
        for text in texts:
            response = await client.embeddings.create(
                model=self.model_name,
                input=f"{self.prefix}{text}",
                dimensions=settings.EMBEDDING_DIM,
            )
            vectors.append(response.data[0].embedding)
        return np.array(vectors)


async def indexing_llm_func(
    prompt: str,
    system_prompt: str = None,
    history: List[dict] = None,
    **kwargs,
) -> str:
    client = get_indexing_llm_client()
    messages = _base_messages(prompt, system_prompt=system_prompt, history=history)
    request_kwargs = {k: v for k, v in kwargs.items() if k != "model"}
    extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
    extra_body["thinking"] = {"type": settings.INDEXING_LLM_THINKING_MODE}

    response = await client.chat.completions.create(
        model=settings.INDEXING_LLM_MODEL,
        messages=messages,
        extra_body=extra_body,
        **request_kwargs,
    )
    return _stream_or_content(response, stream=bool(request_kwargs.get("stream")))


async def answer_llm_func(
    prompt: str,
    system_prompt: str = None,
    history: List[dict] = None,
    **kwargs,
) -> str:
    client = get_answer_llm_client()
    messages = _base_messages(prompt, system_prompt=system_prompt, history=history)

    response = await client.chat.completions.create(
        model=settings.ANSWER_LLM_MODEL,
        messages=messages,
        **kwargs,
    )
    return _stream_or_content(response, stream=bool(kwargs.get("stream")))
```

- [ ] **Step 4: Run the llm-service tests to verify they pass**

Run:

```bash
pytest backend/tests/test_multi_provider_llm_services.py -q
```

Expected: PASS with `4 passed`.

- [ ] **Step 5: Commit the provider-specific service layer**

Run:

```bash
git add backend/core/llm_services.py backend/tests/test_multi_provider_llm_services.py
git commit -m "feat: split llm and embedding clients by role"
```

### Task 3: Replace the singleton RAG engine with dual shared-storage instances

**Files:**
- Create: `backend/tests/test_rag_engine_manager.py`
- Modify: `backend/core/rag_engine.py`

- [ ] **Step 1: Write the failing RAG engine manager tests**

Create `backend/tests/test_rag_engine_manager.py` with this exact content:

```python
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import backend.core.rag_engine as rag_engine


def setup_function():
    rag_engine.RAGEngine._indexing_instance = None
    rag_engine.RAGEngine._query_instance = None


def test_getters_raise_before_initialize():
    with pytest.raises(RuntimeError, match="Indexing RAG engine not initialized"):
        rag_engine.RAGEngine.get_indexing_instance()

    with pytest.raises(RuntimeError, match="Query RAG engine not initialized"):
        rag_engine.RAGEngine.get_query_instance()


def test_initialize_builds_two_role_specific_instances(monkeypatch):
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

    asyncio.run(rag_engine.RAGEngine.initialize())

    assert len(created) == 2
    assert created[0]["llm_model_func"] is rag_engine.indexing_llm_func
    assert created[1]["llm_model_func"] is rag_engine.answer_llm_func
    assert created[0]["embedding_func"]["embedding_dim"] == 1024
    assert created[1]["embedding_func"]["embedding_dim"] == 1024
    assert created[0]["embedding_func"]["func"] == "embed:"
    assert created[1]["embedding_func"]["func"] == "embed:search_query: "
```

- [ ] **Step 2: Run the RAG engine manager tests to verify they fail**

Run:

```bash
pytest backend/tests/test_rag_engine_manager.py -q
```

Expected: FAIL because `RAGEngine` only exposes the old single-instance API.

- [ ] **Step 3: Replace `backend/core/rag_engine.py` with a dual-engine manager**

Replace `backend/core/rag_engine.py` with this exact content:

```python
import os

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

from backend.config import settings
from backend.core.llm_services import VLLMEmbeddingFunc, answer_llm_func, indexing_llm_func


class RAGEngine:
    _indexing_instance = None
    _query_instance = None

    @classmethod
    def _apply_postgres_environment(cls):
        os.environ["POSTGRES_HOST"] = settings.POSTGRES_HOST
        os.environ["POSTGRES_PORT"] = str(settings.POSTGRES_PORT)
        os.environ["POSTGRES_USER"] = settings.POSTGRES_USER
        os.environ["POSTGRES_PASSWORD"] = settings.POSTGRES_PASSWORD
        os.environ["POSTGRES_DATABASE"] = settings.POSTGRES_DATABASE

    @classmethod
    def _build_embedding_func(cls, prefix: str):
        return EmbeddingFunc(
            embedding_dim=settings.EMBEDDING_DIM,
            max_token_size=settings.EMBEDDING_MAX_TOKEN_SIZE,
            func=VLLMEmbeddingFunc(prefix=prefix),
            model_name=settings.EMBEDDING_MODEL,
        )

    @classmethod
    async def initialize(cls):
        if cls._indexing_instance is not None and cls._query_instance is not None:
            return cls._indexing_instance, cls._query_instance

        cls._apply_postgres_environment()

        cls._indexing_instance = LightRAG(
            working_dir=settings.LIGHTRAG_WORKING_DIR,
            llm_model_func=indexing_llm_func,
            embedding_func=cls._build_embedding_func(settings.EMBEDDING_DOCUMENT_PREFIX),
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="PGGraphStorage",
            doc_status_storage="PGDocStatusStorage",
            addon_params={
                "language": settings.SUMMARY_LANGUAGE,
                "entity_types": settings.ENTITY_TYPES,
            },
        )
        await cls._indexing_instance.initialize_storages()

        cls._query_instance = LightRAG(
            working_dir=settings.LIGHTRAG_WORKING_DIR,
            llm_model_func=answer_llm_func,
            embedding_func=cls._build_embedding_func(settings.EMBEDDING_QUERY_PREFIX),
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="PGGraphStorage",
            doc_status_storage="PGDocStatusStorage",
            addon_params={
                "language": settings.SUMMARY_LANGUAGE,
                "entity_types": settings.ENTITY_TYPES,
            },
        )
        await cls._query_instance.initialize_storages()

        return cls._indexing_instance, cls._query_instance

    @classmethod
    def get_indexing_instance(cls):
        if cls._indexing_instance is None:
            raise RuntimeError("Indexing RAG engine not initialized. Call RAGEngine.initialize() first.")
        return cls._indexing_instance

    @classmethod
    def get_query_instance(cls):
        if cls._query_instance is None:
            raise RuntimeError("Query RAG engine not initialized. Call RAGEngine.initialize() first.")
        return cls._query_instance


rag_engine = RAGEngine()
```

- [ ] **Step 4: Run the RAG engine manager tests to verify they pass**

Run:

```bash
pytest backend/tests/test_rag_engine_manager.py -q
```

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit the dual-engine manager**

Run:

```bash
git add backend/core/rag_engine.py backend/tests/test_rag_engine_manager.py
git commit -m "feat: add dual lightrag engine manager"
```

### Task 4: Route upload and chat through the correct engine instances

**Files:**
- Modify: `backend/tests/test_upload_route.py`
- Create: `backend/tests/test_chat_route_provider_split.py`
- Modify: `backend/api/routes.py`

- [ ] **Step 1: Write the failing route tests**

Replace `backend/tests/test_upload_route.py` with this exact content:

```python
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.api.routes as routes


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    return TestClient(app)


def test_pdf_upload_uses_indexing_rag(monkeypatch, tmp_path):
    client = make_client()
    monkeypatch.setattr(routes.settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))

    fake_rag = Mock()
    fake_rag.ainsert = AsyncMock()
    fake_parser = AsyncMock(return_value="# Điều 1\n\nNội dung")

    monkeypatch.setattr(routes.RAGEngine, "get_indexing_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "parse_pdf_to_markdown", fake_parser)

    response = client.post(
        "/api/upload",
        files={"file": ("law.pdf", b"%PDF-1.4\n", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    fake_parser.assert_awaited_once_with(str(tmp_path / "law.pdf"))
    fake_rag.ainsert.assert_awaited_once_with("# Điều 1\n\nNội dung", file_paths=["law.pdf"])


def test_txt_upload_uses_indexing_rag_without_pdf_parser(monkeypatch, tmp_path):
    client = make_client()
    monkeypatch.setattr(routes.settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))

    fake_rag = Mock()
    fake_rag.ainsert = AsyncMock()
    fake_parser = AsyncMock()

    monkeypatch.setattr(routes.RAGEngine, "get_indexing_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "parse_pdf_to_markdown", fake_parser)

    response = client.post(
        "/api/upload",
        files={"file": ("law.txt", "Điều 1: tốc độ tối đa".encode("utf-8"), "text/plain")},
    )

    assert response.status_code == 200
    fake_parser.assert_not_called()
    fake_rag.ainsert.assert_awaited_once_with("Điều 1: tốc độ tối đa", file_paths=["law.txt"])
```

Create `backend/tests/test_chat_route_provider_split.py` with this exact content:

```python
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
```

- [ ] **Step 2: Run the route tests to verify they fail**

Run:

```bash
pytest backend/tests/test_upload_route.py backend/tests/test_chat_route_provider_split.py -q
```

Expected: FAIL because the routes still call `RAGEngine.get_instance()`.

- [ ] **Step 3: Update `backend/api/routes.py` to use the split engines**

Modify `backend/api/routes.py` as follows:

1. Change the `/chat` handler to fetch the query engine:

```python
    rag = RAGEngine.get_query_instance()
```

2. Change `/documents` to fetch the query engine:

```python
    rag = RAGEngine.get_query_instance()
```

3. Change `/upload` to fetch the indexing engine:

```python
        rag = RAGEngine.get_indexing_instance()
```

Do not change the existing SSE event payload structure or the `comparison_mode` branching.

- [ ] **Step 4: Run the route tests to verify they pass**

Run:

```bash
pytest backend/tests/test_upload_route.py backend/tests/test_chat_route_provider_split.py -q
```

Expected: PASS with `5 passed`.

- [ ] **Step 5: Commit the route split**

Run:

```bash
git add backend/api/routes.py backend/tests/test_upload_route.py backend/tests/test_chat_route_provider_split.py
git commit -m "feat: split upload and chat rag engines"
```

### Task 5: Update runtime configuration, docs, and regression tests

**Files:**
- Create: `backend/tests/test_multi_provider_docs.py`
- Modify: `.env.example`
- Modify: `docker-compose.yml`
- Modify: `README.md`

- [ ] **Step 1: Write the failing documentation/runtime regression tests**

Create `backend/tests/test_multi_provider_docs.py` with this exact content:

```python
from pathlib import Path


def test_env_example_describes_split_providers():
    env_example = Path(".env.example").read_text(encoding="utf-8")

    assert "INDEXING_LLM_MODEL=deepseek-v4-flash" in env_example
    assert "ANSWER_LLM_MODEL=openai/gpt-oss-120b" in env_example
    assert "EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B" in env_example
    assert "EMBEDDING_BASE_URL=http://host.docker.internal:8002/v1" in env_example
    assert "openai/text-embedding-3-small" not in env_example


def test_docker_compose_exposes_host_gateway_and_embedding_dim_1024():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert 'host.docker.internal:host-gateway' in compose
    assert 'EMBEDDING_DIM=1024' in compose
    assert 'EMBEDDING_BINDING_HOST=${EMBEDDING_BASE_URL}' in compose
    assert 'EXTRACT_LLM_MODEL=${INDEXING_LLM_MODEL}' in compose
    assert 'QUERY_LLM_MODEL=${ANSWER_LLM_MODEL}' in compose


def test_readme_describes_deepseek_openrouter_and_local_vllm():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "DeepSeek-V4-Flash" in readme
    assert "openai/gpt-oss-120b" in readme
    assert "Qwen/Qwen3-Embedding-0.6B" in readme
    assert "vLLM" in readme
    assert "text-embedding-3-small" not in readme
```

- [ ] **Step 2: Run the docs/runtime tests to verify they fail**

Run:

```bash
pytest backend/tests/test_multi_provider_docs.py -q
```

Expected: FAIL because the current docs and Compose files still describe OpenRouter embeddings and the old model defaults.

- [ ] **Step 3: Update `.env.example`, `docker-compose.yml`, and `README.md`**

Replace `.env.example` with this exact content:

```dotenv
# Backend Settings
DATABASE_URL=postgresql://postgres:postgres@db:5432/law_assistant
POSTGRES_HOST=db
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DATABASE=law_assistant
REDIS_URL=redis://redis:6379/0

# Shared answer fallback
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Indexing LLM: DeepSeek direct API
INDEXING_LLM_BASE_URL=https://api.deepseek.com
INDEXING_LLM_API_KEY=your_deepseek_api_key_here
INDEXING_LLM_MODEL=deepseek-v4-flash
INDEXING_LLM_THINKING_MODE=disabled

# Answer LLM: OpenRouter
ANSWER_LLM_BASE_URL=https://openrouter.ai/api/v1
ANSWER_LLM_API_KEY=
ANSWER_LLM_MODEL=openai/gpt-oss-120b

# Local embedding service via vLLM
EMBEDDING_BASE_URL=http://host.docker.internal:8002/v1
EMBEDDING_API_KEY=EMPTY
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DIM=1024
EMBEDDING_MAX_TOKEN_SIZE=512
EMBEDDING_QUERY_PREFIX=Instruct: Given a legal query, retrieve relevant statutes and legal passages.\nQuery:
EMBEDDING_DOCUMENT_PREFIX=

# LightRAG Settings
LIGHTRAG_WORKING_DIR=./backend/data
```

Modify the `backend` service section in `docker-compose.yml` to include these lines:

```yaml
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

and replace the current backend `environment:` block with:

```yaml
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DATABASE}
      - POSTGRES_HOST=db
      - POSTGRES_PORT=5432
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DATABASE=${POSTGRES_DATABASE}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - INDEXING_LLM_BASE_URL=${INDEXING_LLM_BASE_URL}
      - INDEXING_LLM_API_KEY=${INDEXING_LLM_API_KEY}
      - INDEXING_LLM_MODEL=${INDEXING_LLM_MODEL}
      - INDEXING_LLM_THINKING_MODE=${INDEXING_LLM_THINKING_MODE}
      - ANSWER_LLM_BASE_URL=${ANSWER_LLM_BASE_URL}
      - ANSWER_LLM_API_KEY=${ANSWER_LLM_API_KEY}
      - ANSWER_LLM_MODEL=${ANSWER_LLM_MODEL}
      - EMBEDDING_BASE_URL=${EMBEDDING_BASE_URL}
      - EMBEDDING_API_KEY=${EMBEDDING_API_KEY}
      - EMBEDDING_MODEL=${EMBEDDING_MODEL}
      - EMBEDDING_DIM=${EMBEDDING_DIM}
      - EMBEDDING_MAX_TOKEN_SIZE=${EMBEDDING_MAX_TOKEN_SIZE}
      - EMBEDDING_QUERY_PREFIX=${EMBEDDING_QUERY_PREFIX}
      - EMBEDDING_DOCUMENT_PREFIX=${EMBEDDING_DOCUMENT_PREFIX}
      - LIGHTRAG_WORKING_DIR=/app/backend/data
      - SUMMARY_LANGUAGE=Vietnamese
      - 'ENTITY_TYPES=["Văn bản pháp luật", "Điều khoản", "Cơ quan ban hành", "Đối tượng áp dụng", "Hành vi vi phạm", "Hình thức xử phạt", "Thời hạn", "Khái niệm pháp lý"]'
```

Replace the current `rag-ui` `environment:` block with:

```yaml
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DATABASE}
      - POSTGRES_HOST=db
      - POSTGRES_PORT=5432
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DATABASE=${POSTGRES_DATABASE}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - INDEXING_LLM_BASE_URL=${INDEXING_LLM_BASE_URL}
      - INDEXING_LLM_API_KEY=${INDEXING_LLM_API_KEY}
      - INDEXING_LLM_MODEL=${INDEXING_LLM_MODEL}
      - ANSWER_LLM_BASE_URL=${ANSWER_LLM_BASE_URL}
      - ANSWER_LLM_API_KEY=${ANSWER_LLM_API_KEY}
      - ANSWER_LLM_MODEL=${ANSWER_LLM_MODEL}
      - EMBEDDING_BASE_URL=${EMBEDDING_BASE_URL}
      - EMBEDDING_API_KEY=${EMBEDDING_API_KEY}
      - EMBEDDING_MODEL=${EMBEDDING_MODEL}
      - EMBEDDING_DIM=1024
      - LIGHTRAG_WORKING_DIR=/app/backend/data
      - SUMMARY_LANGUAGE=Vietnamese
      - 'ENTITY_TYPES=["Văn bản pháp luật", "Điều khoản", "Cơ quan ban hành", "Đối tượng áp dụng", "Hành vi vi phạm", "Hình thức xử phạt", "Thời hạn", "Khái niệm pháp lý"]'
      - LIGHTRAG_KV_STORAGE=PGKVStorage
      - LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
      - LIGHTRAG_GRAPH_STORAGE=PGGraphStorage
      - LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage
      - LLM_BINDING=openai
      - LLM_BINDING_API_KEY=${OPENROUTER_API_KEY}
      - LLM_BINDING_HOST=${ANSWER_LLM_BASE_URL}
      - LLM_MODEL=${ANSWER_LLM_MODEL}
      - EXTRACT_LLM_BINDING=openai
      - EXTRACT_LLM_BINDING_API_KEY=${INDEXING_LLM_API_KEY}
      - EXTRACT_LLM_BINDING_HOST=${INDEXING_LLM_BASE_URL}
      - EXTRACT_LLM_MODEL=${INDEXING_LLM_MODEL}
      - QUERY_LLM_BINDING=openai
      - QUERY_LLM_BINDING_API_KEY=${OPENROUTER_API_KEY}
      - QUERY_LLM_BINDING_HOST=${ANSWER_LLM_BASE_URL}
      - QUERY_LLM_MODEL=${ANSWER_LLM_MODEL}
      - EMBEDDING_BINDING=openai
      - EMBEDDING_BINDING_API_KEY=${EMBEDDING_API_KEY}
      - EMBEDDING_BINDING_HOST=${EMBEDDING_BASE_URL}
```

Update the `README.md` sections that mention the old embedding and answer models so they read as follows:

1. In the feature list, add a bullet:

```md
- **Role-Specific Inference Stack**: Uses DeepSeek-V4-Flash for KG extraction, `openai/gpt-oss-120b` for answer generation, and local vLLM-served Qwen3 embeddings for retrieval.
```

2. Replace the `LLM/Embeddings` line in the tech stack with:

```md
- **LLM/Embeddings**: DeepSeek-V4-Flash (KG indexing), `openai/gpt-oss-120b` via OpenRouter (answers), `Qwen/Qwen3-Embedding-0.6B` via local vLLM (embeddings)
```

3. Replace the sample environment block with the new provider-specific variables from `.env.example`.

4. Add a setup note under running instructions:

```md
3. **Start the local embedding service (Host Machine)**:
   ~~~bash
   vllm serve Qwen/Qwen3-Embedding-0.6B --port 8002 --task embed
   ~~~

4. **Start the Frontend (Locally)**:
   ~~~bash
   cd frontend
   npm install
   npm run dev
   ~~~
```

5. Add a short warning note after the environment section:

```md
> [!IMPORTANT]
> Changing the embedding model from `text-embedding-3-small` to `Qwen/Qwen3-Embedding-0.6B` changes the embedding dimension from `1536` to `1024`. Existing vector data must be re-indexed after this migration.
```

- [ ] **Step 4: Run the docs/runtime tests to verify they pass**

Run:

```bash
pytest backend/tests/test_multi_provider_docs.py -q
```

Expected: PASS with `3 passed`.

- [ ] **Step 5: Commit the runtime and documentation changes**

Run:

```bash
git add .env.example docker-compose.yml README.md backend/tests/test_multi_provider_docs.py
git commit -m "docs: document multi-provider rag setup"
```

### Task 6: Run focused verification for the full provider split

**Files:**
- No file changes in this task

- [ ] **Step 1: Run the focused backend test suite**

Run:

```bash
pytest \
  backend/tests/test_multi_provider_config.py \
  backend/tests/test_multi_provider_llm_services.py \
  backend/tests/test_rag_engine_manager.py \
  backend/tests/test_upload_route.py \
  backend/tests/test_chat_route_provider_split.py \
  backend/tests/test_multi_provider_docs.py \
  -q
```

Expected: PASS with all targeted tests green.

- [ ] **Step 2: Start the local embedding service on the host**

Run:

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B --port 8002 --task embed
```

Expected: vLLM starts an OpenAI-compatible embeddings server and logs that it is serving requests on port `8002`.

- [ ] **Step 3: Start the Docker services**

Run:

```bash
docker compose up --build backend rag-ui db
```

Expected: backend and rag-ui boot without embedding-dimension mismatch errors.

- [ ] **Step 4: Verify the local embedding endpoint directly**

Run:

```bash
curl http://localhost:8002/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{"model":"Qwen/Qwen3-Embedding-0.6B","input":"Điều 1 tốc độ tối đa là bao nhiêu?","dimensions":1024}'
```

Expected: HTTP `200` with an embedding vector in the JSON payload.

- [ ] **Step 5: Smoke-test indexing and answer generation**

Run:

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@/absolute/path/to/sample-law.txt"
```

Expected: JSON `{"status":"success", ...}` and backend logs showing DeepSeek indexing activity.

Run:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Tốc độ tối đa trong khu dân cư là bao nhiêu?","stream":false,"comparison_mode":false}'
```

Expected: JSON `{"response":"...", "mode":"hybrid"}` and backend logs showing the query engine path, not the indexing engine.

- [ ] **Step 6: Commit the verified migration**

Run:

```bash
git add backend/config.py backend/core/llm_services.py backend/core/rag_engine.py backend/api/routes.py backend/tests .env.example docker-compose.yml README.md
git commit -m "feat: split rag providers for indexing answer and embeddings"
```

## Self-Review

Spec coverage check:

- DeepSeek direct indexing: covered by Tasks 1, 2, 3, 4, and 6.
- OpenRouter `openai/gpt-oss-120b` answer path: covered by Tasks 1, 2, 3, 4, and 5.
- Local vLLM `Qwen/Qwen3-Embedding-0.6B` embeddings: covered by Tasks 1, 2, 3, 5, and 6.
- `1024`-dimension migration and re-index warning: covered by Tasks 1 and 5.
- `rag-ui` compatibility: covered by Task 5.
- SSE contract preservation and comparison mode semantics: covered by Task 4 and Task 6.

Placeholder scan:

- No `TODO`, `TBD`, or unresolved references remain.
- Every code-changing step includes explicit code or exact replacement instructions.

Type consistency:

- `VLLMEmbeddingFunc`, `indexing_llm_func`, and `answer_llm_func` are named consistently across tests and implementation tasks.
- `get_indexing_instance()` and `get_query_instance()` are the route-facing API throughout the plan.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-07-multi-provider-rag.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
