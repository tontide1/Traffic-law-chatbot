# NVIDIA Answer Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move answer generation from OpenRouter to NVIDIA's OpenAI-compatible endpoint while keeping the existing `ANSWER_LLM_*` app contract, `chat.completions.create(...)` flow, and frontend chat/SSE behavior unchanged.

**Architecture:** Keep the answer path on `ANSWER_LLM_*`, change its default base URL to NVIDIA, and remove all answer-path fallback to `OPENROUTER_API_KEY`. To make the spec's fail-fast requirement real rather than aspirational, validate `ANSWER_LLM_API_KEY` during `RAGEngine.initialize()` so stale environments fail during app startup instead of on the first answer request.

**Tech Stack:** Python 3.11, FastAPI, LightRAG, OpenAI Python SDK, pytest, Docker Compose, Markdown docs

---

## Scope Check

This is one subsystem: answer-provider migration. Do not expand into prompt redesign, API migration to `responses.create(...)`, frontend provider switching, or embedding/indexing changes.

## File Map

- Modify: `backend/config.py`  
  Default `ANSWER_LLM_BASE_URL` to NVIDIA and make `ANSWER_LLM_API_KEY` the only accepted answer credential.

- Modify: `backend/core/rag_engine.py`  
  Validate the answer API key during startup so stale environments stop booting before the app begins serving requests.

- Verify only: `backend/core/llm_services.py`  
  Keep the current `get_answer_llm_client()` and `answer_llm_func()` implementation unless tests reveal provider-specific behavior.

- Modify: `backend/tests/test_multi_provider_config.py`  
  Replace fallback-oriented tests with NVIDIA-default and fail-fast answer-key coverage.

- Modify: `backend/tests/test_multi_provider_llm_services.py`  
  Replace OpenRouter-specific answer-client assertions with NVIDIA-specific ones while preserving the existing `chat.completions.create(...)` coverage.

- Modify: `backend/tests/test_rag_engine_manager.py`  
  Add startup-validation coverage proving `RAGEngine.initialize()` now rejects missing `ANSWER_LLM_API_KEY`.

- Modify: `.env.example`  
  Document NVIDIA as the answer provider and remove the OpenRouter answer-fallback section.

- Modify: `docker-compose.yml`  
  Change answer defaults to NVIDIA, keep `ANSWER_LLM_API_KEY` flowing into both `backend` and `rag-ui`, and rewire `rag-ui` answer/query bindings away from `OPENROUTER_API_KEY`.

- Modify: `README.md`  
  Describe NVIDIA as the current answer provider and keep the embedding wording aligned with `AITeamVN/Vietnamese_Embedding_v2`.

- Modify: `backend/tests/test_multi_provider_docs.py`  
  Add narrow regression checks for `.env.example`, `docker-compose.yml`, and `README.md` that focus on this answer-provider migration instead of unrelated hardware-specific docs.

## Notes Before Execution

- Do not preserve backward compatibility with `OPENROUTER_API_KEY` for answers.
- Do not merely change the error message in `Settings`; also add startup validation in `RAGEngine.initialize()`.
- Do not edit `backend/core/llm_services.py` unless tests force it. The provider migration is config and wiring work, not an answer-flow rewrite.
- Keep the answer model as `openai/gpt-oss-120b`.

### Task 1: Tighten the backend contract and make startup truly fail fast

**Files:**
- Modify: `backend/tests/test_multi_provider_config.py`
- Modify: `backend/tests/test_multi_provider_llm_services.py`
- Modify: `backend/tests/test_rag_engine_manager.py`
- Modify: `backend/config.py`
- Modify: `backend/core/rag_engine.py`
- Verify only: `backend/core/llm_services.py`

- [ ] **Step 1: Replace the config and answer-client regression tests with NVIDIA expectations**

Replace `backend/tests/test_multi_provider_config.py` with this exact content:

```python
from backend.config import Settings


def test_multi_provider_defaults_match_the_new_architecture():
    settings = Settings(_env_file=None)

    assert settings.INDEXING_LLM_BASE_URL == "https://api.deepseek.com"
    assert settings.INDEXING_LLM_MODEL == "deepseek-v4-flash"
    assert settings.INDEXING_LLM_THINKING_MODE == "disabled"
    assert settings.ANSWER_LLM_BASE_URL == "https://integrate.api.nvidia.com/v1"
    assert settings.ANSWER_LLM_MODEL == "openai/gpt-oss-120b"
    assert settings.EMBEDDING_BASE_URL == "http://host.docker.internal:8002/v1"
    assert settings.EMBEDDING_MODEL == "AITeamVN/Vietnamese_Embedding_v2"
    assert settings.EMBEDDING_DIM == 1024
    assert settings.EMBEDDING_DOCUMENT_PREFIX == ""
    assert "Given a legal query" in settings.EMBEDDING_QUERY_PREFIX


def test_answer_key_requires_answer_llm_api_key():
    settings = Settings(_env_file=None, ANSWER_LLM_API_KEY="nvidia-key")

    assert settings.get_answer_llm_api_key() == "nvidia-key"


def test_answer_key_does_not_fall_back_to_openrouter_key():
    settings = Settings(
        _env_file=None,
        OPENROUTER_API_KEY="openrouter-key",
        ANSWER_LLM_API_KEY=None,
    )

    try:
        settings.get_answer_llm_api_key()
    except ValueError as exc:
        assert str(exc) == "ANSWER_LLM_API_KEY is required"
    else:
        raise AssertionError("Expected answer API key lookup to ignore OPENROUTER_API_KEY")


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
        assert str(exc) == "ANSWER_LLM_API_KEY is required"
    else:
        raise AssertionError("Expected answer API key lookup to raise")


def test_entity_types_string_parsing_still_works():
    settings = Settings(
        _env_file=None,
        RAG_ENTITY_TYPES='["Điều khoản", "Văn bản pháp luật"]',
    )

    assert settings.RAG_ENTITY_TYPES == ["Điều khoản", "Văn bản pháp luật"]


def test_google_studio_defaults_match_indexing_provider_selector_plan():
    settings = Settings(_env_file=None)

    assert settings.GOOGLE_STUDIO_BASE_URL == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert settings.GOOGLE_STUDIO_MODEL == "gemini-2.5-flash"


def test_missing_google_studio_key_raises_clear_error():
    settings = Settings(_env_file=None, GOOGLE_STUDIO_API_KEY=None)

    try:
        settings.get_google_studio_api_key()
    except ValueError as exc:
        assert str(exc) == "GOOGLE_STUDIO_API_KEY is required"
    else:
        raise AssertionError("Expected Google Studio API key lookup to raise")
```

In `backend/tests/test_multi_provider_llm_services.py`, replace the old OpenRouter-specific answer-client tests with these exact blocks:

```python
def test_get_answer_llm_client_uses_nvidia_settings(monkeypatch):
    created = []

    class DummyClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(llm_services.openai, "AsyncOpenAI", DummyClient)
    monkeypatch.setattr(llm_services.settings, "ANSWER_LLM_API_KEY", "nvidia-key")
    monkeypatch.setattr(llm_services.settings, "ANSWER_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")

    llm_services.get_answer_llm_client()

    assert created == [{"api_key": "nvidia-key", "base_url": "https://integrate.api.nvidia.com/v1"}]
```

```python
def test_answer_client_factory_memoizes_until_reset(monkeypatch):
    created = []

    class DummyClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(llm_services.openai, "AsyncOpenAI", DummyClient)
    monkeypatch.setattr(llm_services.settings, "ANSWER_LLM_API_KEY", "nvidia-key")
    monkeypatch.setattr(llm_services.settings, "ANSWER_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")

    first = llm_services.get_answer_llm_client()
    second = llm_services.get_answer_llm_client()

    assert first is second
    assert len(created) == 1

    llm_services.reset_client_caches()
    third = llm_services.get_answer_llm_client()

    assert third is not first
    assert len(created) == 2
```

In `backend/tests/test_rag_engine_manager.py`, add this exact line inside `test_initialize_builds_two_indexing_engines_and_one_query_engine()` before `asyncio.run(rag_engine.RAGEngine.initialize())`:

```python
    monkeypatch.setattr(rag_engine.settings, "get_answer_llm_api_key", lambda: "nvidia-key")
```

Append this exact test to `backend/tests/test_rag_engine_manager.py`:

```python
def test_initialize_requires_answer_llm_api_key_before_boot(monkeypatch):
    monkeypatch.setattr(rag_engine.settings, "get_answer_llm_api_key", lambda: (_ for _ in ()).throw(ValueError("ANSWER_LLM_API_KEY is required")))

    with pytest.raises(ValueError, match="ANSWER_LLM_API_KEY is required"):
        asyncio.run(rag_engine.RAGEngine.initialize())
```

- [ ] **Step 2: Run the focused backend tests to confirm the current implementation still allows lazy answer-key failure**

Run:

```bash
pytest backend/tests/test_multi_provider_config.py backend/tests/test_multi_provider_llm_services.py backend/tests/test_rag_engine_manager.py -q
```

Expected: FAIL because `ANSWER_LLM_BASE_URL` still defaults to OpenRouter, `get_answer_llm_api_key()` still accepts `OPENROUTER_API_KEY`, and `RAGEngine.initialize()` does not yet validate the answer key at startup.

- [ ] **Step 3: Replace `backend/config.py` with the NVIDIA-only answer-key contract**

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

    INDEXING_LLM_BASE_URL: str = "https://api.deepseek.com"
    INDEXING_LLM_API_KEY: Optional[str] = None
    INDEXING_LLM_MODEL: str = "deepseek-v4-flash"
    INDEXING_LLM_THINKING_MODE: str = "disabled"

    ANSWER_LLM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    ANSWER_LLM_API_KEY: Optional[str] = None
    ANSWER_LLM_MODEL: str = "openai/gpt-oss-120b"

    GOOGLE_STUDIO_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GOOGLE_STUDIO_API_KEY: Optional[str] = None
    GOOGLE_STUDIO_MODEL: str = "gemini-2.5-flash"

    EMBEDDING_BASE_URL: str = "http://host.docker.internal:8002/v1"
    EMBEDDING_API_KEY: str = "EMPTY"
    EMBEDDING_MODEL: str = "AITeamVN/Vietnamese_Embedding_v2"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_MAX_TOKEN_SIZE: int = 512
    EMBEDDING_QUERY_PREFIX: str = (
        "Instruct: Given a legal query, retrieve relevant statutes and legal passages.\nQuery: "
    )
    EMBEDDING_DOCUMENT_PREFIX: str = ""

    SUMMARY_LANGUAGE: str = "Vietnamese"
    RAG_ENTITY_TYPES: list[str] = [
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

    @field_validator("RAG_ENTITY_TYPES", mode="before")
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
        if not self.ANSWER_LLM_API_KEY:
            raise ValueError("ANSWER_LLM_API_KEY is required")
        return self.ANSWER_LLM_API_KEY

    def get_google_studio_api_key(self) -> str:
        if not self.GOOGLE_STUDIO_API_KEY:
            raise ValueError("GOOGLE_STUDIO_API_KEY is required")
        return self.GOOGLE_STUDIO_API_KEY

    def get_embedding_api_key(self) -> str:
        return self.EMBEDDING_API_KEY or "EMPTY"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
```

- [ ] **Step 4: Add explicit startup validation in `backend/core/rag_engine.py`**

In `backend/core/rag_engine.py`, insert this exact validation line near the start of `RAGEngine.initialize()` after `_apply_postgres_environment()`:

```python
        settings.get_answer_llm_api_key()
```

The method should look like this:

```python
    @classmethod
    async def initialize(cls):
        if (
            cls._deepseek_indexing_instance is not None
            and cls._query_instance is not None
        ):
            return cls._deepseek_indexing_instance, cls._query_instance

        cls._apply_postgres_environment()
        settings.get_answer_llm_api_key()

        cls._deepseek_indexing_instance = LightRAG(
            working_dir=settings.LIGHTRAG_WORKING_DIR,
            llm_model_func=indexing_llm_func,
            embedding_func=cls._build_embedding_func(settings.EMBEDDING_DOCUMENT_PREFIX),
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="PGGraphStorage",
            doc_status_storage="PGDocStatusStorage",
            entity_extraction_use_json=True,
            addon_params={
                "language": settings.SUMMARY_LANGUAGE,
                "entity_types": settings.RAG_ENTITY_TYPES,
                "entity_types_guidance": ", ".join(settings.RAG_ENTITY_TYPES),
            },
        )
        await cls._deepseek_indexing_instance.initialize_storages()

        cls._query_instance = LightRAG(
            working_dir=settings.LIGHTRAG_WORKING_DIR,
            llm_model_func=answer_llm_func,
            embedding_func=cls._build_embedding_func(settings.EMBEDDING_QUERY_PREFIX),
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="PGGraphStorage",
            doc_status_storage="PGDocStatusStorage",
            entity_extraction_use_json=True,
            addon_params={
                "language": settings.SUMMARY_LANGUAGE,
                "entity_types": settings.RAG_ENTITY_TYPES,
                "entity_types_guidance": ", ".join(settings.RAG_ENTITY_TYPES),
            },
        )
        await cls._query_instance.initialize_storages()

        return cls._deepseek_indexing_instance, cls._query_instance
```

- [ ] **Step 5: Re-run the focused backend tests**

Run:

```bash
pytest backend/tests/test_multi_provider_config.py backend/tests/test_multi_provider_llm_services.py backend/tests/test_rag_engine_manager.py -q
```

Expected: PASS. This proves the migration now both removes the OpenRouter fallback and enforces the fail-fast startup behavior promised by the spec.

- [ ] **Step 6: Commit the backend contract and startup validation**

Run:

```bash
git add backend/config.py backend/core/rag_engine.py backend/tests/test_multi_provider_config.py backend/tests/test_multi_provider_llm_services.py backend/tests/test_rag_engine_manager.py
git commit -m "refactor: switch answer provider defaults to nvidia"
```

### Task 2: Rewire compose and docs to NVIDIA with narrower regression coverage

**Files:**
- Modify: `backend/tests/test_multi_provider_docs.py`
- Modify: `.env.example`
- Modify: `docker-compose.yml`
- Modify: `README.md`

- [ ] **Step 1: Replace the docs regression test with scope-focused NVIDIA assertions**

Replace `backend/tests/test_multi_provider_docs.py` with this exact content:

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_env_example_describes_nvidia_answer_provider():
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "INDEXING_LLM_MODEL=deepseek-v4-flash" in env_example
    assert "ANSWER_LLM_BASE_URL=https://integrate.api.nvidia.com/v1" in env_example
    assert "ANSWER_LLM_API_KEY=your_nvidia_api_key_here" in env_example
    assert "ANSWER_LLM_MODEL=openai/gpt-oss-120b" in env_example
    assert "# Answer LLM: NVIDIA-hosted OpenAI-compatible endpoint" in env_example
    assert "GOOGLE_STUDIO_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/" in env_example
    assert "GOOGLE_STUDIO_API_KEY=your_google_studio_api_key_here" in env_example
    assert "GOOGLE_STUDIO_MODEL=gemini-2.5-flash" in env_example
    assert "EMBEDDING_MODEL=AITeamVN/Vietnamese_Embedding_v2" in env_example
    assert "OPENROUTER_API_KEY=your_openrouter_api_key_here" not in env_example


def test_docker_compose_routes_answer_traffic_to_nvidia():
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "ANSWER_LLM_BASE_URL=${ANSWER_LLM_BASE_URL:-https://integrate.api.nvidia.com/v1}" in compose
    assert "ANSWER_LLM_MODEL=${ANSWER_LLM_MODEL:-openai/gpt-oss-120b}" in compose
    assert compose.count("ANSWER_LLM_API_KEY=${ANSWER_LLM_API_KEY}") == 2
    assert "LLM_BINDING_API_KEY=${ANSWER_LLM_API_KEY}" in compose
    assert "LLM_BINDING_HOST=${ANSWER_LLM_BASE_URL:-https://integrate.api.nvidia.com/v1}" in compose
    assert "QUERY_LLM_BINDING_API_KEY=${ANSWER_LLM_API_KEY}" in compose
    assert "QUERY_LLM_BINDING_HOST=${ANSWER_LLM_BASE_URL:-https://integrate.api.nvidia.com/v1}" in compose
    assert "GOOGLE_STUDIO_BASE_URL=${GOOGLE_STUDIO_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}" in compose
    assert "GOOGLE_STUDIO_API_KEY=${GOOGLE_STUDIO_API_KEY}" in compose
    assert "GOOGLE_STUDIO_MODEL=${GOOGLE_STUDIO_MODEL:-gemini-2.5-flash}" in compose
    assert "${OPENROUTER_API_KEY}" not in compose


def test_readme_describes_deepseek_nvidia_and_current_embedding_stack():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "DeepSeek-V4-Flash" in readme
    assert "NVIDIA-hosted OpenAI-compatible endpoint" in readme
    assert "openai/gpt-oss-120b" in readme
    assert "AITeamVN/Vietnamese_Embedding_v2" in readme
    assert "NVIDIA API Key" in readme
    assert "OpenRouter API Key" not in readme
    assert "via OpenRouter" not in readme


def test_readme_keeps_google_studio_selector_context():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Google Studio" in readme
    assert "left sidebar" in readme
    assert "shared knowledge base" in readme
    assert "Existing documents are not rebuilt automatically" in readme
```

- [ ] **Step 2: Run the docs regression tests to capture the current OpenRouter wording and wiring**

Run:

```bash
pytest backend/tests/test_multi_provider_docs.py -q
```

Expected: FAIL because `.env.example`, `docker-compose.yml`, and `README.md` still describe OpenRouter or still wire `rag-ui` answer/query bindings through `OPENROUTER_API_KEY`.

- [ ] **Step 3: Replace `.env.example` with the NVIDIA answer-provider configuration**

Replace `.env.example` with this exact content:

```env
# Backend Settings
DATABASE_URL=postgresql://postgres:postgres@db:5432/law_assistant
POSTGRES_HOST=db
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DATABASE=law_assistant
REDIS_URL=redis://redis:6379/0

# Indexing LLM: DeepSeek direct API
INDEXING_LLM_BASE_URL=https://api.deepseek.com
INDEXING_LLM_API_KEY=your_deepseek_api_key_here
INDEXING_LLM_MODEL=deepseek-v4-flash
INDEXING_LLM_THINKING_MODE=disabled

# Google Studio indexing provider
GOOGLE_STUDIO_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GOOGLE_STUDIO_API_KEY=your_google_studio_api_key_here
GOOGLE_STUDIO_MODEL=gemini-2.5-flash

# Answer LLM: NVIDIA-hosted OpenAI-compatible endpoint
ANSWER_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
ANSWER_LLM_API_KEY=your_nvidia_api_key_here
ANSWER_LLM_MODEL=openai/gpt-oss-120b

# Local embedding service via vLLM
EMBEDDING_BASE_URL=http://host.docker.internal:8002/v1
EMBEDDING_API_KEY=EMPTY
EMBEDDING_MODEL=AITeamVN/Vietnamese_Embedding_v2
EMBEDDING_DIM=1024
EMBEDDING_MAX_TOKEN_SIZE=512

# LightRAG Settings
LIGHTRAG_WORKING_DIR=./backend/data
```

- [ ] **Step 4: Update `docker-compose.yml` so backend and `rag-ui` answer bindings all point at NVIDIA**

In the `backend` service, make the answer-related environment entries match this exact snippet:

```yaml
      - GOOGLE_STUDIO_BASE_URL=${GOOGLE_STUDIO_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}
      - GOOGLE_STUDIO_API_KEY=${GOOGLE_STUDIO_API_KEY}
      - GOOGLE_STUDIO_MODEL=${GOOGLE_STUDIO_MODEL:-gemini-2.5-flash}
      - ANSWER_LLM_BASE_URL=${ANSWER_LLM_BASE_URL:-https://integrate.api.nvidia.com/v1}
      - ANSWER_LLM_API_KEY=${ANSWER_LLM_API_KEY}
      - ANSWER_LLM_MODEL=${ANSWER_LLM_MODEL:-openai/gpt-oss-120b}
```

In the `rag-ui` service, make the answer-related environment entries match this exact snippet:

```yaml
      - ANSWER_LLM_BASE_URL=${ANSWER_LLM_BASE_URL:-https://integrate.api.nvidia.com/v1}
      - ANSWER_LLM_API_KEY=${ANSWER_LLM_API_KEY}
      - ANSWER_LLM_MODEL=${ANSWER_LLM_MODEL:-openai/gpt-oss-120b}
      - LLM_BINDING=openai
      - LLM_BINDING_API_KEY=${ANSWER_LLM_API_KEY}
      - LLM_BINDING_HOST=${ANSWER_LLM_BASE_URL:-https://integrate.api.nvidia.com/v1}
      - LLM_MODEL=${ANSWER_LLM_MODEL:-openai/gpt-oss-120b}
      - QUERY_LLM_BINDING=openai
      - QUERY_LLM_BINDING_API_KEY=${ANSWER_LLM_API_KEY}
      - QUERY_LLM_BINDING_HOST=${ANSWER_LLM_BASE_URL:-https://integrate.api.nvidia.com/v1}
      - QUERY_LLM_MODEL=${ANSWER_LLM_MODEL:-openai/gpt-oss-120b}
```

Remove every occurrence of these OpenRouter answer-path lines from `docker-compose.yml`:

```yaml
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - LLM_BINDING_API_KEY=${OPENROUTER_API_KEY}
      - QUERY_LLM_BINDING_API_KEY=${OPENROUTER_API_KEY}
```

- [ ] **Step 5: Replace the OpenRouter wording in `README.md` with NVIDIA wording and fix the embedding bullet**

In the feature list, replace the current answer-stack bullet with this exact line:

```markdown
- **Role-Specific Inference Stack**: Uses DeepSeek-V4-Flash for KG extraction, `openai/gpt-oss-120b` through the NVIDIA-hosted OpenAI-compatible endpoint for answer generation, and `AITeamVN/Vietnamese_Embedding_v2` via local vLLM for retrieval.
```

In the tech-stack list, replace the current LLM bullet with this exact line:

```markdown
- **LLM/Embeddings**: DeepSeek-V4-Flash (KG indexing), `openai/gpt-oss-120b` via NVIDIA-hosted OpenAI-compatible endpoint (answers), `AITeamVN/Vietnamese_Embedding_v2` via local vLLM (embeddings)
```

Replace the current prerequisites and environment-setup section with this exact block:

````markdown
### Prerequisites

- Docker and Docker Compose
- DeepSeek API Key
- NVIDIA API Key
- Optional: Google Studio API Key for sidebar indexing-provider selection

### Environment Setup

Create a `.env` file in the root directory (refer to `.env.example`):

```bash
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DATABASE=law_assistant
INDEXING_LLM_BASE_URL=https://api.deepseek.com
INDEXING_LLM_API_KEY=your_deepseek_key_here
INDEXING_LLM_MODEL=deepseek-v4-flash
ANSWER_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
ANSWER_LLM_API_KEY=your_nvidia_key_here
ANSWER_LLM_MODEL=openai/gpt-oss-120b
EMBEDDING_BASE_URL=http://host.docker.internal:8002/v1
EMBEDDING_MODEL=AITeamVN/Vietnamese_Embedding_v2
EMBEDDING_DIM=1024
```

Google Studio indexing uses the Gemini OpenAI-compatible endpoint. Set `GOOGLE_STUDIO_API_KEY` in `.env` if you want the sidebar selector to support Google Studio builds in addition to DeepSeek. Answer generation runs through NVIDIA's OpenAI-compatible endpoint using `ANSWER_LLM_*`.
````

- [ ] **Step 6: Re-run the docs regression tests**

Run:

```bash
pytest backend/tests/test_multi_provider_docs.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the compose/docs migration**

Run:

```bash
git add .env.example docker-compose.yml README.md backend/tests/test_multi_provider_docs.py
git commit -m "docs: switch answer provider references to nvidia"
```

### Task 3: Run the targeted verification bundle

**Files:**
- Verify: `backend/config.py`
- Verify: `backend/core/rag_engine.py`
- Verify: `backend/core/llm_services.py`
- Verify: `.env.example`
- Verify: `docker-compose.yml`
- Verify: `README.md`
- Verify: `backend/tests/test_multi_provider_config.py`
- Verify: `backend/tests/test_multi_provider_llm_services.py`
- Verify: `backend/tests/test_rag_engine_manager.py`
- Verify: `backend/tests/test_multi_provider_docs.py`

- [ ] **Step 1: Run the full targeted regression suite**

Run:

```bash
pytest backend/tests/test_multi_provider_config.py backend/tests/test_multi_provider_llm_services.py backend/tests/test_rag_engine_manager.py backend/tests/test_multi_provider_docs.py -q
```

Expected: PASS.

- [ ] **Step 2: Byte-compile the touched backend modules**

Run:

```bash
python -m py_compile backend/config.py backend/core/rag_engine.py backend/core/llm_services.py
```

Expected: No output and exit code `0`.

- [ ] **Step 3: Render the Compose file and verify the answer-key wiring explicitly**

Run:

```bash
docker compose config >/tmp/nvidia-answer-provider-compose.yaml
rg -n "ANSWER_LLM_API_KEY|LLM_BINDING_API_KEY|QUERY_LLM_BINDING_API_KEY|OPENROUTER_API_KEY" /tmp/nvidia-answer-provider-compose.yaml
```

Expected:

```text
backend:
  ANSWER_LLM_API_KEY: ...
rag-ui:
  ANSWER_LLM_API_KEY: ...
  LLM_BINDING_API_KEY: ...
  QUERY_LLM_BINDING_API_KEY: ...
```

Expected also: no `OPENROUTER_API_KEY` matches in `/tmp/nvidia-answer-provider-compose.yaml`.

- [ ] **Step 4: Confirm no answer-path OpenRouter references remain in source and docs**

Run:

```bash
rg -n "OpenRouter|openrouter|OPENROUTER_API_KEY" backend/config.py backend/core/rag_engine.py backend/tests/test_multi_provider_config.py backend/tests/test_multi_provider_llm_services.py backend/tests/test_rag_engine_manager.py backend/tests/test_multi_provider_docs.py .env.example README.md docker-compose.yml
```

Expected: no matches. `rg` should exit with code `1`.

- [ ] **Step 5: Do not create another commit unless verification forces a follow-up fix**

If all checks pass exactly as expected, stop here. The two commits from Task 1 and Task 2 are sufficient for this migration.
