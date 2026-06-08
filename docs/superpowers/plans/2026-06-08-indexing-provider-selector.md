# Indexing Provider Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent left-sidebar indexing-provider selector that lets users choose `DeepSeek` or `Google Studio` for future knowledge-graph builds, keeps one shared graph, and shows document-level provider provenance without changing chat behavior.

**Architecture:** Keep the existing shared-storage LightRAG architecture, add a second indexing-only engine for Google Studio, and route `/api/upload` to the provider-specific indexing engine chosen by the frontend. Persist the selected provider in browser `localStorage`, store per-document provider metadata in a small JSON file under `LIGHTRAG_WORKING_DIR`, and surface that metadata through `/api/documents`.

**Tech Stack:** Python 3.11, FastAPI, LightRAG, OpenAI-compatible async clients, Google Gemini OpenAI-compat endpoint, React 18, TypeScript, Vite, Vitest, React Testing Library, pytest, Docker Compose

---

## File Map

- Modify: `backend/config.py`  
  Keep current DeepSeek indexing settings, add Google Studio configuration, and expose an explicit Google Studio API-key helper.

- Modify: `backend/core/llm_services.py`  
  Add a Google Studio OpenAI-compatible client cache and a provider-specific indexing LLM function while keeping the answer path unchanged.

- Create: `backend/core/indexing_provider_store.py`  
  Persist and read document-to-provider provenance in a small JSON file under `LIGHTRAG_WORKING_DIR`.

- Modify: `backend/core/rag_engine.py`  
  Initialize three LightRAG instances: `deepseek` indexing, `google_studio` indexing, and the existing query engine.

- Modify: `backend/api/routes.py`  
  Accept a `provider` form field on upload, validate it, route to the correct indexing engine, write provider metadata, and return `indexed_provider` from `/documents`.

- Modify: `backend/tests/test_multi_provider_config.py`  
  Extend configuration tests for the new Google Studio settings.

- Modify: `backend/tests/test_multi_provider_llm_services.py`  
  Add tests for Google Studio client construction and request parameters.

- Create: `backend/tests/test_indexing_provider_store.py`  
  Unit tests for the auxiliary provenance store.

- Modify: `backend/tests/test_rag_engine_manager.py`  
  Verify that the engine manager builds both indexing engines plus the query engine.

- Modify: `backend/tests/test_upload_route.py`  
  Verify provider-aware upload behavior, defaulting, invalid-provider rejection, and metadata writes.

- Modify: `backend/tests/test_chat_route_provider_split.py`  
  Verify `/documents` now returns `indexed_provider` while `/chat` behavior stays unchanged.

- Modify: `frontend/package.json`  
  Add a minimal frontend unit-test stack and scripts.

- Create: `frontend/vitest.config.ts`  
  Configure Vitest for the Vite React frontend.

- Create: `frontend/src/test/setup.ts`  
  Install `jest-dom` matchers and any tiny test setup needed by jsdom.

- Create: `frontend/src/lib/indexingProvider.ts`  
  Centralize provider constants, labels, validation, and `localStorage` helpers.

- Create: `frontend/src/lib/indexingProvider.test.ts`  
  Test provider persistence helpers and fallback behavior.

- Modify: `frontend/src/components/FileUpload.tsx`  
  Accept the selected provider and upload-state callback, send provider in `FormData`, show provider-aware status copy, and lock provider switching during active indexing.

- Create: `frontend/src/components/FileUpload.test.tsx`  
  Verify the upload request includes the provider and that upload copy reflects the chosen provider.

- Modify: `frontend/src/App.tsx`  
  Own provider state in the sidebar, persist it, disable provider changes during upload, show provider provenance in the document list, and replace the misleading static footer copy.

- Create: `frontend/src/App.test.tsx`  
  Verify provider restoration from `localStorage` and provider badge rendering.

- Modify: `.env.example`  
  Document Google Studio configuration alongside the existing DeepSeek/OpenRouter settings.

- Modify: `docker-compose.yml`  
  Pass the Google Studio variables through the backend container environment.

- Modify: `README.md`  
  Document the new indexing-provider selector, the shared-graph semantics, and Google Studio setup.

- Modify: `backend/tests/test_multi_provider_docs.py`  
  Add regression coverage for the new environment variables and README copy.

## Notes Before Execution

- Use the official Google Gemini OpenAI-compatible endpoint for the indexing path instead of switching to a second SDK family. Google documents the compatibility base URL as `https://generativelanguage.googleapis.com/v1beta/openai/`.
- Keep `INDEXING_LLM_*` as the DeepSeek provider configuration to minimize disruption. Add `GOOGLE_STUDIO_*` settings rather than renaming existing DeepSeek variables.
- Do not split the graph by provider. Both indexing engines must continue writing to the same shared LightRAG storage.
- Use `file.filename` and `status_obj.file_path` as the shared provenance key for v1. Missing metadata must render as `legacy`, not be guessed from the current selector.
- Add no frontend runtime dependencies beyond what is needed to run focused unit tests.

### Task 1: Extend backend config and provider client tests

**Files:**
- Modify: `backend/tests/test_multi_provider_config.py`
- Modify: `backend/tests/test_multi_provider_llm_services.py`
- Modify: `backend/config.py`
- Modify: `backend/core/llm_services.py`

- [ ] **Step 1: Write the failing config tests for Google Studio settings**

Append these tests to `backend/tests/test_multi_provider_config.py`:

```python
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

- [ ] **Step 2: Write the failing llm-service tests for Google Studio**

Append these tests to `backend/tests/test_multi_provider_llm_services.py`:

```python
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
```

- [ ] **Step 3: Run the focused backend config/service tests to confirm they fail**

Run:

```bash
pytest backend/tests/test_multi_provider_config.py backend/tests/test_multi_provider_llm_services.py -q
```

Expected: FAIL because `GOOGLE_STUDIO_*`, `get_google_studio_api_key()`, `get_google_studio_llm_client()`, and `google_studio_indexing_llm_func()` do not exist yet.

- [ ] **Step 4: Add Google Studio settings in `backend/config.py`**

Update `backend/config.py` by inserting these fields and helper inside `Settings`:

```python
    GOOGLE_STUDIO_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GOOGLE_STUDIO_API_KEY: Optional[str] = None
    GOOGLE_STUDIO_MODEL: str = "gemini-2.5-flash"

    def get_google_studio_api_key(self) -> str:
        if not self.GOOGLE_STUDIO_API_KEY:
            raise ValueError("GOOGLE_STUDIO_API_KEY is required")
        return self.GOOGLE_STUDIO_API_KEY
```

- [ ] **Step 5: Add the Google Studio client path in `backend/core/llm_services.py`**

Make these changes in `backend/core/llm_services.py`:

```python
_google_studio_client: Optional[openai.AsyncOpenAI] = None
```

```python
def reset_client_caches():
    global _indexing_client, _google_studio_client, _answer_client, _embedding_client
    _indexing_client = None
    _google_studio_client = None
    _answer_client = None
    _embedding_client = None
```

```python
def get_google_studio_llm_client():
    global _google_studio_client
    if _google_studio_client is None:
        _google_studio_client = _build_async_client(
            api_key=settings.get_google_studio_api_key(),
            base_url=settings.GOOGLE_STUDIO_BASE_URL,
        )
    return _google_studio_client
```

```python
async def google_studio_indexing_llm_func(
    prompt: str,
    system_prompt: str = None,
    history: List[dict] = None,
    **kwargs,
) -> str | AsyncIterator[str]:
    client = get_google_studio_llm_client()
    messages = _base_messages(prompt, system_prompt=system_prompt, history=history)
    request_kwargs = {k: v for k, v in kwargs.items() if k != "model"}

    response = await client.chat.completions.create(
        model=settings.GOOGLE_STUDIO_MODEL,
        messages=messages,
        **request_kwargs,
    )
    return _stream_or_content(response, stream=bool(request_kwargs.get("stream")))
```

- [ ] **Step 6: Re-run the focused backend config/service tests**

Run:

```bash
pytest backend/tests/test_multi_provider_config.py backend/tests/test_multi_provider_llm_services.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the backend provider-config foundation**

Run:

```bash
git add backend/config.py backend/core/llm_services.py backend/tests/test_multi_provider_config.py backend/tests/test_multi_provider_llm_services.py
git commit -m "feat: add google studio indexing provider config"
```

### Task 2: Add the document-provider provenance store

**Files:**
- Create: `backend/tests/test_indexing_provider_store.py`
- Create: `backend/core/indexing_provider_store.py`

- [ ] **Step 1: Write the failing provenance-store tests**

Create `backend/tests/test_indexing_provider_store.py` with this exact content:

```python
import asyncio
import json

from backend.core.indexing_provider_store import IndexingProviderStore


def test_store_returns_legacy_when_metadata_file_is_missing(tmp_path):
    store = IndexingProviderStore(str(tmp_path))

    assert asyncio.run(store.get_provider("law.pdf")) == "legacy"


def test_store_persists_and_reads_provider_by_file_path(tmp_path):
    store = IndexingProviderStore(str(tmp_path))

    asyncio.run(store.set_provider("law.pdf", "google_studio"))

    assert asyncio.run(store.get_provider("law.pdf")) == "google_studio"
    payload = json.loads((tmp_path / "indexed_providers.json").read_text(encoding="utf-8"))
    assert payload == {"law.pdf": "google_studio"}


def test_store_preserves_existing_entries_when_adding_new_one(tmp_path):
    store = IndexingProviderStore(str(tmp_path))

    asyncio.run(store.set_provider("a.pdf", "deepseek"))
    asyncio.run(store.set_provider("b.pdf", "google_studio"))

    assert asyncio.run(store.get_provider("a.pdf")) == "deepseek"
    assert asyncio.run(store.get_provider("b.pdf")) == "google_studio"
```

- [ ] **Step 2: Run the provenance-store tests to confirm they fail**

Run:

```bash
pytest backend/tests/test_indexing_provider_store.py -q
```

Expected: FAIL because `backend.core.indexing_provider_store` does not exist yet.

- [ ] **Step 3: Implement the provenance store**

Create `backend/core/indexing_provider_store.py` with this exact content:

```python
import asyncio
import json
from pathlib import Path


class IndexingProviderStore:
    _lock = asyncio.Lock()  # Class-level lock shared by all instances

    def __init__(self, working_dir: str):
        self._path = Path(working_dir) / "indexed_providers.json"

    async def _read(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    async def get_provider(self, file_path: str) -> str:
        async with self._lock:
            payload = await self._read()
            return payload.get(file_path, "legacy")

    async def set_provider(self, file_path: str, provider: str) -> None:
        async with self._lock:
            payload = await self._read()
            payload[file_path] = provider
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
```

- [ ] **Step 4: Re-run the provenance-store tests**

Run:

```bash
pytest backend/tests/test_indexing_provider_store.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the provenance-store helper**

Run:

```bash
git add backend/core/indexing_provider_store.py backend/tests/test_indexing_provider_store.py
git commit -m "feat: persist document indexing providers"
```

### Task 3: Make the engine manager provider-aware

**Files:**
- Modify: `backend/tests/test_rag_engine_manager.py`
- Modify: `backend/core/rag_engine.py`

- [ ] **Step 1: Write the failing engine-manager test**

Replace the main initialization test in `backend/tests/test_rag_engine_manager.py` with this version:

```python
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

    asyncio.run(rag_engine.RAGEngine.initialize())

    assert len(created) == 3
    assert created[0]["llm_model_func"] is rag_engine.indexing_llm_func
    assert created[1]["llm_model_func"] is rag_engine.google_studio_indexing_llm_func
    assert created[2]["llm_model_func"] is rag_engine.answer_llm_func
    assert created[0]["embedding_func"]["func"] == "embed:"
    assert created[1]["embedding_func"]["func"] == "embed:"
    assert created[2]["embedding_func"]["func"] == "embed:search_query: "
```

Add this new accessor test at the end of the file:

```python
def test_get_indexing_instance_selects_provider_after_initialize(monkeypatch):
    rag_engine.RAGEngine._deepseek_indexing_instance = "deepseek-engine"
    rag_engine.RAGEngine._google_studio_indexing_instance = "google-engine"

    assert rag_engine.RAGEngine.get_indexing_instance("deepseek") == "deepseek-engine"
    assert rag_engine.RAGEngine.get_indexing_instance("google_studio") == "google-engine"
```

- [ ] **Step 2: Run the engine-manager tests to confirm they fail**

Run:

```bash
pytest backend/tests/test_rag_engine_manager.py -q
```

Expected: FAIL because the engine manager still builds only two LightRAG instances and does not support provider selection.

- [ ] **Step 3: Implement provider-aware indexing engines**

Update `backend/core/rag_engine.py` to follow this structure:

```python
from backend.core.llm_services import (
    VLLMEmbeddingFunc,
    answer_llm_func,
    google_studio_indexing_llm_func,
    indexing_llm_func,
)
```

```python
class RAGEngine:
    _deepseek_indexing_instance = None
    _google_studio_indexing_instance = None
    _query_instance = None
```

```python
    @classmethod
    async def initialize(cls):
        if (
            cls._deepseek_indexing_instance is not None
            and cls._google_studio_indexing_instance is not None
            and cls._query_instance is not None
        ):
            return (
                cls._deepseek_indexing_instance,
                cls._google_studio_indexing_instance,
                cls._query_instance,
            )

        cls._apply_postgres_environment()

        cls._deepseek_indexing_instance = LightRAG(
            working_dir=settings.LIGHTRAG_WORKING_DIR,
            llm_model_func=indexing_llm_func,
            embedding_func=cls._build_embedding_func(settings.EMBEDDING_DOCUMENT_PREFIX),
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="PGGraphStorage",
            doc_status_storage="PGDocStatusStorage",
            addon_params={"language": settings.SUMMARY_LANGUAGE, "entity_types": settings.ENTITY_TYPES},
        )
        await cls._deepseek_indexing_instance.initialize_storages()

        cls._google_studio_indexing_instance = LightRAG(
            working_dir=settings.LIGHTRAG_WORKING_DIR,
            llm_model_func=google_studio_indexing_llm_func,
            embedding_func=cls._build_embedding_func(settings.EMBEDDING_DOCUMENT_PREFIX),
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="PGGraphStorage",
            doc_status_storage="PGDocStatusStorage",
            addon_params={"language": settings.SUMMARY_LANGUAGE, "entity_types": settings.ENTITY_TYPES},
        )
        await cls._google_studio_indexing_instance.initialize_storages()

        cls._query_instance = LightRAG(
            working_dir=settings.LIGHTRAG_WORKING_DIR,
            llm_model_func=answer_llm_func,
            embedding_func=cls._build_embedding_func(settings.EMBEDDING_QUERY_PREFIX),
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="PGGraphStorage",
            doc_status_storage="PGDocStatusStorage",
            addon_params={"language": settings.SUMMARY_LANGUAGE, "entity_types": settings.ENTITY_TYPES},
        )
        await cls._query_instance.initialize_storages()

        return (
            cls._deepseek_indexing_instance,
            cls._google_studio_indexing_instance,
            cls._query_instance,
        )
```

```python
    @classmethod
    def get_indexing_instance(cls, provider: str = "deepseek"):
        if provider == "deepseek":
            if cls._deepseek_indexing_instance is None:
                raise RuntimeError("DeepSeek indexing RAG engine not initialized. Call RAGEngine.initialize() first.")
            return cls._deepseek_indexing_instance
        if provider == "google_studio":
            if cls._google_studio_indexing_instance is None:
                raise RuntimeError("Google Studio indexing RAG engine not initialized. Call RAGEngine.initialize() first.")
            return cls._google_studio_indexing_instance
        raise ValueError(f"Unsupported indexing provider: {provider}")
```

- [ ] **Step 4: Re-run the engine-manager tests**

Run:

```bash
pytest backend/tests/test_rag_engine_manager.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the provider-aware engine manager**

Run:

```bash
git add backend/core/rag_engine.py backend/tests/test_rag_engine_manager.py
git commit -m "feat: add provider-aware indexing engines"
```

### Task 4: Route uploads through the chosen provider and expose document provenance

**Files:**
- Modify: `backend/tests/test_upload_route.py`
- Modify: `backend/tests/test_chat_route_provider_split.py`
- Modify: `backend/api/routes.py`

- [ ] **Step 1: Write the failing route tests**

Update the existing lambda mocks in `backend/tests/test_upload_route.py` from `lambda: fake_rag` to `lambda *args, **kwargs: fake_rag` to keep compatibility with positional parameters.

Then append these tests to `backend/tests/test_upload_route.py`:

```python
def test_upload_defaults_provider_to_deepseek(monkeypatch, tmp_path):
    client = make_client()
    monkeypatch.setattr(routes.settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))

    fake_rag = Mock()
    fake_rag.ainsert = AsyncMock()
    fake_store = Mock()
    fake_store.set_provider = AsyncMock()

    monkeypatch.setattr(routes.RAGEngine, "get_indexing_instance", lambda provider="deepseek": fake_rag)
    monkeypatch.setattr(routes, "parse_pdf_to_markdown", AsyncMock(return_value="content"))
    monkeypatch.setattr(routes, "IndexingProviderStore", lambda working_dir: fake_store)

    response = client.post(
        "/api/upload",
        files={"file": ("law.pdf", b"%PDF-1.4\n", "application/pdf")},
    )

    assert response.status_code == 200
    fake_store.set_provider.assert_awaited_once_with("law.pdf", "deepseek")


def test_upload_routes_google_studio_provider(monkeypatch, tmp_path):
    client = make_client()
    monkeypatch.setattr(routes.settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))

    fake_google_rag = Mock()
    fake_google_rag.ainsert = AsyncMock()
    fake_store = Mock()
    fake_store.set_provider = AsyncMock()
    selected = []

    def fake_get_indexing_instance(provider="deepseek"):
        selected.append(provider)
        return fake_google_rag

    monkeypatch.setattr(routes.RAGEngine, "get_indexing_instance", fake_get_indexing_instance)
    monkeypatch.setattr(routes, "parse_pdf_to_markdown", AsyncMock(return_value="content"))
    monkeypatch.setattr(routes, "IndexingProviderStore", lambda working_dir: fake_store)

    response = client.post(
        "/api/upload",
        data={"provider": "google_studio"},
        files={"file": ("law.pdf", b"%PDF-1.4\n", "application/pdf")},
    )

    assert response.status_code == 200
    assert selected == ["google_studio"]
    fake_store.set_provider.assert_awaited_once_with("law.pdf", "google_studio")


def test_upload_rejects_invalid_provider(monkeypatch, tmp_path):
    client = make_client()
    monkeypatch.setattr(routes.settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))

    response = client.post(
        "/api/upload",
        data={"provider": "not-real"},
        files={"file": ("law.pdf", b"%PDF-1.4\n", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported indexing provider: not-real"
```

Replace the documents-route test in `backend/tests/test_chat_route_provider_split.py` with this version:

```python
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
    fake_store.get_provider = AsyncMock(return_value="google_studio")

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "IndexingProviderStore", lambda working_dir: fake_store)

    response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json()[0]["id"] == "doc-1"
    assert response.json()[0]["indexed_provider"] == "google_studio"
```

- [ ] **Step 2: Run the route tests to confirm they fail**

Run:

```bash
pytest backend/tests/test_upload_route.py backend/tests/test_chat_route_provider_split.py -q
```

Expected: FAIL because `/api/upload` does not accept `provider`, does not validate it, and `/api/documents` does not return `indexed_provider`.

- [ ] **Step 3: Implement provider-aware upload and documents routes**

Update the imports and helpers in `backend/api/routes.py` like this:

```python
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
```

```python
from backend.core.indexing_provider_store import IndexingProviderStore
```

```python
SUPPORTED_INDEXING_PROVIDERS = {"deepseek", "google_studio"}


def resolve_indexing_provider(provider: str | None) -> str:
    value = (provider or "deepseek").strip().lower()
    if value not in SUPPORTED_INDEXING_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported indexing provider: {provider}")
    return value
```

Update `/documents` to include provider provenance:

```python
@router.get("/documents")
async def list_documents():
    rag = RAGEngine.get_query_instance()
    provider_store = IndexingProviderStore(settings.LIGHTRAG_WORKING_DIR)
    try:
        docs_tuple, _ = await rag.doc_status.get_docs_paginated()
        result = []
        for doc_id, status_obj in docs_tuple:
            status_str = "unknown"
            if hasattr(status_obj.status, "value"):
                status_str = status_obj.status.value
            elif isinstance(status_obj.status, str):
                status_str = status_obj.status

            source = status_obj.file_path or "unknown"
            result.append(
                {
                    "id": doc_id,
                    "status": status_str,
                    "source": source,
                    "content_summary": (status_obj.content_summary[:100] + "...") if status_obj.content_summary else "",
                    "indexed_provider": await provider_store.get_provider(source),
                }
            )
        return result
    except Exception as e:
        print(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

Update `/upload` to accept and use `provider`:

```python
@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), provider: str = Form("deepseek")):
    resolved_provider = resolve_indexing_provider(provider)

    if not file.filename.endswith(".pdf") and not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    file_path = os.path.join(settings.LIGHTRAG_WORKING_DIR, file.filename)
    os.makedirs(settings.LIGHTRAG_WORKING_DIR, exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        rag = RAGEngine.get_indexing_instance(resolved_provider)
        provider_store = IndexingProviderStore(settings.LIGHTRAG_WORKING_DIR)

        if file.filename.endswith(".pdf"):
            content = await parse_pdf_to_markdown(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        if not content.strip():
            raise ValueError("File is empty or no text could be extracted")

        await rag.ainsert(content, file_paths=[file.filename])
        await provider_store.set_provider(file.filename, resolved_provider)

        return UploadResponse(
            filename=file.filename,
            status="success",
            message=f"File uploaded and indexed with {resolved_provider}",
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error indexing uploaded file {file.filename}: {e}")
        return UploadResponse(
            filename=file.filename,
            status="error",
            message=f"Failed to index file: {str(e)}",
        )
```

- [ ] **Step 4: Re-run the route tests**

Run:

```bash
pytest backend/tests/test_upload_route.py backend/tests/test_chat_route_provider_split.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the provider-aware routes**

Run:

```bash
git add backend/api/routes.py backend/tests/test_upload_route.py backend/tests/test_chat_route_provider_split.py
git commit -m "feat: route uploads by indexing provider"
```

### Task 5: Add a minimal frontend unit-test harness and provider helpers

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/vitest.config.ts`
- Create: `frontend/src/test/setup.ts`
- Create: `frontend/src/lib/indexingProvider.ts`
- Create: `frontend/src/lib/indexingProvider.test.ts`

- [ ] **Step 1: Write the failing frontend helper tests**

Create `frontend/src/lib/indexingProvider.test.ts` with this exact content:

```ts
import { describe, expect, it } from 'vitest'

import {
  DEFAULT_INDEXING_PROVIDER,
  getIndexingProviderLabel,
  getStoredIndexingProvider,
  isIndexingProvider,
  setStoredIndexingProvider,
} from './indexingProvider'

describe('indexingProvider helpers', () => {
  it('falls back to deepseek for missing or invalid storage values', () => {
    const storage = {
      getItem: () => 'not-real',
      setItem: () => undefined,
    }

    expect(getStoredIndexingProvider(storage as Storage)).toBe(DEFAULT_INDEXING_PROVIDER)
  })

  it('persists the selected provider', () => {
    let stored = ''
    const storage = {
      getItem: () => stored,
      setItem: (_key: string, value: string) => {
        stored = value
      },
    }

    setStoredIndexingProvider('google_studio', storage as Storage)

    expect(stored).toBe('google_studio')
    expect(getStoredIndexingProvider(storage as Storage)).toBe('google_studio')
  })

  it('returns friendly labels for supported and legacy providers', () => {
    expect(isIndexingProvider('deepseek')).toBe(true)
    expect(isIndexingProvider('google_studio')).toBe(true)
    expect(isIndexingProvider('legacy')).toBe(false)
    expect(getIndexingProviderLabel('deepseek')).toBe('DeepSeek')
    expect(getIndexingProviderLabel('google_studio')).toBe('Google Studio')
    expect(getIndexingProviderLabel('legacy')).toBe('Legacy')
    expect(getIndexingProviderLabel('unknown')).toBe('Unknown')
  })
})
```

- [ ] **Step 2: Add the frontend test harness before running tests**

Update `frontend/package.json` to merge the test scripts and dev dependencies with the existing tools (vite, typescript, tailwind, types):

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "preview": "vite preview",
    "test:unit": "vitest run"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/react": "^16.0.1",
    "@testing-library/user-event": "^14.5.2",
    "@types/node": "^20.11.16",
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.17",
    "jsdom": "^25.0.1",
    "postcss": "^8.4.35",
    "tailwindcss": "^3.4.1",
    "typescript": "^5.2.2",
    "vite": "^5.0.8",
    "vitest": "^2.1.1"
  }
}
```

Create `frontend/vitest.config.ts`:

```ts
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    globals: true,
  },
})
```

Create `frontend/src/test/setup.ts`:

```ts
import '@testing-library/jest-dom'
```

- [ ] **Step 3: Run the helper tests to confirm they fail for the right reason**

Run:

```bash
cd frontend && npm install && npm run test:unit -- src/lib/indexingProvider.test.ts
```

Expected: FAIL because `src/lib/indexingProvider.ts` does not exist yet.

- [ ] **Step 4: Implement the shared frontend provider helpers**

Create `frontend/src/lib/indexingProvider.ts` with this exact content:

```ts
export type IndexingProvider = 'deepseek' | 'google_studio'
export type IndexedProvider = IndexingProvider | 'legacy' | 'unknown'

export const INDEXING_PROVIDER_STORAGE_KEY = 'indexing-provider'
export const DEFAULT_INDEXING_PROVIDER: IndexingProvider = 'deepseek'

export function isIndexingProvider(value: string): value is IndexingProvider {
  return value === 'deepseek' || value === 'google_studio'
}

export function getStoredIndexingProvider(storage: Storage = window.localStorage): IndexingProvider {
  const stored = storage.getItem(INDEXING_PROVIDER_STORAGE_KEY)
  return stored && isIndexingProvider(stored) ? stored : DEFAULT_INDEXING_PROVIDER
}

export function setStoredIndexingProvider(provider: IndexingProvider, storage: Storage = window.localStorage) {
  storage.setItem(INDEXING_PROVIDER_STORAGE_KEY, provider)
}

export function getIndexingProviderLabel(provider: IndexedProvider): string {
  switch (provider) {
    case 'deepseek':
      return 'DeepSeek'
    case 'google_studio':
      return 'Google Studio'
    case 'legacy':
      return 'Legacy'
    default:
      return 'Unknown'
  }
}
```

- [ ] **Step 5: Re-run the helper tests**

Run:

```bash
cd frontend && npm run test:unit -- src/lib/indexingProvider.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit the frontend test harness and helpers**

Run:

```bash
git add frontend/package.json frontend/vitest.config.ts frontend/src/test/setup.ts frontend/src/lib/indexingProvider.ts frontend/src/lib/indexingProvider.test.ts
git commit -m "test: add frontend provider helper coverage"
```

### Task 6: Make `FileUpload` send provider and show provider-aware status copy

**Files:**
- Create: `frontend/src/components/FileUpload.test.tsx`
- Modify: `frontend/src/components/FileUpload.tsx`

- [ ] **Step 1: Write the failing `FileUpload` tests**

Create `frontend/src/components/FileUpload.test.tsx` with this exact content:

```tsx
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import FileUpload from './FileUpload'
import client from '../api/client'

vi.mock('../api/client', () => ({
  default: {
    post: vi.fn().mockResolvedValue({ data: { message: 'ok' } }),
  },
}))

describe('FileUpload', () => {
  it('sends the selected provider in FormData', async () => {
    render(<FileUpload selectedProvider="google_studio" />)

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(['hello'], 'law.txt', { type: 'text/plain' })

    fireEvent.change(input, { target: { files: [file] } })
    fireEvent.click(screen.getByRole('button', { name: /begin indexing/i }))

    await waitFor(() => expect(client.post).toHaveBeenCalledTimes(1))

    const [, formData] = vi.mocked(client.post).mock.calls[0]
    expect((formData as FormData).get('provider')).toBe('google_studio')
  })

  it('shows provider-aware helper and loading copy', async () => {
    let resolveRequest: ((value: { data: { message: string } }) => void) | undefined
    vi.mocked(client.post).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveRequest = resolve
        }) as Promise<{ data: { message: string } }>
    )

    render(<FileUpload selectedProvider="deepseek" />)

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(['hello'], 'law.txt', { type: 'text/plain' })

    fireEvent.change(input, { target: { files: [file] } })

    expect(screen.getByText('Will build with: DeepSeek')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /begin indexing/i }))

    expect(screen.getByText('Building knowledge graph with DeepSeek...')).toBeInTheDocument()

    resolveRequest?.({ data: { message: 'ok' } })
    await waitFor(() => expect(screen.getByText('ok')).toBeInTheDocument())
  })
})
```

- [ ] **Step 2: Run the `FileUpload` tests to confirm they fail**

Run:

```bash
cd frontend && npm run test:unit -- src/components/FileUpload.test.tsx
```

Expected: FAIL because `FileUpload` does not accept `selectedProvider`, does not append `provider` to `FormData`, and does not show provider-aware status copy.

- [ ] **Step 3: Implement provider-aware upload behavior in `FileUpload.tsx`**

Update `frontend/src/components/FileUpload.tsx` to follow this structure:

```tsx
import { useState } from 'react'
import { Upload, File, X, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react'
import client from '../api/client'
import { IndexingProvider, getIndexingProviderLabel } from '../lib/indexingProvider'

interface FileUploadProps {
  onSuccess?: () => void
  onUploadStateChange?: (isUploading: boolean) => void
  selectedProvider: IndexingProvider
}

export default function FileUpload({ onSuccess, onUploadStateChange, selectedProvider }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null)
  const [status, setStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle')
  const [message, setMessage] = useState('')

  const providerLabel = getIndexingProviderLabel(selectedProvider)
```

```tsx
  const handleUpload = async () => {
    if (!file) return

    setStatus('uploading')
    onUploadStateChange?.(true)
    const formData = new FormData()
    formData.append('file', file)
    formData.append('provider', selectedProvider)

    try {
      const response = await client.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      setStatus('success')
      setMessage(response.data.message)
      setFile(null)
      onSuccess?.()
    } catch (error: any) {
      setStatus('error')
      setMessage(error.response?.data?.detail || 'Upload failed')
    } finally {
      onUploadStateChange?.(false)
    }
  }
```

Add the provider helper text near the button:

```tsx
      {file && status === 'idle' && (
        <>
          <p className="text-[10px] text-muted-foreground px-1">
            {`Will build with: ${providerLabel}`}
          </p>
          <button
            onClick={handleUpload}
            className="w-full py-2 bg-primary text-primary-foreground rounded-lg text-xs font-bold hover:opacity-90 transition-all shadow-lg shadow-primary/20"
          >
            Begin Indexing
          </button>
        </>
      )}
```

Replace the loading copy with:

```tsx
      {status === 'uploading' && (
        <div className="flex items-center justify-center gap-2 text-xs font-medium text-muted-foreground py-2">
          <Loader2 className="w-3 h-3 animate-spin" />
          <span>{`Building knowledge graph with ${providerLabel}...`}</span>
        </div>
      )}
```

- [ ] **Step 4: Re-run the `FileUpload` tests**

Run:

```bash
cd frontend && npm run test:unit -- src/components/FileUpload.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit the provider-aware upload component**

Run:

```bash
git add frontend/src/components/FileUpload.tsx frontend/src/components/FileUpload.test.tsx
git commit -m "feat: send indexing provider from upload form"
```

### Task 7: Add the sidebar selector and document-provider badges in `App.tsx`

**Files:**
- Create: `frontend/src/App.test.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Write the failing `App` tests**

Create `frontend/src/App.test.tsx` with this exact content:

```tsx
import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import App from './App'
import client from './api/client'

vi.mock('./api/client', () => ({
  default: {
    get: vi.fn().mockResolvedValue({
      data: [
        { id: '1', source: 'a.pdf', status: 'processed', indexed_provider: 'google_studio' },
        { id: '2', source: 'b.pdf', status: 'processed', indexed_provider: 'legacy' },
      ],
    }),
  },
}))

vi.mock('./components/ChatInterface', () => ({
  default: () => <div>Chat Interface</div>,
}))

describe('App', () => {
  it('restores the selected provider from localStorage', async () => {
    window.localStorage.setItem('indexing-provider', 'google_studio')

    render(<App />)

    expect(screen.getByRole('button', { name: 'Google Studio' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('renders provider badges from the documents API', async () => {
    render(<App />)

    await waitFor(() => expect(client.get).toHaveBeenCalledWith('/documents'))

    expect(screen.getByText('Google Studio')).toBeInTheDocument()
    expect(screen.getByText('Legacy')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run the `App` tests to confirm they fail**

Run:

```bash
cd frontend && npm run test:unit -- src/App.test.tsx
```

Expected: FAIL because `App.tsx` does not expose a provider selector, does not restore from `localStorage`, and does not render provider labels from `/documents`.

- [ ] **Step 3: Implement the sidebar provider selector in `App.tsx`**

Update `frontend/src/App.tsx` to add shared provider state and upload locking:

```tsx
import { useState, useEffect } from 'react'
import ChatInterface from './components/ChatInterface'
import FileUpload from './components/FileUpload'
import { Scale, Database, Shield, Share2, FileText, ExternalLink, Columns } from 'lucide-react'
import client from './api/client'
import {
  DEFAULT_INDEXING_PROVIDER,
  IndexedProvider,
  IndexingProvider,
  getIndexingProviderLabel,
  getStoredIndexingProvider,
  setStoredIndexingProvider,
} from './lib/indexingProvider'

interface IndexedDocument {
  id: string
  status: string
  source: string
  content_summary?: string
  indexed_provider?: IndexedProvider
}
```

```tsx
  const [documents, setDocuments] = useState<IndexedDocument[]>([])
  const [comparisonMode, setComparisonMode] = useState(false)
  const [selectedIndexingProvider, setSelectedIndexingProvider] = useState<IndexingProvider>(() => {
    if (typeof window === 'undefined') return DEFAULT_INDEXING_PROVIDER
    return getStoredIndexingProvider()
  })
  const [isUploading, setIsUploading] = useState(false)
```

Persist provider changes:

```tsx
  useEffect(() => {
    setStoredIndexingProvider(selectedIndexingProvider)
  }, [selectedIndexingProvider])
```

Insert this new selector section under `RAG Settings`:

```tsx
            <div className="mt-4 space-y-2">
              <div className="flex items-center justify-between px-1">
                <span className="text-sm font-medium text-foreground">Indexing Provider</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {(['deepseek', 'google_studio'] as IndexingProvider[]).map((provider) => {
                  const active = selectedIndexingProvider === provider
                  return (
                    <button
                      key={provider}
                      type="button"
                      aria-pressed={active}
                      disabled={isUploading}
                      onClick={() => setSelectedIndexingProvider(provider)}
                      className={`rounded-xl border px-3 py-2 text-xs font-semibold transition-all ${
                        active
                          ? 'bg-primary/10 border-primary text-primary'
                          : 'bg-muted/50 border-border text-muted-foreground hover:bg-muted'
                      } ${isUploading ? 'cursor-not-allowed opacity-60' : ''}`}
                    >
                      {getIndexingProviderLabel(provider)}
                    </button>
                  )
                })}
              </div>
              <p className="text-[10px] text-muted-foreground px-1">
                Applies to new knowledge graph builds only. Existing documents are not rebuilt automatically.
              </p>
            </div>
```

Pass the provider and upload-state callback to `FileUpload`:

```tsx
            <FileUpload
              onSuccess={fetchDocuments}
              onUploadStateChange={setIsUploading}
              selectedProvider={selectedIndexingProvider}
            />
```

Render provider badges in the document list:

```tsx
                      <p className="text-[10px] text-muted-foreground flex items-center gap-2">
                        <span className="capitalize">{doc.status}</span>
                        <span>&bull;</span>
                        <span>{getIndexingProviderLabel(doc.indexed_provider || 'legacy')}</span>
                      </p>
```

Replace the misleading footer line with neutral copy:

```tsx
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Shield className="w-4 h-4" />
            <span>DeepSeek or Google Studio indexing</span>
          </div>
```

- [ ] **Step 4: Re-run the `App` tests**

Run:

```bash
cd frontend && npm run test:unit -- src/App.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit the sidebar selector UI**

Run:

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx
git commit -m "feat: add persistent indexing provider selector"
```

### Task 8: Update runtime docs and environment examples

**Files:**
- Modify: `.env.example`
- Modify: `docker-compose.yml`
- Modify: `README.md`
- Modify: `backend/tests/test_multi_provider_docs.py`

- [ ] **Step 1: Write the failing docs regression tests**

Append these tests to `backend/tests/test_multi_provider_docs.py`:

```python
def test_env_example_describes_google_studio_selector_settings():
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "GOOGLE_STUDIO_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/" in env_example
    assert "GOOGLE_STUDIO_API_KEY=your_google_studio_api_key_here" in env_example
    assert "GOOGLE_STUDIO_MODEL=gemini-2.5-flash" in env_example


def test_compose_passes_google_studio_environment_to_backend():
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "GOOGLE_STUDIO_BASE_URL=${GOOGLE_STUDIO_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}" in compose
    assert "GOOGLE_STUDIO_API_KEY=${GOOGLE_STUDIO_API_KEY}" in compose
    assert "GOOGLE_STUDIO_MODEL=${GOOGLE_STUDIO_MODEL:-gemini-2.5-flash}" in compose


def test_readme_mentions_shared_graph_and_sidebar_selector():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Google Studio" in readme
    assert "left sidebar" in readme
    assert "shared knowledge base" in readme
    assert "Existing documents are not rebuilt automatically" in readme
```

- [ ] **Step 2: Run the docs regression tests to confirm they fail**

Run:

```bash
pytest backend/tests/test_multi_provider_docs.py -q
```

Expected: FAIL because the current docs and runtime config do not mention Google Studio selector support.

- [ ] **Step 3: Update `.env.example`, `docker-compose.yml`, and `README.md`**

Add these lines to `.env.example` after the DeepSeek indexing block:

```dotenv
# Google Studio indexing provider
GOOGLE_STUDIO_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GOOGLE_STUDIO_API_KEY=your_google_studio_api_key_here
GOOGLE_STUDIO_MODEL=gemini-2.5-flash
```

Add these lines to the backend service environment in `docker-compose.yml`:

```yaml
      - GOOGLE_STUDIO_BASE_URL=${GOOGLE_STUDIO_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}
      - GOOGLE_STUDIO_API_KEY=${GOOGLE_STUDIO_API_KEY}
      - GOOGLE_STUDIO_MODEL=${GOOGLE_STUDIO_MODEL:-gemini-2.5-flash}
```

Add this paragraph to the feature/architecture section of `README.md`:

```md
- **Selectable Indexing Providers**: Users can choose `DeepSeek` or `Google Studio` from the left sidebar for new knowledge-graph builds. The project keeps one shared knowledge base, and switching providers affects only future uploads. Existing documents are not rebuilt automatically.
```

Add this note to the setup section:

```md
Google Studio indexing uses the Gemini OpenAI-compatible endpoint. Set `GOOGLE_STUDIO_API_KEY` in `.env` if you want the sidebar selector to support Google Studio builds in addition to DeepSeek.
```

- [ ] **Step 4: Re-run the docs regression tests**

Run:

```bash
pytest backend/tests/test_multi_provider_docs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the docs/runtime updates**

Run:

```bash
git add .env.example docker-compose.yml README.md backend/tests/test_multi_provider_docs.py
git commit -m "docs: document indexing provider selector"
```

### Task 9: Run focused verification for the full feature

**Files:**
- Verify only; no new files

- [ ] **Step 1: Run the focused backend test suite**

Run:

```bash
pytest \
  backend/tests/test_multi_provider_config.py \
  backend/tests/test_multi_provider_llm_services.py \
  backend/tests/test_indexing_provider_store.py \
  backend/tests/test_rag_engine_manager.py \
  backend/tests/test_upload_route.py \
  backend/tests/test_chat_route_provider_split.py \
  backend/tests/test_multi_provider_docs.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run the focused frontend unit tests**

Run:

```bash
cd frontend && npm run test:unit -- src/lib/indexingProvider.test.ts src/components/FileUpload.test.tsx src/App.test.tsx
```

Expected: PASS.

- [ ] **Step 3: Run the frontend build**

Run:

```bash
cd frontend && npm run build
```

Expected: PASS.

- [ ] **Step 4: Run frontend lint**

Run:

```bash
cd frontend && npm run lint
```

Expected: PASS.

- [ ] **Step 5: Smoke-check the selector contract manually**

Run the app, then verify these exact behaviors:

```text
1. Reload the page after choosing Google Studio and confirm the sidebar still shows Google Studio.
2. Upload a new TXT file and confirm the network request includes provider=google_studio.
3. Switch back to DeepSeek and upload another file.
4. Confirm /api/documents shows different indexed_provider values for the two files.
5. Confirm chat responses still behave exactly the same as before.
```

- [ ] **Step 6: Commit the final verified feature**

Run:

```bash
git status --short
git add backend frontend .env.example docker-compose.yml README.md
git commit -m "feat: add selectable indexing provider for new graph builds"
```

## Self-Review

- Spec coverage: this plan covers backend config, provider-aware indexing engines, upload contract changes, provenance storage, frontend persistence, upload UX, document badges, and docs/runtime updates. No spec requirement is left unassigned.
- Placeholder scan: no `TBD`, `TODO`, or “implement later” markers remain.
- Type consistency: the provider values are consistently `deepseek`, `google_studio`, `legacy`, and `unknown`; the upload route contract and frontend helpers use the same names.

## External References

- Google Gemini OpenAI compatibility docs: https://ai.google.dev/gemini-api/docs/openai
- Google Gemini SDK/library guidance: https://ai.google.dev/gemini-api/docs/libraries

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-08-indexing-provider-selector.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
