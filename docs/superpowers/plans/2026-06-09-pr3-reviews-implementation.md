# PR #3 Review Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement code quality and stability improvements recommended by Copilot and Sourcery reviews on PR #3, including lazy initialization of Google Studio, process-isolated temp files for atomic writes, best-effort metadata storage, and extra unit test coverage.

**Architecture:**
- Make `RAGEngine.get_indexing_instance` asynchronous to enable lazy instantiation and database storage setup for Google Studio.
- Update atomic writes in `IndexingProviderStore` to append the process ID (`os.getpid()`) to prevent temporary file collisions across multiple workers.
- Wrap metadata writes in routes with try-except to avoid failing document upload if only metadata persistence fails.
- Add focused unit tests for TXT uploads under Google Studio.

**Tech Stack:** Python 3.11, FastAPI, LightRAG, pytest

---

## File Map

- Modify: `backend/core/rag_engine.py`
- Modify: `backend/core/indexing_provider_store.py`
- Modify: `backend/api/routes.py`
- Modify: `backend/tests/test_rag_engine_manager.py`
- Modify: `backend/tests/test_upload_route.py`

---

### Task 1: Lazy initialization of Google Studio in RAGEngine

**Files:**
- Modify: `backend/core/rag_engine.py`
- Modify: `backend/tests/test_rag_engine_manager.py`

- [ ] **Step 1: Update `RAGEngine` to lazily initialize Google Studio**

Update `backend/core/rag_engine.py` to add `_google_init_lock` as class-level attribute, make `get_indexing_instance` async, and perform initialization of Google Studio inside it:

```python
class RAGEngine:
    _deepseek_indexing_instance = None
    _google_studio_indexing_instance = None
    _query_instance = None
    _google_init_lock = asyncio.Lock()  # Dynamic lock registry will bind on the correct loop at runtime
```

Update `initialize` to only instantiate `_deepseek_indexing_instance` and `_query_instance`:

```python
    @classmethod
    async def initialize(cls):
        if (
            cls._deepseek_indexing_instance is not None
            and cls._query_instance is not None
        ):
            return cls._deepseek_indexing_instance, cls._query_instance

        cls._apply_postgres_environment()

        cls._deepseek_indexing_instance = LightRAG(
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
        await cls._deepseek_indexing_instance.initialize_storages()

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

        return cls._deepseek_indexing_instance, cls._query_instance
```

Update `get_indexing_instance` to be async, using lock:

```python
    @classmethod
    async def get_indexing_instance(cls, provider: str = "deepseek"):
        if provider == "deepseek":
            if cls._deepseek_indexing_instance is None:
                raise RuntimeError("DeepSeek indexing RAG engine not initialized. Call RAGEngine.initialize() first.")
            return cls._deepseek_indexing_instance
        if provider == "google_studio":
            if cls._google_studio_indexing_instance is None:
                async with cls._google_init_lock:
                    if cls._google_studio_indexing_instance is None:
                        cls._apply_postgres_environment()
                        cls._google_studio_indexing_instance = LightRAG(
                            working_dir=settings.LIGHTRAG_WORKING_DIR,
                            llm_model_func=google_studio_indexing_llm_func,
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
                        await cls._google_studio_indexing_instance.initialize_storages()
            return cls._google_studio_indexing_instance
        raise ValueError(f"Unsupported indexing provider: {provider}")
```

- [ ] **Step 2: Update `backend/tests/test_rag_engine_manager.py` for async getters**

Update the unit tests since `get_indexing_instance` is now async:

```python
def test_getters_raise_before_initialize():
    with pytest.raises(RuntimeError, match="DeepSeek indexing RAG engine not initialized"):
        asyncio.run(rag_engine.RAGEngine.get_indexing_instance("deepseek"))

    with pytest.raises(RuntimeError, match="Query RAG engine not initialized"):
        rag_engine.RAGEngine.get_query_instance()
```

```python
def test_get_indexing_instance_selects_provider_after_initialize(monkeypatch):
    rag_engine.RAGEngine._deepseek_indexing_instance = "deepseek-engine"
    rag_engine.RAGEngine._google_studio_indexing_instance = "google-engine"

    assert asyncio.run(rag_engine.RAGEngine.get_indexing_instance("deepseek")) == "deepseek-engine"
    assert asyncio.run(rag_engine.RAGEngine.get_indexing_instance("google_studio")) == "google-engine"
```

And update `test_initialize_builds_two_indexing_engines_and_one_query_engine` to reflect that `initialize` only instantiates two instances: DeepSeek indexing and query engine (assert `len(created) == 2`).

- [ ] **Step 3: Verify and Commit**

Run: `PYTHONPATH=. pytest backend/tests/test_rag_engine_manager.py`
Commit: `git add backend/core/rag_engine.py backend/tests/test_rag_engine_manager.py && git commit -m "refactor: lazily initialize google studio indexing engine"`

---

### Task 2: Process-isolated temp file in IndexingProviderStore

**Files:**
- Modify: `backend/core/indexing_provider_store.py`

- [ ] **Step 1: Update `IndexingProviderStore` write atomic method**

In `backend/core/indexing_provider_store.py`, append `os.getpid()` to the temporary filename prefix to prevent temp file collisions:

```python
            def _write_atomic():
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = self._path.with_suffix(f".tmp.{os.getpid()}")
                tmp_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                os.replace(tmp_path, self._path)
```

- [ ] **Step 2: Verify and Commit**

Run: `PYTHONPATH=. pytest backend/tests/test_indexing_provider_store.py`
Commit: `git add backend/core/indexing_provider_store.py && git commit -m "fix: use process-isolated temp file for atomic writes"`

---

### Task 3: Best-effort metadata storage and async routes update

**Files:**
- Modify: `backend/api/routes.py`

- [ ] **Step 1: Await `get_indexing_instance` and try-except metadata writes**

In `backend/api/routes.py`, update `upload_file` to:
1. `await RAGEngine.get_indexing_instance(resolved_provider)` since it's now async.
2. Wrap `set_provider` in a try-except block to make metadata storage best-effort, logging any warning/error.

```python
    try:
        rag = await RAGEngine.get_indexing_instance(resolved_provider)
        provider_store = IndexingProviderStore(settings.LIGHTRAG_WORKING_DIR)

        if file.filename.endswith(".pdf"):
            content = await parse_pdf_to_markdown(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        if not content.strip():
            raise ValueError("File is empty or no text could be extracted")

        await rag.ainsert(content, file_paths=[file.filename])

        try:
            await provider_store.set_provider(file.filename, resolved_provider)
        except Exception as e:
            print(f"WARNING: Failed to save document provider metadata for {file.filename}: {e}")

        return UploadResponse(
            filename=file.filename,
            status="success",
            message=f"File uploaded and indexed with {resolved_provider}",
        )
```

- [ ] **Step 2: Commit**

Commit: `git add backend/api/routes.py && git commit -m "feat: make provider metadata storage best-effort"`

---

### Task 4: Update test mocks and add TXT routing tests

**Files:**
- Modify: `backend/tests/test_upload_route.py`

- [ ] **Step 1: Update upload route tests and mocks**

In `backend/tests/test_upload_route.py`, since `get_indexing_instance` is async, we must mock it using an async function or a mock returning a coroutine:

```python
    monkeypatch.setattr(routes.RAGEngine, "get_indexing_instance", AsyncMock(return_value=fake_rag))
```

Update all mocks of `get_indexing_instance` in `test_upload_route.py` to be async:
1. In `test_pdf_upload_uses_indexing_rag`:
   ```python
   monkeypatch.setattr(routes.RAGEngine, "get_indexing_instance", AsyncMock(return_value=fake_rag))
   ```
2. In `test_pdf_upload_returns_error_when_docling_parse_fails`:
   ```python
   monkeypatch.setattr(routes.RAGEngine, "get_indexing_instance", AsyncMock(return_value=fake_rag))
   ```
3. In `test_txt_upload_uses_indexing_rag_without_pdf_parser`:
   ```python
   monkeypatch.setattr(routes.RAGEngine, "get_indexing_instance", AsyncMock(return_value=fake_rag))
   ```
4. In `test_upload_defaults_provider_to_deepseek`:
   ```python
   monkeypatch.setattr(routes.RAGEngine, "get_indexing_instance", AsyncMock(return_value=fake_rag))
   ```
5. In `test_upload_routes_google_studio_provider`:
   ```python
   monkeypatch.setattr(routes.RAGEngine, "get_indexing_instance", AsyncMock(return_value=fake_google_rag))
   ```

- [ ] **Step 2: Add TXT upload routing test for Google Studio**

Add the following unit test `test_txt_upload_routes_google_studio_provider` checking that TXT uploads are routed correctly and metadata saved:

```python
def test_txt_upload_routes_google_studio_provider(monkeypatch, tmp_path):
    client = make_client()
    monkeypatch.setattr(routes.settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))

    fake_google_rag = Mock()
    fake_google_rag.ainsert = AsyncMock()
    fake_store = Mock()
    fake_store.set_provider = AsyncMock()
    selected = []

    async def fake_get_indexing_instance(provider="deepseek"):
        selected.append(provider)
        return fake_google_rag

    monkeypatch.setattr(routes.RAGEngine, "get_indexing_instance", fake_get_indexing_instance)
    monkeypatch.setattr(routes, "IndexingProviderStore", lambda working_dir: fake_store)

    response = client.post(
        "/api/upload",
        data={"provider": "google_studio"},
        files={"file": ("note.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 200
    assert selected == ["google_studio"]
    fake_google_rag.ainsert.assert_awaited_once_with("hello world", file_paths=["note.txt"])
    fake_store.set_provider.assert_awaited_once_with("note.txt", "google_studio")
```

Also, add a unit test verifying best-effort behavior if `set_provider` fails:

```python
def test_upload_success_even_if_metadata_store_fails(monkeypatch, tmp_path):
    client = make_client()
    monkeypatch.setattr(routes.settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))

    fake_rag = Mock()
    fake_rag.ainsert = AsyncMock()
    fake_store = Mock()
    fake_store.set_provider = AsyncMock(side_effect=IOError("disk full"))

    monkeypatch.setattr(routes.RAGEngine, "get_indexing_instance", AsyncMock(return_value=fake_rag))
    monkeypatch.setattr(routes, "IndexingProviderStore", lambda working_dir: fake_store)

    response = client.post(
        "/api/upload",
        files={"file": ("law.txt", b"content", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    fake_rag.ainsert.assert_awaited_once()
    fake_store.set_provider.assert_awaited_once()
```

- [ ] **Step 3: Verify and Commit**

Run: `PYTHONPATH=. pytest backend/tests/test_upload_route.py`
Commit: `git add backend/tests/test_upload_route.py && git commit -m "test: add txt upload provider routing and best-effort upload tests"`
