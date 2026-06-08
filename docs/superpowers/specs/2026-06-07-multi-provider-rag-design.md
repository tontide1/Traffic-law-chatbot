# Design Spec: Split Indexing, Answer, and Embedding Providers

Date: 2026-06-07
Status: Draft approved in conversation, written for review

## Summary

Split the current single-provider LightRAG backend into three explicit provider roles:

- `DeepSeek-V4-Flash` via the direct DeepSeek API for knowledge-graph extraction and document indexing
- `openai/gpt-oss-120b` via OpenRouter for final answer generation
- `Qwen/Qwen3-Embedding-0.6B` served locally through vLLM at an OpenAI-compatible `/v1/embeddings` endpoint for document and query embeddings

This change is broader than the earlier Docling parser change, but it is still intentionally bounded. It focuses on provider separation for indexing, answering, and embeddings. It does not redesign the frontend, the retrieval modes, or the document parser.

## Context

The current backend couples all LightRAG LLM behavior to a single model provider path:

- `backend/core/rag_engine.py` initializes one `LightRAG` instance with a single `llm_model_func`.
- `backend/core/llm_services.py` hardcodes a single OpenAI-compatible client pointed at OpenRouter.
- `backend/core/llm_services.py` uses that client both for chat completions and embeddings.
- `backend/config.py` exposes only one `LLM_MODEL` and one `EMBEDDING_MODEL`.
- `backend/api/routes.py` uses the same `LightRAG` instance for both `rag.ainsert(...)` and `rag.aquery(...)`.
- `docker-compose.yml` configures backend and `rag-ui` around OpenRouter-based LLM and embedding bindings.

This architecture is simple, but it prevents the project from using:

- a cheaper and extraction-oriented model for graph building
- a separate answer model with different quality and latency trade-offs
- a local embedding service independent of OpenRouter

The desired target architecture is:

`upload PDF/TXT -> Docling/TXT loader -> DeepSeek indexing LLM -> local Qwen3 embeddings -> Postgres/LightRAG storage`

`user question -> local Qwen3 query embedding -> LightRAG retrieval -> gpt-oss-120b answer synthesis`

## Goals

- Use `deepseek-v4-flash` direct API for knowledge-graph extraction and indexing.
- Use `openai/gpt-oss-120b` through OpenRouter for answer generation.
- Remove OpenRouter from the embedding path entirely.
- Serve embeddings locally with `Qwen/Qwen3-Embedding-0.6B` through vLLM's OpenAI-compatible embeddings endpoint.
- Keep the existing upload route, chat route, and SSE response contract intact.
- Keep comparison mode semantics as `naive vs hybrid`, not `provider A vs provider B`.
- Limit code changes to the provider/configuration layer, RAG engine wiring, and deployment/docs updates.

## Non-Goals

- Replacing the Docling parser introduced in the earlier spec.
- Redesigning frontend UI or chat rendering.
- Replacing PostgreSQL, pgvector, or Apache AGE storage.
- Containerizing vLLM in this first pass.
- Adding bulk re-index tooling for all historic documents in the same implementation.
- Changing `ChatResponse.sources` behavior in this spec.
- Introducing a new reranker in the same change.

## Why This Split Is Recommended

Three architectural directions were considered:

1. Keep one provider for everything
2. Split providers by role
3. Build a more general multi-role provider framework with many future roles

Option 2 is the recommended choice.

Keeping one provider for everything is simpler on paper but fails the project goal. It keeps extraction quality, answer quality, and embedding locality coupled together when their trade-offs are different.

Building a fully general provider framework is a reasonable future direction, but it is too large for the current migration. The codebase only needs three stable roles right now:

- indexing LLM
- answer LLM
- embedding model

This spec therefore defines explicit role boundaries without turning the change into a generalized inference platform.

## Proposed Architecture

The backend will move from one `LightRAG` instance to two coordinated instances with shared storage:

- `indexing_rag`: used only for document insertion and KG extraction
- `query_rag`: used only for query embedding, retrieval, and final answer generation

Both instances will:

- point to the same `LIGHTRAG_WORKING_DIR`
- point to the same Postgres-backed storage
- use the same embedding model and embedding dimension

They will differ only in their answer/extraction LLM wiring:

- `indexing_rag` uses `deepseek-v4-flash`
- `query_rag` uses `openai/gpt-oss-120b`

This avoids relying on undocumented runtime behavior inside the current core `LightRAG` object for role switching. It keeps the route semantics explicit:

- `/api/upload` uses `indexing_rag`
- `/api/chat` uses `query_rag`
- `/api/documents` can use either instance because doc status storage is shared, but should use one consistently for clarity

## Module Boundaries

### `backend/config.py`

Replace the current single-model configuration with role-specific settings.

Recommended settings:

- `INDEXING_LLM_BASE_URL=https://api.deepseek.com`
- `INDEXING_LLM_API_KEY=...`
- `INDEXING_LLM_MODEL=deepseek-v4-flash`
- `INDEXING_LLM_THINKING_MODE=off`

- `ANSWER_LLM_BASE_URL=https://openrouter.ai/api/v1`
- `ANSWER_LLM_API_KEY=...`
- `ANSWER_LLM_MODEL=openai/gpt-oss-120b`

- `EMBEDDING_BASE_URL=http://host.docker.internal:8002/v1`
- `EMBEDDING_API_KEY=EMPTY`
- `EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B`
- `EMBEDDING_DIM=1024`
- `EMBEDDING_QUERY_PREFIX=Instruct: Given a legal query, retrieve relevant statutes and legal passages.\nQuery: `
- `EMBEDDING_DOCUMENT_PREFIX=`

Compatibility behavior for the first rollout is recommended:

- if `ANSWER_LLM_API_KEY` is absent, allow fallback to `OPENROUTER_API_KEY`
- keep `LLM_MODEL` and the old embedding defaults only as deprecated compatibility paths during migration
- prefer explicit failure over silent provider fallback once the new settings are present

### `backend/core/llm_services.py`

This module should stop exposing one global OpenRouter client and instead expose role-aware client factories.

Recommended responsibilities:

- `get_indexing_llm_client()`
- `get_answer_llm_client()`
- `get_embedding_client()`
- `indexing_llm_func(...)`
- `answer_llm_func(...)`
- `QwenEmbeddingFunc` or a renamed equivalent dedicated to the local embedding service

Important behavior changes:

- indexing requests go directly to `https://api.deepseek.com`
- answer requests go to OpenRouter
- embedding requests go to local vLLM

The current heuristic that adds a query prefix when the text ends with `?` is too weak for the new embedding setup. The replacement should use explicit config-driven query and document prefixes so behavior is deterministic and not punctuation-dependent.

### `backend/core/rag_engine.py`

This module should evolve from a single singleton into a small engine manager that owns two initialized `LightRAG` instances.

Recommended public API:

- `initialize()`
- `get_indexing_instance()`
- `get_query_instance()`

Both `LightRAG` instances must use:

- identical storage backends
- identical embedding function configuration
- identical `EMBEDDING_DIM`

Only the `llm_model_func` should differ.

This module should also stop hardcoding `embedding_dim=1536` and read the value from settings.

### `backend/api/routes.py`

Route responsibilities after the split:

- `/upload` obtains `indexing_rag`
- `/chat` obtains `query_rag`
- `/documents` uses the shared doc status storage through one of the initialized instances
- `/health` remains unchanged, though future enhancement could expose provider reachability if desired

The SSE stream payload shape must remain unchanged:

- `start`
- `chunk`
- `error`
- `done`

This preserves frontend compatibility.

## Provider-Specific Behavior

### DeepSeek for KG Extraction and Indexing

Indexing should use `deepseek-v4-flash` in non-thinking mode.

Rationale:

- KG extraction does not benefit from verbose reasoning traces
- non-thinking mode reduces token cost and noisy output
- extraction is more reliable when the model is instructed to produce compact, structured content

Implementation guidance:

- do not rely on legacy model aliases like `deepseek-chat`
- use the direct DeepSeek API base URL
- pass any provider-specific non-thinking toggle through the client in a way that survives the existing kwargs filtering
- prefer JSON-friendly extraction behavior where supported by the LightRAG extraction prompts

### OpenRouter gpt-oss-120b for Answers

Answer generation should use `openai/gpt-oss-120b` on OpenRouter.

Rationale:

- it keeps answer integration OpenAI-compatible
- it avoids adding a second SDK family for answer generation
- it reduces migration complexity compared with introducing Google AI Studio in the same change

Recommended answer behavior:

- keep streamed responses enabled
- keep prompts concise and grounding-oriented
- avoid changing frontend-visible answer structure in this migration

### Qwen3-Embedding-0.6B via vLLM

Embeddings should be served locally via vLLM on an OpenAI-compatible endpoint.

Recommended host-side command shape:

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B --runner pooling --port 8002 --dtype float16 --gpu-memory-utilization 0.75
```

The backend then points its embeddings client to:

```text
http://host.docker.internal:8002/v1
```

If the backend runs outside Docker, the local URL can instead be:

```text
http://localhost:8002/v1
```

This spec intentionally keeps vLLM out of `docker-compose` in the first pass. Running it on the host reduces GPU-container complexity and keeps the migration smaller.

## Docker and Runtime Networking

Because the backend currently runs inside Docker, a host-served vLLM process is not reachable through `localhost` from the container.

The backend service should therefore be updated to support host access on Linux, for example by adding:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

Equivalent documentation updates are required for developers who run the backend outside Docker.

The `rag-ui` service also needs a compatibility update because it currently assumes OpenRouter embeddings and `EMBEDDING_DIM=1536`.

Minimum required compatibility changes for `rag-ui`:

- point embedding binding at the local embedding service
- set `EMBEDDING_DIM=1024`
- keep storage bindings aligned with the main backend

If `rag-ui` is used only for graph visualization, its query-time LLM binding can remain simple, but it must not be left on a mismatched embedding dimension.

## Data Migration and Re-Indexing

This change requires a full embedding re-index.

Reason:

- current vectors are built with `text-embedding-3-small` at `1536` dimensions
- new vectors will be built with `Qwen3-Embedding-0.6B` at `1024` dimensions

Consequences:

- existing vector data cannot be assumed compatible
- deployment must include a planned re-index of documents
- retrieval quality must be evaluated after re-index, not before

This spec does not require implementing a bulk re-index command in the same change, but it does require documenting that old embeddings are invalid after the dimension switch.

## Comparison Mode Semantics

Comparison mode in `/api/chat` currently means `naive` versus `hybrid`.

This meaning should stay unchanged.

The migration must not repurpose comparison mode into:

- answer model comparison
- provider comparison
- indexing model comparison

Both comparison branches should use the same `query_rag` answer provider and differ only by LightRAG retrieval mode.

## Error Handling

The provider split increases the number of failure modes, so errors must become more explicit.

Required handling:

- missing DeepSeek key: indexing initialization fails clearly
- missing OpenRouter key: answer initialization fails clearly
- embedding endpoint unavailable: startup or first embedding call fails clearly with the local endpoint in the error message
- mismatched embedding dimension: fail loudly during configuration or storage initialization, not later during retrieval
- host networking misconfiguration for vLLM: produce a direct message indicating the attempted base URL

Logging should distinguish:

- indexing LLM failure
- answer LLM failure
- embedding service failure
- storage initialization failure

## Documentation Changes

Update:

- `.env.example`
- `README.md`
- `docker-compose.yml` comments and environment values
- any setup notes that currently imply OpenRouter is responsible for embeddings

The new documentation should clearly describe:

- DeepSeek for KG extraction
- OpenRouter `gpt-oss-120b` for answer generation
- local vLLM for embeddings
- the requirement to run the vLLM embedding service before indexing or querying
- the need to re-index after changing embedding dimension

## Verification Plan

The smallest relevant verification for this scope is:

1. configuration parsing for all three provider roles
2. backend startup with both `LightRAG` instances initialized
3. local embedding smoke test against the vLLM `/v1/embeddings` endpoint
4. upload a representative PDF or TXT and confirm indexing succeeds through the DeepSeek path
5. query the indexed content and confirm answers stream through the OpenRouter path
6. confirm comparison mode still behaves as `naive vs hybrid`
7. confirm `GET /api/documents` still works
8. confirm `rag-ui` remains compatible with the new embedding dimension or document that it is temporarily unsupported until updated

Recommended focused tests:

- config unit tests for new env variables and fallbacks
- llm service tests for provider-specific client selection
- route tests asserting `/upload` and `/chat` call different engine accessors
- smoke tests for SSE answer streaming

## Risks and Trade-Offs

### Risk: Two LightRAG instances diverge in behavior

Impact:
- subtle indexing/query inconsistencies

Mitigation:
- share the same storage backends
- share the same embedding function and dimension
- limit differences strictly to role-specific LLM functions

### Risk: Local vLLM availability becomes a hard dependency

Impact:
- app startup or retrieval failure when the local embedding service is down

Mitigation:
- document startup order clearly
- fail fast with actionable error messages
- avoid silent fallback to OpenRouter embeddings

### Risk: Embedding migration breaks existing retrieval

Impact:
- stale vectors, retrieval failures, or poor results

Mitigation:
- make the dimension switch explicit
- require re-indexing
- validate on a representative legal dataset after migration

### Risk: `rag-ui` becomes incompatible with backend assumptions

Impact:
- graph UI stops working or behaves inconsistently

Mitigation:
- update `rag-ui` embedding configuration in the same rollout
- keep role-specific settings aligned where required

### Risk: Scope creep into a full inference platform rewrite

Impact:
- implementation expands beyond the current migration

Mitigation:
- keep roles fixed to three
- do not introduce generic plugin abstractions or dynamic routing in this change

## Out of Scope Follow-Ups

Potential future work, not part of this spec:

- bulk re-index tooling for historical documents
- adding a reranker
- containerizing vLLM in Docker Compose
- benchmarking alternative local embedding models like legal-domain Vietnamese-specific embeddings
- exposing provider diagnostics in `/health`
- revisiting `ChatResponse.sources` population

## Implementation Notes

Preferred code shape:

- update `backend/config.py`
- refactor `backend/core/llm_services.py`
- refactor `backend/core/rag_engine.py` into a dual-instance manager
- update `backend/api/routes.py` to use the correct engine per route
- update `.env.example`
- update `docker-compose.yml`
- update `README.md`
- add or update focused backend tests

The change should remain surgical. It should not rewrite route contracts, frontend state management, or the Docling parser.

## Spec Review Notes

Self-review completed for:

- placeholders: none remain
- consistency: provider roles, route usage, and migration steps align
- scope: focused on provider separation and local embeddings, not a broader architecture rewrite
- ambiguity: the dual-`LightRAG` approach is stated explicitly because the current codebase does not use the LightRAG API server role-override mode

## Environment Limitation

The design document has been written locally in the repository. It was not committed automatically because this step was not explicitly requested in the current task.
