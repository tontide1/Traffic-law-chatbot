# Design Spec: Global Indexing Provider Selector

Date: 2026-06-08
Status: Draft approved in conversation, written for review

## Summary

Add a global indexing-provider selector to the left sidebar so users can choose which provider builds the knowledge graph for future document uploads.

The first version supports two providers:

- `DeepSeek`
- `Google Studio`

The selected provider:

- is stored in the browser and restored after reload
- applies only to new knowledge-graph builds
- does not automatically rebuild existing documents

All documents continue to live in one shared LightRAG knowledge base. Documents indexed at different times may therefore carry different indexing provenance, so each indexed document should expose metadata showing which provider was used.

## Context

The current application already separates query-time and indexing-time concerns on the backend:

- `backend/api/routes.py` sends `/api/chat` through the query engine
- `backend/api/routes.py` sends `/api/upload` through the indexing engine
- `backend/core/rag_engine.py` manages separate LightRAG instances for indexing and querying
- `frontend/src/App.tsx` already owns application-level UI settings in the left sidebar
- `frontend/src/components/FileUpload.tsx` handles upload initiation and user-facing indexing status

At the moment, the indexing path is still effectively fixed to DeepSeek via the indexing LLM settings in `backend/config.py` and the indexing client path in `backend/core/llm_services.py`.

This makes it impossible for the user to decide which provider should process the next upload without changing deployment configuration.

## Goals

- Let the user choose the indexing provider from the left sidebar.
- Persist the choice in the browser and restore it on reload.
- Apply the selected provider to future uploads only.
- Keep all indexed documents in one shared knowledge base.
- Expose document-level metadata so the UI can show which provider indexed each document.
- Preserve the existing chat behavior, SSE contract, and comparison mode semantics.
- Keep the first implementation small and predictable.

## Non-Goals

- Automatically re-indexing existing documents when the provider changes.
- Splitting the knowledge base into one graph per provider.
- Making the provider selector affect `/api/chat`.
- Adding bulk rebuild jobs, background workers, retry orchestration, or progress dashboards.
- Redesigning the existing upload flow beyond the provider-selection UX.

## Product Decision

Three approaches were considered:

1. Shared graph, provider applies only to new uploads
2. Separate graph per provider
3. Shared graph with immediate automatic rebuild of all existing documents on provider change

Option 1 is the recommended and approved direction.

Why it is recommended:

- It keeps the implementation aligned with the existing shared-storage LightRAG architecture.
- It avoids turning a simple UI choice into a large and expensive background re-index operation.
- It keeps token costs proportional to new uploads instead of reprocessing the entire corpus.
- It preserves the current query path and avoids multiplying chat/retrieval complexity.

Trade-off:

- The shared graph may contain documents indexed by different providers, so extraction style may not be fully uniform across the corpus.

That trade-off is acceptable for the first version as long as provenance is visible in the UI.

## Shared-Graph Semantics

The system will keep one shared LightRAG storage and graph.

Example:

- Upload document `A` while `DeepSeek` is selected
- Switch the selector to `Google Studio`
- Upload document `B`

Result:

- `A` is indexed into the shared graph using DeepSeek
- `B` is indexed into the same shared graph using Google Studio
- `B` does not replace or rebuild `A`
- `/api/chat` continues querying the shared knowledge base as a whole

This means the graph becomes a mixed-provenance knowledge base rather than multiple isolated graphs.

## Proposed Architecture

The implementation should keep one query engine and introduce provider-aware indexing engine selection.

Recommended backend structure:

- one query engine, unchanged in behavior
- one indexing engine configured for `deepseek`
- one indexing engine configured for `google_studio`

All engines should:

- point at the same storage backends
- point at the same `LIGHTRAG_WORKING_DIR`
- share the same embedding configuration

Only the indexing LLM provider wiring should differ between the two indexing engines.

This keeps provider choice explicit at upload time without coupling it to query behavior.

## Backend Contract

### `POST /api/upload`

Add a provider field to the upload request:

- `provider=deepseek`
- `provider=google_studio`

Compatibility behavior:

- if the frontend does not send `provider`, backend defaults to `deepseek`

Route behavior:

- validate the incoming provider value
- select the matching indexing engine
- parse the uploaded PDF or TXT as before
- run `ainsert(...)` through the selected indexing engine
- persist provider metadata for the uploaded document

### `GET /api/documents`

Extend each returned document item with:

- `indexed_provider`

Expected values:

- `deepseek`
- `google_studio`
- `legacy` or `unknown` for pre-existing documents without metadata

The route should remain otherwise backward-compatible with the existing list shape.

## Provider Metadata Storage

The current LightRAG document status path should not be forced to carry provider provenance unless the underlying object already supports it cleanly.

For the first version, use a small auxiliary metadata store in the LightRAG working directory keyed by document identity.

Requirements:

- simple to read and write from the upload and documents routes
- tolerant of older documents that have no provider entry
- isolated from chat/query logic
- easy to replace later if a more formal metadata store is introduced

The implementation should prefer a minimal local mapping over a broad storage redesign.

## Frontend UX

### Sidebar Placement

Place the provider selector in the left sidebar inside the existing `RAG Settings` section in `frontend/src/App.tsx`.

Rationale:

- the setting is global to the current browser session
- the sidebar already contains application-level toggles
- the selector should be visually separate from chat content to avoid implying it changes answer generation

### Control Type

Use a two-option segmented control:

- `DeepSeek`
- `Google Studio`

Do not use a dropdown for this version because there are only two options and the current state should be visible immediately.

### Helper Copy

Recommended label:

- `Indexing Provider`

Recommended helper text:

- `Applies to new knowledge graph builds only. Existing documents are not rebuilt automatically.`

### Persistence

Store the selected provider in `localStorage` and restore it when the page loads.

Recommended default:

- `deepseek`

### Upload-Time Behavior

When a file is selected but not yet uploaded, the upload area should show which provider will be used.

Examples:

- `Will build with: DeepSeek`
- `Will build with: Google Studio`

While an upload is actively indexing:

- disable the provider selector
- keep the currently displayed provider visible

This avoids ambiguous cases where the user changes providers in the middle of an active build.

### Upload Status Copy

The current loading text in `frontend/src/components/FileUpload.tsx` is too generic.

Recommended replacement:

- `Building knowledge graph with DeepSeek...`
- `Building knowledge graph with Google Studio...`

Success messages should also mention the provider that was used when practical.

## Indexed Documents UI

Each document card in the `Indexed Documents` list should display provider provenance.

Recommended presentation:

- show the existing status
- add a small provider badge or inline label next to it

Examples:

- `processed • DeepSeek`
- `processed • Google Studio`
- `processed • Legacy`

If a document has no provider metadata, the UI should not pretend it was indexed with the current selector. It should show a legacy or unknown state explicitly.

## Module Changes

### `frontend/src/App.tsx`

Responsibilities:

- own the global `selectedIndexingProvider` state
- restore it from `localStorage`
- persist changes back to `localStorage`
- pass the provider into `FileUpload`
- use the provider metadata from `/api/documents` to render the document list

### `frontend/src/components/FileUpload.tsx`

Responsibilities:

- receive the selected provider as a prop
- include it in the upload request
- show provider-aware status copy
- respect a disabled state while indexing is in progress

### `backend/api/routes.py`

Responsibilities:

- accept the provider input for uploads
- resolve the correct indexing engine
- write document provider metadata after successful insertion
- return provider metadata from `/documents`

### `backend/core/rag_engine.py`

Responsibilities:

- expose provider-aware indexing engine access
- keep one shared query engine
- ensure all engines still share the same storage configuration

### `backend/core/llm_services.py`

Responsibilities:

- expose the provider-specific indexing LLM call paths required by the indexing engines
- keep the existing answer path behavior unchanged

### `backend/config.py`

Responsibilities:

- expose the settings needed for both indexing providers
- keep the selected provider itself as a request/UI concern, not a process-wide mutable global

## Error Handling

The first version should fail explicitly for invalid provider input.

Recommended behavior:

- invalid provider -> `400 Bad Request`
- missing provider -> default to `deepseek`
- provider metadata missing for existing document -> return `legacy` or `unknown`, not an error

The system should not silently remap `google_studio` to another provider.

## Testing

Targeted tests should cover:

1. frontend persistence of provider selection
2. upload request includes the selected provider
3. backend upload route selects the correct indexing engine per provider
4. backend rejects invalid provider values
5. `/api/documents` returns `indexed_provider`
6. legacy documents without provider metadata still render safely

The verification strategy should stay focused. Do not require a full end-to-end re-index workflow test for the first pass.

## Risks

### Risk: Mixed-provider graph quality differs across documents

This is the main architectural trade-off of the approved approach.

Mitigation:

- make provider provenance visible
- avoid automatic rebuild behavior
- add manual rebuild tooling later only if corpus consistency becomes a real problem

### Risk: Provider selector appears to affect chat answers

Mitigation:

- keep the selector in the sidebar settings area
- use helper copy that says it applies to new knowledge-graph builds only
- do not change chat-side provider labels in this feature

### Risk: Existing footer/provider copy becomes misleading

The current footer and other static labels that imply a fixed provider setup may become inaccurate once Google Studio is selectable for indexing.

Mitigation:

- update those labels to describe the system more neutrally
- avoid hardcoded copy that implies all indexing always uses DeepSeek

## Future Extensions

Out of scope for this implementation, but compatible with this design:

- a manual `Rebuild all with current provider` action
- richer document metadata such as indexing timestamp and provider display name
- admin-only provider diagnostics
- corpus-level analytics showing provider distribution

## Acceptance Criteria

The feature is complete when all of the following are true:

- the user can choose `DeepSeek` or `Google Studio` from the left sidebar
- the selection survives page reload in the same browser
- new uploads send the selected provider to the backend
- the backend indexes new uploads with the selected provider
- existing indexed documents are not rebuilt when the selector changes
- the documents list shows which provider indexed each document, or a legacy state when unknown
- chat behavior and comparison mode remain unchanged
