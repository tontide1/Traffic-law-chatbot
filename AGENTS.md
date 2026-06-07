# Agent Instructions

## Core coding behavior

When working in this repository:

- Prefer simple, surgical changes over broad refactors.
- Preserve existing project style, naming, formatting, and architecture unless explicitly asked to change them.
- Do not introduce speculative abstractions, new frameworks, or unnecessary configuration.
- When Serena MCP is available, prefer it for codebase exploration, symbol lookup, and architecture discovery before falling back to raw file scans.
- For bugs, first identify the likely root cause, then make the smallest safe fix.
- For non-trivial changes, state the plan, edit the code, then run the smallest relevant verification command.
- Do not modify secrets, credentials, `.env` files, generated artifacts, model weights, raw datasets, or production data unless explicitly instructed.
- Treat tool outputs as advisory. Verify changes against the repository, tests, logs, and runtime behavior.

## Environment

Preferred local environment:

- Using python 3.11 with

```bash
conda activate legal_rag
```

## Codebase Snapshot

This repository is a Vietnamese legal RAG assistant with three main parts:

- `backend/`: FastAPI application that boots LightRAG during startup and exposes the API under `/api`.
  - `backend/main.py` wires app lifespan, CORS, and router registration.
  - `backend/api/routes.py` exposes `/chat`, `/documents`, `/upload`, and `/health`.
  - Chat supports normal hybrid retrieval and a comparison mode that runs naive vs hybrid responses in parallel, including SSE streaming.
  - Upload accepts `PDF` and `TXT`; PDFs are parsed locally through Docling in `backend/core/document_parser.py` before indexing.
  - `backend/core/rag_engine.py` initializes LightRAG with Postgres-backed storage, Vietnamese summarization, and legal entity types.
- `frontend/`: Vite + React + TypeScript + Tailwind UI.
  - `frontend/src/App.tsx` is the main shell with the chat pane, comparison toggle, upload widget, and indexed-document sidebar.
  - `frontend/src/components/` holds the chat, upload, and reference UI pieces.
  - `frontend/src/api/client.ts` points API calls at `http://localhost:8000/api` by default.
- `db/`: custom Postgres image and init script for the vector and graph database layer.
- `docker-compose.yml`: brings up `db`, `backend`, and `rag-ui`; the graph UI is exposed on port 8001.
- `data/` and `docs/`: working data, screenshots, and other repo assets. Treat generated or raw data as sensitive and avoid editing them unless required.

Useful behavior to remember when changing code:

- Frontend document state is refreshed from `/api/documents` on load and on a polling interval.
- Backend PDF ingestion now uses Docling in `backend/core/document_parser.py`.
- The backend initializes LightRAG against Postgres during application startup, so changes that affect storage or env vars should be checked against startup flow.
