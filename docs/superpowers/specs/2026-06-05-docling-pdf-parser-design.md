# Design Spec: Replace Qwen-VL PDF Parsing with Docling

Date: 2026-06-05
Status: Draft approved in conversation, written for review

## Summary

Replace the current PDF ingestion path that converts PDF pages to images and sends them to Qwen-VL via OpenRouter with a local Docling-based parser. The new parser will extract structured Markdown from text-based legal PDFs, run a light normalization step for Vietnamese legal document structure, then pass that Markdown into LightRAG unchanged as a single string payload.

This change is intentionally narrow. It removes Qwen-VL from the PDF parsing pipeline only. It does not remove OpenRouter from chat, answer generation, or embedding flows elsewhere in the project.

## Context

The current upload flow is:

`upload PDF -> qwen_vl_parse_pdf() -> raw text -> LightRAG`

Current behavior and constraints observed in the codebase:

- `backend/api/routes.py` uploads the file and calls `qwen_vl_parse_pdf(...)` for PDFs.
- `backend/core/llm_services.py` implements the parser by converting the first five pages of a PDF to images with `pdf2image`, then sending those images to `qwen/qwen3-vl-235b-a22b-instruct` via OpenRouter.
- The response is inserted into LightRAG with `rag.ainsert(content, file_paths=[...])`.
- The project goal for this change is local execution and removal of the OpenRouter dependency from PDF parsing.
- The expected document type is primarily text-based Vietnamese legal PDFs, not scan-heavy image PDFs.

## Goals

- Remove Qwen-VL entirely from the PDF parsing pipeline.
- Parse PDF files locally with Docling.
- Preserve legal document structure as much as practical for downstream LightRAG graph extraction.
- Keep integration changes small by preserving the existing `rag.ainsert(str)` contract.
- Update user-visible messages and docs so PDF parsing is described as Docling-based, not Qwen-VL-based.

## Non-Goals

- Removing OpenRouter from chat or answer-generation flows.
- Replacing the embedding provider.
- Building a custom structured ingestion pipeline that feeds JSON objects directly into LightRAG.
- Optimizing for scan-heavy PDFs in this change.

## Why Markdown Instead of Plain Text or JSON

Three output forms were considered:

1. Plain text
2. Structured Markdown
3. Full Docling JSON or document object transformation

Structured Markdown is the recommended choice because it keeps the integration surface simple while preserving headings, numbering, lists, and table structure that matter in Vietnamese legal documents. For LightRAG graph extraction, this is a better trade-off than plain text, which flattens structure, and lower risk than a JSON-first pipeline, which would require broader redesign.

## Proposed Architecture

The new upload flow will be:

`upload PDF -> parse_pdf_to_markdown() -> normalize_legal_markdown() -> LightRAG`

### Module boundaries

#### `backend/core/document_parser.py`

New module responsible for document parsing and normalization. It will expose:

- `async parse_pdf_to_markdown(file_path: str) -> str`
- `normalize_legal_markdown(content: str) -> str`

Responsibilities:

- call Docling locally to parse the PDF
- export Markdown
- apply light legal-document normalization
- return a final string suitable for `rag.ainsert(...)`

This module exists to keep PDF parsing concerns out of `llm_services.py`. PDF parsing is no longer an LLM-provider responsibility.

#### `backend/api/routes.py`

Change the PDF branch of `/upload` to call `parse_pdf_to_markdown(...)` from the new parser module.

TXT ingestion remains unchanged.

The success message should become provider-neutral and accurate, for example:

`File uploaded and indexed via Docling (...)`

#### `backend/core/llm_services.py`

Remove `qwen_vl_parse_pdf(...)`.

Keep the rest of the file as-is unless there is directly related cleanup required for imports or dead code. OpenRouter-backed chat and embedding code remains in scope only if still used by the rest of the application.

## Parsing and Normalization Rules

Docling output should be exported as Markdown, then normalized lightly.

Normalization must stay conservative. The purpose is to improve stability, not reinterpret legal content.

Expected normalization behavior:

- trim repeated blank lines
- normalize whitespace around headings and list items
- preserve section hierarchy such as `Phần`, `Chương`, `Mục`, `Điều`, `Khoản`, `Điểm` when they appear clearly in the parsed output
- preserve table and list formatting as much as possible
- avoid semantic rewrites, summarization, or content inference

The normalization step must not:

- collapse all content into plain prose
- renumber clauses
- rewrite legal phrasing
- delete sections because they look repetitive

## Dependency and Runtime Changes

### Python dependencies

Expected dependency updates:

- add `docling`
- remove `pdf2image` if there are no remaining usages in the backend

Other existing PDF-related dependencies should only be removed if they are confirmed unused after implementation.

### System packages

The backend container may need runtime adjustments depending on Docling's actual dependency chain in this environment. The implementation should verify the minimum required packages and only add what is necessary.

The current `backend/Dockerfile` already installs `poppler-utils` because of `pdf2image`. Whether that remains necessary should be decided after the new parser is working in-container.

## Error Handling

The new flow should make parser failures explicit and easier to diagnose.

Cases to handle:

- unsupported file type: unchanged behavior
- Docling parse failure: return an error message that clearly indicates PDF parsing failed
- empty parsed content: keep rejecting the upload with a message indicating no text could be extracted
- LightRAG insert failure: keep returning upload failure while preserving the underlying error detail for logs

Logging should distinguish:

- upload/write failure
- Docling parsing failure
- LightRAG insertion failure

## Documentation Changes

Update project documentation to reflect the new parser:

- README feature list
- README tech stack or setup notes if they mention Qwen 3 VL for PDF parsing
- any user-facing messaging that implies PDF parsing depends on OpenRouter vision

The docs should clearly state that PDF parsing is local and Docling-based, while avoiding claims that all LLM-related features are local if they are not.

## Verification Plan

Use the smallest relevant verification for this scope:

1. install and import Docling successfully in the backend environment
2. run upload flow with a representative text-based legal PDF
3. confirm parsed Markdown is non-empty
4. confirm `rag.ainsert(...)` still accepts the content without contract changes
5. confirm there are no remaining call paths to `qwen_vl_parse_pdf(...)`
6. confirm user-visible upload success text no longer references Qwen 3 VL

If a sample legal PDF is not available in-repo, verification should at minimum cover parser import, route wiring, and a focused parser smoke test with a local text-based PDF fixture created for testing.

## Risks and Trade-Offs

### Risk: Markdown structure may vary between PDFs

Impact:
- inconsistent chunk boundaries and graph quality

Mitigation:
- keep normalization focused and deterministic
- verify with a representative legal PDF, not only a trivial sample

### Risk: Some legal tables may render noisily

Impact:
- lower retrieval quality for sanction tables or structured penalties

Mitigation:
- preserve the table output first
- only add table-specific cleanup if real samples show a problem

### Risk: Scope creep into full provider migration

Impact:
- turns a narrow parser replacement into a much larger infrastructure change

Mitigation:
- explicitly keep OpenRouter for chat and embeddings in this spec
- limit this implementation to PDF parsing concerns

## Out of Scope Follow-Ups

Potential future work, not part of this spec:

- scan-heavy PDF fallback strategy
- custom chunking by `Điều/Khoản/Điểm`
- direct graph-oriented preprocessing before LightRAG insertion
- replacing OpenRouter for chat or embeddings with local or alternate providers

## Implementation Notes

Preferred code shape:

- create `backend/core/document_parser.py`
- update `backend/api/routes.py`
- remove the obsolete parser from `backend/core/llm_services.py`
- update `backend/requirements.txt`
- update `backend/Dockerfile` only as required
- update `README.md`

Changes should stay surgical and preserve the existing architecture outside the PDF ingestion path.

## Spec Review Notes

Self-review completed for:

- placeholders: none remain
- consistency: architecture, scope, and non-goals align
- scope: focused enough for a single implementation plan
- ambiguity: scope explicitly limited to PDF parsing, not full OpenRouter removal

## Environment Limitation

This workspace does not currently expose a valid Git repository, so the design document could not be committed after writing. The file has been saved locally at the path above and is ready for review.
