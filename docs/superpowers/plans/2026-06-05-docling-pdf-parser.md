# Docling PDF Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Qwen-VL PDF parsing path with a local Docling Markdown parser for text-based legal PDFs without changing the existing LightRAG ingest contract.

**Architecture:** Add a focused `backend/core/document_parser.py` module that owns Docling PDF conversion and conservative legal Markdown normalization. Update the upload route to use that parser, remove the old Qwen-VL PDF code from `llm_services.py`, and verify behavior with targeted parser and upload-route tests plus container-level runtime checks.

**Tech Stack:** Python 3.11, FastAPI, LightRAG, Docling, pytest, unittest.mock, Docker Compose

---

## File Map

- Create: `backend/core/document_parser.py`  
  Local Docling PDF conversion, converter caching, and legal Markdown normalization.

- Create: `backend/tests/test_document_parser.py`  
  Unit tests for normalization, Docling conversion wiring, empty-output handling, and parser error wrapping.

- Create: `backend/tests/test_upload_route.py`  
  Route tests for PDF and TXT upload behavior using mocked parser and mocked LightRAG insertions.

- Create: `backend/tests/test_pdf_ingest_cleanup.py`  
  Cleanup assertions that old Qwen-VL PDF parsing references are gone from code/docs/dependencies.

- Modify: `backend/api/routes.py`  
  Swap the PDF upload branch from `qwen_vl_parse_pdf(...)` to `parse_pdf_to_markdown(...)`; update success/error text.

- Modify: `backend/core/llm_services.py`  
  Remove the obsolete Qwen-VL PDF parser and any dead imports left behind.

- Modify: `backend/requirements.txt`  
  Add `docling` plus test dependencies used by this repo; remove `pdf2image`.

- Modify: `backend/Dockerfile`  
  Remove `poppler-utils` and add the minimal runtime package set needed for Docling in a headless Debian container.

- Modify: `README.md`  
  Replace Qwen-VL/OpenRouter PDF parsing descriptions with Docling/local parsing descriptions while leaving chat and embedding provider notes intact.

## Notes Before Execution

- The current workspace does not expose a valid `.git` repository. Commit steps are still included because they are part of the implementation workflow, but they must be executed from a proper checkout where `git status` works.
- This plan keeps OpenRouter in place for chat and embeddings. It only replaces the PDF parsing path.
- The plan assumes there is no existing backend test suite. It introduces a minimal pytest-based test surface because TDD is required and the project currently has no tests.

### Task 1: Add the Docling parser module behind focused unit tests

**Files:**
- Create: `backend/tests/test_document_parser.py`
- Create: `backend/core/document_parser.py`
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Write the failing parser tests**

Create `backend/tests/test_document_parser.py` with this exact content:

```python
import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from backend.core import document_parser


class FakeDocument:
    def __init__(self, markdown: str):
        self._markdown = markdown

    def export_to_markdown(self) -> str:
        return self._markdown


class FakeResult:
    def __init__(self, markdown: str):
        self.document = FakeDocument(markdown)


def test_normalize_legal_markdown_collapses_extra_blank_lines():
    raw = "Chương I\\n\\n\\nĐiều 1\\nNội dung\\n\\n\\n\\nKhoản 1\\n"

    normalized = document_parser.normalize_legal_markdown(raw)

    assert normalized == "Chương I\\n\\nĐiều 1\\nNội dung\\n\\nKhoản 1"


def test_parse_pdf_to_markdown_uses_docling_and_normalizes_output(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\\n")

    fake_converter = Mock()
    fake_converter.convert.return_value = FakeResult("Điều 1\\n\\n\\nNội dung")
    monkeypatch.setattr(document_parser, "get_pdf_converter", lambda: fake_converter)

    content = asyncio.run(document_parser.parse_pdf_to_markdown(str(pdf_path)))

    assert content == "Điều 1\\n\\nNội dung"
    fake_converter.convert.assert_called_once_with(Path(pdf_path))


def test_parse_pdf_to_markdown_rejects_empty_markdown(monkeypatch, tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\\n")

    fake_converter = Mock()
    fake_converter.convert.return_value = FakeResult(" \\n\\n ")
    monkeypatch.setattr(document_parser, "get_pdf_converter", lambda: fake_converter)

    with pytest.raises(ValueError, match="Docling extracted no text from PDF"):
        asyncio.run(document_parser.parse_pdf_to_markdown(str(pdf_path)))


def test_parse_pdf_to_markdown_wraps_docling_errors(monkeypatch, tmp_path):
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\\n")

    fake_converter = Mock()
    fake_converter.convert.side_effect = RuntimeError("boom")
    monkeypatch.setattr(document_parser, "get_pdf_converter", lambda: fake_converter)

    with pytest.raises(RuntimeError, match="Docling failed to parse PDF: boom"):
        asyncio.run(document_parser.parse_pdf_to_markdown(str(pdf_path)))
```

- [ ] **Step 2: Run the parser tests to verify they fail**

Run:

```bash
pytest backend/tests/test_document_parser.py -q
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'backend.core.document_parser'` or equivalent import failure because the parser module does not exist yet.

- [ ] **Step 3: Add the minimal Docling parser implementation**

Create `backend/core/document_parser.py` with this exact content:

```python
import asyncio
import re
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_pdf_converter: Optional[DocumentConverter] = None


def get_pdf_converter() -> DocumentConverter:
    global _pdf_converter
    if _pdf_converter is None:
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
        )
        _pdf_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            }
        )
    return _pdf_converter


def normalize_legal_markdown(content: str) -> str:
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = re.sub(r"[ \t]+\n", "\n", content)
    content = re.sub(r"\n{3,}", "\n\n", content)

    major_heading_pattern = re.compile(r"^(Phần|Chương|Mục|Điều)\b", re.IGNORECASE)
    normalized_lines: list[str] = []

    for raw_line in content.split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()
        if major_heading_pattern.match(stripped) and normalized_lines and normalized_lines[-1] != "":
            normalized_lines.append("")
        normalized_lines.append(line)

    return "\n".join(normalized_lines).strip()


async def parse_pdf_to_markdown(file_path: str) -> str:
    path = Path(file_path)
    converter = get_pdf_converter()

    try:
        result = await asyncio.to_thread(converter.convert, path)
    except Exception as exc:
        raise RuntimeError(f"Docling failed to parse PDF: {exc}") from exc

    markdown = normalize_legal_markdown(result.document.export_to_markdown())
    if not markdown.strip():
        raise ValueError("Docling extracted no text from PDF")

    return markdown
```

Update `backend/requirements.txt` by adding these lines near the end of the file:

```txt
docling>=2.0.0
pytest>=8.2.0
httpx>=0.27.0
```

- [ ] **Step 4: Run the parser tests to verify they pass**

Run:

```bash
pytest backend/tests/test_document_parser.py -q
```

Expected: PASS with `4 passed`.

- [ ] **Step 5: Commit the parser foundation**

Run:

```bash
git add backend/core/document_parser.py backend/tests/test_document_parser.py backend/requirements.txt
git commit -m "feat: add local docling pdf parser"
```

### Task 2: Wire the upload route to the Docling parser behind route tests

**Files:**
- Create: `backend/tests/test_upload_route.py`
- Modify: `backend/api/routes.py`

- [ ] **Step 1: Write the failing upload route tests**

Create `backend/tests/test_upload_route.py` with this exact content:

```python
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.api.routes as routes


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    return TestClient(app)


def test_pdf_upload_uses_docling_parser(monkeypatch, tmp_path):
    client = make_client()
    monkeypatch.setattr(routes.settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))

    fake_rag = Mock()
    fake_rag.ainsert = AsyncMock()
    fake_parser = AsyncMock(return_value="# Điều 1\n\nNội dung")

    monkeypatch.setattr(routes.RAGEngine, "get_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "parse_pdf_to_markdown", fake_parser)

    response = client.post(
        "/api/upload",
        files={"file": ("law.pdf", b"%PDF-1.4\n", "application/pdf")},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["filename"] == "law.pdf"
    assert body["status"] == "success"
    assert "Docling" in body["message"]
    fake_parser.assert_awaited_once_with(str(tmp_path / "law.pdf"))
    fake_rag.ainsert.assert_awaited_once_with("# Điều 1\n\nNội dung", file_paths=["law.pdf"])


def test_pdf_upload_returns_error_when_docling_parse_fails(monkeypatch, tmp_path):
    client = make_client()
    monkeypatch.setattr(routes.settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))

    fake_rag = Mock()
    fake_rag.ainsert = AsyncMock()
    fake_parser = AsyncMock(side_effect=RuntimeError("Docling failed to parse PDF: boom"))

    monkeypatch.setattr(routes.RAGEngine, "get_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "parse_pdf_to_markdown", fake_parser)

    response = client.post(
        "/api/upload",
        files={"file": ("law.pdf", b"%PDF-1.4\n", "application/pdf")},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["filename"] == "law.pdf"
    assert body["status"] == "error"
    assert "Docling failed to parse PDF" in body["message"]
    fake_rag.ainsert.assert_not_awaited()


def test_txt_upload_still_reads_local_text_file(monkeypatch, tmp_path):
    client = make_client()
    monkeypatch.setattr(routes.settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))

    fake_rag = Mock()
    fake_rag.ainsert = AsyncMock()
    fake_parser = AsyncMock()

    monkeypatch.setattr(routes.RAGEngine, "get_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "parse_pdf_to_markdown", fake_parser)

    response = client.post(
        "/api/upload",
        files={"file": ("law.txt", "Điều 1: tốc độ tối đa".encode("utf-8"), "text/plain")},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["filename"] == "law.txt"
    assert body["status"] == "success"
    fake_parser.assert_not_called()
    fake_rag.ainsert.assert_awaited_once_with("Điều 1: tốc độ tối đa", file_paths=["law.txt"])
```

- [ ] **Step 2: Run the upload route tests to verify they fail**

Run:

```bash
pytest backend/tests/test_upload_route.py -q
```

Expected: FAIL because `backend.api.routes` still imports and calls `qwen_vl_parse_pdf(...)` instead of exposing and using `parse_pdf_to_markdown(...)`.

- [ ] **Step 3: Update the upload route to use the Docling parser**

In `backend/api/routes.py`, add this import near the top-level imports:

```python
from backend.core.document_parser import parse_pdf_to_markdown
```

Then replace the entire `upload_file(...)` function with this exact version:

```python
@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf") and not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    file_path = os.path.join(settings.LIGHTRAG_WORKING_DIR, file.filename)
    os.makedirs(settings.LIGHTRAG_WORKING_DIR, exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        rag = RAGEngine.get_instance()

        if file.filename.endswith(".pdf"):
            content = await parse_pdf_to_markdown(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        if not content.strip():
            raise ValueError("File is empty or no text could be extracted")

        await rag.ainsert(content, file_paths=[file.filename])

        return UploadResponse(
            filename=file.filename,
            status="success",
            message=f"File uploaded and indexed via Docling ({len(content)} characters)"
        )
    except Exception as e:
        print(f"Error indexing uploaded file {file.filename}: {e}")
        return UploadResponse(
            filename=file.filename,
            status="error",
            message=f"Failed to index file: {str(e)}"
        )
```

- [ ] **Step 4: Run the upload route tests to verify they pass**

Run:

```bash
pytest backend/tests/test_upload_route.py -q
```

Expected: PASS with `3 passed`.

- [ ] **Step 5: Commit the route wiring**

Run:

```bash
git add backend/api/routes.py backend/tests/test_upload_route.py
git commit -m "feat: route pdf uploads through docling parser"
```

### Task 3: Remove the old Qwen-VL parser path and clean up dependencies, container runtime, and docs

**Files:**
- Create: `backend/tests/test_pdf_ingest_cleanup.py`
- Modify: `backend/core/llm_services.py`
- Modify: `backend/requirements.txt`
- Modify: `backend/Dockerfile`
- Modify: `README.md`

- [ ] **Step 1: Write the failing cleanup assertions**

Create `backend/tests/test_pdf_ingest_cleanup.py` with this exact content:

```python
from pathlib import Path

import backend.core.llm_services as llm_services


def test_qwen_vl_parser_is_removed_from_llm_services():
    assert not hasattr(llm_services, "qwen_vl_parse_pdf")


def test_backend_requirements_replace_pdf2image_with_docling():
    requirements = Path("backend/requirements.txt").read_text(encoding="utf-8")
    assert "docling" in requirements
    assert "pdf2image" not in requirements


def test_readme_describes_docling_for_pdf_parsing():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Local PDF Parsing with Docling" in readme
    assert "Qwen 3 VL" not in readme
```

- [ ] **Step 2: Run the cleanup assertions to verify they fail**

Run:

```bash
pytest backend/tests/test_pdf_ingest_cleanup.py -q
```

Expected: FAIL because `qwen_vl_parse_pdf(...)` still exists, `pdf2image` is still listed, and `README.md` still advertises Qwen 3 VL for PDF parsing.

- [ ] **Step 3: Apply the cleanup changes**

Replace `backend/core/llm_services.py` with this exact content:

```python
import openai
from typing import List, Optional

from backend.config import settings

# Global client cache to avoid pickling issues and redundant connections
_async_client: Optional[openai.AsyncOpenAI] = None


def get_openai_client():
    global _async_client
    if _async_client is None:
        _async_client = openai.AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
    return _async_client


class QwenEmbeddingFunc:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL

    def _get_prefix(self, is_query: bool) -> str:
        if is_query:
            return "Instruct: Given a legal query, retrieve relevant statutes...\nQuery: "
        return ""

    async def __call__(self, texts: List[str]):
        import numpy as np

        client = get_openai_client()
        results = []
        for text in texts:
            prefix = self._get_prefix(is_query=text.strip().endswith("?"))

            response = await client.embeddings.create(
                model=self.model_name,
                input=prefix + text
            )
            results.append(response.data[0].embedding)
        return np.array(results)


async def deepseek_llm_func(
    prompt: str,
    system_prompt: str = None,
    history: List[dict] = None,
    **kwargs
) -> str:
    client = get_openai_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": prompt})

    extra_headers = {
        "HTTP-Referer": "https://github.com/traffic/law-assistant",
        "X-Title": "Traffic Law Assistant",
    }

    allowed_params = [
        "model", "messages", "stream", "temperature", "top_p", "n", "stop", "max_tokens",
        "presence_penalty", "frequency_penalty", "logit_bias", "user", "response_format",
        "seed", "tools", "tool_choice", "parallel_tool_calls"
    ]
    api_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}

    response = await client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        extra_headers=extra_headers,
        **api_kwargs
    )

    if api_kwargs.get("stream"):
        async def stream_generator():
            print("LLM: Starting stream generator")
            try:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        c = chunk.choices[0].delta.content
                        print(f"LLM CHUNK: {c}")
                        yield c
            except Exception as e:
                print(f"LLM STREAM ERROR: {str(e)}")
            print("LLM: Stream generator finished")

        return stream_generator()
    else:
        return response.choices[0].message.content
```

In `backend/requirements.txt`, replace the current document-processing section with this exact block and append the test dependencies shown below:

```txt
# Document Processing
pypdf>=4.0.0
python-docx>=1.1.0
openpyxl>=3.1.0
python-pptx>=0.6.0
docling>=2.0.0

# Test Support
pytest>=8.2.0
httpx>=0.27.0
```

Replace `backend/Dockerfile` with this exact content:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
  build-essential \
  libpq-dev \
  gcc \
  libgl1 \
  && rm -rf /var/lib/apt/lists/*

# Use uv for faster dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --system -r requirements.txt

COPY . ./backend

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

In `README.md`, replace the feature and stack bullets that mention Qwen 3 VL with these exact lines:

```md
- **Local PDF Parsing with Docling**: Uses **Docling** to extract structured Markdown from text-based legal PDFs entirely locally.
```

```md
- **LLM/Embeddings**: DeepSeek V3, OpenAI Embeddings (via OpenRouter)
```

Also update the opening description sentence to this exact version:

```md
An advanced legal document assistant powered by **LightRAG**, localized for Vietnamese law and featuring high-fidelity Knowledge Graph visualization. This project uses **FastAPI** for the backend, **React** for the frontend, and **PostgreSQL (Apache AGE + pgvector)** for graph and vector storage. PDF ingestion is handled locally with **Docling**.
```

- [ ] **Step 4: Run the cleanup and regression tests**

Run:

```bash
pytest backend/tests/test_document_parser.py backend/tests/test_upload_route.py backend/tests/test_pdf_ingest_cleanup.py -q
```

Expected: PASS with `10 passed`.

- [ ] **Step 5: Commit the cleanup and docs updates**

Run:

```bash
git add backend/core/llm_services.py backend/requirements.txt backend/Dockerfile README.md backend/tests/test_pdf_ingest_cleanup.py
git commit -m "refactor: remove qwen vl pdf ingestion path"
```

### Task 4: Run the verification sweep for runtime and cleanup safety

**Files:**
- No new files

- [ ] **Step 1: Run the focused backend test suite**

Run:

```bash
pytest backend/tests/test_document_parser.py backend/tests/test_upload_route.py backend/tests/test_pdf_ingest_cleanup.py -q
```

Expected: PASS with `10 passed`.

- [ ] **Step 2: Build the backend image with the new dependencies**

Run:

```bash
docker compose build backend
```

Expected: SUCCESS. The image should build without `pdf2image`, and the dependency install step should complete with `docling` present.

- [ ] **Step 3: Smoke-test Docling import inside the backend container**

Run:

```bash
docker compose run --rm backend python -c "from backend.core.document_parser import get_pdf_converter; print(type(get_pdf_converter()).__name__)"
```

Expected: prints `DocumentConverter`.

- [ ] **Step 4: Run static cleanup checks against backend code and docs**

Run:

```bash
rg -n "qwen_vl_parse_pdf|pdf2image|Qwen 3 VL" backend README.md
```

Expected: no matches.

## Self-Review

- Spec coverage: parser module, route wiring, error handling, dependency/runtime changes, docs changes, and verification are all covered by Tasks 1-4.
- Placeholder scan: no `TODO`, `TBD`, or deferred “fill this in later” instructions remain.
- Type consistency: the plan consistently uses `parse_pdf_to_markdown(file_path: str) -> str` and `normalize_legal_markdown(content: str) -> str` everywhere.

