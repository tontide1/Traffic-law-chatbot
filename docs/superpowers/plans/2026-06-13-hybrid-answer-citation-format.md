# Hybrid Answer-First Citation Format Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `hybrid` return answer-first text with inline legal citations in both single mode and comparison mode, without changing the API payload shape.

**Architecture:** Keep the existing controlled retrieval pipeline in `backend/core/controlled_chat.py`, but replace the direct-definition short-circuit and fallback renderer with an answer-first formatter. Prompting stays backend-only: the curated answer prompt should bias the LLM toward sentence-level inline citations, and the fallback renderer should produce the same format when generation drifts.

**Tech Stack:** Python 3.11, FastAPI, LightRAG `QueryParam`, pytest

---

## File Structure

| File | Responsibility |
|------|----------------|
| Modify: `backend/core/controlled_chat.py` | Replace heading-based direct-definition rendering with answer-first inline-citation rendering and align semantic-drift fallback |
| Modify: `backend/core/legal_prompts.py` | Update curated answer prompt to require answer-first inline citation formatting |
| Modify: `backend/tests/test_controlled_chat.py` | Lock direct-definition output shape and fallback shape with unit tests |
| Modify: `backend/tests/test_legal_prompts.py` | Lock prompt rules for inline citation and no-heading simple answers |
| Modify: `backend/tests/test_chat_legal_prompt_integration.py` | Verify single-mode and comparison-mode routes preserve the new `hybrid` response contract |

---

### Task 1: Lock The New Output Contract With Failing Tests

**Files:**
- Modify: `backend/tests/test_controlled_chat.py`
- Modify: `backend/tests/test_legal_prompts.py`
- Modify: `backend/tests/test_chat_legal_prompt_integration.py`

- [ ] **Step 1: Add failing unit tests for direct-definition output shape**

Edit `backend/tests/test_controlled_chat.py` so the direct-definition tests assert answer-first inline citations instead of `Căn cứ chính` headings. Replace the current expectation blocks with these assertions:

```python
    assert result.startswith("Hành lang an toàn đường bộ là dải đất dọc hai bên đất của đường bộ")
    assert "(Điều 2 khoản 5" in result
    assert "Luật Đường bộ 35/2024/QH15" in result
    assert "Căn cứ chính" not in result
    assert "Liên kết pháp lý liên quan" not in result
    assert "Điều 4. Chính sách" not in result
    assert rag.aquery.await_count == 1
```

Apply the same shape assertions to:

- `test_direct_definition_uses_context_retrieval_before_extractive_answer`
- `test_direct_definition_returns_extractive_answer_without_semantic_drift`
- `test_direct_definition_unwraps_json_context_before_rendering`
- `test_direct_definition_unwraps_consecutive_json_context_before_rendering`

For the semantic drift fallback test, replace the current heading-based assertion block with:

```python
    assert result.startswith("Hành lang an toàn đường bộ là dải đất dọc hai bên đất của đường bộ")
    assert "(Điều 2 khoản 5" in result
    assert "Theo định nghĩa trên" not in result
    assert "mặt đường" not in result
    assert "được quy hoạch và quản lý" not in result
    assert "các phương tiện lưu thông" not in result
    assert "Căn cứ chính" not in result
    assert rag.aquery.await_count == 2
```

- [ ] **Step 2: Add failing prompt tests for answer-first inline citation rules**

Append these tests to `backend/tests/test_legal_prompts.py`:

```python
def test_curated_answer_prompt_requires_inline_citations():
    template = build_curated_answer_system_prompt().lower()

    assert "trích dẫn" in template or "citation" in template
    assert "inline" in template or "ngay cuối câu" in template
    assert "cuối câu" in template


def test_curated_answer_prompt_avoids_headings_for_simple_answers():
    template = build_curated_answer_system_prompt().lower()

    assert "không dùng" in template or "tránh dùng" in template
    assert "căn cứ chính" in template
    assert "liên kết pháp lý liên quan" in template
    assert "câu hỏi đơn giản" in template or "một căn cứ" in template
```

- [ ] **Step 3: Add failing integration tests for single-mode and comparison-mode `hybrid` content**

Append these tests to `backend/tests/test_chat_legal_prompt_integration.py`:

```python
def test_single_hybrid_chat_returns_answer_first_inline_citation(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="unused"))
    controlled = AsyncMock(
        return_value=(
            "Hành lang an toàn đường bộ là dải đất dọc hai bên đất của đường bộ. "
            "(Điều 2 khoản 5, Luật Đường bộ 35/2024/QH15)."
        )
    )

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Hành lang an toàn đường bộ là gì?", "stream": False, "comparison_mode": False},
    )

    assert response.status_code == 200
    assert response.json()["response"].startswith("Hành lang an toàn đường bộ là dải đất dọc hai bên đất của đường bộ")
    assert "(Điều 2 khoản 5, Luật Đường bộ 35/2024/QH15)." in response.json()["response"]
    assert "Căn cứ chính" not in response.json()["response"]


def test_comparison_hybrid_chat_returns_answer_first_inline_citation(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="Naive answer"))
    controlled = AsyncMock(
        return_value=(
            "Hành lang an toàn đường bộ là dải đất dọc hai bên đất của đường bộ. "
            "(Điều 2 khoản 5, Luật Đường bộ 35/2024/QH15)."
        )
    )

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Hành lang an toàn đường bộ là gì?", "stream": False, "comparison_mode": True},
    )

    assert response.status_code == 200
    assert response.json()["hybrid"]["response"].startswith("Hành lang an toàn đường bộ là dải đất dọc hai bên đất của đường bộ")
    assert "(Điều 2 khoản 5, Luật Đường bộ 35/2024/QH15)." in response.json()["hybrid"]["response"]
    assert "Căn cứ chính" not in response.json()["hybrid"]["response"]
```

- [ ] **Step 4: Run targeted tests to verify failure**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest \
  backend/tests/test_controlled_chat.py \
  backend/tests/test_legal_prompts.py \
  backend/tests/test_chat_legal_prompt_integration.py -v
```

Expected:

- `test_controlled_chat.py` fails because `answer_controlled_chat()` still emits heading-based direct-definition output
- `test_legal_prompts.py` fails because the curated prompt does not yet require inline citations or discourage headings for simple one-basis answers
- route integration tests should already pass structurally, but the new content-shape assertions establish the contract for the backend implementation

- [ ] **Step 5: Commit the red test baseline**

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
git add \
  backend/tests/test_controlled_chat.py \
  backend/tests/test_legal_prompts.py \
  backend/tests/test_chat_legal_prompt_integration.py
git commit -m "test: lock hybrid answer citation format contract"
```

---

### Task 2: Implement Answer-First Inline Citation Rendering

**Files:**
- Modify: `backend/core/controlled_chat.py`

- [ ] **Step 1: Replace the heading-based direct-definition renderer with an inline-citation renderer**

In `backend/core/controlled_chat.py`, replace the existing `ExtractedDefinition` dataclass and rendering helpers with a formatter that can render:

- direct-definition answer-first output
- semantic-drift fallback output

Use this shape:

```python
@dataclass(frozen=True)
class ExtractedDefinition:
    source: str
    text: str


def _clean_definition_text(text: str) -> str:
    cleaned = text.strip().strip("“”\"' ").rstrip(".")
    return cleaned


def _clean_citation_source(source: str) -> str:
    cleaned = " ".join(source.strip().rstrip(":").split())
    cleaned = re.sub(r"^[,;:\s]+", "", cleaned)
    return cleaned


def _render_inline_citation_answer(extracted: ExtractedDefinition) -> str:
    definition = _clean_definition_text(extracted.text)
    citation = _clean_citation_source(extracted.source)

    if citation:
        return f"{definition}. ({citation})."
    return f"{definition}."


def _render_direct_definition_answer(candidates: list[ContextCandidate]) -> str | None:
    extracted = _extract_direct_definition(candidates)
    if extracted is None:
        return None
    return _render_inline_citation_answer(extracted)
```

Delete the old output branch that returned:

```python
return f"Căn cứ chính{source}\n\n“{extracted.text}”"
```

- [ ] **Step 2: Keep the anti-drift fallback but align it to the new output format**

In `answer_controlled_chat()`, keep the current structure, but change the direct-definition and fallback branches to:

```python
    direct_definition_answer = _render_direct_definition_answer(selected)
    if query_class == QueryClass.DIRECT_DEFINITION and direct_definition_answer is not None:
        return direct_definition_answer

    answer = await rag.aquery(
        _build_answer_query(message, curated_context),
        param=QueryParam(mode="bypass", stream=stream),
        system_prompt=build_curated_answer_system_prompt(),
    )

    if (
        not stream
        and isinstance(answer, str)
        and _has_semantic_drift(answer, curated_context)
        and direct_definition_answer is not None
    ):
        return direct_definition_answer

    return answer
```

This preserves the current safety model while ensuring both the short-circuit path and the fallback path use answer-first inline-citation rendering.

- [ ] **Step 3: Make source extraction robust enough for inline citations**

Keep `_split_source_prefix()` and `_extract_direct_definition()` mostly intact, but tighten the source handling so the citation string is usable inline. The helper should continue to accept inputs such as:

```python
"Luật Đường bộ 35/2024/QH15, Điều 2 khoản 5: Hành lang an toàn đường bộ là dải đất..."
"Điều 2 khoản 5: Hành lang an toàn đường bộ là dải đất..."
```

and produce a readable final citation:

```python
"Luật Đường bộ 35/2024/QH15, Điều 2 khoản 5"
```

or:

```python
"Điều 2 khoản 5"
```

If the extracted source still comes out reversed or awkward, normalize it in `_clean_citation_source()` instead of changing the retrieval pipeline.

- [ ] **Step 4: Run unit tests for controlled chat**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_controlled_chat.py -v
```

Expected:

- direct-definition tests pass with answer-first inline citations
- semantic-drift fallback test passes with answer-first inline citations
- relational and streaming tests remain green

- [ ] **Step 5: Commit the formatter change**

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
git add backend/core/controlled_chat.py backend/tests/test_controlled_chat.py
git commit -m "feat: render hybrid definitions with inline citations"
```

---

### Task 3: Update Prompt Rules And Verify Route-Level Behavior

**Files:**
- Modify: `backend/core/legal_prompts.py`
- Modify: `backend/tests/test_legal_prompts.py`
- Modify: `backend/tests/test_chat_legal_prompt_integration.py`

- [ ] **Step 1: Update the curated answer prompt to require answer-first inline citations**

In `backend/core/legal_prompts.py`, update `build_curated_answer_system_prompt()` to this exact body:

```python
def build_curated_answer_system_prompt() -> str:
    """Return a plain system prompt for bypass-mode answers from curated context."""
    return """Bạn là trợ lý pháp lý tiếng Việt.
Trả lời dựa duy nhất trên câu hỏi và "Ngữ cảnh pháp lý đã được chọn lọc" trong user message.
Không dùng kiến thức nền ngoài ngữ cảnh đã được chọn lọc.
Giữ nguyên thuật ngữ pháp lý, không thay bằng từ đồng nghĩa.
Không viết các câu diễn giải như "Theo định nghĩa trên", "có thể hiểu là", "nói cách khác", hoặc các câu tương đương nếu câu hỏi chỉ cần căn cứ trực tiếp.
Không thêm từ/cụm từ không xuất hiện trong ngữ cảnh đã được chọn lọc để giải thích lại thuật ngữ pháp lý.
Trả lời trực tiếp vào câu hỏi trước, sau đó đặt trích dẫn pháp lý ngay cuối câu hoặc cuối ý mà nó hỗ trợ.
Với câu hỏi chỉ có một căn cứ trực tiếp, không dùng các mục riêng như "Căn cứ chính" hoặc "Liên kết pháp lý liên quan".
Chỉ dùng mục hoặc bullet khi câu hỏi thực sự cần nhiều ý độc lập, và mỗi ý phải có trích dẫn ngay cuối ý đó.
Nếu ngữ cảnh nói không tìm thấy căn cứ trực tiếp đủ liên quan, hãy nói rõ là chưa đủ căn cứ trực tiếp.
Trả lời hoàn toàn bằng tiếng Việt."""
```

- [ ] **Step 2: Run prompt tests**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_legal_prompts.py -v
```

Expected: all prompt tests pass, including the new inline-citation and no-heading cases.

- [ ] **Step 3: Run route-level integration tests**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_chat_legal_prompt_integration.py -v
```

Expected:

- single-mode `hybrid` route tests pass
- comparison-mode `hybrid` route tests pass
- `naive` path assertions remain unchanged

- [ ] **Step 4: Run the full targeted regression bundle**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest \
  backend/tests/test_controlled_chat.py \
  backend/tests/test_legal_prompts.py \
  backend/tests/test_chat_legal_prompt_integration.py -v
```

Expected: the full targeted bundle passes with no failures.

- [ ] **Step 5: Commit prompt and integration verification changes**

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
git add \
  backend/core/legal_prompts.py \
  backend/tests/test_legal_prompts.py \
  backend/tests/test_chat_legal_prompt_integration.py
git commit -m "feat: enforce answer-first hybrid citation format"
```

---

## Self-Review

### Spec Coverage

- `hybrid` single mode answer-first inline citation: covered by Task 1 integration test and Task 2 renderer change
- `hybrid` comparison mode answer-first inline citation: covered by Task 1 integration test and Task 3 route regression
- no heading-only direct-definition block: covered by Task 1 unit assertions and Task 2 renderer replacement
- fallback formatting aligned with main output contract: covered by Task 1 semantic-drift test and Task 2 fallback branch
- no API payload changes: preserved by only editing `response` string content and route integration tests

### Placeholder Scan

- No `TODO`, `TBD`, or “implement later” placeholders remain
- Every code-changing step points to exact files and includes target code blocks
- Every verification step has exact commands and expected outcomes

### Type Consistency

- `answer_controlled_chat()` still returns `str | AsyncIterator[str]`
- `ChatResponse` and `ComparisonResponse` remain unchanged
- `ExtractedDefinition`, `ContextCandidate`, and `QueryClass` stay in the same modules and are referenced consistently

