# Controlled Hybrid Legal GraphRAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Hybrid GraphRAG over-fetching by tightening legal prompts, routing direct questions to low-expansion retrieval, and applying reranking plus a legal relevance gate before answer generation.

**Architecture:** Keep LightRAG and the current API contract, but add a small backend orchestration layer around chat retrieval. `backend/api/routes.py` should stay thin: classify the query, retrieve context through a controlled service, rerank/filter candidates, then generate the final answer from curated context. Graph schema, ingestion, frontend payloads, and SSE event shape stay unchanged.

**Tech Stack:** Python 3.11, FastAPI, LightRAG `QueryParam`, pytest, optional `FlagEmbedding` runtime wrapper for `BAAI/bge-reranker-v2-m3`

---

## File Structure

| File | Responsibility |
|------|----------------|
| Modify: `backend/core/legal_prompts.py` | Strengthen hybrid/naive prompt guard rules and add a bypass-safe curated answer prompt |
| Create: `backend/core/query_router.py` | Classify Vietnamese legal questions and map each class to a retrieval policy |
| Create: `backend/core/legal_relevance.py` | Represent context candidates, split LightRAG context, label direct/supporting/background/noise evidence, select final context |
| Create: `backend/core/reranker.py` | Reranker protocol, no-op reranker, lazy `BAAI/bge-reranker-v2-m3` implementation wrapper |
| Modify: `backend/config.py` | Add reranker configuration with safe defaults and disable switch |
| Create: `backend/core/controlled_chat.py` | Orchestrate retrieval, rerank, legal gate, and final answer generation |
| Modify: `backend/api/routes.py` | Delegate hybrid chat paths to controlled chat service while preserving response/SSE shape |
| Create: `backend/tests/test_query_router.py` | Router classification and policy tests |
| Create: `backend/tests/test_legal_relevance.py` | Candidate splitting, labeling, background demotion, and final context selection tests |
| Create: `backend/tests/test_reranker.py` | Reranker factory and mock ordering tests |
| Create: `backend/tests/test_controlled_chat.py` | Service-level orchestration tests with fake LightRAG/reranker |
| Modify: `backend/tests/test_legal_prompts.py` | Prompt guard regression tests |
| Modify: `backend/tests/test_chat_legal_prompt_integration.py` | Route tests updated for controlled hybrid orchestration |
| Modify: `data/legal_benchmark.json` | Preserve existing benchmark items and add routing/forbidden-source metadata |
| Modify: `backend/core/benchmark_checker.py` | Benchmark loader and deterministic term/source checks |
| Modify: `scripts/run_regression_check.py` | Manual live-system regression script for curated benchmark questions |

---

### Task 1: Strengthen Legal Prompt Guard

**Files:**
- Modify: `backend/core/legal_prompts.py`
- Modify: `backend/tests/test_legal_prompts.py`

- [ ] **Step 1: Add failing prompt guard tests**

Append these tests to `backend/tests/test_legal_prompts.py`:

```python
def test_hybrid_prompt_rejects_broad_background_for_direct_questions():
    template = build_legal_system_prompt("hybrid")

    assert "Hiến pháp" in template
    assert "Điều 1" in template
    assert "Điều 4" in template
    assert "phạm vi điều chỉnh" in template.lower()
    assert "chính sách chung" in template.lower()
    assert "không dẫn" in template.lower() or "không trích dẫn" in template.lower()


def test_hybrid_prompt_limits_related_links_when_direct_basis_is_enough():
    template = build_legal_system_prompt("hybrid")
    lowered = template.lower()

    assert "liên kết pháp lý liên quan" in lowered
    assert "tối đa 1-2" in lowered
    assert "căn cứ trực tiếp đã đủ" in lowered
    assert "bỏ" in lowered or "không cần" in lowered


def test_naive_prompt_preserves_direct_answer_contract():
    template = build_legal_system_prompt("naive")
    lowered = template.lower()

    assert "căn cứ chính" in lowered
    assert "căn cứ trực tiếp" in lowered
    assert "giữ nguyên" in lowered
    assert "thuật ngữ pháp lý" in lowered


def test_curated_answer_prompt_has_no_lightrag_context_placeholders():
    from backend.core.legal_prompts import build_curated_answer_system_prompt

    template = build_curated_answer_system_prompt()

    assert "{context_data}" not in template
    assert "{content_data}" not in template
    assert "{response_type}" not in template
    assert "ngữ cảnh pháp lý đã được chọn lọc" in template.lower()
```

- [ ] **Step 2: Run prompt tests to verify failure**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_legal_prompts.py -v
```

Expected: at least the new hybrid background/prompt limit tests fail because the prompt does not yet contain those guard rules.

- [ ] **Step 3: Update legal prompt templates**

Modify `backend/core/legal_prompts.py` so the shared prefix and suffix contain these rules:

```python
"""Mode-aware legal answer system prompts for LightRAG."""

from typing import Literal

Mode = Literal["hybrid", "naive"]

_PROMPT_PREFIX = """Bạn là trợ lý pháp lý tiếng Việt.
Hãy trả lời hoàn toàn bằng tiếng Việt.
Giữ nguyên thuật ngữ pháp lý của nguồn, không thay bằng từ đồng nghĩa, không diễn giải lại theo kiểu paraphrase, và không tự ý lược bỏ sắc thái pháp lý.

Yêu cầu đầu ra:
- Giữ nguyên tên gọi, cụm từ, và thuật ngữ pháp lý quan trọng như trong nguồn.
- Dùng các nhãn mục ổn định: "Căn cứ chính" và "Liên kết pháp lý liên quan".
- "Căn cứ chính" chỉ gồm căn cứ trực tiếp trả lời câu hỏi của người dùng.
- Nếu căn cứ trực tiếp đã đủ trả lời, bỏ mục "Liên kết pháp lý liên quan" hoặc ghi ngắn gọn rằng không cần bổ sung.
- "Liên kết pháp lý liên quan" chỉ dùng khi quy định khác trực tiếp làm rõ, thu hẹp, mở rộng, áp dụng, định nghĩa, nêu điều kiện, ngoại lệ, thẩm quyền, chế tài, hoặc hệ quả cần thiết cho câu trả lời.
- Với câu hỏi trực tiếp hoặc câu hỏi định nghĩa, mục "Liên kết pháp lý liên quan" tối đa 1-2 ý và mỗi ý phải nói rõ nó làm rõ phần nào của câu hỏi.
- Không dẫn Hiến pháp, Điều 1, Điều 4, phạm vi điều chỉnh, chính sách chung, căn cứ ban hành, hoặc node cấp văn bản nếu người dùng không hỏi trực tiếp về các nội dung đó.
- Nếu dữ liệu chưa đủ căn cứ trực tiếp, nói rõ là chưa đủ căn cứ thay vì dùng căn cứ nền để lấp chỗ trống.
- Phần trả lời phải theo định dạng {response_type}.
- Bám sát câu hỏi của người dùng: {user_prompt}
"""

_PROMPT_SUFFIX = """

Trình bày:
- Căn cứ chính: nêu căn cứ pháp lý trực tiếp, giữ nguyên thuật ngữ pháp lý khi có thể.
- Liên kết pháp lý liên quan: chỉ dùng khi cần nối nhiều quy định hoặc điều khoản liên quan trực tiếp; không dùng để liệt kê bối cảnh chung.
"""

_HYBRID_TEMPLATE = _PROMPT_PREFIX + """

Ngữ cảnh truy xuất đã được chọn lọc:
{context_data}
""" + _PROMPT_SUFFIX

_NAIVE_TEMPLATE = _PROMPT_PREFIX + """

Dữ liệu nội dung đã được chọn lọc:
{content_data}
""" + _PROMPT_SUFFIX

_TEMPLATES: dict[Mode, str] = {
    "hybrid": _HYBRID_TEMPLATE,
    "naive": _NAIVE_TEMPLATE,
}


def build_legal_system_prompt(mode: Mode) -> str:
    """Return the legal system prompt template for the requested LightRAG mode."""
    try:
        return _TEMPLATES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode: {mode!r}") from exc


def build_curated_answer_system_prompt() -> str:
    """Return a plain system prompt for bypass-mode answers from curated context."""
    return """Bạn là trợ lý pháp lý tiếng Việt.
Trả lời dựa duy nhất trên câu hỏi và "Ngữ cảnh pháp lý đã được chọn lọc" trong user message.
Không dùng kiến thức nền ngoài ngữ cảnh đã được chọn lọc.
Giữ nguyên thuật ngữ pháp lý, không thay bằng từ đồng nghĩa.
Dùng mục "Căn cứ chính".
Chỉ dùng mục "Liên kết pháp lý liên quan" nếu ngữ cảnh đã chọn lọc có căn cứ liên quan trực tiếp.
Nếu ngữ cảnh nói không tìm thấy căn cứ trực tiếp đủ liên quan, hãy nói rõ là chưa đủ căn cứ trực tiếp.
Trả lời hoàn toàn bằng tiếng Việt."""
```

- [ ] **Step 4: Run prompt tests to verify pass**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_legal_prompts.py -v
```

Expected: all prompt tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
git add backend/core/legal_prompts.py backend/tests/test_legal_prompts.py
git commit -m "feat: tighten legal answer prompt guards"
```

---

### Task 2: Add Rule-Based Query Router

**Files:**
- Create: `backend/core/query_router.py`
- Create: `backend/tests/test_query_router.py`

- [ ] **Step 1: Write failing router tests**

Create `backend/tests/test_query_router.py`:

```python
from backend.core.query_router import QueryClass, classify_query, retrieval_policy_for


def test_classifies_direct_definition_question():
    question = "Hành lang an toàn đường bộ được xác định từ đâu và nhằm mục đích gì?"

    assert classify_query(question) == QueryClass.DIRECT_DEFINITION


def test_classifies_relational_question():
    question = "Quy định về hành lang an toàn đường bộ liên quan đến trách nhiệm của cơ quan nào?"

    assert classify_query(question) == QueryClass.RELATIONAL


def test_classifies_direct_rule_question():
    question = "Người điều khiển xe máy vi phạm nồng độ cồn bị phạt bao nhiêu?"

    assert classify_query(question) == QueryClass.DIRECT_RULE


def test_defaults_uncertain_question_to_open_ended():
    question = "Hãy phân tích các quy định mới trong luật đường bộ"

    assert classify_query(question) == QueryClass.OPEN_ENDED


def test_direct_definition_policy_uses_low_expansion_retrieval():
    policy = retrieval_policy_for(QueryClass.DIRECT_DEFINITION, stream=False)

    assert policy.query_class == QueryClass.DIRECT_DEFINITION
    assert policy.retrieval_mode == "naive"
    assert policy.answer_mode == "hybrid"
    assert policy.top_k <= 10
    assert policy.chunk_top_k <= 5
    assert policy.final_top_n == 3
    assert policy.max_relation_tokens <= 1000
    assert policy.stream is False


def test_relational_policy_keeps_hybrid_retrieval():
    policy = retrieval_policy_for(QueryClass.RELATIONAL, stream=True)

    assert policy.query_class == QueryClass.RELATIONAL
    assert policy.retrieval_mode == "hybrid"
    assert policy.answer_mode == "hybrid"
    assert policy.final_top_n == 6
    assert policy.stream is True
```

- [ ] **Step 2: Run router tests to verify failure**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_query_router.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.core.query_router'`.

- [ ] **Step 3: Implement query router**

Create `backend/core/query_router.py`:

```python
"""Rule-based routing for Vietnamese legal RAG questions."""

from dataclasses import dataclass
from enum import StrEnum


class QueryClass(StrEnum):
    DIRECT_DEFINITION = "direct_definition"
    DIRECT_RULE = "direct_rule"
    RELATIONAL = "relational"
    OPEN_ENDED = "open_ended"


@dataclass(frozen=True)
class RetrievalPolicy:
    query_class: QueryClass
    retrieval_mode: str
    answer_mode: str
    top_k: int
    chunk_top_k: int
    max_entity_tokens: int
    max_relation_tokens: int
    max_total_tokens: int
    final_top_n: int
    stream: bool = False


_DIRECT_DEFINITION_SIGNALS = (
    "là gì",
    "khái niệm",
    "được hiểu như thế nào",
    "được xác định từ đâu",
    "nhằm mục đích gì",
    "có nghĩa là gì",
)

_RELATIONAL_SIGNALS = (
    "liên quan đến",
    "căn cứ nào",
    "so sánh",
    "khác nhau",
    "trong trường hợp",
    "ngoại lệ",
    "hậu quả pháp lý",
    "trách nhiệm của",
)

_DIRECT_RULE_SIGNALS = (
    "bị phạt",
    "mức phạt",
    "bao nhiêu",
    "thời hạn",
    "điều kiện",
    "nghĩa vụ",
    "thẩm quyền",
    "bị cấm",
    "phải thực hiện",
    "phải chịu",
)


def _normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


def _contains_any(text: str, signals: tuple[str, ...]) -> bool:
    return any(signal in text for signal in signals)


def classify_query(query: str) -> QueryClass:
    normalized = _normalize_query(query)

    if _contains_any(normalized, _RELATIONAL_SIGNALS):
        return QueryClass.RELATIONAL
    if _contains_any(normalized, _DIRECT_DEFINITION_SIGNALS):
        return QueryClass.DIRECT_DEFINITION
    if _contains_any(normalized, _DIRECT_RULE_SIGNALS):
        return QueryClass.DIRECT_RULE
    return QueryClass.OPEN_ENDED


def retrieval_policy_for(query_class: QueryClass, stream: bool = False) -> RetrievalPolicy:
    if query_class == QueryClass.DIRECT_DEFINITION:
        return RetrievalPolicy(
            query_class=query_class,
            retrieval_mode="naive",
            answer_mode="hybrid",
            top_k=8,
            chunk_top_k=4,
            max_entity_tokens=1000,
            max_relation_tokens=500,
            max_total_tokens=6000,
            final_top_n=3,
            stream=stream,
        )
    if query_class == QueryClass.DIRECT_RULE:
        return RetrievalPolicy(
            query_class=query_class,
            retrieval_mode="naive",
            answer_mode="hybrid",
            top_k=12,
            chunk_top_k=6,
            max_entity_tokens=1500,
            max_relation_tokens=1000,
            max_total_tokens=8000,
            final_top_n=4,
            stream=stream,
        )
    if query_class == QueryClass.RELATIONAL:
        return RetrievalPolicy(
            query_class=query_class,
            retrieval_mode="hybrid",
            answer_mode="hybrid",
            top_k=30,
            chunk_top_k=15,
            max_entity_tokens=4000,
            max_relation_tokens=5000,
            max_total_tokens=16000,
            final_top_n=6,
            stream=stream,
        )
    return RetrievalPolicy(
        query_class=query_class,
        retrieval_mode="hybrid",
        answer_mode="hybrid",
        top_k=20,
        chunk_top_k=10,
        max_entity_tokens=2500,
        max_relation_tokens=3000,
        max_total_tokens=12000,
        final_top_n=5,
        stream=stream,
    )
```

- [ ] **Step 4: Run router tests to verify pass**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_query_router.py -v
```

Expected: all router tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
git add backend/core/query_router.py backend/tests/test_query_router.py
git commit -m "feat: add legal query router"
```

---

### Task 3: Add Legal Relevance Gate

**Files:**
- Create: `backend/core/legal_relevance.py`
- Create: `backend/tests/test_legal_relevance.py`

- [ ] **Step 1: Write failing legal relevance tests**

Create `backend/tests/test_legal_relevance.py`:

```python
from backend.core.legal_relevance import (
    CandidateLabel,
    ContextCandidate,
    label_candidate,
    render_curated_context,
    select_relevant_candidates,
    split_lightrag_context,
)
from backend.core.query_router import QueryClass


QUESTION = "Hành lang an toàn đường bộ được xác định từ đâu và việc thiết lập hành lang này nhằm mục đích gì?"


def test_label_direct_definition_candidate_as_direct():
    candidate = ContextCandidate(
        id="c1",
        text="Điều 2 khoản 5: Hành lang an toàn đường bộ là dải đất dọc hai bên đất của đường bộ, tính từ mép ngoài phần đất để bảo vệ, bảo trì đường bộ ra hai bên để bảo đảm an toàn giao thông đường bộ.",
        source="Luật Đường bộ 35/2024/QH15",
        score=0.9,
    )

    assert label_candidate(QUESTION, candidate, QueryClass.DIRECT_DEFINITION) == CandidateLabel.DIRECT


def test_demotes_constitution_and_policy_background_for_direct_definition():
    candidates = [
        ContextCandidate(id="direct", text="Điều 2 khoản 5: Hành lang an toàn đường bộ là dải đất dọc hai bên đất của đường bộ.", source="Luật Đường bộ", score=0.95),
        ContextCandidate(id="constitution", text="Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam là căn cứ ban hành Luật Đường bộ.", source="Hiến pháp", score=0.93),
        ContextCandidate(id="policy", text="Điều 4. Chính sách phát triển kết cấu hạ tầng đường bộ.", source="Luật Đường bộ", score=0.92),
    ]

    selected = select_relevant_candidates(
        question=QUESTION,
        candidates=candidates,
        query_class=QueryClass.DIRECT_DEFINITION,
        final_top_n=3,
    )

    assert [item.id for item in selected] == ["direct"]


def test_allows_background_when_user_explicitly_asks_for_scope():
    candidate = ContextCandidate(
        id="scope",
        text="Điều 1. Phạm vi điều chỉnh của Luật Đường bộ.",
        source="Luật Đường bộ",
        score=0.8,
    )

    assert label_candidate(
        "Điều 1 quy định phạm vi điều chỉnh như thế nào?",
        candidate,
        QueryClass.DIRECT_RULE,
    ) != CandidateLabel.BACKGROUND


def test_relational_query_allows_supporting_candidates():
    question = "Quy định về hành lang an toàn đường bộ liên quan đến trách nhiệm của cơ quan nào?"
    candidates = [
        ContextCandidate(id="direct", text="Hành lang an toàn đường bộ là dải đất dọc hai bên đất của đường bộ.", source="Luật Đường bộ", score=0.95),
        ContextCandidate(id="supporting", text="Trách nhiệm của cơ quan quản lý đường bộ bao gồm quản lý, bảo vệ kết cấu hạ tầng đường bộ.", source="Luật Đường bộ", score=0.9),
    ]

    selected = select_relevant_candidates(
        question=question,
        candidates=candidates,
        query_class=QueryClass.RELATIONAL,
        final_top_n=6,
    )

    assert [item.id for item in selected] == ["direct", "supporting"]


def test_split_lightrag_context_returns_non_empty_candidates():
    context = "-----Document 1-----\nĐiều 2 khoản 5: Hành lang an toàn đường bộ là dải đất.\n-----Document 2-----\nĐiều 4. Chính sách phát triển."

    candidates = split_lightrag_context(context)

    assert len(candidates) == 2
    assert candidates[0].id == "ctx-1"
    assert "Điều 2" in candidates[0].text


def test_split_lightrag_context_handles_leading_separator():
    context = "-----Chunks-----\nĐiều 2 khoản 5: Hành lang an toàn đường bộ là dải đất.\n\nĐiều 3. Nguyên tắc hoạt động đường bộ."

    candidates = split_lightrag_context(context)

    assert candidates
    assert "Điều 2" in candidates[0].text


def test_short_vietnamese_legal_terms_are_not_dropped():
    candidate = ContextCandidate(
        id="fine",
        text="Người điều khiển xe máy vi phạm nồng độ cồn bị phạt tiền.",
        source="Nghị định xử phạt",
        score=0.8,
    )

    assert label_candidate(
        "Xe máy bị phạt bao nhiêu?",
        candidate,
        QueryClass.DIRECT_RULE,
    ) == CandidateLabel.DIRECT


def test_render_curated_context_numbers_candidates():
    candidates = [
        ContextCandidate(id="c1", text="Điều 2 khoản 5: Hành lang an toàn đường bộ là dải đất.", source="Luật Đường bộ", score=0.8),
    ]

    rendered = render_curated_context(candidates)

    assert "[1]" in rendered
    assert "Luật Đường bộ" in rendered
    assert "Điều 2 khoản 5" in rendered
```

- [ ] **Step 2: Run legal relevance tests to verify failure**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_legal_relevance.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.core.legal_relevance'`.

- [ ] **Step 3: Implement relevance gate**

Create `backend/core/legal_relevance.py`:

```python
"""Legal relevance filtering for retrieved LightRAG context."""

import re
from dataclasses import dataclass
from enum import StrEnum

from backend.core.query_router import QueryClass


class CandidateLabel(StrEnum):
    DIRECT = "direct"
    SUPPORTING = "supporting"
    BACKGROUND = "background"
    NOISE = "noise"


@dataclass(frozen=True)
class ContextCandidate:
    id: str
    text: str
    source: str = ""
    score: float = 0.0
    label: CandidateLabel | None = None


_BACKGROUND_PATTERNS = (
    "hiến pháp",
    "phạm vi điều chỉnh",
    "chính sách phát triển",
    "chính sách chung",
    "căn cứ ban hành",
    "quản lý nhà nước",
    "điều 1",
    "điều 4",
)

_BACKGROUND_REQUEST_PATTERNS = (
    "phạm vi điều chỉnh",
    "chính sách",
    "hiến pháp",
    "căn cứ ban hành",
    "điều 1",
    "điều 4",
)

_DIRECT_DEFINITION_PATTERNS = (
    "là ",
    "được hiểu là",
    "có nghĩa là",
    "được xác định",
    "tính từ",
    "nhằm",
    "để bảo đảm",
)

_SUPPORTING_PATTERNS = (
    "trách nhiệm",
    "thẩm quyền",
    "điều kiện",
    "ngoại lệ",
    "chế tài",
    "xử phạt",
    "nghĩa vụ",
    "quản lý",
    "bảo vệ",
)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _looks_background(text: str) -> bool:
    normalized = _normalize(text)
    return any(pattern in normalized for pattern in _BACKGROUND_PATTERNS)


def _question_requests_background(question: str) -> bool:
    normalized = _normalize(question)
    return any(pattern in normalized for pattern in _BACKGROUND_REQUEST_PATTERNS)


def _shares_key_terms(question: str, candidate_text: str) -> bool:
    stopwords = {
        "những",
        "được",
        "như",
        "thế",
        "nào",
        "bao",
        "nhiêu",
        "của",
        "và",
        "cho",
        "người",
    }
    question_terms = {
        token
        for token in re.findall(r"[\wÀ-ỹ]+", _normalize(question))
        if len(token) >= 3 and token not in stopwords
    }
    candidate_terms = set(re.findall(r"[\wÀ-ỹ]+", _normalize(candidate_text)))
    return bool(question_terms & candidate_terms)


def label_candidate(
    question: str,
    candidate: ContextCandidate,
    query_class: QueryClass,
) -> CandidateLabel:
    text = _normalize(f"{candidate.source} {candidate.text}")

    if (
        _looks_background(text)
        and query_class in {QueryClass.DIRECT_DEFINITION, QueryClass.DIRECT_RULE}
        and not _question_requests_background(question)
    ):
        return CandidateLabel.BACKGROUND

    if not _shares_key_terms(question, candidate.text):
        return CandidateLabel.NOISE

    if query_class == QueryClass.DIRECT_DEFINITION:
        if any(pattern in text for pattern in _DIRECT_DEFINITION_PATTERNS):
            return CandidateLabel.DIRECT
        return CandidateLabel.SUPPORTING

    if query_class == QueryClass.DIRECT_RULE:
        if any(pattern in text for pattern in _SUPPORTING_PATTERNS) or "phạt" in text:
            return CandidateLabel.DIRECT
        return CandidateLabel.SUPPORTING

    if any(pattern in text for pattern in _SUPPORTING_PATTERNS):
        return CandidateLabel.SUPPORTING
    return CandidateLabel.DIRECT


def select_relevant_candidates(
    question: str,
    candidates: list[ContextCandidate],
    query_class: QueryClass,
    final_top_n: int,
) -> list[ContextCandidate]:
    labeled: list[ContextCandidate] = []
    for candidate in candidates:
        label = label_candidate(question, candidate, query_class)
        if label in {CandidateLabel.DIRECT, CandidateLabel.SUPPORTING}:
            labeled.append(
                ContextCandidate(
                    id=candidate.id,
                    text=candidate.text,
                    source=candidate.source,
                    score=candidate.score,
                    label=label,
                )
            )

    direct = [item for item in labeled if item.label == CandidateLabel.DIRECT]
    supporting = [item for item in labeled if item.label == CandidateLabel.SUPPORTING]

    if query_class in {QueryClass.DIRECT_DEFINITION, QueryClass.DIRECT_RULE} and not direct:
        return []

    ordered = sorted(direct, key=lambda item: item.score, reverse=True)
    if query_class not in {QueryClass.DIRECT_DEFINITION, QueryClass.DIRECT_RULE}:
        ordered.extend(sorted(supporting, key=lambda item: item.score, reverse=True))
    else:
        remaining_slots = max(final_top_n - len(ordered), 0)
        ordered.extend(sorted(supporting, key=lambda item: item.score, reverse=True)[:remaining_slots])

    return ordered[:final_top_n]


def split_lightrag_context(context: str) -> list[ContextCandidate]:
    stripped = context.strip()
    if not stripped:
        return []

    separator_re = r"(?:^|\n)-{3,}[^\n]*-{3,}(?:\n|$)"
    parts = [
        part.strip()
        for part in re.split(separator_re, stripped)
        if part.strip()
    ]
    if len(parts) <= 1:
        parts = [part.strip() for part in re.split(r"\n\s*\n", stripped) if part.strip()]

    return [
        ContextCandidate(id=f"ctx-{index}", text=part, source=_infer_source(part))
        for index, part in enumerate(parts, start=1)
    ]


def _infer_source(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    if "luật" in first_line.casefold() or "điều" in first_line.casefold():
        return first_line[:160]
    return ""


def render_curated_context(candidates: list[ContextCandidate]) -> str:
    if not candidates:
        return "Không tìm thấy căn cứ trực tiếp đủ liên quan trong ngữ cảnh truy xuất."

    sections = []
    for index, candidate in enumerate(candidates, start=1):
        source = f" ({candidate.source})" if candidate.source else ""
        sections.append(f"[{index}]{source}\n{candidate.text.strip()}")
    return "\n\n".join(sections)
```

- [ ] **Step 4: Run legal relevance tests to verify pass**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_legal_relevance.py -v
```

Expected: all legal relevance tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
git add backend/core/legal_relevance.py backend/tests/test_legal_relevance.py
git commit -m "feat: add legal relevance gate"
```

---

### Task 4: Add Reranker Configuration and Wrapper

**Files:**
- Modify: `backend/config.py`
- Modify: `backend/requirements.txt`
- Create: `backend/core/reranker.py`
- Create: `backend/tests/test_reranker.py`

- [ ] **Step 1: Write failing reranker tests**

Create `backend/tests/test_reranker.py`:

```python
from backend.core.legal_relevance import ContextCandidate
from backend.core.reranker import NoOpReranker, RerankerConfig, rerank_candidates


def test_noop_reranker_preserves_candidate_order_and_scores():
    reranker = NoOpReranker()
    candidates = [
        ContextCandidate(id="a", text="A", score=0.1),
        ContextCandidate(id="b", text="B", score=0.2),
    ]

    result = reranker.rerank("question", candidates)

    assert result == candidates


def test_rerank_candidates_sorts_by_mock_scores():
    class FakeReranker:
        def rerank(self, question, candidates):
            return [
                ContextCandidate(id="b", text="B", score=0.9),
                ContextCandidate(id="a", text="A", score=0.2),
            ]

    candidates = [
        ContextCandidate(id="a", text="A"),
        ContextCandidate(id="b", text="B"),
    ]

    result = rerank_candidates("question", candidates, FakeReranker())

    assert [item.id for item in result] == ["b", "a"]
    assert result[0].score == 0.9


def test_reranker_config_defaults_to_bge_model():
    config = RerankerConfig()

    assert config.enabled is True
    assert config.model == "BAAI/bge-reranker-v2-m3"
    assert config.max_length == 1024
```

- [ ] **Step 2: Run reranker tests to verify failure**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_reranker.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.core.reranker'`.

- [ ] **Step 3: Add config settings**

Add these fields to `Settings` in `backend/config.py`:

```python
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_MAX_LENGTH: int = 1024
    RERANKER_DEVICE: str = "auto"
    RERANKER_TOP_N_DIRECT: int = 3
    RERANKER_TOP_N_RELATIONAL: int = 6
```

- [ ] **Step 4: Add reranker dependency**

Add this line to `backend/requirements.txt` near the model/runtime dependencies:

```text
FlagEmbedding>=1.2.11
```

The model documentation for `BAAI/bge-reranker-v2-m3` uses `FlagReranker` from `FlagEmbedding`; the implementation must call `compute_score(..., max_length=1024)` through the configurable `RERANKER_MAX_LENGTH`.

- [ ] **Step 5: Implement reranker wrapper**

Create `backend/core/reranker.py`:

```python
"""Reranker wrappers for legal context candidates."""

from dataclasses import dataclass
from typing import Protocol

from backend.config import settings
from backend.core.legal_relevance import ContextCandidate


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = True
    model: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = 1024
    device: str = "auto"


class Reranker(Protocol):
    def rerank(self, question: str, candidates: list[ContextCandidate]) -> list[ContextCandidate]:
        ...


class NoOpReranker:
    def rerank(self, question: str, candidates: list[ContextCandidate]) -> list[ContextCandidate]:
        return candidates


class BGEReranker:
    def __init__(self, config: RerankerConfig):
        self.config = config
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                raise RuntimeError(
                    "FlagEmbedding is required for BAAI/bge-reranker-v2-m3. "
                    "Install backend requirements or set RERANKER_ENABLED=false."
                ) from exc

            use_fp16 = self.config.device != "cpu"
            self._model = FlagReranker(self.config.model, use_fp16=use_fp16)
        return self._model

    def rerank(self, question: str, candidates: list[ContextCandidate]) -> list[ContextCandidate]:
        if not candidates:
            return []

        model = self._load_model()
        pairs = [
            [question, candidate.text[: self.config.max_length * 4]]
            for candidate in candidates
        ]
        scores = model.compute_score(
            pairs,
            normalize=True,
            max_length=self.config.max_length,
        )
        if isinstance(scores, float):
            scores = [scores]

        rescored = [
            ContextCandidate(
                id=candidate.id,
                text=candidate.text,
                source=candidate.source,
                score=float(score),
                label=candidate.label,
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sorted(rescored, key=lambda item: item.score, reverse=True)


def build_reranker() -> Reranker:
    if not settings.RERANKER_ENABLED:
        return NoOpReranker()
    return BGEReranker(
        RerankerConfig(
            enabled=settings.RERANKER_ENABLED,
            model=settings.RERANKER_MODEL,
            max_length=settings.RERANKER_MAX_LENGTH,
            device=settings.RERANKER_DEVICE,
        )
    )


def rerank_candidates(
    question: str,
    candidates: list[ContextCandidate],
    reranker: Reranker,
) -> list[ContextCandidate]:
    return reranker.rerank(question, candidates)
```

- [ ] **Step 6: Run reranker tests to verify pass**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_reranker.py -v
```

Expected: all reranker tests pass without loading the real BAAI model.

- [ ] **Step 7: Commit**

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
git add backend/config.py backend/requirements.txt backend/core/reranker.py backend/tests/test_reranker.py
git commit -m "feat: add configurable legal reranker"
```

---

### Task 5: Add Controlled Chat Orchestrator

**Files:**
- Create: `backend/core/controlled_chat.py`
- Create: `backend/tests/test_controlled_chat.py`

- [ ] **Step 1: Write failing controlled chat tests**

Create `backend/tests/test_controlled_chat.py`:

```python
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import backend.core.controlled_chat as controlled_chat
from backend.core.controlled_chat import answer_controlled_chat
from backend.core.legal_relevance import ContextCandidate
from backend.core.query_router import QueryClass


class FakeReranker:
    def rerank(self, question, candidates):
        return [
            ContextCandidate(id=candidate.id, text=candidate.text, source=candidate.source, score=1.0 - index / 10)
            for index, candidate in enumerate(candidates)
        ]


@pytest.mark.asyncio
async def test_direct_definition_uses_context_retrieval_before_final_answer(monkeypatch):
    monkeypatch.setattr(
        controlled_chat,
        "classify_query",
        lambda message: QueryClass.DIRECT_DEFINITION,
    )

    async def fake_aquery(query, param=None, system_prompt=None):
        if param.only_need_context:
            assert param.mode == "naive"
            assert system_prompt is None
            return "Điều 2 khoản 5: Hành lang an toàn đường bộ là dải đất dọc hai bên đất của đường bộ.\n\nĐiều 4. Chính sách phát triển kết cấu hạ tầng đường bộ."
        assert param.mode == "bypass"
        assert "Điều 2 khoản 5" in query
        assert "Điều 4. Chính sách" not in query
        return "Căn cứ chính\nĐiều 2 khoản 5..."

    rag = SimpleNamespace(aquery=AsyncMock(side_effect=fake_aquery))

    result = await answer_controlled_chat(
        rag=rag,
        message="Hành lang an toàn đường bộ được xác định từ đâu và nhằm mục đích gì?",
        stream=False,
        reranker=FakeReranker(),
    )

    assert result == "Căn cứ chính\nĐiều 2 khoản 5..."
    assert rag.aquery.await_count == 2


@pytest.mark.asyncio
async def test_relational_query_uses_hybrid_context_retrieval(monkeypatch):
    monkeypatch.setattr(
        controlled_chat,
        "classify_query",
        lambda message: QueryClass.RELATIONAL,
    )

    async def fake_aquery(query, param=None, system_prompt=None):
        if param.only_need_context:
            assert param.mode == "hybrid"
            assert system_prompt is None
            return "Trách nhiệm của cơ quan quản lý đường bộ bao gồm quản lý, bảo vệ kết cấu hạ tầng đường bộ."
        assert param.mode == "bypass"
        return "Căn cứ chính\nTrách nhiệm..."

    rag = SimpleNamespace(aquery=AsyncMock(side_effect=fake_aquery))

    result = await answer_controlled_chat(
        rag=rag,
        message="Quy định về hành lang an toàn đường bộ liên quan đến trách nhiệm của cơ quan nào?",
        stream=False,
        reranker=FakeReranker(),
    )

    assert "Trách nhiệm" in result
    assert rag.aquery.await_count == 2
```

- [ ] **Step 2: Run controlled chat tests to verify failure**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_controlled_chat.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.core.controlled_chat'`.

- [ ] **Step 3: Implement controlled chat service**

Create `backend/core/controlled_chat.py`:

```python
"""Controlled retrieval and answer generation for legal chat."""

from collections.abc import AsyncIterator

from lightrag import QueryParam

from backend.core.legal_prompts import build_curated_answer_system_prompt
from backend.core.legal_relevance import (
    render_curated_context,
    select_relevant_candidates,
    split_lightrag_context,
)
from backend.core.query_router import classify_query, retrieval_policy_for
from backend.core.reranker import Reranker, build_reranker, rerank_candidates


def _build_retrieval_param(policy) -> QueryParam:
    return QueryParam(
        mode=policy.retrieval_mode,
        only_need_context=True,
        top_k=policy.top_k,
        chunk_top_k=policy.chunk_top_k,
        max_entity_tokens=policy.max_entity_tokens,
        max_relation_tokens=policy.max_relation_tokens,
        max_total_tokens=policy.max_total_tokens,
        stream=False,
    )


def _build_answer_query(message: str, curated_context: str) -> str:
    return (
        "Câu hỏi của người dùng:\n"
        f"{message}\n\n"
        "Ngữ cảnh pháp lý đã được chọn lọc:\n"
        f"{curated_context}"
    )


async def answer_controlled_chat(
    rag,
    message: str,
    stream: bool = False,
    reranker: Reranker | None = None,
) -> str | AsyncIterator[str]:
    query_class = classify_query(message)
    policy = retrieval_policy_for(query_class, stream=stream)
    active_reranker = reranker or build_reranker()

    raw_context = await rag.aquery(
        message,
        param=_build_retrieval_param(policy),
        system_prompt=None,
    )

    candidates = split_lightrag_context(str(raw_context))
    reranked = rerank_candidates(message, candidates, active_reranker)
    selected = select_relevant_candidates(
        question=message,
        candidates=reranked,
        query_class=query_class,
        final_top_n=policy.final_top_n,
    )
    curated_context = render_curated_context(selected)

    return await rag.aquery(
        _build_answer_query(message, curated_context),
        param=QueryParam(mode="bypass", stream=stream),
        system_prompt=build_curated_answer_system_prompt(),
    )
```

- [ ] **Step 4: Run controlled chat tests to verify pass**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_controlled_chat.py -v
```

Expected: all controlled chat tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
git add backend/core/controlled_chat.py backend/tests/test_controlled_chat.py
git commit -m "feat: add controlled legal chat pipeline"
```

---

### Task 6: Wire Controlled Pipeline into Chat Route

**Files:**
- Modify: `backend/api/routes.py`
- Modify: `backend/tests/test_chat_legal_prompt_integration.py`
- Modify: `backend/tests/test_chat_route_provider_split.py`

- [ ] **Step 1: Add failing route integration test for controlled hybrid path**

Append this test to `backend/tests/test_chat_legal_prompt_integration.py`:

```python
def test_single_hybrid_chat_uses_controlled_pipeline(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="unused direct rag answer"))
    controlled = AsyncMock(return_value="Controlled answer")

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Hành lang an toàn đường bộ được xác định từ đâu?", "stream": False, "comparison_mode": False},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "Controlled answer"
    controlled.assert_awaited_once_with(
        rag=fake_rag,
        message="Hành lang an toàn đường bộ được xác định từ đâu?",
        stream=False,
    )
    fake_rag.aquery.assert_not_awaited()
```

- [ ] **Step 2: Add failing streaming route test for controlled hybrid path**

Append this test to `backend/tests/test_chat_route_provider_split.py`:

```python
def test_chat_streaming_uses_controlled_pipeline_and_keeps_sse_shape(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="unused"))

    async def controlled_stream(**kwargs):
        async def gen():
            yield "controlled-1"
        return gen()

    controlled = AsyncMock(side_effect=controlled_stream)

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Cho tôi câu trả lời", "stream": True, "comparison_mode": False},
    )

    assert response.status_code == 200
    assert 'data: {"type": "chunk", "mode": "hybrid", "content": "controlled-1"}' in response.text
    assert 'data: {"type": "done"}' in response.text
    controlled.assert_awaited_once()
```

- [ ] **Step 3: Run route tests to verify failure**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_chat_legal_prompt_integration.py backend/tests/test_chat_route_provider_split.py -v
```

Expected: FAIL because `backend.api.routes` has no `answer_controlled_chat` import and still calls `rag.aquery()` directly for single hybrid chat.

- [ ] **Step 4: Wire non-comparison hybrid route through controlled service**

Modify imports in `backend/api/routes.py`:

```python
from backend.core.controlled_chat import answer_controlled_chat
```

In the non-streaming single-response branch, replace the direct hybrid `rag.aquery(...)` call with:

```python
                response = await answer_controlled_chat(
                    rag=rag,
                    message=request.message,
                    stream=False,
                )
                return ChatResponse(response=response, mode="hybrid")
```

In the streaming single-response branch, replace the direct hybrid `rag.aquery(...)` call with:

```python
                generator = await answer_controlled_chat(
                    rag=rag,
                    message=request.message,
                    stream=True,
                )
                if hasattr(generator, '__aiter__'):
                    async for chunk in generator:
                        yield f"data: {json.dumps({'type': 'chunk', 'mode': 'hybrid', 'content': chunk})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'chunk', 'mode': 'hybrid', 'content': str(generator)})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
```

Keep comparison mode unchanged in this task so Naive vs current Hybrid remains available while the controlled single-chat path is stabilized. A later task will switch comparison hybrid after tests cover it.

- [ ] **Step 5: Update existing single-chat route test expectations**

In `backend/tests/test_chat_legal_prompt_integration.py`, update `test_chat_passes_hybrid_legal_prompt_for_non_streaming_requests` and `test_chat_passes_hybrid_legal_prompt_for_streaming_requests` so they patch `answer_controlled_chat` instead of asserting direct `rag.aquery()` for the single hybrid path:

```python
def test_chat_passes_hybrid_legal_prompt_for_non_streaming_requests(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="unused"))
    controlled = AsyncMock(return_value="Câu trả lời")

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Tốc độ tối đa là bao nhiêu?", "stream": False, "comparison_mode": False},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "Câu trả lời"
    controlled.assert_awaited_once_with(
        rag=fake_rag,
        message="Tốc độ tối đa là bao nhiêu?",
        stream=False,
    )
```

```python
def test_chat_passes_hybrid_legal_prompt_for_streaming_requests(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="unused"))

    async def controlled_stream(**kwargs):
        async def gen():
            yield "chunk-1"
        return gen()

    controlled = AsyncMock(side_effect=controlled_stream)

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Cho tôi câu trả lời", "stream": True, "comparison_mode": False},
    )

    assert response.status_code == 200
    assert 'data: {"type": "chunk", "mode": "hybrid", "content": "chunk-1"}' in response.text
    assert 'data: {"type": "done"}' in response.text
    controlled.assert_awaited_once_with(
        rag=fake_rag,
        message="Cho tôi câu trả lời",
        stream=True,
    )
```

In `backend/tests/test_chat_route_provider_split.py`, replace `test_chat_uses_query_rag_for_non_streaming_requests` with:

```python
def test_chat_uses_query_rag_for_non_streaming_requests(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="unused"))
    controlled = AsyncMock(return_value="Câu trả lời")

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Tốc độ tối đa là bao nhiêu?", "stream": False, "comparison_mode": False},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "Câu trả lời"
    controlled.assert_awaited_once_with(
        rag=fake_rag,
        message="Tốc độ tối đa là bao nhiêu?",
        stream=False,
    )
    fake_rag.aquery.assert_not_awaited()
```

In `backend/tests/test_chat_route_provider_split.py`, replace `test_chat_streaming_keeps_sse_event_shape` with:

```python
def test_chat_streaming_keeps_sse_event_shape(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="unused"))

    async def controlled_stream(**kwargs):
        async def gen():
            yield "chunk-1"
            yield "chunk-2"
        return gen()

    controlled = AsyncMock(side_effect=controlled_stream)

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Cho tôi câu trả lời", "stream": True, "comparison_mode": False},
    )

    assert response.status_code == 200
    assert 'data: {"type": "chunk", "mode": "hybrid", "content": "chunk-1"}' in response.text
    assert 'data: {"type": "done"}' in response.text
    controlled.assert_awaited_once_with(
        rag=fake_rag,
        message="Cho tôi câu trả lời",
        stream=True,
    )
```

- [ ] **Step 6: Run route tests to verify pass**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_chat_legal_prompt_integration.py backend/tests/test_chat_route_provider_split.py -v
```

Expected: route tests pass and SSE event shape remains unchanged.

- [ ] **Step 7: Commit**

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
git add backend/api/routes.py backend/tests/test_chat_legal_prompt_integration.py backend/tests/test_chat_route_provider_split.py
git commit -m "feat: route single hybrid chat through controlled pipeline"
```

---

### Task 7: Switch Comparison Hybrid Column to Controlled Pipeline

**Files:**
- Modify: `backend/api/routes.py`
- Modify: `backend/tests/test_chat_legal_prompt_integration.py`

- [ ] **Step 1: Add failing comparison test**

Append this test to `backend/tests/test_chat_legal_prompt_integration.py`:

```python
def test_comparison_uses_naive_direct_call_and_controlled_hybrid(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="Naive answer"))
    controlled = AsyncMock(return_value="Controlled hybrid answer")

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Hành lang an toàn đường bộ được xác định từ đâu?", "stream": False, "comparison_mode": True},
    )

    assert response.status_code == 200
    assert response.json()["naive"]["response"] == "Naive answer"
    assert response.json()["hybrid"]["response"] == "Controlled hybrid answer"
    _assert_legal_query_call(
        _call_by_mode(fake_rag.aquery, "naive"),
        "Hành lang an toàn đường bộ được xác định từ đâu?",
        "naive",
        build_legal_system_prompt("naive"),
    )
    controlled.assert_awaited_once_with(
        rag=fake_rag,
        message="Hành lang an toàn đường bộ được xác định từ đâu?",
        stream=False,
    )
```

- [ ] **Step 2: Run comparison test to verify failure**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_chat_legal_prompt_integration.py::test_comparison_uses_naive_direct_call_and_controlled_hybrid -v
```

Expected: FAIL because comparison mode still calls direct `rag.aquery(... mode="hybrid")`.

- [ ] **Step 3: Update non-streaming comparison hybrid branch**

In `backend/api/routes.py`, replace the hybrid comparison `rag.aquery(...)` call with:

```python
                hybrid_response = await answer_controlled_chat(
                    rag=rag,
                    message=request.message,
                    stream=False,
                )
```

Keep the naive comparison branch unchanged.

- [ ] **Step 4: Update streaming comparison hybrid branch**

In the streaming comparison branch, replace the hybrid `stream_wrapper(rag.aquery(...), "hybrid")` task with:

```python
                    stream_wrapper(
                        answer_controlled_chat(
                            rag=rag,
                            message=request.message,
                            stream=True,
                        ),
                        "hybrid",
                    )
```

Keep the naive streaming comparison branch unchanged.

- [ ] **Step 5: Update existing comparison test expectations**

In `backend/tests/test_chat_legal_prompt_integration.py`, replace `test_chat_passes_mode_specific_legal_prompts_for_comparison_requests` with:

```python
def test_chat_passes_mode_specific_legal_prompts_for_comparison_requests(monkeypatch):
    client = make_client()
    fake_rag = SimpleNamespace(aquery=AsyncMock(return_value="Naive answer"))
    controlled = AsyncMock(return_value="Hybrid answer")

    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Quy định nồng độ cồn thế nào?", "stream": False, "comparison_mode": True},
    )

    assert response.status_code == 200
    assert response.json()["naive"]["response"] == "Naive answer"
    assert response.json()["hybrid"]["response"] == "Hybrid answer"
    _assert_legal_query_call(
        _call_by_mode(fake_rag.aquery, "naive"),
        "Quy định nồng độ cồn thế nào?",
        "naive",
        build_legal_system_prompt("naive"),
    )
    controlled.assert_awaited_once_with(
        rag=fake_rag,
        message="Quy định nồng độ cồn thế nào?",
        stream=False,
    )
```

In `backend/tests/test_chat_legal_prompt_integration.py`, replace `test_chat_passes_mode_specific_legal_prompts_for_streaming_comparison_requests` with:

```python
def test_chat_passes_mode_specific_legal_prompts_for_streaming_comparison_requests(monkeypatch):
    client = make_client()

    async def naive_stream():
        yield "naive-1"

    async def hybrid_stream():
        yield "hybrid-1"

    async def fake_aquery(query, param=None, system_prompt=None):
        if param.mode == "naive":
            return naive_stream()
        raise AssertionError(f"Unexpected direct rag mode: {param.mode!r}")

    controlled = AsyncMock(return_value=hybrid_stream())
    fake_rag = SimpleNamespace(aquery=AsyncMock(side_effect=fake_aquery))
    monkeypatch.setattr(routes.RAGEngine, "get_query_instance", lambda: fake_rag)
    monkeypatch.setattr(routes, "answer_controlled_chat", controlled)

    response = client.post(
        "/api/chat",
        json={"message": "Quy định nồng độ cồn thế nào?", "stream": True, "comparison_mode": True},
    )

    assert response.status_code == 200
    assert 'data: {"type": "chunk", "mode": "naive", "content": "naive-1"}' in response.text
    assert 'data: {"type": "chunk", "mode": "hybrid", "content": "hybrid-1"}' in response.text
    assert 'data: {"type": "done"}' in response.text
    _assert_legal_query_call(
        _call_by_mode(fake_rag.aquery, "naive"),
        "Quy định nồng độ cồn thế nào?",
        "naive",
        build_legal_system_prompt("naive"),
        stream=True,
    )
    controlled.assert_awaited_once_with(
        rag=fake_rag,
        message="Quy định nồng độ cồn thế nào?",
        stream=True,
    )
```

- [ ] **Step 6: Run comparison route tests**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_chat_legal_prompt_integration.py -v
```

Expected: comparison tests pass, with naive direct retrieval and hybrid controlled pipeline.

- [ ] **Step 7: Commit**

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
git add backend/api/routes.py backend/tests/test_chat_legal_prompt_integration.py
git commit -m "feat: use controlled pipeline for comparison hybrid answers"
```

---

### Task 8: Add Benchmark Fixture and Deterministic Checker

**Files:**
- Modify: `data/legal_benchmark.json`
- Modify: `backend/core/benchmark_checker.py`
- Modify: `backend/tests/test_benchmark_checker.py`

- [ ] **Step 1: Add failing benchmark checker tests without removing existing coverage**

Append these tests to `backend/tests/test_benchmark_checker.py`. Keep the existing tests for `check_required_terms`, `check_forbidden_paraphrases`, `check_expected_sources`, and `check_expected_answer_points`.

Also add `check_forbidden_sources` to the existing import list from `backend.core.benchmark_checker`.

```python
def test_check_forbidden_sources_pass():
    answer = "Căn cứ chính: Điều 2 khoản 5 Luật Đường bộ."

    result = check_forbidden_sources(answer, ["Hiến pháp", "Điều 4"])

    assert result["pass"] is True
    assert result["found"] == []


def test_check_forbidden_sources_fail():
    answer = "Liên kết pháp lý liên quan: Hiến pháp là căn cứ ban hành Luật Đường bộ."

    result = check_forbidden_sources(answer, ["Hiến pháp", "Điều 4"])

    assert result["pass"] is False
    assert result["found"] == ["Hiến pháp"]


def test_check_answer_combines_forbidden_sources_with_existing_checks():
    answer = (
        "Hành lang an toàn đường bộ là dải đất dọc hai bên đất của đường bộ.\n\n"
        "**Tham chiếu**\n"
        "- [1] Luật Đường bộ\n\n"
        "Liên kết pháp lý liên quan: Hiến pháp là căn cứ chung."
    )
    item = {
        "required_terms": ["dải đất dọc hai bên đất của đường bộ"],
        "forbidden_paraphrases": [],
        "forbidden_sources": ["Hiến pháp", "Điều 4"],
        "expected_sources": ["Luật Đường bộ"],
        "expected_answer_points": [],
    }

    result = check_answer(answer, item)

    assert result["required_terms"]["pass"] is True
    assert result["forbidden_sources"]["pass"] is False
    assert result["forbidden_sources"]["found"] == ["Hiến pháp"]
    assert result["overall_pass"] is False
```

- [ ] **Step 2: Run benchmark checker tests to verify failure**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_benchmark_checker.py -v
```

Expected: FAIL because `check_forbidden_sources` does not exist and `check_answer` does not yet include a `forbidden_sources` result.

- [ ] **Step 3: Implement benchmark checker**

Modify `backend/core/benchmark_checker.py` without changing the existing public signature `check_answer(answer: str, benchmark_item: dict) -> dict`.

Add this function after `check_forbidden_paraphrases`:

```python
def check_forbidden_sources(answer: str, sources: list[str]) -> dict:
    found = _find_matching_terms(answer, sources)
    return {"pass": not found, "found": found}
```

Update `check_answer` to preserve the old schema and add `forbidden_sources`:

```python
def check_answer(answer: str, benchmark_item: dict) -> dict:
    required_terms = benchmark_item.get("required_terms", [])
    forbidden_paraphrases = [
        *benchmark_item.get("forbidden_paraphrases", []),
        *benchmark_item.get("forbidden_terms", []),
    ]
    expected_sources = [
        *benchmark_item.get("expected_sources", []),
        *benchmark_item.get("required_sources", []),
    ]
    expected_answer_points = benchmark_item.get("expected_answer_points", [])
    forbidden_sources = benchmark_item.get("forbidden_sources", [])

    required_result = check_required_terms(answer, required_terms)
    forbidden_result = check_forbidden_paraphrases(answer, forbidden_paraphrases)
    expected_sources_result = check_expected_sources(answer, expected_sources)
    expected_answer_points_result = check_expected_answer_points(answer, expected_answer_points)
    forbidden_sources_result = check_forbidden_sources(answer, forbidden_sources)

    return {
        "required_terms": required_result,
        "forbidden_paraphrases": forbidden_result,
        "expected_sources": expected_sources_result,
        "expected_answer_points": expected_answer_points_result,
        "forbidden_sources": forbidden_sources_result,
        "overall_pass": (
            required_result["pass"]
            and forbidden_result["pass"]
            and expected_sources_result["pass"]
            and expected_answer_points_result["pass"]
            and forbidden_sources_result["pass"]
        ),
    }
```

- [ ] **Step 4: Add benchmark fixture metadata without deleting existing benchmark items**

Replace `data/legal_benchmark.json` with the same 10-item fixture shape, preserving `category`, `expected_sources`, `forbidden_paraphrases`, and `expected_answer_points`, while adding `id`, `query_class`, and `forbidden_sources`:

```json
[
  {
    "id": "road-safety-corridor-definition",
    "question": "Hành lang an toàn đường bộ được xác định từ đâu và việc thiết lập hành lang này nhằm mục đích gì?",
    "category": "exactness",
    "query_class": "direct_definition",
    "expected_sources": ["Luật Đường bộ"],
    "required_terms": [
      "dải đất dọc hai bên đất của đường bộ",
      "tính từ mép ngoài phần đất để bảo vệ, bảo trì đường bộ",
      "bảo đảm an toàn giao thông đường bộ",
      "bảo đảm tầm nhìn xe chạy",
      "hạn chế ảnh hưởng đến môi trường xung quanh"
    ],
    "forbidden_paraphrases": ["lề đường", "giảm ảnh hưởng", "bảo vệ tầm nhìn"],
    "forbidden_sources": ["Hiến pháp", "Điều 1", "Điều 4"],
    "expected_answer_points": ["tính từ mép ngoài phần đất để bảo vệ, bảo trì đường bộ", "bảo đảm an toàn giao thông đường bộ"]
  },
  {
    "id": "road-land-definition",
    "question": "Đất của đường bộ bao gồm những gì?",
    "category": "exactness",
    "query_class": "direct_definition",
    "expected_sources": ["Luật Đường bộ"],
    "required_terms": ["đất của đường bộ", "phần đất"],
    "forbidden_paraphrases": ["lề đường thay cho đất của đường bộ"],
    "forbidden_sources": ["Hiến pháp"],
    "expected_answer_points": ["xây dựng công trình đường bộ"]
  },
  {
    "id": "traffic-participant-driver-definition",
    "question": "Người điều khiển phương tiện tham gia giao thông đường bộ được định nghĩa như thế nào?",
    "category": "exactness",
    "query_class": "direct_definition",
    "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ"],
    "required_terms": ["người điều khiển phương tiện", "tham gia giao thông"],
    "forbidden_paraphrases": [],
    "forbidden_sources": ["Hiến pháp"],
    "expected_answer_points": ["điều khiển phương tiện tham gia giao thông đường bộ"]
  },
  {
    "id": "driver-definition-highway-incident-responsibility",
    "question": "Người điều khiển phương tiện tham gia giao thông đường bộ được định nghĩa như thế nào và họ có những trách nhiệm gì khi gặp sự cố trên đường cao tốc?",
    "category": "graph_strength",
    "query_class": "relational",
    "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ"],
    "required_terms": ["người điều khiển phương tiện"],
    "forbidden_paraphrases": [],
    "forbidden_sources": [],
    "expected_answer_points": ["định nghĩa người điều khiển", "trách nhiệm khi gặp sự cố", "đường cao tốc"]
  },
  {
    "id": "compare-scope-and-common-prohibitions",
    "question": "Phân biệt phạm vi điều chỉnh giữa Luật Trật tự, an toàn giao thông đường bộ và Luật Đường bộ. Có những hành vi nào bị cấm chung trong cả hai luật này không?",
    "category": "graph_strength",
    "query_class": "relational",
    "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ", "Luật Đường bộ"],
    "required_terms": ["hành vi bị cấm"],
    "forbidden_paraphrases": [],
    "forbidden_sources": [],
    "expected_answer_points": ["phạm vi điều chỉnh Luật 35", "phạm vi điều chỉnh Luật 36", "hành vi bị cấm chung"]
  },
  {
    "id": "road-state-management-responsibility",
    "question": "Cơ quan nào chịu trách nhiệm chính về quản lý nhà nước đối với đường bộ và sự phối hợp giữa Bộ Công an với Bộ Giao thông vận tải được quy định ra sao?",
    "category": "graph_strength",
    "query_class": "relational",
    "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ", "Luật Đường bộ"],
    "required_terms": ["Bộ Công an", "Bộ Giao thông vận tải"],
    "forbidden_paraphrases": [],
    "forbidden_sources": [],
    "expected_answer_points": ["trách nhiệm quản lý nhà nước", "phối hợp giữa hai bộ"]
  },
  {
    "id": "expressway-operation-condition-safety",
    "question": "Điều kiện để đưa đường cao tốc vào khai thác là gì và các quy định này liên quan thế nào đến an toàn giao thông?",
    "category": "graph_strength",
    "query_class": "relational",
    "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ", "Luật Đường bộ"],
    "required_terms": ["đường cao tốc"],
    "forbidden_paraphrases": [],
    "forbidden_sources": [],
    "expected_answer_points": ["điều kiện khai thác", "an toàn giao thông"]
  },
  {
    "id": "traffic-accident-handling-responsibility",
    "question": "Khi xảy ra tai nạn giao thông trên đường bộ, quy trình xử lý và trách nhiệm của các bên liên quan được quy định như thế nào?",
    "category": "graph_strength",
    "query_class": "relational",
    "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ"],
    "required_terms": ["tai nạn giao thông"],
    "forbidden_paraphrases": [],
    "forbidden_sources": [],
    "expected_answer_points": ["quy trình xử lý", "trách nhiệm người điều khiển", "trách nhiệm cơ quan chức năng"]
  },
  {
    "id": "corridor-prohibited-acts-road-land",
    "question": "Hành lang an toàn đường bộ liên quan gì đến các hành vi bị cấm trong phạm vi đất của đường bộ?",
    "category": "graph_strength",
    "query_class": "relational",
    "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ", "Luật Đường bộ"],
    "required_terms": ["hành lang an toàn đường bộ", "đất của đường bộ"],
    "forbidden_paraphrases": ["lề đường"],
    "forbidden_sources": [],
    "expected_answer_points": ["hành vi bị cấm trong hành lang", "liên quan đến đất của đường bộ"]
  },
  {
    "id": "expressway-speed-distance-legal-basis",
    "question": "Quy định về tốc độ và khoảng cách an toàn trên đường cao tốc khác gì so với đường bộ thông thường, và cơ sở pháp lý nào chi phối?",
    "category": "graph_strength",
    "query_class": "relational",
    "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ"],
    "required_terms": ["tốc độ", "khoảng cách an toàn"],
    "forbidden_paraphrases": [],
    "forbidden_sources": [],
    "expected_answer_points": ["quy định tốc độ đường cao tốc", "quy định khoảng cách", "cơ sở pháp lý"]
  }
]
```

- [ ] **Step 5: Run benchmark checker tests to verify pass**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_benchmark_checker.py -v
```

Expected: all benchmark checker tests pass.

- [ ] **Step 6: Commit**

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
git add data/legal_benchmark.json backend/core/benchmark_checker.py backend/tests/test_benchmark_checker.py
git commit -m "test: add legal rag benchmark checks"
```

---

### Task 9: Add Live Regression Script

**Files:**
- Modify: `scripts/run_regression_check.py`
- Modify: `backend/tests/test_run_regression_check.py`

- [ ] **Step 1: Write failing script argument test**

Append this test to `backend/tests/test_run_regression_check.py`. Keep the existing category/filter tests.

```python
from scripts.run_regression_check import build_payload


def test_build_payload_defaults_to_stream_false_and_comparison_mode():
    payload = build_payload("Câu hỏi")

    assert payload == {
        "message": "Câu hỏi",
        "stream": False,
        "comparison_mode": True,
    }
```

- [ ] **Step 2: Run script test to verify failure**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_run_regression_check.py -v
```

Expected: FAIL because the script does not expose `build_payload`.

- [ ] **Step 3: Implement regression script**

Modify `scripts/run_regression_check.py` without removing `--index`, `--category`, or comparison-mode checking.

Add this helper near `_preview`:

```python
def build_payload(question: str) -> dict[str, object]:
    return {
        "message": question,
        "stream": False,
        "comparison_mode": True,
    }
```

Then replace the inline request JSON in `main()`:

```python
                response = client.post(
                    "/api/chat",
                    json=build_payload(question),
                )
```

Update `_print_check()` to report forbidden sources in addition to the existing checks:

```python
    forbidden_sources = check.get("forbidden_sources", {"pass": True, "found": []})
```

Add this block after the forbidden paraphrase print block:

```python
    if not forbidden_sources["pass"]:
        print(f"    forbidden sources found: {forbidden_sources['found']}")
```

- [ ] **Step 4: Run script unit test to verify pass**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest backend/tests/test_run_regression_check.py -v
```

Expected: script unit test passes.

- [ ] **Step 5: Commit**

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
git add scripts/run_regression_check.py backend/tests/test_run_regression_check.py
git commit -m "test: add live legal rag regression script"
```

---

### Task 10: Final Verification

**Files:**
- No code changes expected

- [ ] **Step 1: Run focused backend tests**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m pytest \
  backend/tests/test_legal_prompts.py \
  backend/tests/test_query_router.py \
  backend/tests/test_legal_relevance.py \
  backend/tests/test_reranker.py \
  backend/tests/test_controlled_chat.py \
  backend/tests/test_chat_legal_prompt_integration.py \
  backend/tests/test_chat_route_provider_split.py \
  backend/tests/test_benchmark_checker.py \
  backend/tests/test_run_regression_check.py \
  -v
```

Expected: all listed tests pass.

- [ ] **Step 2: Run syntax check for touched backend modules**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python -m py_compile \
  backend/core/legal_prompts.py \
  backend/core/query_router.py \
  backend/core/legal_relevance.py \
  backend/core/reranker.py \
  backend/core/controlled_chat.py \
  backend/api/routes.py \
  backend/config.py \
  scripts/run_regression_check.py
```

Expected: command exits with code 0 and prints no syntax errors.

- [ ] **Step 3: Optional live regression check**

Only run after the backend, database, embedding server, and required API keys are available.

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
conda run -n legal_rag python scripts/run_regression_check.py --index 0
```

Expected: both comparison checks pass for the first benchmark item; the hybrid answer does not contain `Hiến pháp`, `Điều 1`, or `Điều 4` for the direct-definition case.

- [ ] **Step 4: Review git diff**

Run:

```bash
cd /home/tontide1/coding/traffic-law-assistant-main
git status --short
git diff --stat
```

Expected: changed files match this plan; no `.env*`, generated artifacts, DB dumps, raw datasets, model weights, or frontend payload changes are included.
