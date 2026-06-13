"""Controlled retrieval and answer generation for legal chat."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
import re
import unicodedata

from lightrag import QueryParam

from backend.core.legal_prompts import build_curated_answer_system_prompt
from backend.core.legal_relevance import (
    ContextCandidate,
    render_curated_context,
    select_relevant_candidates,
    split_lightrag_context,
)
from backend.core.query_router import (
    QueryClass,
    RetrievalPolicy,
    classify_query,
    retrieval_policy_for,
)
from backend.core.reranker import Reranker, build_reranker, rerank_candidates

_DIRECT_DEFINITION_MARKERS = (
    " là ",
    " được hiểu là ",
    " có nghĩa là ",
    " được xác định ",
)

_DIRECT_DEFINITION_DRIFT_MARKERS = (
    "theo định nghĩa trên",
    "có thể hiểu",
    "nói cách khác",
    "tức là",
    "mặt đường",
    "được quy hoạch và quản lý",
    "các phương tiện lưu thông",
)


@dataclass(frozen=True)
class ExtractedDefinition:
    source: str
    text: str


def _build_retrieval_param(policy: RetrievalPolicy) -> QueryParam:
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


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.casefold().split())


def _strip_markdown_table_line(text: str) -> str:
    return text.strip().strip("|").strip()


def _sentence_candidates(text: str) -> list[str]:
    normalized_lines = [
        _strip_markdown_table_line(line)
        for line in text.splitlines()
        if _strip_markdown_table_line(line)
    ]
    joined = " ".join(normalized_lines)
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", joined)
        if sentence.strip()
    ] or ([joined.strip()] if joined.strip() else [])


def _split_source_prefix(sentence: str, fallback_source: str) -> tuple[str, str]:
    if ":" not in sentence:
        return fallback_source.strip(), sentence.strip()

    prefix, definition = sentence.split(":", 1)
    normalized_prefix = _normalize(prefix)
    if "điều" not in normalized_prefix and "luật" not in normalized_prefix:
        return fallback_source.strip(), sentence.strip()

    source = prefix.strip() or fallback_source.strip()
    return source, definition.strip()


def _extract_direct_definition(candidates: list[ContextCandidate]) -> ExtractedDefinition | None:
    for candidate in candidates:
        for sentence in _sentence_candidates(candidate.text):
            normalized_sentence = f" {_normalize(sentence)} "
            if not any(marker in normalized_sentence for marker in _DIRECT_DEFINITION_MARKERS):
                continue

            source, definition = _split_source_prefix(sentence, candidate.source)
            if definition:
                return ExtractedDefinition(source=source, text=definition)
    return None


def _clean_definition_text(text: str) -> str:
    cleaned = text.strip().strip("“”\"' ").rstrip(".")
    return cleaned


def _clean_citation_source(source: str) -> str:
    cleaned = " ".join(source.strip().rstrip(":").split())
    cleaned = re.sub(r"^[,;:\s]+", "", cleaned)

    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) >= 2:
        p0_lower = parts[0].lower()
        p1_lower = parts[1].lower()
        law_keywords = ("luật", "nghị định", "thông tư", "hiến pháp", "pháp lệnh", "quyết định", "nghị quyết")
        article_keywords = ("điều", "khoản", "điểm", "chương", "mục")

        if any(kw in p0_lower for kw in law_keywords) and any(kw in p1_lower for kw in article_keywords):
            parts[0], parts[1] = parts[1], parts[0]
            cleaned = ", ".join(parts)

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


def _has_semantic_drift(answer: str, source_context: str) -> bool:
    normalized_answer = _normalize(answer)
    normalized_context = _normalize(source_context)
    return any(
        marker in normalized_answer and marker not in normalized_context
        for marker in _DIRECT_DEFINITION_DRIFT_MARKERS
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

    candidates = split_lightrag_context(str(raw_context)) if raw_context else []
    reranked = rerank_candidates(message, candidates, active_reranker)
    selected = select_relevant_candidates(
        question=message,
        candidates=reranked,
        query_class=query_class,
        final_top_n=policy.final_top_n,
    )
    curated_context = render_curated_context(selected)

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
