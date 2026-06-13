"""Vietnamese legal citation extraction and text formatting helper utilities."""

from dataclasses import dataclass
import re
import unicodedata

from backend.core.legal_relevance import ContextCandidate

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


def normalize_text(text: str) -> str:
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
    normalized_prefix = normalize_text(prefix)
    if "điều" not in normalized_prefix and "luật" not in normalized_prefix:
        return fallback_source.strip(), sentence.strip()

    source = prefix.strip() or fallback_source.strip()
    return source, definition.strip()


def _extract_direct_definition(candidates: list[ContextCandidate]) -> ExtractedDefinition | None:
    for candidate in candidates:
        for sentence in _sentence_candidates(candidate.text):
            normalized_sentence = f" {normalize_text(sentence)} "
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


def render_direct_definition_answer(candidates: list[ContextCandidate]) -> str | None:
    extracted = _extract_direct_definition(candidates)
    if extracted is None:
        return None
    return _render_inline_citation_answer(extracted)


def has_semantic_drift(answer: str, source_context: str) -> bool:
    normalized_answer = normalize_text(answer)
    normalized_context = normalize_text(source_context)
    return any(
        marker in normalized_answer and marker not in normalized_context
        for marker in _DIRECT_DEFINITION_DRIFT_MARKERS
    )
