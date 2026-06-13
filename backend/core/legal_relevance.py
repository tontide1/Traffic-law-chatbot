"""Legal relevance filtering for retrieved LightRAG context."""

import json
import re
import unicodedata
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


_BACKGROUND_REGEXES = (
    re.compile(r"hiến pháp"),
    re.compile(r"phạm vi điều chỉnh"),
    re.compile(r"chính sách phát triển"),
    re.compile(r"chính sách chung"),
    re.compile(r"căn cứ ban hành"),
    re.compile(r"quản lý nhà nước"),
    re.compile(r"\bđiều 1(?!\d)"),
    re.compile(r"\bđiều 4(?!\d)"),
)

_BACKGROUND_REQUEST_REGEXES = (
    re.compile(r"phạm vi điều chỉnh"),
    re.compile(r"chính sách"),
    re.compile(r"hiến pháp"),
    re.compile(r"căn cứ ban hành"),
    re.compile(r"\bđiều 1(?!\d)"),
    re.compile(r"\bđiều 4(?!\d)"),
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
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.casefold().split())


def _looks_background(text: str) -> bool:
    normalized = _normalize(text)
    if (
        re.search(r"\bđiều 2(?!\d)", normalized)
        or re.search(r"\bđiều 3(?!\d)", normalized)
    ):
        return False
    return any(pattern.search(normalized) for pattern in _BACKGROUND_REGEXES)


def _question_requests_background(question: str) -> bool:
    normalized = _normalize(question)
    return any(pattern.search(normalized) for pattern in _BACKGROUND_REQUEST_REGEXES)


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
        if len(token) >= 2 and token not in stopwords
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
        if supporting:
            return sorted(supporting, key=lambda item: item.score, reverse=True)[:final_top_n]
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

    cleaned_parts = [
        cleaned
        for part in parts
        for cleaned in _unwrap_context_payload(part)
        if cleaned
    ]

    return [
        ContextCandidate(id=f"ctx-{index}", text=part, source=_infer_source(part))
        for index, part in enumerate(cleaned_parts, start=1)
    ]


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if not lines:
        return stripped
    if lines[-1].strip() == "```":
        lines = lines[1:-1]
    else:
        lines = lines[1:]
    return "\n".join(lines).strip()


def _extract_content_items(payload: object) -> list[str]:
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            return [content.strip()]
        return []

    if isinstance(payload, list):
        return [
            item["content"].strip()
            for item in payload
            if (
                isinstance(item, dict)
                and isinstance(item.get("content"), str)
                and item["content"].strip()
            )
        ]

    return []


def _extract_consecutive_json_contents(text: str) -> list[str]:
    decoder = json.JSONDecoder()
    contents: list[str] = []
    index = 0

    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break

        try:
            payload, next_index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return []

        contents.extend(_extract_content_items(payload))
        index = next_index

    return contents


def _unwrap_context_payload(text: str) -> list[str]:
    stripped = _strip_json_fence(text)
    if stripped.casefold().startswith("json "):
        stripped = stripped[5:].strip()

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        contents = _extract_consecutive_json_contents(stripped)
        return contents or [stripped]

    contents = _extract_content_items(payload)
    return contents or [stripped]


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
