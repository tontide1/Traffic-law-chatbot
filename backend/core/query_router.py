"""Rule-based routing for Vietnamese legal RAG questions."""

from dataclasses import dataclass
from enum import StrEnum
import unicodedata


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
    "định nghĩa",
    "bao gồm những gì",
    "thế nào là",
)

_RELATIONAL_SIGNALS = (
    "liên quan đến",
    "căn cứ nào",
    "so sánh",
    "khác nhau",
    "ngoại lệ",
    "hậu quả pháp lý",
    "trách nhiệm của",
    "phân biệt",
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
    normalized = unicodedata.normalize("NFC", query)
    return " ".join(normalized.casefold().split())


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
