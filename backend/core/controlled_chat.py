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
