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


@pytest.mark.anyio
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


@pytest.mark.anyio
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
