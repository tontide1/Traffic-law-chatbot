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
        text="xe máy đi bộ",
        source="Luật",
        score=0.8,
    )

    assert label_candidate(
        "xe đi bộ",
        candidate,
        QueryClass.RELATIONAL,
    ) == CandidateLabel.DIRECT


def test_fallback_to_supporting_when_no_direct_candidates():
    candidates = [
        ContextCandidate(id="supporting", text="Trách nhiệm của cơ quan quản lý đường bộ.", source="Luật", score=0.9),
    ]

    selected = select_relevant_candidates(
        question="Hành lang an toàn đường bộ được xác định từ đâu?",
        candidates=candidates,
        query_class=QueryClass.DIRECT_DEFINITION,
        final_top_n=3,
    )

    assert [item.id for item in selected] == ["supporting"]


def test_render_curated_context_numbers_candidates():
    candidates = [
        ContextCandidate(id="c1", text="Điều 2 khoản 5: Hành lang an toàn đường bộ là dải đất.", source="Luật Đường bộ", score=0.8),
    ]

    rendered = render_curated_context(candidates)

    assert "[1]" in rendered
    assert "Luật Đường bộ" in rendered
    assert "Điều 2 khoản 5" in rendered
