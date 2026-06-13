import pytest

from backend.core.legal_prompts import build_legal_system_prompt


def test_build_legal_system_prompt_hybrid_uses_context_data_placeholder():
    template = build_legal_system_prompt("hybrid")

    assert "{response_type}" in template
    assert "{user_prompt}" in template
    assert "{context_data}" in template
    assert "{content_data}" not in template


def test_build_legal_system_prompt_naive_uses_content_data_placeholder():
    template = build_legal_system_prompt("naive")

    assert "{response_type}" in template
    assert "{user_prompt}" in template
    assert "{content_data}" in template
    assert "{context_data}" not in template


@pytest.mark.parametrize("mode", ["hybrid", "naive"])
def test_build_legal_system_prompt_mentions_vietnamese_and_legal_fidelity(mode):
    template = build_legal_system_prompt(mode)
    lowered = template.lower()

    assert "tiếng việt" in lowered
    assert "đồng nghĩa" in lowered


@pytest.mark.parametrize("mode", ["hybrid", "naive"])
def test_build_legal_system_prompt_has_stable_section_labels(mode):
    template = build_legal_system_prompt(mode)

    assert "Căn cứ chính" in template
    assert "Liên kết pháp lý liên quan" in template


def test_build_legal_system_prompt_rejects_unsupported_mode():
    with pytest.raises(ValueError):
        build_legal_system_prompt("unsupported")


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


def test_curated_answer_prompt_rejects_definition_paraphrase_markers():
    from backend.core.legal_prompts import build_curated_answer_system_prompt

    template = build_curated_answer_system_prompt().lower()

    assert "theo định nghĩa trên" in template
    assert "nói cách khác" in template
    assert "không viết" in template or "không dùng" in template
