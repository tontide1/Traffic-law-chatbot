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
