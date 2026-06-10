import pytest

from backend.config import Settings


def test_multi_provider_defaults_match_the_new_architecture():
    settings = Settings(_env_file=None)

    assert settings.INDEXING_LLM_BASE_URL == "https://api.deepseek.com"
    assert settings.INDEXING_LLM_MODEL == "deepseek-v4-flash"
    assert settings.INDEXING_LLM_THINKING_MODE == "disabled"
    assert settings.ANSWER_LLM_BASE_URL == "https://integrate.api.nvidia.com/v1"
    assert settings.ANSWER_LLM_MODEL == "openai/gpt-oss-120b"
    assert settings.EMBEDDING_BASE_URL == "http://host.docker.internal:8002/v1"
    assert settings.EMBEDDING_MODEL == "AITeamVN/Vietnamese_Embedding_v2"
    assert settings.EMBEDDING_DIM == 1024
    assert settings.EMBEDDING_DOCUMENT_PREFIX == ""
    assert "Given a legal query" in settings.EMBEDDING_QUERY_PREFIX


def test_answer_key_requires_answer_llm_api_key():
    settings = Settings(_env_file=None, ANSWER_LLM_API_KEY="nvidia-key")

    assert settings.get_answer_llm_api_key() == "nvidia-key"


def test_answer_key_does_not_fall_back_to_alternative_key():
    settings = Settings(_env_file=None)

    with pytest.raises(ValueError, match="ANSWER_LLM_API_KEY is required"):
        settings.get_answer_llm_api_key()


def test_missing_required_provider_keys_raise_clear_errors():
    settings = Settings(_env_file=None)

    try:
        settings.get_indexing_llm_api_key()
    except ValueError as exc:
        assert str(exc) == "INDEXING_LLM_API_KEY is required"
    else:
        raise AssertionError("Expected indexing API key lookup to raise")

    try:
        settings.get_answer_llm_api_key()
    except ValueError as exc:
        assert str(exc) == "ANSWER_LLM_API_KEY is required"
    else:
        raise AssertionError("Expected answer API key lookup to raise")


def test_entity_types_string_parsing_still_works():
    settings = Settings(
        _env_file=None,
        RAG_ENTITY_TYPES='["Điều khoản", "Văn bản pháp luật"]',
    )

    assert settings.RAG_ENTITY_TYPES == ["Điều khoản", "Văn bản pháp luật"]


def test_google_studio_defaults_match_indexing_provider_selector_plan():
    settings = Settings(_env_file=None)

    assert settings.GOOGLE_STUDIO_BASE_URL == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert settings.GOOGLE_STUDIO_MODEL == "gemini-2.5-flash"


def test_missing_google_studio_key_raises_clear_error():
    settings = Settings(_env_file=None, GOOGLE_STUDIO_API_KEY=None)

    try:
        settings.get_google_studio_api_key()
    except ValueError as exc:
        assert str(exc) == "GOOGLE_STUDIO_API_KEY is required"
    else:
        raise AssertionError("Expected Google Studio API key lookup to raise")
