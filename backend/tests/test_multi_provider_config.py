from backend.config import Settings


def test_multi_provider_defaults_match_the_new_architecture():
    settings = Settings(_env_file=None)

    assert settings.INDEXING_LLM_BASE_URL == "https://api.deepseek.com"
    assert settings.INDEXING_LLM_MODEL == "deepseek-v4-flash"
    assert settings.INDEXING_LLM_THINKING_MODE == "disabled"
    assert settings.ANSWER_LLM_BASE_URL == "https://openrouter.ai/api/v1"
    assert settings.ANSWER_LLM_MODEL == "openai/gpt-oss-120b"
    assert settings.EMBEDDING_BASE_URL == "http://host.docker.internal:8002/v1"
    assert settings.EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.EMBEDDING_DIM == 1024
    assert settings.EMBEDDING_DOCUMENT_PREFIX == ""
    assert "Given a legal query" in settings.EMBEDDING_QUERY_PREFIX


def test_answer_key_falls_back_to_openrouter_key():
    settings = Settings(
        _env_file=None,
        OPENROUTER_API_KEY="openrouter-key",
        ANSWER_LLM_API_KEY=None,
    )

    assert settings.get_answer_llm_api_key() == "openrouter-key"


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
        assert str(exc) == "ANSWER_LLM_API_KEY or OPENROUTER_API_KEY is required"
    else:
        raise AssertionError("Expected answer API key lookup to raise")


def test_entity_types_string_parsing_still_works():
    settings = Settings(
        _env_file=None,
        ENTITY_TYPES='["Điều khoản", "Văn bản pháp luật"]',
    )

    assert settings.ENTITY_TYPES == ["Điều khoản", "Văn bản pháp luật"]
