from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_env_example_describes_split_providers():
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "INDEXING_LLM_MODEL=deepseek-v4-flash" in env_example
    assert "ANSWER_LLM_MODEL=openai/gpt-oss-120b" in env_example
    assert "EMBEDDING_MODEL=AITeamVN/Vietnamese_Embedding_v2" in env_example
    assert "EMBEDDING_BASE_URL=http://host.docker.internal:8002/v1" in env_example
    assert "openai/text-embedding-3-small" not in env_example


def test_docker_compose_exposes_host_gateway_and_embedding_dim_1024():
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert 'host.docker.internal:host-gateway' in compose
    assert 'INDEXING_LLM_BASE_URL=${INDEXING_LLM_BASE_URL:-https://api.deepseek.com}' in compose
    assert 'INDEXING_LLM_MODEL=${INDEXING_LLM_MODEL:-deepseek-v4-flash}' in compose
    assert 'INDEXING_LLM_THINKING_MODE=${INDEXING_LLM_THINKING_MODE:-disabled}' in compose
    assert 'ANSWER_LLM_BASE_URL=${ANSWER_LLM_BASE_URL:-https://openrouter.ai/api/v1}' in compose
    assert 'ANSWER_LLM_MODEL=${ANSWER_LLM_MODEL:-openai/gpt-oss-120b}' in compose
    assert 'EMBEDDING_BASE_URL=${EMBEDDING_BASE_URL:-http://host.docker.internal:8002/v1}' in compose
    assert 'EMBEDDING_MODEL=${EMBEDDING_MODEL:-AITeamVN/Vietnamese_Embedding_v2}' in compose
    assert 'EMBEDDING_DIM=${EMBEDDING_DIM:-1024}' in compose
    assert 'EMBEDDING_MAX_TOKEN_SIZE=${EMBEDDING_MAX_TOKEN_SIZE:-512}' in compose
    assert 'EMBEDDING_BINDING_HOST=${EMBEDDING_BASE_URL:-http://host.docker.internal:8002/v1}' in compose
    assert 'EXTRACT_LLM_MODEL=${INDEXING_LLM_MODEL:-deepseek-v4-flash}' in compose
    assert 'QUERY_LLM_MODEL=${ANSWER_LLM_MODEL:-openai/gpt-oss-120b}' in compose


def test_readme_describes_deepseek_openrouter_and_local_vllm():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "DeepSeek-V4-Flash" in readme
    assert "openai/gpt-oss-120b" in readme
    assert "AITeamVN/Vietnamese_Embedding_v2" in readme
    assert "vLLM" in readme
    assert "--runner pooling" in readme
    assert "--dtype float16" in readme
    assert "--gpu-memory-utilization 0.75" in readme
    assert "--max-model-len 2048" in readme
    assert "--attention-backend TRITON_ATTN" in readme
    assert "GTX 1660 Super" in readme
    assert "Turing-based GTX 16-series" in readme
    assert "--task embed" not in readme
    assert "text-embedding-3-small" not in readme


def test_env_example_describes_google_studio_selector_settings():
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "GOOGLE_STUDIO_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/" in env_example
    assert "GOOGLE_STUDIO_API_KEY=your_google_studio_api_key_here" in env_example
    assert "GOOGLE_STUDIO_MODEL=gemini-2.5-flash" in env_example


def test_compose_passes_google_studio_environment_to_backend():
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "GOOGLE_STUDIO_BASE_URL=${GOOGLE_STUDIO_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}" in compose
    assert "GOOGLE_STUDIO_API_KEY=${GOOGLE_STUDIO_API_KEY}" in compose
    assert "GOOGLE_STUDIO_MODEL=${GOOGLE_STUDIO_MODEL:-gemini-2.5-flash}" in compose


def test_readme_mentions_shared_graph_and_sidebar_selector():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Google Studio" in readme
    assert "left sidebar" in readme
    assert "shared knowledge base" in readme
    assert "Existing documents are not rebuilt automatically" in readme
