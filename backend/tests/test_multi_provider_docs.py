from pathlib import Path


def test_env_example_describes_split_providers():
    env_example = Path(".env.example").read_text(encoding="utf-8")

    assert "INDEXING_LLM_MODEL=deepseek-v4-flash" in env_example
    assert "ANSWER_LLM_MODEL=openai/gpt-oss-120b" in env_example
    assert "EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B" in env_example
    assert "EMBEDDING_BASE_URL=http://host.docker.internal:8002/v1" in env_example
    assert "openai/text-embedding-3-small" not in env_example


def test_docker_compose_exposes_host_gateway_and_embedding_dim_1024():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert 'host.docker.internal:host-gateway' in compose
    assert 'EMBEDDING_DIM=1024' in compose
    assert 'EMBEDDING_BINDING_HOST=${EMBEDDING_BASE_URL}' in compose
    assert 'EXTRACT_LLM_MODEL=${INDEXING_LLM_MODEL}' in compose
    assert 'QUERY_LLM_MODEL=${ANSWER_LLM_MODEL}' in compose


def test_readme_describes_deepseek_openrouter_and_local_vllm():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "DeepSeek-V4-Flash" in readme
    assert "openai/gpt-oss-120b" in readme
    assert "Qwen/Qwen3-Embedding-0.6B" in readme
    assert "vLLM" in readme
    assert "text-embedding-3-small" not in readme
