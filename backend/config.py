import json
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

NVIDIA_OPENAI_COMPATIBLE_BASE_URL = "https://integrate.api.nvidia.com/v1"


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/law_assistant"

    # Postgres individual components for LightRAG
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DATABASE: str = "law_assistant"

    REDIS_URL: Optional[str] = None

    INDEXING_LLM_BASE_URL: str = "https://api.deepseek.com"
    INDEXING_LLM_API_KEY: Optional[str] = None
    INDEXING_LLM_MODEL: str = "deepseek-v4-flash"
    INDEXING_LLM_THINKING_MODE: str = "disabled"

    ANSWER_LLM_BASE_URL: str = NVIDIA_OPENAI_COMPATIBLE_BASE_URL
    ANSWER_LLM_API_KEY: Optional[str] = None
    ANSWER_LLM_MODEL: str = "openai/gpt-oss-120b"

    GOOGLE_STUDIO_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GOOGLE_STUDIO_API_KEY: Optional[str] = None
    GOOGLE_STUDIO_MODEL: str = "gemini-2.5-flash"

    EMBEDDING_BASE_URL: str = "http://host.docker.internal:8002/v1"
    EMBEDDING_API_KEY: str = "EMPTY"
    EMBEDDING_MODEL: str = "AITeamVN/Vietnamese_Embedding_v2"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_MAX_TOKEN_SIZE: int = 512
    EMBEDDING_QUERY_PREFIX: str = (
        "Instruct: Given a legal query, retrieve relevant statutes and legal passages.\nQuery: "
    )
    EMBEDDING_DOCUMENT_PREFIX: str = ""

    SUMMARY_LANGUAGE: str = "Vietnamese"
    RAG_ENTITY_TYPES: list[str] = [
        "Văn bản pháp luật",
        "Điều khoản",
        "Cơ quan ban hành",
        "Đối tượng áp dụng",
        "Hành vi vi phạm",
        "Hình thức xử phạt",
        "Thời hạn",
        "Khái niệm pháp lý",
    ]

    LIGHTRAG_WORKING_DIR: str = "./backend/data"

    @field_validator("RAG_ENTITY_TYPES", mode="before")
    @classmethod
    def parse_entity_types(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in value.split(",")]
        return value

    def get_indexing_llm_api_key(self) -> str:
        if not self.INDEXING_LLM_API_KEY:
            raise ValueError("INDEXING_LLM_API_KEY is required")
        return self.INDEXING_LLM_API_KEY

    def get_answer_llm_api_key(self) -> str:
        if not self.ANSWER_LLM_API_KEY:
            raise ValueError("ANSWER_LLM_API_KEY is required")
        return self.ANSWER_LLM_API_KEY

    def get_google_studio_api_key(self) -> str:
        if not self.GOOGLE_STUDIO_API_KEY:
            raise ValueError("GOOGLE_STUDIO_API_KEY is required")
        return self.GOOGLE_STUDIO_API_KEY

    def get_embedding_api_key(self) -> str:
        return self.EMBEDDING_API_KEY or "EMPTY"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
