import asyncio
import os

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

from backend.config import settings
from backend.core.llm_services import (
    VLLMEmbeddingFunc,
    answer_llm_func,
    google_studio_indexing_llm_func,
    indexing_llm_func,
)


class RAGEngine:
    _deepseek_indexing_instance = None
    _google_studio_indexing_instance = None
    _query_instance = None
    _google_init_lock = asyncio.Lock()

    @classmethod
    def _apply_postgres_environment(cls):
        os.environ["POSTGRES_HOST"] = settings.POSTGRES_HOST
        os.environ["POSTGRES_PORT"] = str(settings.POSTGRES_PORT)
        os.environ["POSTGRES_USER"] = settings.POSTGRES_USER
        os.environ["POSTGRES_PASSWORD"] = settings.POSTGRES_PASSWORD
        os.environ["POSTGRES_DATABASE"] = settings.POSTGRES_DATABASE

    @classmethod
    def _build_embedding_func(cls, prefix: str):
        return EmbeddingFunc(
            embedding_dim=settings.EMBEDDING_DIM,
            max_token_size=settings.EMBEDDING_MAX_TOKEN_SIZE,
            func=VLLMEmbeddingFunc(prefix=prefix),
            model_name=settings.EMBEDDING_MODEL,
        )

    @classmethod
    async def initialize(cls):
        if (
            cls._deepseek_indexing_instance is not None
            and cls._query_instance is not None
        ):
            return cls._deepseek_indexing_instance, cls._query_instance

        cls._apply_postgres_environment()

        cls._deepseek_indexing_instance = LightRAG(
            working_dir=settings.LIGHTRAG_WORKING_DIR,
            llm_model_func=indexing_llm_func,
            embedding_func=cls._build_embedding_func(settings.EMBEDDING_DOCUMENT_PREFIX),
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="PGGraphStorage",
            doc_status_storage="PGDocStatusStorage",
            addon_params={
                "language": settings.SUMMARY_LANGUAGE,
                "entity_types": settings.ENTITY_TYPES,
            },
        )
        await cls._deepseek_indexing_instance.initialize_storages()

        cls._query_instance = LightRAG(
            working_dir=settings.LIGHTRAG_WORKING_DIR,
            llm_model_func=answer_llm_func,
            embedding_func=cls._build_embedding_func(settings.EMBEDDING_QUERY_PREFIX),
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="PGGraphStorage",
            doc_status_storage="PGDocStatusStorage",
            addon_params={
                "language": settings.SUMMARY_LANGUAGE,
                "entity_types": settings.ENTITY_TYPES,
            },
        )
        await cls._query_instance.initialize_storages()

        return cls._deepseek_indexing_instance, cls._query_instance

    @classmethod
    async def get_indexing_instance(cls, provider: str = "deepseek"):
        if provider == "deepseek":
            if cls._deepseek_indexing_instance is None:
                raise RuntimeError("DeepSeek indexing RAG engine not initialized. Call RAGEngine.initialize() first.")
            return cls._deepseek_indexing_instance
        if provider == "google_studio":
            if cls._google_studio_indexing_instance is None:
                async with cls._google_init_lock:
                    if cls._google_studio_indexing_instance is None:
                        cls._apply_postgres_environment()
                        cls._google_studio_indexing_instance = LightRAG(
                            working_dir=settings.LIGHTRAG_WORKING_DIR,
                            llm_model_func=google_studio_indexing_llm_func,
                            embedding_func=cls._build_embedding_func(settings.EMBEDDING_DOCUMENT_PREFIX),
                            kv_storage="PGKVStorage",
                            vector_storage="PGVectorStorage",
                            graph_storage="PGGraphStorage",
                            doc_status_storage="PGDocStatusStorage",
                            addon_params={
                                "language": settings.SUMMARY_LANGUAGE,
                                "entity_types": settings.ENTITY_TYPES,
                            },
                        )
                        await cls._google_studio_indexing_instance.initialize_storages()
            return cls._google_studio_indexing_instance
        raise ValueError(f"Unsupported indexing provider: {provider}")

    @classmethod
    def get_query_instance(cls):
        if cls._query_instance is None:
            raise RuntimeError("Query RAG engine not initialized. Call RAGEngine.initialize() first.")
        return cls._query_instance


rag_engine = RAGEngine()
