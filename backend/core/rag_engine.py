import os
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from backend.core.llm_services import QwenEmbeddingFunc, deepseek_llm_func
from backend.config import settings

class RAGEngine:
    _instance = None

    @classmethod
    async def initialize(cls):
        """Asynchronously initialize the LightRAG storage pools."""
        if cls._instance is None:
            # Initialize custom embedding function
            embedding_func = QwenEmbeddingFunc()
            
            # Set environment variables for LightRAG Postgres compatibility
            os.environ["POSTGRES_HOST"] = settings.POSTGRES_HOST
            os.environ["POSTGRES_PORT"] = str(settings.POSTGRES_PORT)
            os.environ["POSTGRES_USER"] = settings.POSTGRES_USER
            os.environ["POSTGRES_PASSWORD"] = settings.POSTGRES_PASSWORD
            os.environ["POSTGRES_DATABASE"] = settings.POSTGRES_DATABASE

            # LightRAG initialization with native Postgres storage
            cls._instance = LightRAG(
                working_dir=settings.LIGHTRAG_WORKING_DIR,
                llm_model_func=deepseek_llm_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=1536,
                    max_token_size=512,
                    func=embedding_func,
                    model_name=settings.EMBEDDING_MODEL
                ),
                # Use strings for storage types (LightRAG will instantiate them)
                kv_storage="PGKVStorage",
                vector_storage="PGVectorStorage",
                graph_storage="PGGraphStorage",
                doc_status_storage="PGDocStatusStorage",
                addon_params={
                    "language": settings.SUMMARY_LANGUAGE,
                    "entity_types": settings.ENTITY_TYPES
                }
            )
            # CRITICAL: Initialize Postgres connection pools
            await cls._instance.initialize_storages()
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise RuntimeError("RAGEngine not initialized. Call RAGEngine.initialize() first.")
        return cls._instance

rag_engine = RAGEngine()
