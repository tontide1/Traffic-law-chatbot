from collections.abc import AsyncIterator
import openai
from typing import List, Optional

import numpy as np

from backend.config import settings

_indexing_client: Optional[openai.AsyncOpenAI] = None
_google_studio_client: Optional[openai.AsyncOpenAI] = None
_answer_client: Optional[openai.AsyncOpenAI] = None
_embedding_client: Optional[openai.AsyncOpenAI] = None


def reset_client_caches():
    global _indexing_client, _google_studio_client, _answer_client, _embedding_client
    _indexing_client = None
    _google_studio_client = None
    _answer_client = None
    _embedding_client = None


def _build_async_client(api_key: str, base_url: str) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)


def get_indexing_llm_client():
    global _indexing_client
    if _indexing_client is None:
        _indexing_client = _build_async_client(
            api_key=settings.get_indexing_llm_api_key(),
            base_url=settings.INDEXING_LLM_BASE_URL,
        )
    return _indexing_client


def get_google_studio_llm_client():
    global _google_studio_client
    if _google_studio_client is None:
        _google_studio_client = _build_async_client(
            api_key=settings.get_google_studio_api_key(),
            base_url=settings.GOOGLE_STUDIO_BASE_URL,
        )
    return _google_studio_client


def get_answer_llm_client():
    global _answer_client
    if _answer_client is None:
        _answer_client = _build_async_client(
            api_key=settings.get_answer_llm_api_key(),
            base_url=settings.ANSWER_LLM_BASE_URL,
        )
    return _answer_client


def get_embedding_client():
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = _build_async_client(
            api_key=settings.get_embedding_api_key(),
            base_url=settings.EMBEDDING_BASE_URL,
        )
    return _embedding_client


# LightRAG passes these internal kwargs to every LLM func — strip them before
# forwarding to the OpenAI client, which doesn't know about them.
_LIGHTRAG_INTERNAL_KWARGS = {"hashing_kv", "mode", "json_mode"}


def _filter_kwargs(kwargs: dict) -> dict:
    """Remove LightRAG-internal keys and the 'model' override from kwargs."""
    return {k: v for k, v in kwargs.items() if k not in _LIGHTRAG_INTERNAL_KWARGS and k != "model"}


def _base_messages(prompt: str, system_prompt: str = None, history: List[dict] = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


def _stream_or_content(response, stream: bool) -> str | AsyncIterator[str]:
    if stream:
        async def stream_generator():
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return stream_generator()
    return response.choices[0].message.content


class VLLMEmbeddingFunc:
    def __init__(self, prefix: str = ""):
        self.model_name = settings.EMBEDDING_MODEL
        self.prefix = prefix

    async def __call__(self, texts: List[str]):
        client = get_embedding_client()
        vectors = []
        for text in texts:
            response = await client.embeddings.create(
                model=self.model_name,
                input=f"{self.prefix}{text}",
            )
            vectors.append(response.data[0].embedding)
        return np.array(vectors)


async def indexing_llm_func(
    prompt: str,
    system_prompt: str = None,
    history: List[dict] = None,
    history_messages: List[dict] = None,
    **kwargs,
) -> str | AsyncIterator[str]:
    client = get_indexing_llm_client()
    effective_history = history or history_messages
    messages = _base_messages(prompt, system_prompt=system_prompt, history=effective_history)
    request_kwargs = _filter_kwargs(kwargs)
    extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
    extra_body["thinking"] = {"type": settings.INDEXING_LLM_THINKING_MODE}

    response = await client.chat.completions.create(
        model=settings.INDEXING_LLM_MODEL,
        messages=messages,
        extra_body=extra_body,
        **request_kwargs,
    )
    return _stream_or_content(response, stream=bool(request_kwargs.get("stream")))


async def google_studio_indexing_llm_func(
    prompt: str,
    system_prompt: str = None,
    history: List[dict] = None,
    history_messages: List[dict] = None,
    **kwargs,
) -> str | AsyncIterator[str]:
    client = get_google_studio_llm_client()
    effective_history = history or history_messages
    messages = _base_messages(prompt, system_prompt=system_prompt, history=effective_history)
    request_kwargs = _filter_kwargs(kwargs)

    response = await client.chat.completions.create(
        model=settings.GOOGLE_STUDIO_MODEL,
        messages=messages,
        **request_kwargs,
    )
    return _stream_or_content(response, stream=bool(request_kwargs.get("stream")))


async def answer_llm_func(
    prompt: str,
    system_prompt: str = None,
    history: List[dict] = None,
    history_messages: List[dict] = None,
    **kwargs,
) -> str | AsyncIterator[str]:
    client = get_answer_llm_client()
    effective_history = history or history_messages
    messages = _base_messages(prompt, system_prompt=system_prompt, history=effective_history)
    request_kwargs = _filter_kwargs(kwargs)

    response = await client.chat.completions.create(
        model=settings.ANSWER_LLM_MODEL,
        messages=messages,
        **request_kwargs,
    )
    return _stream_or_content(response, stream=bool(request_kwargs.get("stream")))
