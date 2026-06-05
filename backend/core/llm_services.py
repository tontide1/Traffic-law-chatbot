import openai
from typing import List, Optional

from backend.config import settings

# Global client cache to avoid pickling issues and redundant connections
_async_client: Optional[openai.AsyncOpenAI] = None


def get_openai_client():
    global _async_client
    if _async_client is None:
        _async_client = openai.AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
    return _async_client


class QwenEmbeddingFunc:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL

    def _get_prefix(self, is_query: bool) -> str:
        if is_query:
            return "Instruct: Given a legal query, retrieve relevant statutes...\nQuery: "
        return ""

    async def __call__(self, texts: List[str]):
        import numpy as np

        client = get_openai_client()
        results = []
        for text in texts:
            prefix = self._get_prefix(is_query=text.strip().endswith("?"))

            response = await client.embeddings.create(
                model=self.model_name,
                input=prefix + text
            )
            results.append(response.data[0].embedding)
        return np.array(results)


async def deepseek_llm_func(
    prompt: str,
    system_prompt: str = None,
    history: List[dict] = None,
    **kwargs
) -> str:
    client = get_openai_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": prompt})

    extra_headers = {
        "HTTP-Referer": "https://github.com/traffic/law-assistant",
        "X-Title": "Traffic Law Assistant",
    }

    allowed_params = [
        "model", "messages", "stream", "temperature", "top_p", "n", "stop", "max_tokens",
        "presence_penalty", "frequency_penalty", "logit_bias", "user", "response_format",
        "seed", "tools", "tool_choice", "parallel_tool_calls"
    ]
    api_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}

    response = await client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        extra_headers=extra_headers,
        **api_kwargs
    )

    if api_kwargs.get("stream"):
        async def stream_generator():
            print("LLM: Starting stream generator")
            try:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        c = chunk.choices[0].delta.content
                        print(f"LLM CHUNK: {c}")
                        yield c
            except Exception as e:
                print(f"LLM STREAM ERROR: {str(e)}")
            print("LLM: Stream generator finished")

        return stream_generator()
    else:
        return response.choices[0].message.content
