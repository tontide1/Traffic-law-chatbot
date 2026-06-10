# Design Spec: Move Answer LLM Provider from OpenRouter to NVIDIA

Date: 2026-06-10
Status: Draft approved in conversation, written for review

## Summary

Move the answer-generation provider from OpenRouter to NVIDIA's OpenAI-compatible endpoint while keeping the existing backend answer flow and application-level configuration contract stable.

This spec keeps the current `chat.completions.create(...)` path for `ANSWER_LLM`, changes the default answer provider to NVIDIA, removes OpenRouter fallback behavior, and updates both the backend and `rag-ui` answer/query bindings to use the NVIDIA-hosted endpoint.

## Context

The current repository already separates provider roles:

- indexing uses DeepSeek direct API
- answers use `openai/gpt-oss-120b` via OpenRouter
- embeddings use a local OpenAI-compatible vLLM endpoint

This answer-provider setup is implemented across:

- `backend/config.py`
- `backend/core/llm_services.py`
- `docker-compose.yml`
- `.env.example`
- `README.md`
- provider-related backend tests

The current answer path uses the OpenAI Python SDK with an OpenAI-compatible base URL and calls `client.chat.completions.create(...)`. That behavior should remain unchanged for this migration.

The user's goal is not to redesign the answer API surface. The goal is to stop using OpenRouter for `ANSWER_LLM` and instead use NVIDIA's hosted OpenAI-compatible endpoint for the same model:

- base URL: `https://integrate.api.nvidia.com/v1`
- model: `openai/gpt-oss-120b`

## Goals

- Remove OpenRouter from the `ANSWER_LLM` path.
- Use NVIDIA's OpenAI-compatible endpoint for answer generation.
- Keep the app-level answer configuration names stable:
  - `ANSWER_LLM_BASE_URL`
  - `ANSWER_LLM_API_KEY`
  - `ANSWER_LLM_MODEL`
- Change the default `ANSWER_LLM_BASE_URL` value to `https://integrate.api.nvidia.com/v1`.
- Keep `ANSWER_LLM_MODEL` defaulted to `openai/gpt-oss-120b`.
- Keep the existing `chat.completions.create(...)` request path.
- Remove fallback to `OPENROUTER_API_KEY`.
- Move both backend answer traffic and `rag-ui` query/answer bindings to NVIDIA.
- Update docs to state clearly that the current answer provider is NVIDIA.

## Non-Goals

- Do not migrate answer generation to `responses.create(...)`.
- Do not redesign prompt/history handling.
- Do not change frontend chat contracts or SSE payload shape.
- Do not change indexing provider behavior.
- Do not change embedding provider behavior.
- Do not add answer-provider switching in the UI.
- Do not preserve backward compatibility with `OPENROUTER_API_KEY`.

## Options Considered

### Option 1: Config-only switch

Only update env defaults and docs from OpenRouter to NVIDIA.

This is insufficient because the current code and compose setup still encode OpenRouter-specific fallback and `rag-ui` answer/query key wiring.

### Option 2: Provider switch with stable app contract

Keep the existing `ANSWER_LLM_*` configuration interface, but switch the underlying provider defaults and wiring to NVIDIA.

This is the recommended option because it:

- solves the actual provider migration request
- avoids unnecessary API-contract changes inside the backend
- minimizes risk to streaming, comparison mode, and current tests
- keeps future provider changes possible without renaming app-level settings again

### Option 3: Provider switch plus API migration

Move to NVIDIA and also rewrite the answer path to `responses.create(...)`.

This is not recommended because it expands scope into request-shape, history mapping, streaming parsing, and response-shape changes that are unrelated to the provider migration itself.

## Recommended Design

Adopt Option 2.

The repository should continue to treat answer generation as an app-level role called `ANSWER_LLM`, but the provider behind that role becomes NVIDIA by default.

The key design rule is:

- stable app contract
- changed provider default
- no OpenRouter fallback

## Configuration Design

### `backend/config.py`

Keep the answer-setting names unchanged:

- `ANSWER_LLM_BASE_URL`
- `ANSWER_LLM_API_KEY`
- `ANSWER_LLM_MODEL`

Change the default answer provider values to:

```python
ANSWER_LLM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
ANSWER_LLM_API_KEY: Optional[str] = None
ANSWER_LLM_MODEL: str = "openai/gpt-oss-120b"
```

Keep `OPENROUTER_API_KEY` out of the answer path.

`get_answer_llm_api_key()` should become:

- accept only `ANSWER_LLM_API_KEY`
- raise a clear error if it is missing

Recommended error message:

```text
ANSWER_LLM_API_KEY is required
```

This is an intentional fail-fast behavior. Existing environments that still rely on `OPENROUTER_API_KEY` should stop booting until they are updated.

### `.env.example`

Rewrite the answer-provider section to make NVIDIA explicit:

```env
# Answer LLM: NVIDIA-hosted OpenAI-compatible endpoint
ANSWER_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
ANSWER_LLM_API_KEY=your_nvidia_api_key_here
ANSWER_LLM_MODEL=openai/gpt-oss-120b
```

Remove the OpenRouter answer fallback section from `.env.example`.

If `OPENROUTER_API_KEY` remains in the file for another purpose, it must not be described as part of the answer path.

### `README.md`

Update all answer-provider documentation so it clearly states:

- answers use `openai/gpt-oss-120b`
- the provider is NVIDIA
- the endpoint is OpenAI-compatible

Examples:

- "NVIDIA-hosted OpenAI-compatible endpoint"
- "answer generation runs through NVIDIA"

The README should no longer describe OpenRouter as the default answer provider.

## Runtime Design

### `backend/core/llm_services.py`

Keep the client construction pattern:

```python
openai.AsyncOpenAI(
    api_key=settings.get_answer_llm_api_key(),
    base_url=settings.ANSWER_LLM_BASE_URL,
)
```

Keep the answer request path:

```python
await client.chat.completions.create(
    model=settings.ANSWER_LLM_MODEL,
    messages=messages,
    **request_kwargs,
)
```

Do not change:

- message construction
- history handling
- comparison mode behavior
- stream/non-stream branching

This migration is provider-level only. It is not a request-contract redesign.

### `docker-compose.yml`

Update backend defaults:

- `ANSWER_LLM_BASE_URL=${ANSWER_LLM_BASE_URL:-https://integrate.api.nvidia.com/v1}`
- `ANSWER_LLM_MODEL=${ANSWER_LLM_MODEL:-openai/gpt-oss-120b}`

Update `rag-ui` answer/query bindings so they use `ANSWER_LLM_API_KEY` instead of `OPENROUTER_API_KEY`:

- `LLM_BINDING_API_KEY=${ANSWER_LLM_API_KEY}`
- `QUERY_LLM_BINDING_API_KEY=${ANSWER_LLM_API_KEY}`

Update `rag-ui` host defaults to NVIDIA:

- `LLM_BINDING_HOST=${ANSWER_LLM_BASE_URL:-https://integrate.api.nvidia.com/v1}`
- `QUERY_LLM_BINDING_HOST=${ANSWER_LLM_BASE_URL:-https://integrate.api.nvidia.com/v1}`

The indexing and embedding bindings remain unchanged.

## Files Expected to Change

- `backend/config.py`
- `backend/core/llm_services.py`
- `docker-compose.yml`
- `.env.example`
- `README.md`
- `backend/tests/test_multi_provider_config.py`
- `backend/tests/test_multi_provider_llm_services.py`
- `backend/tests/test_multi_provider_docs.py`

## Verification Plan

Targeted verification should cover configuration, client wiring, and docs.

### Config tests

Update tests to confirm:

- `ANSWER_LLM_BASE_URL` defaults to `https://integrate.api.nvidia.com/v1`
- `ANSWER_LLM_MODEL` remains `openai/gpt-oss-120b`
- `get_answer_llm_api_key()` requires `ANSWER_LLM_API_KEY`
- `OPENROUTER_API_KEY` no longer satisfies answer-key lookup

### LLM service tests

Update tests to confirm:

- answer client creation uses NVIDIA base URL
- answer client creation uses `ANSWER_LLM_API_KEY`
- answer request flow still calls `chat.completions.create(...)`
- answer path still avoids indexing-specific `thinking` injection

### Docs and compose tests

Update tests to confirm:

- `.env.example` documents NVIDIA as the answer provider
- `docker-compose.yml` defaults `ANSWER_LLM_BASE_URL` to NVIDIA
- `rag-ui` consumes `ANSWER_LLM_API_KEY` for answer/query bindings
- `README.md` describes NVIDIA, not OpenRouter, as the answer provider

## Risks

### Environment breakage for stale local configs

Because backward compatibility is intentionally removed, any environment that still depends on `OPENROUTER_API_KEY` for answers will fail until `ANSWER_LLM_API_KEY` is populated.

This is expected and acceptable for this change.

### `rag-ui` startup assumptions

`rag-ui` currently receives answer/query credentials through `OPENROUTER_API_KEY`. Switching those bindings to `ANSWER_LLM_API_KEY` must be validated because startup failures there would break the graph UI's answer/query path.

### Provider behavior differences

Although NVIDIA exposes an OpenAI-compatible endpoint, provider-specific behavior may still differ in areas such as optional parameters, rate limits, or response metadata. This spec intentionally avoids relying on provider-specific extras beyond `base_url`, `api_key`, and the current chat-completions path.

## Rollout Notes

- This migration should be treated as a breaking configuration change for local environments.
- Documentation must be updated in the same change as code and tests.
- No database migration or re-indexing is required because only the answer provider changes.

## Success Criteria

The migration is complete when all of the following are true:

- backend answer generation uses NVIDIA by default
- `rag-ui` answer/query bindings use `ANSWER_LLM_API_KEY`
- no answer-path fallback to `OPENROUTER_API_KEY` remains
- docs consistently describe NVIDIA as the current answer provider
- targeted provider/config/docs tests pass
