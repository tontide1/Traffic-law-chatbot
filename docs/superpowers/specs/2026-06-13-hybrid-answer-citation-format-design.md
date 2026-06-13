# Design Spec: Hybrid Answer-First Citation Format

Date: 2026-06-13
Status: Approved in conversation, written for review

## Summary

Standardize Hybrid GraphRAG output so `hybrid` answers behave the same in:

- comparison mode
- single hybrid mode

The required output contract is:

```text
answer first + citation inline
```

This replaces the current direct-definition short-circuit behavior where `hybrid` can return only an extractive `Căn cứ chính` block with a quoted statute sentence. The new behavior must produce a user-facing answer sentence first, then attach the legal citation directly to that sentence or claim.

## Problem

Current `hybrid` behavior is inconsistent with both user expectations and `naive` comparison output.

Observed regression:

- `naive` produces a readable answer with legal grounding
- `hybrid` can degrade into a bare extractive block
- the same question can therefore look worse in `hybrid` than in `naive`

Example failure shape:

```text
Hành lang an toàn đường bộ là gì?
```

Current `hybrid` output can become:

```text
Căn cứ chính

"Hành lang an toàn đường bộ là dải đất dọc hai bên..."
```

This is legally grounded, but it is not a good answer format for chat. It reads like raw evidence dumping, not an assistant response.

## Goals

- Make `hybrid` answer in the same user-facing format in both comparison mode and single mode.
- Ensure `hybrid` answers start with the answer, not with a heading dump.
- Attach legal citations inline to the corresponding answer sentence or claim.
- Keep legal wording close to the source while still reading like an answer.
- Preserve the controlled retrieval pipeline and its anti-drift safeguards.
- Minimize frontend impact by keeping the payload shape unchanged.

## Non-Goals

- Do not redesign GraphRAG retrieval or graph indexing in this change.
- Do not change the frontend response schema.
- Do not add a separate citation object to the API contract in this phase.
- Do not introduce a rich footnote renderer.
- Do not change `naive` formatting unless needed to keep comparison readable.

## User-Approved Output Rule

The user approved this presentation rule:

```text
When the user asks a question, hybrid should return an answer plus citation.
Citation style: inline with the answer, not a separate citation section.
```

This means the default target format is:

```md
<answer sentence>. (<citation>)
```

For the sample question:

```md
Hành lang an toàn đường bộ là dải đất dọc hai bên đất của đường bộ, tính từ mép ngoài phần đất để bảo vệ, bảo trì đường bộ ra hai bên nhằm bảo đảm an toàn giao thông đường bộ, bảo đảm tầm nhìn xe chạy và hạn chế ảnh hưởng đến môi trường xung quanh. (Điều 2 khoản 5, Luật Đường bộ 35/2024/QH15).
```

## Option Review

### Option 1: Keep extractive short-circuit and only polish headings

Pros:

- very small backend change
- preserves the current safety-first extractive path

Cons:

- still looks like evidence dump
- still inconsistent with `naive`
- still fails the user-approved answer-first contract

Decision:

- reject

### Option 2: Generate answer-first output from curated context and embed citation inline

Pros:

- matches the approved chat format
- keeps the controlled context discipline already added
- works for both direct-definition and broader legal questions
- requires only backend prompt/orchestration changes

Cons:

- must guard against paraphrase drift
- requires tests for exact formatting behavior

Decision:

- adopt

### Option 3: Return answer text plus a structured citation payload and let frontend format it

Pros:

- cleaner long-term architecture
- enables richer rendering later

Cons:

- changes API contract
- adds frontend work
- out of scope for the requested fix

Decision:

- defer

## Recommended Design

### 1. One Hybrid Output Contract

`hybrid` must always return an answer-oriented string, regardless of whether it is used:

- alone
- inside comparison mode

The backend should no longer allow direct-definition questions to exit early with only a quoted extractive block.

Instead:

- direct-definition questions still use controlled retrieval
- direct-definition questions still prefer exact statutory wording
- final rendering must be answer-first with inline citation

### 2. Inline Citation Format

The default citation format should be plain-text inline legal references:

```text
(Điều ..., khoản ..., Luật .../.../QH...)
```

Formatting rules:

- place the citation at the end of the sentence it supports
- use one citation per claim unless multiple provisions are truly necessary
- keep citation text compact and human-readable
- prefer source labels already available from curated context
- do not output a separate `Trích dẫn` or `Căn cứ chính` section for simple direct-definition answers

### 3. Direct-Definition Behavior

For questions such as:

- `... là gì?`
- `... được hiểu như thế nào?`
- `... được xác định từ đâu?`

the answer should:

- start with a direct answer sentence
- preserve the statutory definition as much as possible
- end with an inline citation

It should not:

- start with `Căn cứ chính`
- dump a raw quote with no answer framing
- add unrelated related-law sections

### 4. Multi-Claim Behavior

For questions that legitimately need more than one claim, `hybrid` may use:

- short paragraphs
- flat bullets

But each claim must still carry its own inline citation.

Example shape:

```md
- Hành lang an toàn đường bộ là dải đất dọc hai bên đất của đường bộ ... (Điều 2 khoản 5, Luật Đường bộ 35/2024/QH15).
- Mục đích của hành lang này là bảo đảm an toàn giao thông đường bộ, tầm nhìn xe chạy và hạn chế ảnh hưởng đến môi trường xung quanh. (Điều 2 khoản 5, Luật Đường bộ 35/2024/QH15).
```

### 5. Prompting and Rendering Strategy

The controlled answer prompt should explicitly instruct the model to:

- answer the question directly first
- keep the answer in natural Vietnamese legal prose
- attach the supporting citation inline at the end of the sentence
- avoid headings unless the question genuinely needs a multi-part answer
- avoid separate `Căn cứ chính` and `Liên kết pháp lý liên quan` sections for simple one-basis answers
- avoid paraphrase markers such as `Theo định nghĩa trên`, `nói cách khác`, or similar filler

### 6. Controlled Pipeline Change

`answer_controlled_chat()` should keep the current retrieval and curation path, but change the final rendering logic:

- do not return `_render_direct_definition_answer()` as the final user output
- use extracted definition evidence as a source of truth for citation-safe generation
- if generation drifts semantically, fall back to an answer-first rendered sentence with inline citation, not to a heading-based extractive block

This keeps the anti-drift safeguard but aligns the fallback with the new output contract.

### 7. API Compatibility

No API schema changes are required.

The backend may continue returning:

- `ChatResponse(response=..., mode="hybrid")`
- `ComparisonResponse(naive=..., hybrid=...)`

Only the contents of `response` change.

## Files Likely To Change

- `backend/core/controlled_chat.py`
- `backend/core/legal_prompts.py`
- `backend/tests/test_controlled_chat.py`
- `backend/tests/test_legal_prompts.py`
- optionally `backend/tests/test_chat_legal_prompt_integration.py`

## Acceptance Criteria

- Asking `Hành lang an toàn đường bộ là gì?` in single hybrid mode returns an answer sentence followed by an inline citation.
- Asking the same question in comparison mode returns a `hybrid` answer with the same answer-first citation style.
- `hybrid` does not return a heading-only extractive block for direct-definition questions.
- If fallback rendering is triggered, the fallback still follows the answer-first inline-citation format.
- Existing response payload shapes and streaming event shapes remain unchanged.
- Targeted backend tests cover direct-definition formatting, fallback formatting, and comparison-mode consistency.

## Testing Strategy

- unit test the hybrid formatter for direct-definition output
- unit test semantic-drift fallback output shape
- integration test `/api/chat` for single hybrid non-streaming responses
- integration test `/api/chat` for comparison mode non-streaming responses
- keep streaming contract tests unchanged unless the streamed chunk content shape needs assertion updates

## Risks

- If the prompt is too loose, the model may cite inline but still paraphrase too much.
- If the fallback is too extractive, it will regress back to the current bad UX.
- If the citation formatter depends on noisy source labels, answer readability may suffer.

## Rollout Guidance

- Implement as a backend-only change first.
- Verify against the sample direct-definition question that triggered the complaint.
- Compare `naive` and `hybrid` side by side after the change to confirm `hybrid` is no longer visually worse by default.
