# Design Spec: Controlled Hybrid Legal GraphRAG

Date: 2026-06-11
Status: Draft approved in conversation, written for review

## Summary

Improve Hybrid GraphRAG answer quality by making graph retrieval more disciplined without redesigning the graph index.

The current comparison example shows Naive RAG outperforming Hybrid GraphRAG on a direct legal-definition question. Hybrid retrieves and cites the direct definition correctly, but then over-expands into broad legal background such as scope articles, policy articles, and constitutional basis. Those additions are legally adjacent but not useful for answering the user's question.

This spec focuses on three changes:

1. tighten the hybrid legal answer prompt
2. add a rule-based query router
3. add reranking plus a legal relevance gate using `BAAI/bge-reranker-v2-m3`

The deeper graph-index redesign is explicitly out of scope for this phase.

## Problem

Hybrid GraphRAG currently behaves like a broad context expander. That is useful for multi-hop legal questions, but harmful for direct questions where one provision already answers the user.

Observed failure case:

```text
Hành lang an toàn đường bộ được xác định từ đâu và việc thiết lập hành lang này nhằm mục đích gì?
```

Expected behavior:

- cite the direct definition in Luật Đường bộ, Điều 2, khoản 5
- optionally connect to Điều 3 only if it directly clarifies the stated safety/environment purpose
- keep the answer short and legally grounded

Current Hybrid GraphRAG failure mode:

- cites the direct definition correctly
- adds broad background nodes such as Điều 1, Điều 4, and Hiến pháp
- makes the answer look more complex while adding little legal value

This is a retrieval-discipline problem, not a reason to abandon Hybrid GraphRAG.

## Goals

- Keep Hybrid GraphRAG useful for multi-hop legal questions.
- Prevent Hybrid GraphRAG from adding broad legal background to direct questions.
- Preserve exact legal wording and direct statutory basis for definition-style questions.
- Route simple legal lookup questions away from full graph expansion.
- Rerank retrieved candidates before answer generation.
- Treat reranking as both ordering and filtering, not only as top-k sorting.
- Keep the API response contract stable for the first implementation.
- Keep changes reversible and easy to benchmark against comparison mode.

## Non-Goals

- Do not redesign the graph schema or index construction.
- Do not re-ingest the corpus as part of this phase.
- Do not build a full legal citation engine.
- Do not replace LightRAG.
- Do not add a graph visualization UI.
- Do not change frontend comparison-mode payload shape.
- Do not use an LLM judge as the first relevance gate.
- Do not tune every legal question type in this phase.

## Product Decision

Adopt a controlled Hybrid GraphRAG pipeline:

```text
question
 -> classify query type
 -> choose retrieval policy
 -> retrieve candidate context
 -> rerank candidates
 -> apply legal relevance gate
 -> generate final answer from curated context
```

The important design rule is:

```text
GraphRAG should prove relevance before adding related law.
```

This means a graph-linked provision should not be shown just because it shares the same law, chapter, policy theme, or parent document. It must directly answer the question, modify the answer, define a term used in the answer, state an applicable condition, state an exception, identify a competent authority, or supply a sanction/consequence required by the question.

## Option Review

### Option 1: Prompt-only fix

Tighten the system prompt and ask the model not to include broad legal background.

Pros:

- fastest change
- low risk
- no new dependency

Cons:

- weak protection when noisy context is already present
- answer model may still use irrelevant context because it was retrieved

Decision:

- use this as step 1, but not as the complete solution

### Option 2: Query router plus prompt guard

Classify the query and choose whether it needs full hybrid graph retrieval.

Pros:

- prevents over-engineering for direct questions
- cheap to implement rule-based first
- improves behavior before adding model infrastructure

Cons:

- rule-based classification can miss edge cases
- still needs candidate filtering inside hybrid mode

Decision:

- use this as step 2

### Option 3: Rerank plus legal relevance gate

Retrieve candidate context, rerank it with `BAAI/bge-reranker-v2-m3`, and remove weak or background-only candidates before final answer generation.

Pros:

- directly attacks over-fetching
- keeps LightRAG and the graph index unchanged
- can be benchmarked with deterministic examples
- gives Hybrid GraphRAG room to work on multi-hop questions

Cons:

- adds runtime cost
- needs deployment/configuration for the reranker
- reranker scores alone may still rate broad topical context too highly

Decision:

- use this as step 3
- combine model score with legal relevance rules

## Recommended Design

### 1. Hybrid Prompt Guard

Update the hybrid legal system prompt so the final answer layer treats related context conservatively.

Prompt rules should include:

- Use `Căn cứ chính` for provisions that directly answer the user question.
- Use `Liên kết pháp lý liên quan` only when another provision directly changes, narrows, explains, or applies the answer.
- If the direct legal basis fully answers the question, omit `Liên kết pháp lý liên quan` or state that no additional link is needed.
- Do not cite broad background provisions unless the user asks for them directly.
- Do not cite Hiến pháp, scope articles, policy articles, legislative basis, or whole-law parent nodes for ordinary definition questions.
- Limit `Liên kết pháp lý liên quan` to at most 1-2 items in direct-question cases.
- Preserve statutory wording and legal terms from the source.

Prompt changes are necessary but not sufficient. They are the final guard, not the primary retrieval-control mechanism.

### 2. Rule-Based Query Router

Add a small query classification layer before calling LightRAG.

The first version should be rule-based and deterministic. Do not use an LLM classifier initially.

Recommended query classes:

| Class | Meaning | Retrieval policy |
|---|---|---|
| `direct_definition` | asks for definition, concept, source boundary, purpose directly stated in law | naive or hybrid-lite |
| `direct_rule` | asks for a specific rule, condition, obligation, prohibition, deadline, authority, or sanction | direct retrieval with limited graph context |
| `relational` | asks how provisions relate, asks for related legal basis, exceptions, consequences, comparisons, multi-hop reasoning | hybrid |
| `open_ended` | broad request without clear target | hybrid with strict context budget |

Example Vietnamese signals for `direct_definition`:

- `là gì`
- `khái niệm`
- `được hiểu như thế nào`
- `được xác định từ đâu`
- `nhằm mục đích gì`
- `có nghĩa là gì`

Example Vietnamese signals for `relational`:

- `liên quan đến`
- `căn cứ nào`
- `so sánh`
- `khác nhau`
- `trong trường hợp`
- `ngoại lệ`
- `hậu quả pháp lý`
- `trách nhiệm của`

Default behavior:

- If the router is uncertain, choose the safer low-expansion policy.
- Full hybrid graph expansion should be opt-in by query shape, not the default for every question.

### 3. Retrieval Policies

The router should map each query class to a retrieval policy.

#### `direct_definition`

Purpose:

- answer from direct statutory text
- avoid graph expansion noise

Recommended policy:

- prefer `mode="naive"` or a constrained hybrid context-only path
- use small `top_k` and `chunk_top_k`
- strongly limit entity/relation token budgets if hybrid context is used
- rerank candidates
- allow only 1-3 final context items

Expected behavior for the observed failure:

- keep Điều 2, khoản 5
- optionally keep Điều 3 only if it directly supports the stated purposes
- remove Điều 1, Điều 4, Hiến pháp

#### `direct_rule`

Purpose:

- find the exact applicable legal rule
- allow limited adjacent context only when it affects applicability

Recommended policy:

- use direct retrieval first
- allow supporting provisions only if they define terms, exceptions, sanctions, authority, conditions, or procedure needed for the answer
- rerank and gate candidates
- keep final context small

#### `relational`

Purpose:

- let GraphRAG answer the questions it is built for

Recommended policy:

- use `mode="hybrid"`
- retrieve broader candidate context
- rerank all candidates before answer synthesis
- allow more final items, but still reject broad background-only nodes

#### `open_ended`

Purpose:

- answer broad questions without flooding the answer

Recommended policy:

- use hybrid retrieval
- apply stricter output length and citation limits
- require answer to separate direct basis from connected explanation

## Reranker Design

Use `BAAI/bge-reranker-v2-m3` as the first reranker model.

### Role

The reranker is a second-stage cross-encoder style filter:

```text
candidate retrieval -> pair scoring(question, candidate_text) -> sorted candidates -> threshold gate
```

It should not search the entire corpus directly. It should only score candidates already retrieved by LightRAG.

### Candidate Input

Each candidate should contain enough text for scoring:

- article/clause title when available
- source document title when available
- chunk text or node text
- relationship text if the candidate came from graph context

Candidate text should be trimmed before reranking. Avoid sending huge context blocks as a single candidate because that hides irrelevant material inside a high-scoring passage.

### Score Use

Reranker output should drive two decisions:

- ordering: higher-scoring candidates appear first
- gating: low-scoring candidates are removed before answer generation

Recommended first-pass controls:

- per-query-class final context limits
- minimum score threshold
- relative score threshold compared with the best candidate
- hard caps on broad background categories

The exact numeric thresholds should be tuned from a small benchmark rather than guessed permanently.

### Deployment Shape

Preferred runtime shape:

- encapsulate reranking behind a small backend service/module interface
- make the model name configurable
- default model: `BAAI/bge-reranker-v2-m3`
- allow reranking to be disabled with configuration for local debugging

This avoids scattering model-specific calls through route code.

## Legal Relevance Gate

Reranker score alone is not enough. A broad policy article can be topically similar to a road-infrastructure question while still being poor evidence for the answer.

Add a deterministic legal relevance gate after reranking.

### Candidate Labels

Each candidate should be assigned one of:

| Label | Meaning | Final answer behavior |
|---|---|---|
| `direct` | directly answers at least one slot of the question | include first |
| `supporting` | defines, narrows, qualifies, or applies the direct answer | include only if useful |
| `background` | generally related but not needed | exclude by default |
| `noise` | unrelated or misleading | exclude |

### Question Slot Matching

For direct questions, split the user question into answer slots.

Example:

```text
Hành lang an toàn đường bộ được xác định từ đâu?
Việc thiết lập hành lang này nhằm mục đích gì?
```

A candidate is relevant only if it answers one of those slots or directly supports the slot answer.

### Background Demotion Rules

For `direct_definition` and `direct_rule`, demote or exclude candidates that are only:

- whole-law scope
- legislative basis
- constitutional basis
- general policy
- state management scope
- same-law or same-chapter parent context
- document metadata

These candidates may be allowed only when the user explicitly asks about scope, policy, authority basis, constitutional basis, or legislative context.

### Minimum Evidence Rule

If no candidate passes as `direct`, the system should not fill the answer with background context. It should say that the retrieved context does not provide enough direct legal basis.

## Answer Generation Contract

The final answer should receive only curated context.

Output sections:

- `Căn cứ chính`
- `Liên kết pháp lý liên quan` only when relevant context survives the gate

Rules:

- Direct definition questions should usually produce one concise `Căn cứ chính` section.
- `Liên kết pháp lý liên quan` should be absent or very short when the direct basis is sufficient.
- Every related legal link must state why it is relevant to the question.
- Do not cite provisions just because they are ancestors, siblings, or global basis nodes.
- Preserve legal wording from source context.
- If context is insufficient, say so directly.

The first implementation should keep `ChatResponse.response` as a string and keep SSE chunk shape unchanged.

## Backend Architecture

Recommended modules:

| Module | Responsibility |
|---|---|
| `backend/core/legal_prompts.py` | legal answer prompt templates and prompt guard rules |
| `backend/core/query_router.py` | classify user query into retrieval class and policy |
| `backend/core/reranker.py` | reranker interface and `BAAI/bge-reranker-v2-m3` implementation wrapper |
| `backend/core/legal_relevance.py` | candidate labeling, slot matching, background demotion, final context selection |
| `backend/api/routes.py` | orchestrate chat request using the above services |

The route should remain thin. It should not contain query classification rules, reranker scoring logic, or legal relevance heuristics inline.

## Data Flow

### Non-streaming single response

```text
POST /api/chat
 -> classify query
 -> choose retrieval policy
 -> retrieve candidate context from LightRAG
 -> rerank candidates
 -> legal relevance gate
 -> final LightRAG/bypass answer generation with curated context
 -> ChatResponse(response=..., mode=...)
```

### Streaming single response

Use the same retrieval, reranking, and gating steps before final answer streaming starts.

Streaming should begin only after curated context is ready. This may add a short pre-stream delay, but avoids streaming a noisy answer.

### Comparison mode

Comparison mode should remain useful for evaluation.

Recommended behavior:

- Naive column: current naive retrieval plus tightened naive prompt
- Hybrid column: controlled hybrid pipeline with router, reranker, and gate

For a `direct_definition` query, the hybrid column is allowed to behave similarly to naive if that is the correct legal behavior. Hybrid should not force extra graph links just to appear different.

## Configuration

Add configuration values instead of hardcoding model/runtime behavior.

Recommended settings:

```text
RERANKER_ENABLED=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_MAX_LENGTH=1024
RERANKER_DEVICE=auto
RERANKER_MIN_SCORE_DIRECT=
RERANKER_MIN_SCORE_RELATIONAL=
RERANKER_TOP_N_DIRECT=3
RERANKER_TOP_N_RELATIONAL=6
```

Threshold defaults may start conservative and be tuned with the benchmark. Empty threshold values are acceptable during early implementation if the code uses top-n plus relative ranking first.

## Evaluation

Create a small benchmark focused on the failure mode and the intended graph value.

Benchmark item shape:

```json
{
  "id": "road-safety-corridor-definition",
  "question": "Hành lang an toàn đường bộ được xác định từ đâu và việc thiết lập hành lang này nhằm mục đích gì?",
  "query_class": "direct_definition",
  "required_sources": ["Luật Đường bộ 35/2024/QH15 Điều 2 khoản 5"],
  "allowed_supporting_sources": ["Luật Đường bộ 35/2024/QH15 Điều 3 khoản 1"],
  "forbidden_sources": ["Hiến pháp", "Điều 1", "Điều 4"],
  "required_terms": [
    "dải đất dọc hai bên đất của đường bộ",
    "tính từ mép ngoài phần đất để bảo vệ, bảo trì đường bộ",
    "bảo đảm an toàn giao thông đường bộ",
    "bảo đảm tầm nhìn xe chạy",
    "hạn chế ảnh hưởng đến môi trường xung quanh"
  ],
  "forbidden_terms": [
    "lề đường",
    "giảm ảnh hưởng",
    "bảo vệ tầm nhìn"
  ]
}
```

Recommended split:

- 4 direct-definition/direct-rule questions
- 4 relational questions
- 2 open-ended legal questions

Success metrics:

- Direct questions keep required statutory terms.
- Direct questions do not include forbidden broad sources.
- Relational questions keep more useful supporting provisions than naive mode.
- Hybrid answers are not longer unless additional context is legally justified.

## Testing Strategy

Unit tests:

- query router classifies Vietnamese question patterns correctly
- prompt templates include direct-answer constraints and background-source prohibitions
- legal relevance gate demotes background candidates for `direct_definition`
- reranker wrapper can be mocked and its scores affect final ordering
- context selector enforces per-class final context limits

Route tests:

- `/api/chat` still returns the same response schema
- comparison mode still returns naive and hybrid outputs
- streaming still emits the same SSE event shape
- router policy is passed into the hybrid pipeline

Benchmark/regression checks:

- observed road-safety-corridor question excludes Điều 1, Điều 4, and Hiến pháp from the final answer
- observed question preserves the key terms from Điều 2, khoản 5
- at least one relational question still benefits from hybrid retrieval

## Acceptance Criteria

This phase is successful when:

1. the observed failure question produces a concise answer centered on Điều 2, khoản 5
2. Hybrid no longer cites Hiến pháp, Điều 1, or Điều 4 for that direct-definition question
3. `direct_definition` queries are routed to low-expansion retrieval
4. `relational` queries still use hybrid retrieval
5. reranking can be enabled with `BAAI/bge-reranker-v2-m3`
6. reranking can be disabled for local debugging without breaking chat
7. final answer generation receives curated context rather than raw over-expanded graph context
8. existing frontend response and SSE contracts remain stable

## Risks and Mitigations

### Risk: Rule-based router misclassifies questions

Impact:

- some questions may use too little or too much graph context

Mitigation:

- default uncertain cases to low-expansion retrieval
- keep comparison mode for quick manual inspection
- make routing rules small and test-covered

### Risk: Reranker scores broad topical passages highly

Impact:

- policy or scope articles may survive because they are semantically close

Mitigation:

- combine score with legal relevance labels
- demote known background categories for direct questions
- require direct slot coverage

### Risk: Added reranker latency

Impact:

- chat responses may start later, especially streaming

Mitigation:

- rerank only LightRAG candidates, not the full corpus
- cap candidate count by query class
- allow reranker disablement in configuration

### Risk: Hybrid appears similar to Naive on simple questions

Impact:

- demo may look less differentiated

Mitigation:

- treat this as correct behavior for direct legal lookup
- use relational benchmark questions to demonstrate graph value

## Out-of-Scope Follow-Ups

- graph schema redesign
- relation-type weighting at index time
- corpus re-ingestion
- statute-aware chunking by article/clause/point
- structured citation schema
- frontend graph-path visualization
- LLM-based query classification
- LLM-as-judge evaluation

## Implementation Notes

- Keep edits small and localized.
- Prefer new core modules over adding large logic blocks to `backend/api/routes.py`.
- Do not change existing frontend contracts in this phase.
- Do not change ingestion or storage.
- Use targeted tests before broad end-to-end runs.
- Treat benchmark output as regression evidence, not as a complete legal-quality evaluation.

This spec intentionally prioritizes retrieval discipline over graph sophistication. The goal is to make Hybrid GraphRAG answer only with graph context that earns its place in the legal answer.
