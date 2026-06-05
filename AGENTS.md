# Agent Instructions

## Core coding behavior

When working in this repository:

- Prefer simple, surgical changes over broad refactors.
- Preserve existing project style, naming, formatting, and architecture unless explicitly asked to change them.
- Do not introduce speculative abstractions, new frameworks, or unnecessary configuration.
- For bugs, first identify the likely root cause, then make the smallest safe fix.
- For non-trivial changes, state the plan, edit the code, then run the smallest relevant verification command.
- Do not modify secrets, credentials, `.env` files, generated artifacts, model weights, raw datasets, or production data unless explicitly instructed.
- Treat tool outputs as advisory. Verify changes against the repository, tests, logs, and runtime behavior.

## Environment

Preferred local environment:

- Using python 3.11 with

```bash
conda activate legal-rag
```
