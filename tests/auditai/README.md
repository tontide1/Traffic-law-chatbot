# AuditAI guerrilla scaffold — tontide1/Traffic-law-chatbot

- [ ] Review `dataset.json` — remove TODOs; keep only public-docs questions
- [ ] Prefer real HTTP target when available; else use `mock_adapter.py`
- [ ] Mock returns **empty contexts** on purpose (faithfulness uses dataset contexts)
- [ ] For PR numbers: set `judge.provider` to `xai` or `openai` (not mock)
- [ ] `python tests/auditai/mock_adapter.py` + `auditai run --config tests/auditai/auditai.yml`
- [ ] Check report `judge_usage` (tokens in/out/total)
- [ ] Fill PR body via `fill_pr_body.py` — never invent metrics
- [ ] Badge only if maintainer wants it
- [ ] If push rejects workflows: use `workflow-auditai.yml.example` (needs PAT `workflow` scope)

Commands:

```bash
python tests/auditai/mock_adapter.py &
export XAI_API_KEY=...   # or OPENAI_API_KEY
# edit auditai.yml judge.provider = xai|openai
auditai run --config tests/auditai/auditai.yml
```
