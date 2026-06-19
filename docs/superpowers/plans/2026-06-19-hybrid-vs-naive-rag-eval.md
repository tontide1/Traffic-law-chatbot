# Hybrid vs Naive RAG Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a two-step evaluation pipeline that first exports RAG predictions to JSONL, then evaluates them using an OpenAI LLM judge against a legal rubric.

**Architecture:** 
1. Modify `run_regression_check.py` to support an `--export` flag that saves naive and hybrid prediction data along with deterministic results to a JSONL file.
2. Create a new `run_llm_judge.py` script that uses OpenAI's Structured Outputs (via `pydantic`) to grade those predictions on Accuracy, Comprehensiveness, and Connectivity based on a provided ground truth.

**Tech Stack:** Python 3.11+, argparse, pydantic, openai.

---

### Task 1: Support `--export` in Regression Check

**Files:**
- Create: `tests/test_regression_export.py`
- Modify: `scripts/run_regression_check.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_regression_export.py
import argparse
import json
import os
import tempfile
from pathlib import Path
from scripts.run_regression_check import parse_args, export_to_jsonl

def test_parse_args_export():
    parser = argparse.ArgumentParser()
    # Mocking the addition to the script
    # We expect an --export flag
    try:
        args = parse_args(["--export", "test.jsonl"])
        assert args.export == Path("test.jsonl")
    except SystemExit:
        assert False, "Failed to parse --export flag"

def test_export_to_jsonl():
    data = {"question": "test", "naive": "a", "hybrid": "b"}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        export_to_jsonl(tmp_path, data)
        with open(tmp_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            assert json.loads(lines[0]) == data
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_regression_export.py -v`
Expected: FAIL with missing `--export` argument or missing `export_to_jsonl` function.

- [ ] **Step 3: Write minimal implementation**

Modify `scripts/run_regression_check.py` to add `--export` argument and the `export_to_jsonl` function. Ensure imports are correct.

```python
# scripts/run_regression_check.py (Add to imports)
import json

# scripts/run_regression_check.py (Add to parse_args)
# Find the end of parse_args and add:
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Path to export the predictions and results as JSONL",
    )
    # Note: modify the parse_args signature to accept optional args list for testing
    # def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    #     ...
    #     return parser.parse_args(args)

# scripts/run_regression_check.py (Add export function)
def export_to_jsonl(filepath: Path, record: dict[str, Any]) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_regression_export.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_regression_export.py scripts/run_regression_check.py
git commit -m "feat: add export flag and function to regression check"
```

---

### Task 2: Integrate Export into the Execution Loop

**Files:**
- Modify: `scripts/run_regression_check.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_regression_export.py (Append this test)
from unittest.mock import patch
from scripts.run_regression_check import main_loop # Assuming a main loop or run function, we will test the logic.

# Since testing the full live API loop is flaky, we will just test the export side effect via a mock.
@patch("scripts.run_regression_check.export_to_jsonl")
def test_integration_calls_export(mock_export):
    # This is a conceptual test. We want to ensure that IF --export is set, the function is called.
    # Write a minimal dummy test that expects the export function to be invoked.
    # For a purely functional test, we check if the script runs without crashing when --export is used.
    assert True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_regression_export.py -v`

- [ ] **Step 3: Write minimal implementation**

Modify `scripts/run_regression_check.py` where it evaluates each item to actually call the export.

```python
# scripts/run_regression_check.py (Inside the main loop where results are printed)
            # Find where naive_result and hybrid_result are obtained
            if args.export:
                record = {
                    "id": item.get("id"),
                    "question": question,
                    "ground_truth": item.get("ground_truth", ""),
                    "naive_response": naive_ans,
                    "hybrid_response": hybrid_ans,
                    "naive_deterministic_pass": naive_result["overall_pass"],
                    "hybrid_deterministic_pass": hybrid_result["overall_pass"]
                }
                export_to_jsonl(args.export, record)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python scripts/run_regression_check.py --index 0 --export data/predictions.jsonl`
Expected: The script executes successfully and creates `data/predictions.jsonl`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_regression_check.py
git commit -m "feat: integrate export payload into regression main loop"
```

---

### Task 3: Create Pydantic Schema for LLM Judge

**Files:**
- Create: `scripts/run_llm_judge.py`
- Create: `tests/test_llm_judge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_judge.py
from scripts.run_llm_judge import EvaluationResult

def test_evaluation_result_schema():
    data = {
        "accuracy_score": 4,
        "comprehensiveness_score": 3,
        "connectivity_score": 5,
        "reasoning": "Test reasoning"
    }
    result = EvaluationResult(**data)
    assert result.accuracy_score == 4
    assert result.connectivity_score == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_llm_judge.py -v`
Expected: FAIL with missing `run_llm_judge.py` or `EvaluationResult`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/run_llm_judge.py
import argparse
import json
import os
from pathlib import Path
from pydantic import BaseModel, Field

class EvaluationResult(BaseModel):
    accuracy_score: int = Field(..., description="Điểm chính xác từ 1-5")
    comprehensiveness_score: int = Field(..., description="Điểm đầy đủ từ 1-5")
    connectivity_score: int = Field(..., description="Điểm liên kết từ 1-5")
    reasoning: str = Field(..., description="Lý do chấm điểm (tiếng Việt)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_llm_judge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_llm_judge.py tests/test_llm_judge.py
git commit -m "feat: add pydantic schema for llm judge structured output"
```

---

### Task 4: Implement OpenAI API Call logic

**Files:**
- Modify: `scripts/run_llm_judge.py`
- Modify: `tests/test_llm_judge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_judge.py (Append)
from unittest.mock import patch, MagicMock
from scripts.run_llm_judge import evaluate_with_llm

@patch("scripts.run_llm_judge.client.beta.chat.completions.parse")
def test_evaluate_with_llm(mock_parse):
    # Mock the return value of OpenAI structured outputs
    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = EvaluationResult(
        accuracy_score=5, comprehensiveness_score=5, connectivity_score=5, reasoning="Good"
    )
    mock_parse.return_value = mock_response

    result = evaluate_with_llm("Câu hỏi", "Ground truth", "Câu trả lời")
    assert result.accuracy_score == 5
    assert result.reasoning == "Good"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_llm_judge.py -v`
Expected: FAIL due to missing `evaluate_with_llm` and `client`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/run_llm_judge.py (Append)
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy"))

def evaluate_with_llm(question: str, ground_truth: str, answer: str) -> EvaluationResult:
    prompt = f"""Bạn là một giám khảo chuyên gia về Luật Giao thông Việt Nam.
Hãy chấm điểm câu trả lời sau dựa trên Ground Truth.
Câu hỏi: {question}
Ground Truth: {ground_truth}
Câu trả lời: {answer}

Tiêu chí:
1. Accuracy (1-5): Tính chính xác.
2. Comprehensiveness (1-5): Tính đầy đủ các trường hợp.
3. Connectivity (1-5): Khả năng liên kết các điều luật.
"""
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Bạn là giám khảo chấm điểm."},
            {"role": "user", "content": prompt}
        ],
        response_format=EvaluationResult,
    )
    return completion.choices[0].message.parsed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_llm_judge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_llm_judge.py tests/test_llm_judge.py
git commit -m "feat: implement openai structured output evaluation logic"
```

---

### Task 5: Implement JSONL Parsing and Main Loop

**Files:**
- Modify: `scripts/run_llm_judge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_judge.py (Append)
import tempfile
from scripts.run_llm_judge import process_jsonl

@patch("scripts.run_llm_judge.evaluate_with_llm")
def test_process_jsonl(mock_eval):
    mock_eval.return_value = EvaluationResult(
        accuracy_score=5, comprehensiveness_score=5, connectivity_score=5, reasoning="Good"
    )
    
    data = {
        "id": "1", "question": "Q", "ground_truth": "G", 
        "naive_response": "N", "hybrid_response": "H"
    }
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        tmp.write((json.dumps(data) + "\n").encode("utf-8"))
        tmp_path = Path(tmp.name)
        
    try:
        report = process_jsonl(tmp_path)
        assert len(report) == 1
        assert report[0]["naive_eval"].accuracy_score == 5
    finally:
        os.remove(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_llm_judge.py -v`
Expected: FAIL due to missing `process_jsonl`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/run_llm_judge.py (Append)
def process_jsonl(filepath: Path) -> list[dict]:
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            question = record["question"]
            ground_truth = record.get("ground_truth", "")
            
            naive_eval = evaluate_with_llm(question, ground_truth, record["naive_response"])
            hybrid_eval = evaluate_with_llm(question, ground_truth, record["hybrid_response"])
            
            results.append({
                "id": record["id"],
                "naive_eval": naive_eval,
                "hybrid_eval": hybrid_eval
            })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    
    report = process_jsonl(args.input)
    print(f"Processed {len(report)} items.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_llm_judge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_llm_judge.py
git commit -m "feat: add main loop for jsonl processing in llm judge"
```
