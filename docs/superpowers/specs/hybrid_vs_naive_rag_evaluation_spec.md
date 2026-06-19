# Specification: Hybrid GraphRAG vs Naive RAG Evaluation

## 1. Overview and Objective
This specification defines the evaluation framework to compare the performance of Hybrid GraphRAG against Naive RAG in the Vietnamese Traffic Law Assistant project. The objective is to quantify and analyze the strengths of GraphRAG, particularly in "connecting the dots" across multi-document legal scenarios.

## 2. Evaluation Methodology (Hybrid Evaluation)
We will adopt a **Hybrid Evaluation** approach, combining two layers of checks:
- **Deterministic Layer**: Validates exact legal terms, required phrases, and referenced sources (using the existing `benchmark_checker.py`).
- **Semantic Layer (LLM-as-a-judge)**: Uses an LLM (OpenAI GPT-4o/mini) to evaluate the depth, accuracy, and reasoning of the answers.

## 3. LLM-as-a-judge Criteria
The LLM Judge will score the answers from 1 to 5 based on a custom Vietnamese legal rubric focusing on three specific dimensions:
- **Tính Chính xác (Accuracy)**: Câu trả lời có đúng với Ground Truth không? Có bịa thông tin (hallucinate) hay hiểu sai luật không?
- **Tính Đầy đủ (Comprehensiveness)**: Câu trả lời có bao quát đủ các trường hợp rải rác trong luật không?
- **Khả năng Liên kết (Connectivity)**: Câu trả lời có chỉ ra được mối liên hệ giữa các điều/luật hoặc các chủ thể không? (Đây là tiêu chí cốt lõi để làm nổi bật thế mạnh của GraphRAG).

## 4. Pipeline Architecture (Two-Step Pipeline)
The evaluation will be broken down into two decoupled scripts to optimize API costs and allow for prompt iteration without re-querying the backend.

### Step 1: Answer Generation (`scripts/run_regression_check.py`)
- Reads questions from `data/legal_benchmark.json`.
- Queries the live API (`POST /api/chat` with `comparison_mode=True`).
- Runs the existing deterministic checks.
- **New Feature**: Exports a consolidated results file `data/predictions.jsonl` containing the question, naive_response, hybrid_response, and deterministic results.

### Step 2: LLM Evaluation (`scripts/run_llm_judge.py`)
- Reads the generated `data/predictions.jsonl`.
- Loads the custom Vietnamese evaluation prompt.
- Calls the OpenAI API (GPT-4o) using **Structured Outputs** to guarantee parsable JSON evaluation scores and reasoning.
- Aggregates scores and outputs a final evaluation report comparing Naive vs Hybrid.

## 5. Required Implementation Steps

### Step 5.1: Update Benchmark Data
- **File**: `data/legal_benchmark.json`
- **Action**: Add a `ground_truth` field to benchmark questions where applicable. This provides the LLM Judge with an absolute baseline for accurate evaluation.

### Step 5.2: Update Regression Script
- **File**: `scripts/run_regression_check.py`
- **Action**: Add an `--export` or `--output` flag. When enabled, append the evaluation results and raw answers to `data/predictions.jsonl`.

### Step 5.3: Create LLM Judge Script
- **File**: `scripts/run_llm_judge.py`
- **Action**: 
  - Define the evaluation system prompt.
  - Setup OpenAI API client with Pydantic schemas for structured output:
    ```python
    class EvaluationResult(BaseModel):
        accuracy_score: int
        comprehensiveness_score: int
        connectivity_score: int
        reasoning: str
    ```
  - Parse `predictions.jsonl`, run evaluations concurrently, and print a comparative summary matrix.
