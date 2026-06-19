# scripts/run_llm_judge.py
import argparse
import json
import logging
import os
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI, OpenAIError
from typing import Optional

REQUIRED_FIELDS = ("question", "ground_truth", "naive_response", "hybrid_response")
_openai_client: Optional[OpenAI] = None

class EvaluationResult(BaseModel):
    accuracy_score: int = Field(..., ge=1, le=5, description="Điểm chính xác từ 1-5")
    comprehensiveness_score: int = Field(..., ge=1, le=5, description="Điểm đầy đủ từ 1-5")
    connectivity_score: int = Field(..., ge=1, le=5, description="Điểm liên kết từ 1-5")
    reasoning: str = Field(..., description="Lý do chấm điểm (tiếng Việt)")

def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client

def evaluate_with_llm(
    question: str,
    ground_truth: str,
    answer: str,
    model: str = "gpt-4o-2024-08-06",
    record_id: Optional[str] = None,
    response_kind: Optional[str] = None,
    client: Optional[OpenAI] = None,
) -> Optional[EvaluationResult]:
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
    try:
        client = client or get_openai_client()
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là giám khảo chấm điểm."},
                {"role": "user", "content": prompt}
            ],
            response_format=EvaluationResult,
        )
        return completion.choices[0].message.parsed
    except (OpenAIError, ValidationError) as e:
        logging.error(
            "OpenAI evaluation failed for record_id=%s response_kind=%s: %s",
            record_id,
            response_kind,
            e,
        )
        return None

def validate_record(record: dict) -> list[str]:
    return [field for field in REQUIRED_FIELDS if not record.get(field)]

def mark_failed_evaluation(record: dict, response_kind: str, reason: str) -> None:
    record[f"{response_kind}_scores"] = None
    record[f"{response_kind}_evaluation_status"] = "failed"
    record[f"{response_kind}_evaluation_error"] = reason

def mark_successful_evaluation(record: dict, response_kind: str, result: EvaluationResult) -> None:
    record[f"{response_kind}_scores"] = result.model_dump()
    record[f"{response_kind}_evaluation_status"] = "completed"
    record.pop(f"{response_kind}_evaluation_error", None)

def evaluate_record(record: dict) -> dict:
    naive_eval = evaluate_with_llm(
        record["question"],
        record["ground_truth"],
        record["naive_response"],
        record_id=record.get("id"),
        response_kind="naive",
    )
    if naive_eval:
        mark_successful_evaluation(record, "naive", naive_eval)
    else:
        mark_failed_evaluation(record, "naive", "llm_evaluation_failed")

    hybrid_eval = evaluate_with_llm(
        record["question"],
        record["ground_truth"],
        record["hybrid_response"],
        record_id=record.get("id"),
        response_kind="hybrid",
    )
    if hybrid_eval:
        mark_successful_evaluation(record, "hybrid", hybrid_eval)
    else:
        mark_failed_evaluation(record, "hybrid", "llm_evaluation_failed")

    if record["naive_evaluation_status"] == record["hybrid_evaluation_status"] == "completed":
        record["evaluation_status"] = "completed"
    else:
        record["evaluation_status"] = "failed"

    return record

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG outputs using LLM Judge")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL from regression check")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL with scores")
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.input, "r", encoding="utf-8") as f_in, open(args.output, "w", encoding="utf-8") as f_out:
        for line_number, line in enumerate(f_in, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logging.error("Invalid JSON on line %s: %s", line_number, e)
                record = {
                    "line_number": line_number,
                    "evaluation_status": "invalid_json",
                    "evaluation_error": str(e),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            missing_fields = validate_record(record)
            if missing_fields:
                logging.error(
                    "Skipping invalid record line=%s id=%s missing_fields=%s",
                    line_number,
                    record.get("id"),
                    ",".join(missing_fields),
                )
                record["evaluation_status"] = "invalid_record"
                record["evaluation_error"] = "missing required fields: " + ", ".join(missing_fields)
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            record = evaluate_record(record)
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Evaluated ID {record.get('id')}")

if __name__ == "__main__":
    main()
