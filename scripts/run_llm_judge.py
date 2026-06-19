# scripts/run_llm_judge.py
import argparse
import json
import logging
import os
from pathlib import Path
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Optional

class EvaluationResult(BaseModel):
    accuracy_score: int = Field(..., ge=1, le=5, description="Điểm chính xác từ 1-5")
    comprehensiveness_score: int = Field(..., ge=1, le=5, description="Điểm đầy đủ từ 1-5")
    connectivity_score: int = Field(..., ge=1, le=5, description="Điểm liên kết từ 1-5")
    reasoning: str = Field(..., description="Lý do chấm điểm (tiếng Việt)")

def evaluate_with_llm(question: str, ground_truth: str, answer: str, model: str = "gpt-4o-2024-08-06") -> Optional[EvaluationResult]:
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
    client = OpenAI()
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là giám khảo chấm điểm."},
                {"role": "user", "content": prompt}
            ],
            response_format=EvaluationResult,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        logging.error(f"Error calling OpenAI: {e}")
        return None

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
        for line in f_in:
            if not line.strip():
                continue
            record = json.loads(line)
            
            # Evaluate Naive
            naive_eval = evaluate_with_llm(record["question"], record["ground_truth"], record["naive_response"])
            if naive_eval:
                record["naive_scores"] = naive_eval.model_dump()
            
            # Evaluate Hybrid
            hybrid_eval = evaluate_with_llm(record["question"], record["ground_truth"], record["hybrid_response"])
            if hybrid_eval:
                record["hybrid_scores"] = hybrid_eval.model_dump()
                
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Evaluated ID {record.get('id')}")

if __name__ == "__main__":
    main()
