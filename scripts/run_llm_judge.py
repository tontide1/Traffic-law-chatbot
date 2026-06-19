# scripts/run_llm_judge.py
import argparse
import json
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

client = OpenAI()

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
        print(f"Error calling OpenAI: {e}")
        return None
