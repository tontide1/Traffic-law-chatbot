# scripts/run_llm_judge.py
import argparse
import json
import os
from pathlib import Path
from pydantic import BaseModel, Field

class EvaluationResult(BaseModel):
    accuracy_score: int = Field(..., ge=1, le=5, description="Điểm chính xác từ 1-5")
    comprehensiveness_score: int = Field(..., ge=1, le=5, description="Điểm đầy đủ từ 1-5")
    connectivity_score: int = Field(..., ge=1, le=5, description="Điểm liên kết từ 1-5")
    reasoning: str = Field(..., description="Lý do chấm điểm (tiếng Việt)")
