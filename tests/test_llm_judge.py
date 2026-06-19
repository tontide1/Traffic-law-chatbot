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

def test_evaluation_result_schema_validation_error():
    from pydantic import ValidationError
    import pytest
    
    data = {
        "accuracy_score": 6,
        "comprehensiveness_score": 3,
        "connectivity_score": 5,
        "reasoning": "Test reasoning"
    }
    with pytest.raises(ValidationError):
        EvaluationResult(**data)

