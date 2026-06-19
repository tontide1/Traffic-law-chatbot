import os
os.environ["OPENAI_API_KEY"] = "dummy"

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

from unittest.mock import patch, MagicMock
from scripts.run_llm_judge import evaluate_with_llm, EvaluationResult

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

@patch("scripts.run_llm_judge.client.beta.chat.completions.parse")
def test_evaluate_with_llm_error(mock_parse):
    # Mock the API throwing an error
    mock_parse.side_effect = Exception("API Error")

    result = evaluate_with_llm("Câu hỏi", "Ground truth", "Câu trả lời")
    assert result is None
