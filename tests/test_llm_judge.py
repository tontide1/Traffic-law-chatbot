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

@patch("scripts.run_llm_judge.OpenAI")
def test_evaluate_with_llm(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_parse = mock_client.beta.chat.completions.parse

    # Mock the return value of OpenAI structured outputs
    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = EvaluationResult(
        accuracy_score=5, comprehensiveness_score=5, connectivity_score=5, reasoning="Good"
    )
    mock_parse.return_value = mock_response

    result = evaluate_with_llm("Câu hỏi", "Ground truth", "Câu trả lời")
    assert result.accuracy_score == 5
    assert result.reasoning == "Good"

@patch("scripts.run_llm_judge.OpenAI")
def test_evaluate_with_llm_error(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_parse = mock_client.beta.chat.completions.parse

    # Mock the API throwing an error
    mock_parse.side_effect = Exception("API Error")

    result = evaluate_with_llm("Câu hỏi", "Ground truth", "Câu trả lời")
    assert result is None

from pathlib import Path
from unittest.mock import mock_open
from scripts.run_llm_judge import main

@patch("scripts.run_llm_judge.evaluate_with_llm")
@patch("builtins.open", new_callable=mock_open, read_data='{"id": "1", "question": "Q", "ground_truth": "GT", "naive_response": "N", "hybrid_response": "H"}\n')
@patch("scripts.run_llm_judge.argparse.ArgumentParser.parse_args")
def test_main_execution_loop(mock_args, mock_file, mock_evaluate):
    # Mock args
    mock_args.return_value = MagicMock(input=Path("input.jsonl"), output=Path("output.jsonl"))
    
    # Mock evaluate_with_llm to return dummy scores
    mock_evaluate.side_effect = [
        EvaluationResult(accuracy_score=5, comprehensiveness_score=5, connectivity_score=5, reasoning="N/A"),
        EvaluationResult(accuracy_score=4, comprehensiveness_score=4, connectivity_score=4, reasoning="N/A")
    ]
    
    main()
    
    assert mock_evaluate.call_count == 2
