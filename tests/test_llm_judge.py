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

import json
from unittest.mock import patch, MagicMock
from openai import OpenAIError
from scripts.run_llm_judge import evaluate_record, evaluate_with_llm, EvaluationResult

def test_evaluate_with_llm():
    mock_client = MagicMock()
    mock_parse = mock_client.beta.chat.completions.parse

    # Mock the return value of OpenAI structured outputs
    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = EvaluationResult(
        accuracy_score=5, comprehensiveness_score=5, connectivity_score=5, reasoning="Good"
    )
    mock_parse.return_value = mock_response

    result = evaluate_with_llm("Câu hỏi", "Ground truth", "Câu trả lời", client=mock_client)
    assert result.accuracy_score == 5
    assert result.reasoning == "Good"

def test_evaluate_with_llm_reuses_single_openai_client():
    with patch("scripts.run_llm_judge._openai_client", None), patch("scripts.run_llm_judge.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.parsed = EvaluationResult(
            accuracy_score=5, comprehensiveness_score=5, connectivity_score=5, reasoning="Good"
        )
        mock_client.beta.chat.completions.parse.return_value = mock_response

        evaluate_with_llm("Q1", "GT", "A1")
        evaluate_with_llm("Q2", "GT", "A2")

    mock_openai_class.assert_called_once()

def test_evaluate_with_llm_error():
    mock_client = MagicMock()
    mock_parse = mock_client.beta.chat.completions.parse

    # Mock the API throwing an error
    mock_parse.side_effect = OpenAIError("API Error")

    result = evaluate_with_llm("Câu hỏi", "Ground truth", "Câu trả lời", client=mock_client)
    assert result is None

@patch("scripts.run_llm_judge.evaluate_with_llm")
def test_evaluate_record_marks_failed_evaluation(mock_evaluate):
    record = {"id": "1", "question": "Q", "ground_truth": "GT", "naive_response": "N", "hybrid_response": "H"}
    mock_evaluate.side_effect = [
        None,
        EvaluationResult(accuracy_score=4, comprehensiveness_score=4, connectivity_score=4, reasoning="N/A"),
    ]

    result = evaluate_record(record)

    assert result["evaluation_status"] == "failed"
    assert result["naive_scores"] is None
    assert result["naive_evaluation_status"] == "failed"
    assert result["naive_evaluation_error"] == "llm_evaluation_failed"
    assert result["hybrid_evaluation_status"] == "completed"

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

@patch("builtins.open", new_callable=mock_open, read_data='{"id": "1", "question": "Q"}\n{bad json}\n')
@patch("scripts.run_llm_judge.argparse.ArgumentParser.parse_args")
def test_main_flags_invalid_records(mock_args, mock_file):
    mock_args.return_value = MagicMock(input=Path("input.jsonl"), output=Path("output.jsonl"))

    main()

    handle = mock_file()
    output_records = [json.loads(call.args[0]) for call in handle.write.call_args_list]
    assert output_records[0]["evaluation_status"] == "invalid_record"
    assert "ground_truth" in output_records[0]["evaluation_error"]
    assert output_records[1]["evaluation_status"] == "invalid_json"
