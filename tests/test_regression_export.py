import argparse
import json
import os
import tempfile
from pathlib import Path
from scripts.run_regression_check import parse_args, export_to_jsonl

def test_parse_args_export():
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

from unittest.mock import patch
import sys
from scripts.run_regression_check import main

@patch("scripts.run_regression_check.export_to_jsonl")
@patch("scripts.run_regression_check.httpx.Client")
@patch("scripts.run_regression_check.load_benchmark")
@patch("scripts.run_regression_check.check_answer")
def test_integration_calls_export(mock_check, mock_load, mock_client_class, mock_export):
    mock_load.return_value = [{
        "id": "123",
        "question": "test q",
        "ground_truth": "truth",
        "category": "test"
    }]
    
    mock_client = mock_client_class.return_value.__enter__.return_value
    mock_client.post.return_value.json.return_value = {
        "naive": {"response": "naive_ans"},
        "hybrid": {"response": "hybrid_ans"}
    }
    
    mock_check.return_value = {
        "overall_pass": True,
        "required_terms": {"pass": True, "missing": []},
        "forbidden_paraphrases": {"pass": True, "found": []},
        "expected_sources": {"pass": True, "missing": []},
        "expected_answer_points": {"pass": True, "missing": []},
        "forbidden_sources": {"pass": True, "found": []}
    }
    
    with patch.object(sys, "argv", ["run_regression_check.py", "--export", "dummy.jsonl"]):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0
            
    mock_export.assert_called_once()
    args, kwargs = mock_export.call_args
    assert args[0] == Path("dummy.jsonl")
    
    record = args[1]
    assert record["id"] == "123"
    assert record["question"] == "test q"
    assert record["ground_truth"] == "truth"
    assert record["naive_response"] == "naive_ans"
    assert record["hybrid_response"] == "hybrid_ans"
    assert record["naive_deterministic_pass"] is True
    assert record["hybrid_deterministic_pass"] is True

