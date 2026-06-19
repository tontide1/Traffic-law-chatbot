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
