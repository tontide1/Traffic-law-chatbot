#!/usr/bin/env python3
"""Manual regression check for the legal benchmark against the live chat API.

This script loads ``data/legal_benchmark.json``, sends each selected benchmark
question to ``POST /api/chat`` with ``comparison_mode=True``, and evaluates both
returned answers with ``backend.core.benchmark_checker.check_answer``.

Usage:
    conda run -n legal_rag python scripts/run_regression_check.py
    conda run -n legal_rag python scripts/run_regression_check.py --index 0
    conda run -n legal_rag python scripts/run_regression_check.py --category exactness
    conda run -n legal_rag python scripts/run_regression_check.py --fixture path/to/file.json
    conda run -n legal_rag python scripts/run_regression_check.py --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core.benchmark_checker import check_answer, load_benchmark

DEFAULT_FIXTURE_PATH = REPO_ROOT / "data" / "legal_benchmark.json"
REQUEST_TIMEOUT_SECONDS = 120.0
PREVIEW_LENGTH = 280


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the legal benchmark against the live /api/chat route",
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--index",
        type=int,
        default=None,
        help="Run only the benchmark item at this 0-based index",
    )
    selection_group.add_argument(
        "--category",
        default=None,
        help="Run only benchmark items in the selected fixture-defined category",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="Path to the benchmark fixture JSON file",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the FastAPI backend",
    )
    return parser.parse_args()


def _normalize_api_url(api_url: str) -> str:
    return api_url.rstrip("/")


def _preview(text: str, limit: int = PREVIEW_LENGTH) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3]}..."


def _extract_comparison_answer(payload: dict[str, Any], mode: str) -> str:
    item = payload.get(mode)
    if not isinstance(item, dict):
        raise KeyError(f"Missing {mode!r} response in comparison payload")

    response = item.get("response")
    if not isinstance(response, str):
        raise KeyError(f"Missing {mode!r}.response string in comparison payload")

    return response


def _available_categories(items: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            category
            for item in items
            if isinstance((category := item.get("category")), str) and category
        }
    )


def _filter_items(items: list[dict[str, Any]], index: int | None, category: str | None) -> list[dict[str, Any]]:
    if index is not None:
        if index < 0 or index >= len(items):
            raise IndexError(f"index {index} out of range (0-{len(items) - 1})")
        return [items[index]]

    if category is not None:
        selected = [item for item in items if item.get("category") == category]
        if not selected:
            available = ", ".join(_available_categories(items)) or "none"
            raise ValueError(f"unknown category {category!r}; available categories: {available}")
        return selected

    return items


def _check_comparison_response(item: dict[str, Any], payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    naive_answer = _extract_comparison_answer(payload, "naive")
    hybrid_answer = _extract_comparison_answer(payload, "hybrid")
    return {
        "naive": {
            "answer": naive_answer,
            "check": check_answer(naive_answer, item),
        },
        "hybrid": {
            "answer": hybrid_answer,
            "check": check_answer(hybrid_answer, item),
        },
    }


def _print_check(mode: str, result: dict[str, Any]) -> bool:
    check = result["check"]
    required_terms = check["required_terms"]
    forbidden_paraphrases = check["forbidden_paraphrases"]
    expected_sources = check["expected_sources"]
    expected_answer_points = check["expected_answer_points"]
    passed = bool(check["overall_pass"])
    status = "PASS" if passed else "FAIL"
    print(f"  [{mode}] {status}")

    if not required_terms["pass"]:
        print(f"    missing required terms: {required_terms['missing']}")
    if not forbidden_paraphrases["pass"]:
        print(f"    forbidden paraphrases found: {forbidden_paraphrases['found']}")
    if not expected_sources["pass"]:
        print(f"    missing expected sources: {expected_sources['missing']}")
    if not expected_answer_points["pass"]:
        print(f"    missing expected answer points: {expected_answer_points['missing']}")

    print(f"    answer preview: {_preview(result['answer'])}")
    return passed


def main() -> int:
    args = parse_args()
    fixture_path = args.fixture.expanduser()

    try:
        items = load_benchmark(fixture_path)
    except FileNotFoundError:
        print(f"ERROR: benchmark fixture not found: {fixture_path}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        print(f"ERROR: failed to load benchmark fixture {fixture_path}: {exc}")
        return 1

    try:
        selected_items = _filter_items(items, args.index, args.category)
    except (IndexError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    if not selected_items:
        if args.category is not None:
            print(f"No benchmark items matched category {args.category!r}.")
        else:
            print("No benchmark items selected.")
        return 0

    api_url = _normalize_api_url(args.api_url)
    chat_url = f"{api_url}/api/chat"

    print(f"Loaded {len(items)} benchmark items from {fixture_path}")
    print(f"Selected {len(selected_items)} item(s)")
    print(f"Using live chat endpoint: {chat_url}")
    print("Sending each question with comparison_mode=True and stream=False")

    passed_checks = 0
    failed_checks = 0

    try:
        with httpx.Client(base_url=api_url, timeout=REQUEST_TIMEOUT_SECONDS) as client:
            for position, item in enumerate(selected_items, start=1):
                question = item.get("question", "")
                category = item.get("category", "unknown")
                print("\n" + "=" * 80)
                print(f"[{position}/{len(selected_items)}] {category}: {question}")
                print("=" * 80)

                response = client.post(
                    "/api/chat",
                    json={
                        "message": question,
                        "stream": False,
                        "comparison_mode": True,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                results = _check_comparison_response(item, payload)

                for mode in ("naive", "hybrid"):
                    if _print_check(mode, results[mode]):
                        passed_checks += 1
                    else:
                        failed_checks += 1
    except httpx.HTTPError as exc:
        print(f"ERROR: request to {chat_url} failed: {exc}")
        return 1
    except KeyError as exc:
        print(f"ERROR: malformed comparison response from {chat_url}: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        print(f"ERROR: unexpected failure while running regression check: {exc}")
        return 1

    total_checks = passed_checks + failed_checks
    print("\n" + "=" * 80)
    print(
        f"SUMMARY: {passed_checks} passed, {failed_checks} failed, {total_checks} total checks"
    )
    if failed_checks:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
