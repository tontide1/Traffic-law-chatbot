import json
import os
from collections import Counter

import pytest

from backend.core.benchmark_checker import (
    check_answer,
    check_expected_answer_points,
    check_expected_sources,
    check_forbidden_paraphrases,
    check_forbidden_sources,
    check_required_terms,
    load_benchmark,
    normalize_text,
)

EXPECTED_REAL_BENCHMARK_CATEGORIES = {"exactness", "graph_strength"}


SAMPLE_FIXTURE = [
    {
        "question": "Hành lang an toàn đường bộ là gì?",
        "category": "exactness",
        "expected_sources": ["Luật Trật tự, an toàn giao thông đường bộ"],
        "required_terms": ["đất của đường bộ", "hành lang an toàn đường bộ"],
        "forbidden_paraphrases": ["lề đường", "bảo vệ tầm nhìn"],
        "expected_answer_points": ["phần đất dọc hai bên đường bộ"],
    }
]


def _write_fixture(tmp_path, data):
    path = os.path.join(tmp_path, "benchmark.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return path


def test_load_benchmark_returns_list(tmp_path):
    path = _write_fixture(str(tmp_path), SAMPLE_FIXTURE)
    items = load_benchmark(path)
    assert len(items) == 1
    assert items[0]["category"] == "exactness"


def test_load_real_benchmark_fixture():
    fixture_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "legal_benchmark.json"
    )
    items = load_benchmark(fixture_path)
    category_counts = Counter(item["category"] for item in items)
    assert items
    assert EXPECTED_REAL_BENCHMARK_CATEGORIES.issubset(category_counts)
    assert all(category_counts[category] > 0 for category in EXPECTED_REAL_BENCHMARK_CATEGORIES)
    for item in items:
        assert "id" in item
        assert "question" in item
        assert "query_class" in item
        assert "required_terms" in item
        assert "forbidden_paraphrases" in item
        assert "forbidden_sources" in item
        assert "expected_answer_points" in item


def test_normalize_text_casefolds_and_collapses_whitespace():
    assert normalize_text("  Đất   Của  ") == "đất của"


def test_check_required_terms_pass():
    answer = "Hành lang an toàn đường bộ là phần đất của đường bộ dọc hai bên."
    result = check_required_terms(answer, ["đất của đường bộ", "hành lang an toàn đường bộ"])
    assert result["pass"] is True
    assert result["missing"] == []


def test_check_required_terms_fail():
    answer = "Hành lang an toàn đường bộ là phần lề đường."
    result = check_required_terms(answer, ["đất của đường bộ"])
    assert result["pass"] is False
    assert "đất của đường bộ" in result["missing"]


def test_check_forbidden_paraphrases_pass():
    answer = "Hành lang an toàn đường bộ là phần đất của đường bộ."
    result = check_forbidden_paraphrases(answer, ["lề đường", "bảo vệ tầm nhìn"])
    assert result["pass"] is True
    assert result["found"] == []


def test_check_forbidden_paraphrases_fail():
    answer = "Hành lang an toàn đường bộ là phần lề đường dọc hai bên."
    result = check_forbidden_paraphrases(answer, ["lề đường"])
    assert result["pass"] is False
    assert "lề đường" in result["found"]


@pytest.mark.parametrize(
    "answer, expected_sources, expected_pass, expected_missing",
    [
        (
            "Hành lang an toàn đường bộ...\n\n"
            "**Tham chiếu**\n"
            "- [1] Luật Trật tự, an toàn giao thông đường bộ\n",
            ["Luật Trật tự, an toàn giao thông đường bộ"],
            True,
            [],
        ),
        (
            "Theo Luật Trật tự, an toàn giao thông đường bộ, hành lang an toàn...",
            ["Luật Trật tự, an toàn giao thông đường bộ"],
            False,
            ["Luật Trật tự, an toàn giao thông đường bộ"],
        ),
        (
            "Hành lang an toàn...\n\n"
            "**Tham chiếu**\n"
            "- [1] Luật Đường bộ\n",
            ["Luật Trật tự, an toàn giao thông đường bộ"],
            False,
            ["Luật Trật tự, an toàn giao thông đường bộ"],
        ),
        (
            "Hành lang an toàn đường bộ là phần đất của đường bộ.",
            ["Luật Trật tự, an toàn giao thông đường bộ"],
            False,
            ["Luật Trật tự, an toàn giao thông đường bộ"],
        ),
        (
            "Hành lang an toàn...\n\n"
            "**Tham chiếu**\n"
            "Theo Luật Trật tự, an toàn giao thông đường bộ...\n",
            ["Luật Trật tự, an toàn giao thông đường bộ"],
            False,
            ["Luật Trật tự, an toàn giao thông đường bộ"],
        ),
    ],
)
def test_check_expected_sources_cases(answer, expected_sources, expected_pass, expected_missing):
    result = check_expected_sources(answer, expected_sources)
    assert result["pass"] is expected_pass
    assert result["missing"] == expected_missing


def test_check_expected_answer_points_pass():
    answer = "Hành lang an toàn đường bộ là phần đất dọc hai bên đường bộ."
    result = check_expected_answer_points(answer, ["phần đất dọc hai bên đường bộ"])
    assert result["pass"] is True
    assert result["missing"] == []


def test_check_expected_answer_points_fail():
    answer = "Hành lang an toàn đường bộ là phần đất của đường bộ."
    result = check_expected_answer_points(answer, ["phần đất dọc hai bên đường bộ"])
    assert result["pass"] is False
    assert result["missing"] == ["phần đất dọc hai bên đường bộ"]


def test_check_answer_combines_all_checks():
    answer = (
        "Hành lang an toàn đường bộ là phần đất của đường bộ. "
        "Đây là phần đất dọc hai bên đường bộ.\n\n"
        "**Tham chiếu**\n"
        "- [1] Luật Trật tự, an toàn giao thông đường bộ\n"
    )
    item = SAMPLE_FIXTURE[0]
    result = check_answer(answer, item)
    assert result["required_terms"]["pass"] is True
    assert result["forbidden_paraphrases"]["pass"] is True
    assert result["expected_sources"]["pass"] is True
    assert result["expected_answer_points"]["pass"] is True
    assert result["overall_pass"] is True


def test_check_answer_fails_on_missing_expected_answer_points():
    answer = (
        "Hành lang an toàn đường bộ là phần đất của đường bộ.\n\n"
        "**Tham chiếu**\n"
        "- [1] Luật Trật tự, an toàn giao thông đường bộ\n"
    )
    item = SAMPLE_FIXTURE[0]
    result = check_answer(answer, item)
    assert result["expected_answer_points"]["pass"] is False
    assert result["overall_pass"] is False


def test_check_answer_fails_on_forbidden():
    answer = (
        "Hành lang an toàn đường bộ là phần đất của đường bộ. Nó bảo vệ tầm nhìn cho người lái.\n\n"
        "**Tham chiếu**\n"
        "- [1] Luật Trật tự, an toàn giao thông đường bộ\n"
    )
    item = SAMPLE_FIXTURE[0]
    result = check_answer(answer, item)
    assert result["forbidden_paraphrases"]["pass"] is False
    assert result["overall_pass"] is False


def test_check_forbidden_sources_pass():
    answer = "Căn cứ chính: Điều 2 khoản 5 Luật Đường bộ."

    result = check_forbidden_sources(answer, ["Hiến pháp", "Điều 4"])

    assert result["pass"] is True
    assert result["found"] == []


def test_check_forbidden_sources_fail():
    answer = "Liên kết pháp lý liên quan: Hiến pháp là căn cứ ban hành Luật Đường bộ."

    result = check_forbidden_sources(answer, ["Hiến pháp", "Điều 4"])

    assert result["pass"] is False
    assert result["found"] == ["Hiến pháp"]


def test_check_answer_combines_forbidden_sources_with_existing_checks():
    answer = (
        "Hành lang an toàn đường bộ là dải đất dọc hai bên đất của đường bộ.\n\n"
        "**Tham chiếu**\n"
        "- [1] Luật Đường bộ\n\n"
        "Liên kết pháp lý liên quan: Hiến pháp là căn cứ chung."
    )
    item = {
        "required_terms": ["dải đất dọc hai bên đất của đường bộ"],
        "forbidden_paraphrases": [],
        "forbidden_sources": ["Hiến pháp", "Điều 4"],
        "expected_sources": ["Luật Đường bộ"],
        "expected_answer_points": [],
    }

    result = check_answer(answer, item)

    assert result["required_terms"]["pass"] is True
    assert result["forbidden_sources"]["pass"] is False
    assert result["forbidden_sources"]["found"] == ["Hiến pháp"]
    assert result["overall_pass"] is False
