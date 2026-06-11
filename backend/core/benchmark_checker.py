"""Deterministic benchmark loader and checker for legal answer evaluation."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


_THAM_CHIEU_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\*\*)?Tham chiếu(?:\*\*)?\s*$", re.IGNORECASE)
_REFERENCE_ENTRY_RE = re.compile(r"^\s*-\s*\[\d+\]\s+(.+?)\s*$")


def load_benchmark(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Benchmark fixture must be a JSON list")
    return data


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _find_matching_terms(answer: str, terms: list[str]) -> list[str]:
    normalized_answer = normalize_text(answer)
    return [term for term in terms if normalize_text(term) in normalized_answer]


def check_required_terms(answer: str, terms: list[str]) -> dict:
    found = _find_matching_terms(answer, terms)
    missing = [term for term in terms if term not in found]
    return {"pass": not missing, "missing": missing}


def check_expected_answer_points(answer: str, points: list[str]) -> dict:
    found = _find_matching_terms(answer, points)
    missing = [point for point in points if point not in found]
    return {"pass": not missing, "missing": missing}


def check_forbidden_paraphrases(answer: str, terms: list[str]) -> dict:
    found = _find_matching_terms(answer, terms)
    return {"pass": not found, "found": found}


def check_forbidden_sources(answer: str, sources: list[str]) -> dict:
    found = _find_matching_terms(answer, sources)
    return {"pass": not found, "found": found}


def _extract_tham_chieu_section(answer: str) -> str:
    lines = answer.splitlines()
    start_index = None
    for index, line in enumerate(lines):
        if _THAM_CHIEU_HEADING_RE.match(line):
            start_index = index + 1
            break
    if start_index is None:
        return ""

    section_lines = []
    for line in lines[start_index:]:
        if line.strip() and re.match(r"^\s*#{1,6}\s+", line):
            break
        section_lines.append(line)
    return "\n".join(section_lines).strip()


def _extract_reference_titles(tham_chieu_section: str) -> list[str]:
    titles = []
    for line in tham_chieu_section.splitlines():
        match = _REFERENCE_ENTRY_RE.match(line)
        if match:
            titles.append(match.group(1))
    return titles


def check_expected_sources(answer: str, expected_sources: list[str]) -> dict:
    section = _extract_tham_chieu_section(answer)
    titles = _extract_reference_titles(section)
    normalized_titles = [normalize_text(title) for title in titles]

    missing = []
    for source in expected_sources:
        normalized_source = normalize_text(source)
        if normalized_source not in normalized_titles:
            missing.append(source)

    return {"pass": not missing, "missing": missing}


def check_answer(answer: str, benchmark_item: dict) -> dict:
    required_terms = benchmark_item.get("required_terms", [])
    forbidden_paraphrases = [
        *benchmark_item.get("forbidden_paraphrases", []),
        *benchmark_item.get("forbidden_terms", []),
    ]
    expected_sources = [
        *benchmark_item.get("expected_sources", []),
        *benchmark_item.get("required_sources", []),
    ]
    expected_answer_points = benchmark_item.get("expected_answer_points", [])
    forbidden_sources = benchmark_item.get("forbidden_sources", [])

    required_result = check_required_terms(answer, required_terms)
    forbidden_result = check_forbidden_paraphrases(answer, forbidden_paraphrases)
    expected_sources_result = check_expected_sources(answer, expected_sources)
    expected_answer_points_result = check_expected_answer_points(answer, expected_answer_points)
    forbidden_sources_result = check_forbidden_sources(answer, forbidden_sources)

    return {
        "required_terms": required_result,
        "forbidden_paraphrases": forbidden_result,
        "expected_sources": expected_sources_result,
        "expected_answer_points": expected_answer_points_result,
        "forbidden_sources": forbidden_sources_result,
        "overall_pass": (
            required_result["pass"]
            and forbidden_result["pass"]
            and expected_sources_result["pass"]
            and expected_answer_points_result["pass"]
            and forbidden_sources_result["pass"]
        ),
    }
