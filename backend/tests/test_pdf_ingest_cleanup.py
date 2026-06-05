from pathlib import Path

import backend.core.llm_services as llm_services


def test_qwen_vl_parser_is_removed_from_llm_services():
    assert not hasattr(llm_services, "qwen_vl_parse_pdf")


def test_backend_requirements_replace_pdf2image_with_docling():
    requirements = Path("backend/requirements.txt").read_text(encoding="utf-8")
    assert "docling" in requirements
    assert "pdf2image" not in requirements


def test_readme_describes_docling_for_pdf_parsing():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Local PDF Parsing with Docling" in readme
    assert "Qwen 3 VL" not in readme
