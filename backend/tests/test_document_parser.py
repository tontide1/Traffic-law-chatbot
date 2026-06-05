import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from backend.core import document_parser


class FakeDocument:
    def __init__(self, markdown: str):
        self._markdown = markdown

    def export_to_markdown(self) -> str:
        return self._markdown


class FakeResult:
    def __init__(self, markdown: str):
        self.document = FakeDocument(markdown)


def test_normalize_legal_markdown_collapses_extra_blank_lines():
    raw = "Chương I\n\n\nĐiều 1\nNội dung\n\n\n\nKhoản 1\n"

    normalized = document_parser.normalize_legal_markdown(raw)

    assert normalized == "Chương I\n\nĐiều 1\nNội dung\n\nKhoản 1"


def test_parse_pdf_to_markdown_uses_docling_and_normalizes_output(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    fake_converter = Mock()
    fake_converter.convert.return_value = FakeResult("Điều 1\n\n\nNội dung")
    monkeypatch.setattr(document_parser, "get_pdf_converter", lambda: fake_converter)

    content = asyncio.run(document_parser.parse_pdf_to_markdown(str(pdf_path)))

    assert content == "Điều 1\n\nNội dung"
    fake_converter.convert.assert_called_once_with(Path(pdf_path))


def test_parse_pdf_to_markdown_rejects_empty_markdown(monkeypatch, tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    fake_converter = Mock()
    fake_converter.convert.return_value = FakeResult(" \n\n ")
    monkeypatch.setattr(document_parser, "get_pdf_converter", lambda: fake_converter)

    with pytest.raises(ValueError, match="Docling extracted no text from PDF"):
        asyncio.run(document_parser.parse_pdf_to_markdown(str(pdf_path)))


def test_parse_pdf_to_markdown_wraps_docling_errors(monkeypatch, tmp_path):
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    fake_converter = Mock()
    fake_converter.convert.side_effect = RuntimeError("boom")
    monkeypatch.setattr(document_parser, "get_pdf_converter", lambda: fake_converter)

    with pytest.raises(RuntimeError, match="Docling failed to parse PDF: boom"):
        asyncio.run(document_parser.parse_pdf_to_markdown(str(pdf_path)))
