import asyncio
import re
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_pdf_converter: Optional[DocumentConverter] = None


def get_pdf_converter() -> DocumentConverter:
    global _pdf_converter
    if _pdf_converter is None:
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
        )
        _pdf_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            }
        )
    return _pdf_converter


def normalize_legal_markdown(content: str) -> str:
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = re.sub(r"[ \t]+\n", "\n", content)
    content = re.sub(r"\n{3,}", "\n\n", content)

    major_heading_pattern = re.compile(r"^(Phần|Chương|Mục|Điều)\b", re.IGNORECASE)
    normalized_lines: list[str] = []

    for raw_line in content.split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()
        if major_heading_pattern.match(stripped) and normalized_lines and normalized_lines[-1] != "":
            normalized_lines.append("")
        normalized_lines.append(line)

    return "\n".join(normalized_lines).strip()


async def parse_pdf_to_markdown(file_path: str) -> str:
    path = Path(file_path)
    converter = get_pdf_converter()

    try:
        result = await asyncio.to_thread(converter.convert, path)
    except Exception as exc:
        raise RuntimeError(f"Docling failed to parse PDF: {exc}") from exc

    markdown = normalize_legal_markdown(result.document.export_to_markdown())
    if not markdown.strip():
        raise ValueError("Docling extracted no text from PDF")

    return markdown
