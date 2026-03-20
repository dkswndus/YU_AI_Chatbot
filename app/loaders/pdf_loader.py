"""PDF 텍스트 추출."""
import fitz
from pathlib import Path


def extract_text(pdf_path: str | Path) -> str:
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)
