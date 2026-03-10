"""PDF 텍스트 추출."""
from pathlib import Path

import fitz  # PyMuPDF


def extract_text(pdf_path: str | Path) -> str:
    """PDF 파일에서 텍스트 추출."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    doc = fitz.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text())
    doc.close()
    return "\n".join(parts).strip()
