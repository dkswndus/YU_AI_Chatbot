"""질문에 답이 될 만한 단위로 청킹. 섹션 감지 후 카테고리별 저장."""
import re
from typing import Any


# 목표 청크 크기(자): 답변 한 덩어리로 적당한 크기
TARGET_CHUNK_CHARS = 550
MAX_CHUNK_CHARS = 900
MIN_CHUNK_CHARS = 120

# 섹션 헤더 패턴: 제1조(목적), 1. 개설목적, 가. 개설목적, 1) 안내
SECTION_HEADER_RE = re.compile(
    r"^(?:제\d+조\s*\([^)]+\)"
    r"|\d+\.\s+.{1,80}"
    r"|[가나다라마바사아]\.\s+.{1,80}"
    r"|\d+\)\s+.{1,80}"
    r"|[\d]+\.\s*[\d가나다라]\.\s+.{1,80})$"
)


def _is_section_header(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 100:
        return False
    if SECTION_HEADER_RE.match(s):
        return True
    # 짧은 줄이고 마침표로 안 끝나면 헤더로 간주
    if len(s) <= 50 and not s.endswith("다.") and not s.endswith("요."):
        if s.endswith(")"):
            return True
    return False


def _normalize_section_name(line: str) -> str:
    """섹션 제목만 추출 (점선·페이지 번호 제거)."""
    s = re.sub(r"·+\s*\d*\s*$", "", line.strip())
    return s[:80].strip() if s else "본문"


def _split_sentences(text: str) -> list[str]:
    """문장 단위로 분리 (한국어 마침표 기준)."""
    parts = re.split(r"(?<=[다요음])\.\s+", text)
    result = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not p.endswith("."):
            p = p + "."
        result.append(p)
    return result if result else [text]


def chunk_text(
    text: str,
    doc_id: str,
    title: str,
    category: str,
) -> list[dict[str, Any]]:
    """
    본문을 답변 단위 청크로 나눔.
    Returns list of {"chunk_id", "doc_id", "title", "category", "section", "text"}.
    """
    # 단락 단위로 분리 (빈 줄 기준, 단일 줄도 유지)
    raw_paras = [p.strip() for p in text.split("\n") if p.strip()]
    if not raw_paras:
        return []

    chunks: list[dict[str, Any]] = []
    current_section = "본문"
    current_lines: list[str] = []
    current_len = 0
    chunk_index = 0

    def flush_chunk(force_section: str | None = None):
        nonlocal chunk_index, current_section, current_lines, current_len
        if not current_lines:
            return
        section = force_section if force_section is not None else current_section
        chunk_text_val = "\n".join(current_lines).strip()
        if not chunk_text_val or len(chunk_text_val) < 30:
            return
        chunk_index += 1
        chunks.append({
            "chunk_id": f"{doc_id}_chunk_{chunk_index:02d}",
            "doc_id": doc_id,
            "title": title,
            "category": category,
            "section": section,
            "text": chunk_text_val,
        })
        current_lines = []
        current_len = 0

    i = 0
    while i < len(raw_paras):
        line = raw_paras[i]
        line_len = len(line) + (1 if current_lines else 0)

        if _is_section_header(line):
            # 기존 내용이 있으면 먼저 flush
            if current_lines:
                flush_chunk()
            current_section = _normalize_section_name(line)
            current_lines = [line]
            current_len = len(line)
            i += 1
            continue

        # 한 단락이 너무 길면 문장 단위로 쪼개서 넣기
        if len(line) > MAX_CHUNK_CHARS:
            if current_lines:
                flush_chunk()
            for sent in _split_sentences(line):
                if current_len + len(sent) > TARGET_CHUNK_CHARS and current_lines:
                    flush_chunk()
                current_lines.append(sent)
                current_len += len(sent)
            i += 1
            continue

        # 누적 시 목표 초과하면 flush 후 새 청크
        if current_len + line_len > TARGET_CHUNK_CHARS and current_lines:
            flush_chunk()
            # 새 청크의 첫 줄이 헤더면 section 갱신
            if _is_section_header(line):
                current_section = _normalize_section_name(line)

        current_lines.append(line)
        current_len += len(line)
        i += 1

    flush_chunk()
    return chunks
