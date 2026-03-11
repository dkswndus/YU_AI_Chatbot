"""질문에 답이 될 만한 단위로 청킹. 섹션 감지 후 카테고리별 저장."""
import re
from typing import Any
from typing import Optional


# 목표 청크 크기(자): 답변 한 덩어리로 적당한 크기
TARGET_CHUNK_CHARS = 550
MAX_CHUNK_CHARS = 900
MIN_CHUNK_CHARS = 120

# academic 시간표(강의시간/강의실) 라인 패턴 예: 화10:15-12:10 무-12422(유도실기장1)
ACADEMIC_TIME_RE = re.compile(
    r"^(?P<day>[월화수목금토일])"
    r"(?P<time>\d{1,2}:\d{2}-\d{1,2}:\d{2})\s+"
    r"(?P<room_code>[^\s(]+)\((?P<room_name>[^)]+)\)\s*$"
)
# 학수번호+과목명 라인 패턴 예: 110065-05 전공실기(메치기의전문동작)
ACADEMIC_COURSE_RE = re.compile(r"^(?P<course_code>\d{6}-\d{2})\s+(?P<course_name>.+)$")
# 교수명 후보(한글 2~6자)
KOREAN_NAME_RE = re.compile(r"^[가-힣]{2,6}$")


def _chunk_academic_timetable(
    text: str,
    doc_id: str,
    title: str,
    category: str,
) -> list[dict[str, Any]]:
    """
    academic(종합강의시간표) 문서를 '수업 1개 = 청크 1개'로 분리.
    time line(요일+시간+강의실)을 기준으로, 직전의 과목/교수 정보를 같이 묶는다.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    chunks: list[dict[str, Any]] = []
    chunk_index = 0

    i = 0
    while i < len(lines):
        m = ACADEMIC_TIME_RE.match(lines[i])
        if not m:
            i += 1
            continue

        day = m.group("day")
        time_range = m.group("time")
        room_code = m.group("room_code")
        room_name = m.group("room_name")
        room = f"{room_code}({room_name})"

        # 과목 라인(학수번호+과목명)을 뒤로 10줄 내에서 찾기
        course_idx = None
        course_code = ""
        course_name = ""
        for j in range(i - 1, max(-1, i - 12), -1):
            cm = ACADEMIC_COURSE_RE.match(lines[j])
            if cm:
                course_idx = j
                course_code = cm.group("course_code")
                course_name = cm.group("course_name").strip()
                break

        # 교수명은 time line 직전(혹은 course_idx 이후)에서 가장 가까운 한글 이름 라인
        teacher_name = ""
        start_search = i - 1
        end_search = (course_idx if course_idx is not None else max(0, i - 12))
        for j in range(start_search, end_search - 1, -1):
            if KOREAN_NAME_RE.match(lines[j]) and not lines[j].startswith("*"):
                teacher_name = lines[j]
                break

        # 청크 범위: course_idx부터 시작(없으면 time line부터), 다음 '학년(숫자만)' + '전공/교필...' 조합 전까지
        start = course_idx if course_idx is not None else i
        end = i + 1
        while end < len(lines):
            if lines[end].isdigit() and (end + 1) < len(lines):
                if lines[end + 1] in {"전공", "교필", "교선", "교직"}:
                    break
            # 다음 시간표 라인이 나오면 끊기 (과목 정보 누락된 경우 대비)
            if end != i and ACADEMIC_TIME_RE.match(lines[end]):
                break
            end += 1

        chunk_index += 1
        raw_block = "\n".join(lines[start:end]).strip()
        summary_parts = []
        if course_name:
            summary_parts.append(course_name)
        if teacher_name:
            summary_parts.append(teacher_name)
        summary_parts.append(f"{day} {time_range}")
        summary_parts.append(room)
        normalized_line = " / ".join(summary_parts)

        chunks.append(
            {
                "chunk_id": f"{doc_id}_chunk_{chunk_index:02d}",
                "doc_id": doc_id,
                "category": category,
                "title": title,
                "teacher_name": teacher_name,
                "course_code": course_code,
                "course_name": course_name,
                "day": day,
                "time_range": time_range,
                "room": room,
                "section": f"{day}{time_range} {room}",
                # text는 검색/임베딩을 위해 원문 블록을 유지하고,
                # normalized_line을 UI 표시/검증용으로 제공
                "text": raw_block,
                "normalized_line": normalized_line,
            }
        )
        i = end

    # 만약 시간표 패턴이 하나도 없으면 기존 방식으로 fallback (caller에서 처리)
    return chunks

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
    # academic 시간표는 구조화 청킹 우선
    if category == "academic":
        academic_chunks = _chunk_academic_timetable(text, doc_id=doc_id, title=title, category=category)
        if academic_chunks:
            return academic_chunks

    # 단락 단위로 분리 (빈 줄 기준, 단일 줄도 유지)
    raw_paras = [p.strip() for p in text.split("\n") if p.strip()]
    if not raw_paras:
        return []

    chunks: list[dict[str, Any]] = []
    current_section = "본문"
    current_lines: list[str] = []
    current_len = 0
    chunk_index = 0

    def flush_chunk(force_section: Optional[str] = None):
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
