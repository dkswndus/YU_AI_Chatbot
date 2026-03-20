"""시간표 PDF 텍스트 파싱.

- Ⅻ. 강의시간표 이전 → 섹션 기반 텍스트 청킹 (type=info)
- Ⅻ. 강의시간표 이후 → 학수번호 기반 구조화 파싱 (type=course)
"""
import re
from typing import Any, Dict, List

# ── 공통 패턴 ────────────────────────────────────────────────
# 학수번호 단독 라인 (예: 910035-01) 또는 학수번호+교과목명 같은 라인
COURSE_NUM_INLINE_RE = re.compile(r"^(\d{6}-\d{2,3})\s+(.+)$")
COURSE_NUM_ONLY_RE = re.compile(r"^(\d{6}-\d{2,3})$")
# 강의시간
TIME_RE = re.compile(r"(월|화|수|목|금|토|일)(\d{1,2}:\d{2}-\d{1,2}:\d{2})")
# 학점(인원)
CREDITS_RE = re.compile(r"^(\d)\((\d+)\)$")
# 이수구분
TYPE_RE = re.compile(r"^(교필|교선|전필|전선|기전|일선|교직|논문|졸작|실습|융합)$")
# 학년
YEAR_RE = re.compile(r"^[1-9]$")
# 수강대상
TARGET_RE = re.compile(r"^(가|나|다|라|마|바|사|훈|가나다)$")
# 강의실 (건물명-호수(...))
ROOM_RE = re.compile(r"^.+-\d+\(.+\)$")
# 무시할 라인
SKIP_RE = re.compile(
    r"^(※|★|\*|-\s*\d+\s*-|·|목\s*차|Ⅰ|Ⅱ|Ⅲ|Ⅳ|Ⅴ|Ⅵ|Ⅶ|Ⅷ|Ⅸ|Ⅹ|Ⅺ|"
    r"학년이수|구분수강|학수번호|교과목명|학점|교수명|강의시간|강\s*의\s*실)"
)
# 섹션 헤더 (정보 파트용)
SECTION_RE = re.compile(
    r"^(Ⅰ|Ⅱ|Ⅲ|Ⅳ|Ⅴ|Ⅵ|Ⅶ|Ⅷ|Ⅸ|Ⅹ|Ⅺ)\.\s*.+$"
    r"|^\s*\d+\.\s*.{2,}$"
    r"|^\s*[가나다라마바사아자차카타파하]\.\s*.+$"
)
# 강의시간표 시작 마커 (Ⅻ. 로 시작하는 줄)
TIMETABLE_START_RE = re.compile(r"^Ⅻ\.")
# 최소 청크 길이
MIN_CHUNK_LEN = 30


# ── 정보 파트: 섹션 기반 청킹 ────────────────────────────────
def _chunk_info(lines: List[str], doc_id: str, title: str, semester: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    current_section = title
    current_lines: List[str] = []
    chunk_index = [1]

    def flush():
        body = "\n".join(current_lines).strip()
        if len(body) >= MIN_CHUNK_LEN:
            chunks.append({
                "type": "info",
                "chunk_id": f"{doc_id}_info_{chunk_index[0]:03d}",
                "doc_id": doc_id,
                "title": title,
                "semester": semester,
                "section": current_section,
                "text": body,
            })
            chunk_index[0] += 1

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if "···" in stripped or "·····" in stripped:
            continue
        if TIMETABLE_START_RE.match(stripped):
            break
        # 섹션 헤더 먼저 체크 (SKIP_RE보다 우선)
        if SECTION_RE.match(stripped):
            flush()
            current_section = stripped
            current_lines = []
            continue
        if SKIP_RE.match(stripped):
            continue
        current_lines.append(stripped)

    flush()
    return chunks


# ── 시간표 파트: 학수번호 기반 구조화 파싱 ──────────────────
def _parse_courses(lines: List[str], doc_id: str, title: str, semester: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    # 학수번호 위치 수집 (인라인 또는 단독)
    course_positions: List[tuple] = []  # (line_idx, course_num, course_name_or_None)
    for i, line in enumerate(lines):
        m = COURSE_NUM_INLINE_RE.match(line)
        if m:
            course_positions.append((i, m.group(1), m.group(2).strip()))
            continue
        m2 = COURSE_NUM_ONLY_RE.match(line)
        if m2:
            # 교과목명은 다음 라인에 있음
            name = lines[i + 1].strip() if i + 1 < len(lines) else ""
            course_positions.append((i, m2.group(1), name))

    for pos_idx, (ci, course_num, course_name) in enumerate(course_positions):
        # 이전 라인에서 학년·이수구분·수강대상 추출
        year = course_type = target = None
        for back in range(1, 6):
            if ci - back < 0:
                break
            prev = lines[ci - back]
            if TYPE_RE.match(prev) and course_type is None:
                course_type = prev
            elif YEAR_RE.match(prev) and year is None:
                year = int(prev)
            elif TARGET_RE.match(prev) and target is None:
                target = prev

        # 다음 블록 범위
        end = course_positions[pos_idx + 1][0] if pos_idx + 1 < len(course_positions) else len(lines)
        # 학수번호 단독 라인이면 교과목명 라인도 블록에서 제외
        start = ci + 2 if COURSE_NUM_ONLY_RE.match(lines[ci]) else ci + 1
        block = lines[start:end]

        credits = capacity = professor = day = time_range = room = None
        j = 0
        while j < len(block):
            line = block[j]
            if not line or SKIP_RE.match(line):
                j += 1
                continue

            # 학점(인원)
            cm = CREDITS_RE.match(line)
            if cm and credits is None:
                credits = int(cm.group(1))
                capacity = int(cm.group(2))
                j += 1
                continue

            # 강의시간 포함 라인
            tm = TIME_RE.search(line)
            if tm and day is None:
                day = tm.group(1)
                time_range = tm.group(2)
                prefix = line[: tm.start()].strip()
                suffix = line[tm.end():].strip()
                if prefix and professor is None:
                    professor = prefix
                if suffix:
                    room = suffix
                j += 1
                if not room and j < len(block):
                    nxt = block[j]
                    if nxt and not SKIP_RE.match(nxt) and not TIME_RE.search(nxt):
                        room = nxt
                        j += 1
                break

            # 강의실 (강의시간 없는 과목)
            if credits is not None and professor is not None and day is None and room is None:
                if ROOM_RE.match(line):
                    room = line
                    j += 1
                    continue

            # 교수명
            if credits is not None and professor is None:
                if not SKIP_RE.match(line) and not ROOM_RE.match(line):
                    # 교수명이 두 줄에 걸칠 수 있음
                    name = line
                    if j + 1 < len(block):
                        nxt = block[j + 1]
                        if (nxt and not CREDITS_RE.match(nxt) and not TIME_RE.search(nxt)
                                and not SKIP_RE.match(nxt) and not ROOM_RE.match(nxt)
                                and not COURSE_NUM_INLINE_RE.match(nxt)
                                and not COURSE_NUM_ONLY_RE.match(nxt)
                                and len(nxt) <= 10):
                            name = name + nxt
                            j += 1
                    professor = name
                    j += 1
                    continue

            j += 1

        if not course_num:
            continue

        records.append({
            "type": "course",
            "doc_id": doc_id,
            "title": title,
            "semester": semester,
            "course_number": course_num,
            "course_name": course_name,
            "year": year,
            "course_type": course_type,
            "target": target,
            "credits": credits,
            "capacity": capacity,
            "professor": professor,
            "day": day,
            "time_range": time_range,
            "room": room,
        })

    return records


# ── 메인 ─────────────────────────────────────────────────────
def parse_timetable(text: str, doc_id: str, title: str, semester: str) -> List[Dict[str, Any]]:
    lines = [l.strip() for l in text.splitlines()]

    # 강의시간표 시작 위치 탐색 (목차 줄은 '···' 포함이라 제외)
    split_idx = None
    for i, line in enumerate(lines):
        if TIMETABLE_START_RE.match(line) and "···" not in line:
            split_idx = i
            break

    if split_idx is None:
        # 시간표 구분선 없으면 전체 코스 파싱만
        return _parse_courses([l for l in lines if l], doc_id, title, semester)

    info_lines = lines[:split_idx]
    course_lines = [l for l in lines[split_idx:] if l]

    info_chunks = _chunk_info(info_lines, doc_id, title, semester)
    course_records = _parse_courses(course_lines, doc_id, title, semester)

    return info_chunks + course_records
