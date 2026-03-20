"""
academic_001.txt → academic_001.md 변환 스크립트
LLM이 잘 이해할 수 있는 마크다운 형식으로 재구조화
"""
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT  = PROJECT_ROOT / "data" / "processed" / "academic_001.txt"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "academic_001.md"

# ───────────────────────────── helpers ──────────────────────────────
PHONE_RE = re.compile(r"^\d{4}$")
ROOM_RE  = re.compile(r"^[가-힣A-Za-z]+-\d+")
DAY_RE   = re.compile(r"^[월화수목금토일]$")
PAGE_RE  = re.compile(r"^-\s*\d+\s*-$")
DATE_RE  = re.compile(r"(\d+)일\(([월화수목금토일])\)")


def clean(line: str) -> str:
    return line.strip()


def is_skip(line: str) -> bool:
    s = clean(line)
    return (not s) or PAGE_RE.match(s) is not None


# ─────────────────── Section splitter ───────────────────────────────
ROMAN = ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ", "Ⅸ", "Ⅹ", "Ⅺ", "Ⅻ"]
SECTION_RE = re.compile(r"^([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ])\.")

def split_sections(lines):
    """Returns dict: roman_numeral -> list of lines"""
    sections = {}
    current = None
    for line in lines:
        m = SECTION_RE.match(clean(line))
        if m:
            current = m.group(1)
            sections[current] = []
        elif current:
            sections[current].append(line)
    return sections


# ─────────────────── Ⅰ. 학사일정 ────────────────────────────────────
def parse_schedule(lines):
    """Returns markdown string for 학사일정"""
    md = []
    semester = None
    month = None
    events = {}  # date_str -> [events]

    i = 0
    raw = [clean(l) for l in lines]

    while i < len(raw):
        line = raw[i]
        if not line or PAGE_RE.match(line):
            i += 1
            continue

        # Semester header
        if "1학기" in line and "2026" in line:
            semester = "1학기"
            i += 1
            continue
        if "2학기" in line and "2026" in line:
            semester = "2학기"
            i += 1
            continue

        # Month
        m_month = re.match(r"^(\d+)월$", line)
        if m_month:
            month = int(m_month.group(1))
            i += 1
            continue

        # Date + event (may be on same line or following lines)
        dates_in_line = DATE_RE.findall(line)
        # Also check for event text after the date
        if dates_in_line and month:
            # Extract event text: everything after the date pattern
            event_text_raw = DATE_RE.sub("", line).strip()
            # Following lines until next date or blank are continuation events
            j = i + 1
            event_lines = [event_text_raw] if event_text_raw else []
            while j < len(raw):
                nxt = raw[j]
                if not nxt or DATE_RE.search(nxt) or re.match(r"^\d+주차", nxt) or re.match(r"^\d+월$", nxt):
                    break
                if re.match(r"^[123456789]\d?$", nxt) or re.match(r"^[①-⑩]", nxt):
                    # calendar date number, skip
                    j += 1
                    continue
                # Skip table headers
                if nxt in ["월", "주차", "요  일", "일   정", "내   용", "일월화수목금토"]:
                    j += 1
                    continue
                event_lines.append(nxt)
                j += 1

            for day_num, _ in dates_in_line:
                date_key = f"{month}월 {day_num}일"
                if event_lines:
                    events.setdefault(date_key, []).extend(
                        [e for e in event_lines if e]
                    )
            i = j
            continue

        i += 1

    # Build markdown
    if events:
        md.append("| 날짜 | 일정 |")
        md.append("|------|------|")
        for date_key in sorted(events.keys(), key=lambda x: (int(re.search(r"(\d+)월", x).group(1)), int(re.search(r"(\d+)일", x).group(1)))):
            evs = " / ".join(dict.fromkeys(events[date_key]))
            md.append(f"| {date_key} | {evs} |")
    return "\n".join(md)


# ─────────────────── Ⅱ.1 전임교원 연구일 ────────────────────────────
def parse_professor_table(lines):
    raw = [clean(l) for l in lines]

    # Stop at 강의실 안내
    stop_markers = ["2. 강의실 안내", "강의실 안내"]
    stop_idx = len(raw)
    for idx, line in enumerate(raw):
        if any(line.startswith(m) for m in stop_markers):
            stop_idx = idx
            break
    raw = raw[:stop_idx]

    # Remove table header repetitions and page markers
    header_words = {"대학", "학과", "교수명", "연구일전화번호", "연구실", "비고"}

    professors = []
    current_college = ""
    current_dept = ""
    i = 0

    while i < len(raw):
        line = raw[i]
        if not line or PAGE_RE.match(line) or line in header_words:
            i += 1
            continue
        # College+dept concatenated (e.g. "무도유도", "체육체육과학대학")
        # or standalone dept name
        if PHONE_RE.match(line):
            # This is a phone number: look back for day and name
            phone = f"031-8020-{line}"
            day = raw[i - 1] if i >= 1 else ""
            name = raw[i - 2] if i >= 2 else ""
            room = ""
            note = ""
            if i + 1 < len(raw) and (ROOM_RE.match(raw[i + 1]) or raw[i + 1] == ""):
                room = raw[i + 1] if ROOM_RE.match(raw[i + 1]) else ""
                i += 1
            if i + 1 < len(raw) and raw[i + 1] in ["연구년", "파견", "휴직", "연구", "년"]:
                note = raw[i + 1]
                i += 1

            if DAY_RE.match(day) and name:
                professors.append({
                    "college": current_college,
                    "dept": current_dept,
                    "name": name,
                    "day": day,
                    "phone": phone,
                    "room": room,
                    "note": note,
                })
            i += 1
            continue

        # Department / college lines
        # Simple heuristic: lines with 대학/학과/학부 and no digits
        if re.search(r"[대학과부]", line) and not re.search(r"\d", line):
            # Could be college or dept
            if "대학" in line and len(line) <= 10:
                current_college = line
                current_dept = ""
            else:
                current_dept = line
        elif not DAY_RE.match(line) and not ROOM_RE.match(line):
            # Treat as potential name or dept update; will be used as "name" when phone found
            pass

        i += 1

    if not professors:
        return ""

    md = ["| 대학 | 학과 | 교수명 | 연구일 | 전화번호 | 연구실 | 비고 |",
          "|------|------|--------|--------|----------|--------|------|"]
    for p in professors:
        md.append(
            f"| {p['college']} | {p['dept']} | {p['name']} | {p['day']}요일 | {p['phone']} | {p['room']} | {p['note']} |"
        )
    return "\n".join(md)


# ─────────────────── Ⅱ.2 강의실 안내 ────────────────────────────────
def parse_rooms(lines):
    raw = [clean(l) for l in lines]

    start = 0
    for idx, line in enumerate(raw):
        if "강의실 안내" in line:
            start = idx + 1
            break
    stop = len(raw)
    for idx in range(start, len(raw)):
        if "학과 전화번호" in raw[idx]:
            stop = idx
            break
    raw = raw[start:stop]

    # Build a simple list grouped by building
    md = []
    current_building = ""
    current_floor = ""
    i = 0
    while i < len(raw):
        line = raw[i]
        if not line or PAGE_RE.match(line):
            i += 1
            continue
        # Floor patterns
        m_floor = re.match(r"^(지층|\d+층)$", line)
        if m_floor:
            current_floor = m_floor.group(1)
            i += 1
            continue
        # Room numbers line (contains digits and commas)
        if re.search(r"\d{4,}", line) or (re.search(r"\d+", line) and ("," in line or "<" in line)):
            md.append(f"  - {current_floor}: {line}")
            i += 1
            continue
        # Building name (Korean text, no digits, possibly multi-line)
        if re.search(r"[가-힣]", line) and not re.search(r"\d{3}", line):
            # Check if it continues on next few lines
            building_parts = [line]
            j = i + 1
            while j < len(raw) and raw[j] and not re.match(r"^(지층|\d+층)$", raw[j]) and not re.search(r"\d{4,}", raw[j]) and len(raw[j]) < 15:
                if re.search(r"[가-힣]", raw[j]):
                    building_parts.append(raw[j])
                j += 1
            building = " ".join(building_parts).replace("\n", "")
            if building != current_building:
                current_building = building
                md.append(f"\n**{current_building}**")
            i = j
            continue
        i += 1

    return "\n".join(md)


# ─────────────────── Ⅱ.3 학과 전화번호 ──────────────────────────────
def parse_phone_numbers(lines):
    raw = [clean(l) for l in lines]

    start = 0
    for idx, line in enumerate(raw):
        if "학과 전화번호" in line:
            start = idx + 1
            break
    raw = raw[start:]

    DEPT_PHONE_RE = re.compile(r"^031-8020-\d{4}$")
    header_words = {"단과대학", "학과", "전화번호"}

    entries = []
    current_college = ""
    i = 0
    while i < len(raw):
        line = raw[i]
        if not line or PAGE_RE.match(line) or line in header_words:
            i += 1
            continue
        if DEPT_PHONE_RE.match(line):
            phone = line
            dept = raw[i - 1] if i >= 1 else ""
            if dept and not dept.startswith("031") and dept not in header_words:
                entries.append({"college": current_college, "dept": dept, "phone": phone})
            i += 1
            continue
        # Check if college name (contains 대학)
        if "대학" in line and len(line) <= 15 and not re.search(r"\d", line):
            current_college = line
        i += 1

    if not entries:
        return ""

    md = ["| 단과대학 | 학과 | 전화번호 |",
          "|---------|------|----------|"]
    for e in entries:
        md.append(f"| {e['college']} | {e['dept']} | {e['phone']} |")
    return "\n".join(md)


# ─────────────────── Generic text section ────────────────────────────
def parse_text_section(lines):
    raw = [clean(l) for l in lines]
    result = []
    prev_blank = False
    for line in raw:
        if PAGE_RE.match(line):
            continue
        if not line:
            if not prev_blank:
                result.append("")
            prev_blank = True
            continue
        prev_blank = False

        # Sub-section headings like "1. xxx", "가. xxx", "나. xxx"
        if re.match(r"^\d+\.\s+", line) and len(line) < 60:
            result.append(f"\n### {line}")
        elif re.match(r"^[가나다라마바사아자차카타파하]\.\s+", line):
            result.append(f"\n#### {line}")
        elif re.match(r"^\s*[①②③④⑤⑥⑦⑧⑨⑩]\s", line):
            result.append(f"- {line.strip()}")
        else:
            result.append(line)
    return "\n".join(result)


# ─────────────────── Main ────────────────────────────────────────────
def main():
    text = INPUT.read_text(encoding="utf-8")
    all_lines = text.splitlines()

    sections = split_sections(all_lines)

    md_parts = ["# 2026학년도 1학기 종합강의시간표\n",
                "> 출처: 용인대학교 2026학년도 1학기 종합강의시간표\n"]

    # Ⅰ. 학사일정
    if "Ⅰ" in sections:
        md_parts.append("## Ⅰ. 학사일정\n")
        md_parts.append(parse_schedule(sections["Ⅰ"]))
        md_parts.append("")

    # Ⅱ. 전임교원 연구일, 강의실, 학과 전화번호
    if "Ⅱ" in sections:
        md_parts.append("## Ⅱ. 전임교원 연구일, 강의실, 학과 전화번호 안내\n")

        md_parts.append("### 1. 전임교원 연구일 안내\n")
        prof_md = parse_professor_table(sections["Ⅱ"])
        md_parts.append(prof_md if prof_md else "_데이터 없음_")
        md_parts.append("")

        md_parts.append("### 2. 강의실 안내\n")
        room_md = parse_rooms(sections["Ⅱ"])
        md_parts.append(room_md if room_md else "_데이터 없음_")
        md_parts.append("")

        md_parts.append("### 3. 학과 전화번호 안내\n")
        phone_md = parse_phone_numbers(sections["Ⅱ"])
        md_parts.append(phone_md if phone_md else "_데이터 없음_")
        md_parts.append("")

    # Ⅲ~Ⅺ: text sections
    section_titles = {
        "Ⅲ": "수강신청 제반사항 안내",
        "Ⅳ": "학사(학적) 안내",
        "Ⅴ": "교직 안내",
        "Ⅵ": "국내 학점교류 안내",
        "Ⅶ": "전체 재학생 군사학 교과목 이수 안내",
        "Ⅷ": "학군사관 후보생 군사학 교과목 이수 안내",
        "Ⅸ": "본교 원격수업 수강신청 및 개설 교과목 안내",
        "Ⅹ": "한국열린사이버대학교(OCU) 학점교류 수강신청 및 개설 교과목 안내",
        "Ⅺ": "한국대학가상교육연합(KCU) 학점교류 수강신청 및 개설 교과목 안내",
    }
    for roman, title in section_titles.items():
        if roman in sections:
            md_parts.append(f"## {roman}. {title}\n")
            md_parts.append(parse_text_section(sections[roman]))
            md_parts.append("")

    # Ⅻ: skip (already handled as structured course records)
    md_parts.append("## Ⅻ. 강의시간표\n")
    md_parts.append("> 강의시간표 데이터는 구조화된 JSON 레코드로 별도 관리됩니다.")
    md_parts.append("")

    output = "\n".join(md_parts)
    OUTPUT.write_text(output, encoding="utf-8")
    print(f"저장 완료: {OUTPUT}")
    print(f"총 {len(output.splitlines())}줄")


if __name__ == "__main__":
    main()
