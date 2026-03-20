"""
academic_001.md → JSON 청크 변환
마크다운 헤더(##, ###) 기준으로 섹션 분할 후 JSON 레코드 생성
"""
import re
from pathlib import Path
from typing import List, Dict

# 마크다운 테이블에서 교수 연구일 행 파싱
PROF_ROW_RE = re.compile(
    r"\|\s*(?P<college>[^|]*?)\s*\|\s*(?P<dept>[^|]*?)\s*\|\s*(?P<name>[^|]+?)\s*\|"
    r"\s*(?P<day>[^|]+?)\s*\|\s*(?P<phone>[^|]+?)\s*\|\s*(?P<room>[^|]*?)\s*\|\s*(?P<note>[^|]*?)\s*\|"
)
PHONE_ROW_RE = re.compile(
    r"\|\s*(?P<college>[^|]+?)\s*\|\s*(?P<dept>[^|]+?)\s*\|\s*(?P<phone>031-\d{4}-\d{4})\s*\|"
)

MAX_CHUNK_CHARS = 1500  # 청크 최대 길이


def _split_by_header(text: str, level: int = 2) -> List[Dict]:
    """## 또는 ### 헤더로 섹션 분할"""
    prefix = "#" * level + " "
    pattern = re.compile(r"^" + re.escape(prefix), re.MULTILINE)

    parts = pattern.split(text)
    result = []
    for part in parts[1:]:  # 첫 번째는 헤더 앞 내용
        lines = part.split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        result.append({"title": title, "body": body})
    return result


def _parse_professor_rows(body: str) -> List[Dict]:
    """교수 연구일 테이블 → 구조화된 레코드"""
    records = []
    for line in body.splitlines():
        m = PROF_ROW_RE.match(line.strip())
        if not m:
            continue
        name = m.group("name").strip()
        # 헤더 행 스킵
        if name in ("교수명", "---"):
            continue
        records.append({
            "type": "professor",
            "college": m.group("college").strip(),
            "dept": m.group("dept").strip(),
            "name": name,
            "day": m.group("day").strip(),
            "phone": m.group("phone").strip(),
            "room": m.group("room").strip(),
            "note": m.group("note").strip(),
        })
    return records


def _parse_phone_rows(body: str) -> List[Dict]:
    """학과 전화번호 테이블 → 구조화된 레코드"""
    records = []
    for line in body.splitlines():
        m = PHONE_ROW_RE.match(line.strip())
        if not m:
            continue
        dept = m.group("dept").strip()
        if dept in ("학과", "---"):
            continue
        records.append({
            "type": "dept_phone",
            "college": m.group("college").strip(),
            "dept": dept,
            "phone": m.group("phone").strip(),
        })
    return records


def _split_long_text(title: str, body: str, section: str, doc_id: str, category: str,
                     title_doc: str, semester: str, max_chars: int = MAX_CHUNK_CHARS) -> List[Dict]:
    """긴 텍스트 섹션을 단락 단위로 분할"""
    # ### 서브섹션으로 먼저 분할
    sub_sections = _split_by_header(body, level=3)
    if sub_sections:
        chunks = []
        for sub in sub_sections:
            sub_title = f"{title} > {sub['title']}"
            sub_body = sub["body"]
            if len(sub_body) <= max_chars:
                chunks.append({
                    "type": "info",
                    "doc_id": doc_id,
                    "category": category,
                    "title": title_doc,
                    "semester": semester,
                    "section": sub_title,
                    "text": f"## {sub_title}\n\n{sub_body}",
                })
            else:
                # 단락 단위 추가 분할
                paras = re.split(r"\n{2,}", sub_body)
                buffer = ""
                chunk_idx = 1
                for para in paras:
                    if len(buffer) + len(para) > max_chars and buffer:
                        chunks.append({
                            "type": "info",
                            "doc_id": doc_id,
                            "category": category,
                            "title": title_doc,
                            "semester": semester,
                            "section": f"{sub_title} (파트{chunk_idx})",
                            "text": f"## {sub_title}\n\n{buffer.strip()}",
                        })
                        chunk_idx += 1
                        buffer = para + "\n\n"
                    else:
                        buffer += para + "\n\n"
                if buffer.strip():
                    chunks.append({
                        "type": "info",
                        "doc_id": doc_id,
                        "category": category,
                        "title": title_doc,
                        "semester": semester,
                        "section": f"{sub_title} (파트{chunk_idx})" if chunk_idx > 1 else sub_title,
                        "text": f"## {sub_title}\n\n{buffer.strip()}",
                    })
        return chunks
    else:
        # 단락 단위 분할
        paras = re.split(r"\n{2,}", body)
        chunks = []
        buffer = ""
        chunk_idx = 1
        for para in paras:
            if len(buffer) + len(para) > max_chars and buffer:
                chunks.append({
                    "type": "info",
                    "doc_id": doc_id,
                    "category": category,
                    "title": title_doc,
                    "semester": semester,
                    "section": f"{title} (파트{chunk_idx})",
                    "text": f"## {title}\n\n{buffer.strip()}",
                })
                chunk_idx += 1
                buffer = para + "\n\n"
            else:
                buffer += para + "\n\n"
        if buffer.strip():
            chunks.append({
                "type": "info",
                "doc_id": doc_id,
                "category": category,
                "title": title_doc,
                "semester": semester,
                "section": f"{title} (파트{chunk_idx})" if chunk_idx > 1 else title,
                "text": f"## {title}\n\n{buffer.strip()}",
            })
        return chunks


def chunk_markdown(md_path: str | Path, doc_id: str, title: str,
                   category: str = "academic", semester: str = "") -> List[Dict]:
    """
    마크다운 파일을 청킹하여 JSON 레코드 리스트 반환
    - professor 타입: 교수 연구일 개별 레코드
    - dept_phone 타입: 학과 전화번호 개별 레코드
    - info 타입: 텍스트 섹션 청크
    """
    text = Path(md_path).read_text(encoding="utf-8")
    top_sections = _split_by_header(text, level=2)
    records = []

    for sec in top_sections:
        sec_title = sec["title"]
        body = sec["body"]

        # 교수 연구일 테이블
        if "전임교원 연구일" in sec_title or "연구일 안내" in sec_title:
            # ### 1. 전임교원 연구일 안내 서브섹션 찾기
            subs = _split_by_header(body, level=3)
            for sub in subs:
                if "연구일" in sub["title"]:
                    profs = _parse_professor_rows(sub["body"])
                    for p in profs:
                        p.update({"doc_id": doc_id, "category": category,
                                  "title": title, "semester": semester})
                        records.append(p)
                elif "전화번호" in sub["title"]:
                    phones = _parse_phone_rows(sub["body"])
                    for ph in phones:
                        ph.update({"doc_id": doc_id, "category": category,
                                   "title": title, "semester": semester})
                        records.append(ph)
                else:
                    # 강의실 안내 등
                    if sub["body"].strip():
                        records.append({
                            "type": "info",
                            "doc_id": doc_id,
                            "category": category,
                            "title": title,
                            "semester": semester,
                            "section": f"{sec_title} > {sub['title']}",
                            "text": f"## {sec_title} > {sub['title']}\n\n{sub['body']}",
                        })
        elif "전임교원" in sec_title:
            # 서브섹션 없이 통째로 온 경우
            subs = _split_by_header(body, level=3)
            if subs:
                for sub in subs:
                    if "연구일" in sub["title"]:
                        profs = _parse_professor_rows(sub["body"])
                        for p in profs:
                            p.update({"doc_id": doc_id, "category": category,
                                      "title": title, "semester": semester})
                            records.append(p)
                    elif "전화번호" in sub["title"]:
                        phones = _parse_phone_rows(sub["body"])
                        for ph in phones:
                            ph.update({"doc_id": doc_id, "category": category,
                                       "title": title, "semester": semester})
                            records.append(ph)
                    else:
                        if sub["body"].strip():
                            records.append({
                                "type": "info",
                                "doc_id": doc_id,
                                "category": category,
                                "title": title,
                                "semester": semester,
                                "section": f"{sec_title} > {sub['title']}",
                                "text": f"## {sec_title} > {sub['title']}\n\n{sub['body']}",
                            })
        elif "강의시간표" in sec_title:
            # 건너뜀 (별도 JSON 레코드로 관리)
            continue
        else:
            # 일반 텍스트 섹션 → 길이에 따라 분할
            if not body.strip():
                continue
            chunks = _split_long_text(sec_title, body, sec_title,
                                      doc_id, category, title, semester)
            records.extend(chunks)

    return records
