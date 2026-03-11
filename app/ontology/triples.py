"""교수–과목 관계(트리플) 추출, 저장, 조회.
관계: (교수, teaches, 과목) + 일정 정보(요일, 시간, 강의실, 학기).
다중 조건 조회(요일+대학 등) 및 그래프 조회 지원.
"""
import json
from pathlib import Path

# 요일 한글 → 코드 (무도대학 수요일 수업 등 파싱용)
DAY_NAME_TO_CODE = {
    "월요일": "월", "월": "월",
    "화요일": "화", "화": "화",
    "수요일": "수", "수": "수",
    "목요일": "목", "목": "목",
    "금요일": "금", "금": "금",
}

# 강의실 접두어 → 대학(계열). "무도대학 수요일 수업" 등 조회 시 사용
COLLEGE_TO_ROOM_PREFIX = {
    "무도대학": "무-",
    "무도": "무-",
    "인문사회": "인문사회-",
    "인문사회융합대학": "인문사회-",
    "용오름": "용오름-",
    "용오름대학": "용오름-",
    "AI바이오": "AI바이오-",
    "AI바이오융합대학": "AI바이오-",
    "종합": "종-",
    "종": "종-",
}


def _chunks_dir_from_project_root():
    """프로젝트 루트 기준 청크 디렉터리."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "chunks" / "academic"


def _default_ontology_path():
    """기본 온톨로지 저장 경로."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "ontology" / "teaches.json"


def build_triples_from_chunks(chunks_dir=None):
    """
    academic 청크 JSON에서 (교수, teaches, 과목) 트리플 목록 생성.
    teacher_name 이 있고, course_name 또는 section 이 있는 청크만 사용.
    중복 제거: (teacher_name, course_name, day, time_range, room, doc_id) 기준.
    """
    if chunks_dir is None:
        chunks_dir = _chunks_dir_from_project_root()
    chunks_dir = Path(chunks_dir)
    if not chunks_dir.is_dir():
        return []

    seen = set()
    triples = []
    for path in sorted(chunks_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(chunks, list):
            continue
        doc_id = path.stem
        for c in chunks:
            teacher = (c.get("teacher_name") or "").strip()
            if not teacher:
                continue
            course = (c.get("course_name") or "").strip()
            if not course and (c.get("section") or c.get("normalized_line")):
                # normalized_line 예: "드래곤실천인성(유도) / 장성호 / 월 13:05-15:00 / ..."
                raw = (c.get("normalized_line") or c.get("section") or "").strip()
                if raw:
                    course = raw.split(" / ")[0].strip() if " / " in raw else raw[:50]
            if not course:
                continue
            day = (c.get("day") or "").strip()
            time_range = (c.get("time_range") or "").strip()
            room = (c.get("room") or "").strip()
            title = (c.get("title") or "").strip()
            key = (teacher, course, day, time_range, room, doc_id)
            if key in seen:
                continue
            seen.add(key)
            triples.append({
                "subject": teacher,
                "predicate": "teaches",
                "object": course,
                "day": day,
                "time_range": time_range,
                "room": room,
                "doc_id": doc_id,
                "title": title,
            })
    return triples


def save_triples(triples, path=None):
    """트리플 목록을 JSON으로 저장."""
    if path is None:
        path = _default_ontology_path()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=2)
    return path


def load_triples(path=None):
    """저장된 트리플 JSON 로드."""
    if path is None:
        path = _default_ontology_path()
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_courses_by_teacher(teacher_name, triples=None, path=None):
    """
    교수명으로 수업 목록 조회.
    triples 를 넘기면 그대로 사용, 아니면 path(또는 기본 경로)에서 로드.
    반환: [ {"course_name", "day", "time_range", "room", "title", "doc_id"}, ... ]
    """
    if triples is None:
        triples = load_triples(path)
    teacher_name = (teacher_name or "").strip()
    if not teacher_name:
        return []
    out = []
    for t in triples:
        if (t.get("subject") or "").strip() != teacher_name:
            continue
        out.append({
            "course_name": t.get("object") or "",
            "day": t.get("day") or "",
            "time_range": t.get("time_range") or "",
            "room": t.get("room") or "",
            "title": t.get("title") or "",
            "doc_id": t.get("doc_id") or "",
        })
    return out


def get_schedule_by_course(course_name, triples=None, path=None):
    """
    과목명으로 수업 일정 조회 (어떤 교수, 요일, 시간, 강의실, 학기).
    과목명은 부분 일치(포함)로 검색.
    반환: [ {"teacher_name", "course_name", "day", "time_range", "room", "title", "doc_id"}, ... ]
    """
    if triples is None:
        triples = load_triples(path)
    course_name = (course_name or "").strip()
    if not course_name:
        return []
    out = []
    for t in triples:
        obj = (t.get("object") or "").strip()
        if course_name not in obj:
            continue
        out.append({
            "teacher_name": t.get("subject") or "",
            "course_name": obj,
            "day": t.get("day") or "",
            "time_range": t.get("time_range") or "",
            "room": t.get("room") or "",
            "title": t.get("title") or "",
            "doc_id": t.get("doc_id") or "",
        })
    return out


def get_schedule_by_filters(
    day=None,
    college=None,
    room_prefix=None,
    teacher=None,
    course_contains=None,
    triples=None,
    path=None,
):
    """
    관계 조합 조회 (요일, 대학, 교수, 과목 등 자유 조합).
    - day: 요일 (예: "수")
    - college: 대학명 (예: "무도대학") → 강의실 접두어로 매칭 (무-)
    - room_prefix: 강의실 접두어 직접 지정 (예: "무-")
    - teacher: 교수명 정확 일치
    - course_contains: 과목명 부분 일치
    반환: [ {"teacher_name", "course_name", "day", "time_range", "room", "title", "doc_id"}, ... ]
    """
    if triples is None:
        triples = load_triples(path)
    use_prefix = room_prefix
    if college and not use_prefix:
        use_prefix = COLLEGE_TO_ROOM_PREFIX.get(college.strip()) or COLLEGE_TO_ROOM_PREFIX.get(
            college.strip().replace("대학", "")
        )
        if not use_prefix and college.strip():
            for k, v in COLLEGE_TO_ROOM_PREFIX.items():
                if k in college or college in k:
                    use_prefix = v
                    break
    out = []
    for t in triples:
        if day is not None and (t.get("day") or "").strip() != str(day).strip():
            continue
        if use_prefix and not (t.get("room") or "").startswith(use_prefix):
            continue
        if teacher is not None and (t.get("subject") or "").strip() != str(teacher).strip():
            continue
        if course_contains and course_contains not in (t.get("object") or ""):
            continue
        out.append({
            "teacher_name": t.get("subject") or "",
            "course_name": (t.get("object") or "").strip(),
            "day": t.get("day") or "",
            "time_range": t.get("time_range") or "",
            "room": t.get("room") or "",
            "title": t.get("title") or "",
            "doc_id": t.get("doc_id") or "",
        })
    return out


def parse_schedule_query(text):
    """
    짧은 질의에서 요일·대학 추출 (예: "무도대학 수요일 수업" → day=수, college=무도대학).
    반환: {"day": "수" or None, "college": "무도대학" or None}
    """
    if not text or not isinstance(text, str):
        return {"day": None, "college": None}
    t = text.strip()
    day = None
    for name, code in DAY_NAME_TO_CODE.items():
        if name in t:
            day = code
            break
    college = None
    for name in COLLEGE_TO_ROOM_PREFIX:
        if name in t:
            college = name
            break
    return {"day": day, "college": college}
