"""Neo4j 그래프 DB 연동: 교수–과목–요일·시간·강의실 관계 저장 및 Cypher 조회."""
import os
from pathlib import Path

from app.ontology.triples import COLLEGE_TO_ROOM_PREFIX, load_triples


def _env(key, default=None):
    return os.environ.get(key, default)


def get_driver(uri=None, user=None, password=None):
    """Neo4j 드라이버 생성. 미지정 시 환경변수 NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD 사용."""
    uri = uri or _env("NEO4J_URI", "bolt://localhost:7687")
    user = user or _env("NEO4J_USER", "neo4j")
    password = password or _env("NEO4J_PASSWORD", "")
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise ImportError("neo4j 패키지가 필요합니다. pip install neo4j")
    return GraphDatabase.driver(uri, auth=(user, password))


def sync_triples_to_neo4j(triples=None, path=None, driver=None):
    """
    트리플을 Neo4j에 반영.
    노드: (:Professor {name}), (:Course {name})
    관계: (Professor)-[:TEACHES {day, time_range, room, title, doc_id}]->(Course)
    기존 Professor, Course, TEACHES 는 삭제 후 재생성.
    """
    if triples is None:
        triples = load_triples(path)
    if not triples:
        return 0

    close_driver = False
    if driver is None:
        driver = get_driver()
        close_driver = True
    try:
        with driver.session() as session:
            session.run("MATCH (n) WHERE n:Professor OR n:Course DETACH DELETE n")
            created = 0
            for t in triples:
                teacher = (t.get("subject") or "").strip()
                course = (t.get("object") or "").strip()
                if not teacher or not course:
                    continue
                session.run(
                    """
                    MERGE (p:Professor {name: $teacher})
                    MERGE (c:Course {name: $course})
                    CREATE (p)-[:TEACHES {day: $day, time_range: $time_range, room: $room, title: $title, doc_id: $doc_id}]->(c)
                    """,
                    teacher=teacher,
                    course=course,
                    day=t.get("day") or "",
                    time_range=t.get("time_range") or "",
                    room=t.get("room") or "",
                    title=t.get("title") or "",
                    doc_id=t.get("doc_id") or "",
                )
                created += 1
        return created
    finally:
        if close_driver:
            driver.close()


def query_schedule_from_neo4j(
    day=None,
    college=None,
    room_prefix=None,
    teacher=None,
    course_contains=None,
    driver=None,
):
    """
    Neo4j에서 Cypher로 수업 일정 조회.
    반환 형식: [ {"teacher_name", "course_name", "day", "time_range", "room", "title", "doc_id"}, ... ]
    """
    use_prefix = room_prefix
    if college and not use_prefix:
        use_prefix = COLLEGE_TO_ROOM_PREFIX.get(college.strip()) or COLLEGE_TO_ROOM_PREFIX.get(
            (college.strip() or "").replace("대학", "")
        )
        if not use_prefix and college.strip():
            for k, v in COLLEGE_TO_ROOM_PREFIX.items():
                if k in college or college in k:
                    use_prefix = v
                    break

    close_driver = False
    if driver is None:
        driver = get_driver()
        close_driver = True
    try:
        with driver.session() as session:
            q = """
            MATCH (p:Professor)-[t:TEACHES]->(c:Course)
            WHERE 1=1
            """
            params = {}
            if day is not None:
                q += " AND t.day = $day"
                params["day"] = str(day).strip()
            if use_prefix:
                q += " AND t.room STARTS WITH $room_prefix"
                params["room_prefix"] = use_prefix
            if teacher is not None:
                q += " AND p.name = $teacher"
                params["teacher"] = str(teacher).strip()
            if course_contains:
                q += " AND c.name CONTAINS $course_contains"
                params["course_contains"] = str(course_contains).strip()
            q += " RETURN p.name AS teacher_name, c.name AS course_name, t.day AS day, t.time_range AS time_range, t.room AS room, t.title AS title, t.doc_id AS doc_id"
            result = session.run(q, params)
            out = []
            for rec in result:
                out.append({
                    "teacher_name": rec.get("teacher_name") or "",
                    "course_name": rec.get("course_name") or "",
                    "day": rec.get("day") or "",
                    "time_range": rec.get("time_range") or "",
                    "room": rec.get("room") or "",
                    "title": rec.get("title") or "",
                    "doc_id": rec.get("doc_id") or "",
                })
            return out
    finally:
        if close_driver:
            driver.close()


def get_courses_by_teacher_neo4j(teacher_name, driver=None):
    """Neo4j에서 교수명으로 수업 목록 조회. 반환 형식은 get_courses_by_teacher 와 동일."""
    out = query_schedule_from_neo4j(teacher=teacher_name, driver=driver)
    return [{"course_name": r["course_name"], "day": r["day"], "time_range": r["time_range"], "room": r["room"], "title": r["title"], "doc_id": r["doc_id"]} for r in out]


def get_schedule_by_course_neo4j(course_name, driver=None):
    """Neo4j에서 과목명(포함)으로 수업 일정 조회. 반환 형식은 get_schedule_by_course 와 동일."""
    return query_schedule_from_neo4j(course_contains=course_name, driver=driver)


def is_neo4j_available():
    """Neo4j 연결 가능 여부 (패키지 설치 + URI 설정 + 실제 연결 시도)."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return False
    uri = _env("NEO4J_URI", "").strip()
    if not uri:
        return False
    try:
        d = get_driver()
        d.verify_connectivity()
        d.close()
        return True
    except Exception:
        return False
