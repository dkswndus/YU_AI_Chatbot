"""
ChromaDB 청크 데이터 → Neo4j 그래프 DB 적재

노드:
  - Professor {name}
  - Course {course_number, course_name, course_type, year, credits, capacity}
  - Room {name}
  - Department {name, phone, college}

관계:
  - (Professor)-[:TEACHES {day, time_range}]->(Course)
  - (Course)-[:HELD_IN]->(Room)
  - (Professor)-[:RESEARCH_DAY {day, phone, room}]->(Professor) -- 자기참조 대신 속성으로
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neo4j import GraphDatabase

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "yu_chatbot_2026"

COURSE_FILE = PROJECT_ROOT / "data" / "processed" / "chunks" / "academic" / "academic_001.json"
MD_FILE     = PROJECT_ROOT / "data" / "processed" / "chunks" / "academic_md" / "academic_001_md.json"


def ingest_courses(tx, records):
    """강의 레코드 → Professor, Course, Room 노드 + 관계"""
    for r in records:
        if r.get("type") != "course":
            continue
        prof_name   = r.get("professor") or ""
        course_num  = r.get("course_number", "")
        course_name = r.get("course_name", "")
        day         = r.get("day") or ""
        time_range  = r.get("time_range") or ""
        room        = r.get("room") or ""
        course_type = r.get("course_type") or ""
        year        = r.get("year") or 0
        credits     = r.get("credits") or 0
        capacity    = r.get("capacity") or 0

        # Course 노드
        tx.run("""
            MERGE (c:Course {course_number: $cn})
            SET c.course_name = $name,
                c.course_type = $ct,
                c.year = $yr,
                c.credits = $cr,
                c.capacity = $cap
        """, cn=course_num, name=course_name, ct=course_type,
             yr=year, cr=credits, cap=capacity)

        # Professor 노드 + TEACHES 관계
        if prof_name:
            tx.run("""
                MERGE (p:Professor {name: $pname})
                WITH p
                MATCH (c:Course {course_number: $cn})
                MERGE (p)-[t:TEACHES]->(c)
                SET t.day = $day, t.time_range = $time_range
            """, pname=prof_name, cn=course_num, day=day, time_range=time_range)

        # Room 노드 + HELD_IN 관계
        if room:
            tx.run("""
                MERGE (r:Room {name: $room})
                WITH r
                MATCH (c:Course {course_number: $cn})
                MERGE (c)-[:HELD_IN]->(r)
            """, room=room, cn=course_num)


def ingest_professors(tx, records):
    """교수 연구일 레코드 → Professor 노드에 연구일 정보 추가"""
    for r in records:
        if r.get("type") != "professor":
            continue
        name  = r.get("name", "")
        day   = r.get("day", "")
        phone = r.get("phone", "")
        room  = r.get("room", "")
        dept  = r.get("dept", "")
        if not name:
            continue
        tx.run("""
            MERGE (p:Professor {name: $name})
            SET p.research_day = $day,
                p.phone = $phone,
                p.office = $room,
                p.dept = $dept
        """, name=name, day=day, phone=phone, room=room, dept=dept)


def ingest_dept_phones(tx, records):
    """학과 전화번호 → Department 노드"""
    for r in records:
        if r.get("type") != "dept_phone":
            continue
        dept    = r.get("dept", "")
        phone   = r.get("phone", "")
        college = r.get("college", "")
        if not dept:
            continue
        tx.run("""
            MERGE (d:Department {name: $dept})
            SET d.phone = $phone, d.college = $college
        """, dept=dept, phone=phone, college=college)


def create_indexes(tx):
    for label, prop in [
        ("Professor", "name"),
        ("Course", "course_number"),
        ("Course", "course_name"),
        ("Room", "name"),
        ("Department", "name"),
    ]:
        tx.run(f"CREATE INDEX {label}_{prop} IF NOT EXISTS FOR (n:{label}) ON (n.{prop})")


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    course_records = json.loads(COURSE_FILE.read_text(encoding="utf-8"))
    md_records     = json.loads(MD_FILE.read_text(encoding="utf-8"))

    with driver.session() as session:
        print("인덱스 생성...")
        session.execute_write(create_indexes)

        print(f"강의 레코드 적재 중... ({len(course_records)}개)")
        # 배치 처리
        BATCH = 200
        for i in range(0, len(course_records), BATCH):
            batch = course_records[i:i + BATCH]
            session.execute_write(ingest_courses, batch)
            print(f"  {min(i+BATCH, len(course_records))}/{len(course_records)}")

        print(f"교수 연구일 적재 중...")
        session.execute_write(ingest_professors, md_records)

        print(f"학과 전화번호 적재 중...")
        session.execute_write(ingest_dept_phones, md_records)

    # 통계
    with driver.session() as session:
        counts = {
            label: session.run(f"MATCH (n:{label}) RETURN count(n) as c").single()["c"]
            for label in ["Professor", "Course", "Room", "Department"]
        }
        teaches = session.run("MATCH ()-[r:TEACHES]->() RETURN count(r) as c").single()["c"]

    driver.close()

    print("\n적재 완료:")
    for label, cnt in counts.items():
        print(f"  {label}: {cnt}개")
    print(f"  TEACHES 관계: {teaches}개")


if __name__ == "__main__":
    main()
