"""ChromaDB 청크 데이터 → Neo4j 그래프 DB 적재 (정규화 스키마).

[스키마 v2 — 정규화]
노드:
  - Professor  {name, phone, office, research_day, dept}
  - Course     {course_number, course_name, course_type, year, credits, capacity}
                                                              # course_number = 섹션 단위 unique
  - Day        {name}                       ← NEW: 월/화/수/목/금
  - Time       {range}                      ← NEW: 시간대별 unique
  - Room       {name}                       ← 기존 유지
  - Department {name, phone, college}

관계:
  (Professor)-[:TEACHES]->(Course)          # day/time_range 속성 제거 (Day/Time 노드로 이동)
  (Course)-[:HELD_ON]->(Day)                ← NEW
  (Course)-[:HAS_TIME]->(Time)              ← NEW
  (Course)-[:LOCATED_IN]->(Room)            # 기존 HELD_IN → LOCATED_IN
  (Professor)-[:BELONGS_TO]->(Department)   ← NEW (기존 property → 관계)

[v1 → v2 전환 이유]
  · "월요일에 강의 있는 교수는?" 같은 역방향/다중 홉 질의 지원
  · Day/Time 을 관계 속성으로 두면 SQL JOIN 유사 표현 → GraphRAG의 이점 상실
  · Cypher 로 자연스러운 다단계 탐색: (Prof)-[:TEACHES]->(Course)-[:HELD_ON]->(Day)

[사용법]
  # 완전 재적재 (기존 데이터 삭제)
  python scripts/rag/run_ingest_neo4j.py --wipe

  # 기존 유지 + upsert (스키마 v2 로 마이그레이션 시에도 사용 가능. 다만 v1 잔재 관계는 남음)
  python scripts/rag/run_ingest_neo4j.py
"""
import argparse
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


def wipe_all(tx):
    """DB 전체 삭제. v1 스키마 잔재 제거용."""
    tx.run("MATCH (n) DETACH DELETE n")


def create_indexes(tx):
    for label, prop in [
        ("Professor",  "name"),
        ("Course",     "course_number"),
        ("Course",     "course_name"),
        ("Day",        "name"),
        ("Time",       "range"),
        ("Room",       "name"),
        ("Department", "name"),
    ]:
        tx.run(f"CREATE INDEX {label}_{prop} IF NOT EXISTS FOR (n:{label}) ON (n.{prop})")


def ingest_courses(tx, records):
    """강의 레코드 → Course + Professor + Day + Time + Room + 관계들."""
    for r in records:
        if r.get("type") != "course":
            continue
        prof_name   = (r.get("professor") or "").strip()
        course_num  = r.get("course_number", "")
        course_name = r.get("course_name", "")
        day         = (r.get("day") or "").strip()
        time_range  = (r.get("time_range") or "").strip()
        room        = (r.get("room") or "").strip()

        if not course_num:
            continue

        # Course 노드 (section 단위 unique)
        tx.run("""
            MERGE (c:Course {course_number: $cn})
            SET c.course_name = $name,
                c.course_type = $ct,
                c.year        = $yr,
                c.credits     = $cr,
                c.capacity    = $cap
        """, cn=course_num,
             name=course_name,
             ct=r.get("course_type") or "",
             yr=r.get("year") or 0,
             cr=r.get("credits") or 0,
             cap=r.get("capacity") or 0)

        if prof_name:
            tx.run("""
                MERGE (p:Professor {name: $pname})
                WITH p
                MATCH (c:Course {course_number: $cn})
                MERGE (p)-[:TEACHES]->(c)
            """, pname=prof_name, cn=course_num)

        if day:
            tx.run("""
                MERGE (d:Day {name: $day})
                WITH d
                MATCH (c:Course {course_number: $cn})
                MERGE (c)-[:HELD_ON]->(d)
            """, day=day, cn=course_num)

        if time_range:
            tx.run("""
                MERGE (t:Time {range: $tr})
                WITH t
                MATCH (c:Course {course_number: $cn})
                MERGE (c)-[:HAS_TIME]->(t)
            """, tr=time_range, cn=course_num)

        if room:
            tx.run("""
                MERGE (r:Room {name: $room})
                WITH r
                MATCH (c:Course {course_number: $cn})
                MERGE (c)-[:LOCATED_IN]->(r)
            """, room=room, cn=course_num)


def ingest_professors(tx, records):
    """교수 연구일/연락처. Professor 노드에 속성 병합."""
    for r in records:
        if r.get("type") != "professor":
            continue
        name = (r.get("name") or "").strip()
        if not name:
            continue
        tx.run("""
            MERGE (p:Professor {name: $name})
            SET p.research_day = $day,
                p.phone        = $phone,
                p.office       = $room,
                p.dept         = $dept
        """, name=name,
             day=r.get("day", ""),
             phone=r.get("phone", ""),
             room=r.get("room", ""),
             dept=r.get("dept", ""))


def ingest_dept_phones(tx, records):
    """학과 전화번호 → Department 노드."""
    for r in records:
        if r.get("type") != "dept_phone":
            continue
        dept = (r.get("dept") or "").strip()
        if not dept:
            continue
        tx.run("""
            MERGE (d:Department {name: $dept})
            SET d.phone   = $phone,
                d.college = $college
        """, dept=dept,
             phone=r.get("phone", ""),
             college=r.get("college", ""))


def link_professor_departments(tx):
    """professor.dept (문자열) 이 Department 노드 이름과 일치하면 BELONGS_TO 관계 생성."""
    tx.run("""
        MATCH (p:Professor)
        WHERE p.dept IS NOT NULL AND p.dept <> ""
        MATCH (d:Department {name: p.dept})
        MERGE (p)-[:BELONGS_TO]->(d)
    """)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wipe", action="store_true",
                    help="적재 전 DB 전체 삭제 (v1 잔재 제거 시 권장)")
    return ap.parse_args()


def main():
    args = parse_args()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    course_records = json.loads(COURSE_FILE.read_text(encoding="utf-8"))
    md_records     = json.loads(MD_FILE.read_text(encoding="utf-8"))

    with driver.session() as session:
        if args.wipe:
            print("⚠  DB 전체 삭제 중...")
            session.execute_write(wipe_all)

        print("인덱스 생성 중...")
        session.execute_write(create_indexes)

        print(f"강의 레코드 적재 중... ({len(course_records)}개, 배치 200)")
        BATCH = 200
        for i in range(0, len(course_records), BATCH):
            batch = course_records[i:i + BATCH]
            session.execute_write(ingest_courses, batch)
            print(f"  {min(i + BATCH, len(course_records))}/{len(course_records)}")

        print("교수 정보 적재 중...")
        session.execute_write(ingest_professors, md_records)

        print("학과 전화번호 적재 중...")
        session.execute_write(ingest_dept_phones, md_records)

        print("Professor ↔ Department 관계 연결 중...")
        session.execute_write(link_professor_departments)

    with driver.session() as session:
        node_counts = {}
        for label in ("Professor", "Course", "Day", "Time", "Room", "Department"):
            node_counts[label] = session.run(
                f"MATCH (n:{label}) RETURN count(n) AS c"
            ).single()["c"]

        rel_counts = {}
        for rel in ("TEACHES", "HELD_ON", "HAS_TIME", "LOCATED_IN", "BELONGS_TO"):
            rel_counts[rel] = session.run(
                f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c"
            ).single()["c"]

    driver.close()

    print("\n[적재 완료]")
    print("노드:")
    for label, cnt in node_counts.items():
        print(f"  {label:12} {cnt:>5}")
    print("관계:")
    for rel, cnt in rel_counts.items():
        print(f"  {rel:12} {cnt:>5}")


if __name__ == "__main__":
    main()
