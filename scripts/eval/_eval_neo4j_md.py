"""Neo4j 키워드 기반 정확도 평가 (교수명/과목명)"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from neo4j import GraphDatabase

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "yu_chatbot_2026"

Q_FILE     = PROJECT_ROOT / "data" / "yongin_univ_questions_1000_student_style.json"
CHUNK_FILE = PROJECT_ROOT / "data" / "processed" / "chunks" / "academic" / "academic_001.json"
OUT_FILE   = PROJECT_ROOT / "_eval_neo4j_result.json"


def load_keywords():
    data = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    professors = set()
    courses = set()
    for r in data:
        if r.get("professor"):
            professors.add(r["professor"])
        if r.get("course_name"):
            courses.add(r["course_name"])
    return professors, courses


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    questions = json.loads(Q_FILE.read_text(encoding="utf-8"))
    professors, courses = load_keywords()

    stats = {
        "prof": {"total": 0, "hit": 0},
        "course": {"total": 0, "hit": 0},
        "both": {"total": 0, "hit": 0},
        "none": 0,
    }

    with driver.session() as session:
        for item in questions:
            q = item.get("question", item) if isinstance(item, dict) else item
            found_profs   = [p for p in professors if p in q]
            found_courses = [c for c in courses if c in q]

            has_prof   = len(found_profs) > 0
            has_course = len(found_courses) > 0

            if not has_prof and not has_course:
                stats["none"] += 1
                continue

            hit = False

            if has_prof:
                for pname in found_profs:
                    result = session.run(
                        "MATCH (p:Professor {name: $name}) RETURN p LIMIT 1",
                        name=pname
                    ).single()
                    if result:
                        hit = True
                        break

            if not hit and has_course:
                for cname in found_courses:
                    result = session.run(
                        "MATCH (c:Course) WHERE c.course_name = $name RETURN c LIMIT 1",
                        name=cname
                    ).single()
                    if result:
                        hit = True
                        break

            if has_prof and has_course:
                stats["both"]["total"] += 1
                if hit: stats["both"]["hit"] += 1
            elif has_prof:
                stats["prof"]["total"] += 1
                if hit: stats["prof"]["hit"] += 1
            else:
                stats["course"]["total"] += 1
                if hit: stats["course"]["hit"] += 1

    driver.close()

    def pct(h, t): return f"{h/t*100:.1f}%" if t else "N/A"

    summary = {
        "교수명 포함 질문": {
            "total": stats["prof"]["total"],
            "hit": stats["prof"]["hit"],
            "accuracy": pct(stats["prof"]["hit"], stats["prof"]["total"]),
        },
        "과목명 포함 질문": {
            "total": stats["course"]["total"],
            "hit": stats["course"]["hit"],
            "accuracy": pct(stats["course"]["hit"], stats["course"]["total"]),
        },
        "교수+과목 포함 질문": {
            "total": stats["both"]["total"],
            "hit": stats["both"]["hit"],
            "accuracy": pct(stats["both"]["hit"], stats["both"]["total"]),
        },
        "키워드 없는 질문": stats["none"],
    }
    OUT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("평가 완료")
    for k, v in list(summary.items()):
        if isinstance(v, dict):
            print(f"  {k}: {v['hit']}/{v['total']} = {v['accuracy']}")
        else:
            print(f"  {k}: {v}개")


if __name__ == "__main__":
    main()
