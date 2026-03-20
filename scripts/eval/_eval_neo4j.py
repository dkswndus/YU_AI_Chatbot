"""Neo4j 키워드 기반 정확도 평가."""
import json
import sys
from pathlib import Path
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "yu_chatbot_2026"

# 청크에서 교수명/과목명 목록 추출
chunks = json.loads(Path("data/processed/chunks/academic/academic_001.json").read_text(encoding="utf-8"))
professors = set(r["professor"] for r in chunks if r.get("type") == "course" and r.get("professor"))
courses = set(r["course_name"] for r in chunks if r.get("type") == "course" and r.get("course_name"))
print(f"교수: {len(professors)}명, 과목: {len(courses)}개")

data = json.loads(Path("data/yongin_univ_questions_1000_student_style.json").read_text(encoding="utf-8"))

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

prof_total, prof_hit = 0, 0
course_total, course_hit = 0, 0
no_keyword = 0
miss_examples = []

print("평가 시작 (1000개)...")
with driver.session() as session:
    for i, item in enumerate(data):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/1000 완료")

        q = item["question"]
        mentioned_profs = [p for p in professors if p in q]
        mentioned_courses = [c for c in courses if c in q]

        if not mentioned_profs and not mentioned_courses:
            no_keyword += 1
            continue

        # 교수명 평가
        if mentioned_profs:
            prof_total += 1
            hit = False
            for prof in mentioned_profs:
                result = session.run(
                    "MATCH (p:Professor {name: $name})-[:TEACHES]->(c:Course) RETURN c.name LIMIT 1",
                    name=prof
                ).single()
                if result:
                    hit = True
                    break
            if hit:
                prof_hit += 1
            else:
                miss_examples.append(("교수 미스", q, mentioned_profs, "Neo4j에 없음"))

        # 과목명 평가
        if mentioned_courses:
            course_total += 1
            hit = False
            for cname in mentioned_courses:
                result = session.run(
                    "MATCH (c:Course {name: $name}) RETURN c.name LIMIT 1",
                    name=cname
                ).single()
                if result:
                    hit = True
                    break
            if hit:
                course_hit += 1
            else:
                miss_examples.append(("과목 미스", q, mentioned_courses, "Neo4j에 없음"))

driver.close()

print(f"\n키워드 없는 질문: {no_keyword}개 (평가 제외)")
print(f"교수명 질문: {prof_total}개 → Hit {prof_hit} ({prof_hit/prof_total*100:.1f}%)" if prof_total else "교수명 질문 없음")
print(f"과목명 질문: {course_total}개 → Hit {course_hit} ({course_hit/course_total*100:.1f}%)" if course_total else "과목명 질문 없음")

# 파일 저장
out_path = Path("docs/할일_정리.md")
existing = out_path.read_text(encoding="utf-8") if out_path.exists() else ""

lines = []
lines.append("# Neo4j 키워드 기반 정확도 평가\n")
lines.append(f"- 평가 일자: 2026-03-20")
lines.append(f"- 데이터: yongin_univ_questions_1000_student_style.json (1000개)")
lines.append(f"- 방법: 질문 내 교수명/과목명 → Neo4j MATCH 쿼리로 존재 여부 확인\n")

lines.append("## 전체 결과\n")
lines.append("| 항목 | 질문 수 | Hit | 정확도 |")
lines.append("|---|---|---|---|")
if prof_total:
    lines.append(f"| 교수명 포함 질문 | {prof_total} | {prof_hit} | **{prof_hit/prof_total*100:.1f}%** |")
if course_total:
    lines.append(f"| 과목명 포함 질문 | {course_total} | {course_hit} | **{course_hit/course_total*100:.1f}%** |")
lines.append(f"| 키워드 없어 평가 불가 | {no_keyword} | - | - |")
lines.append("")

if miss_examples:
    lines.append("## 미스 예시 (Top 15)\n")
    for tag, q, keywords, reason in miss_examples[:15]:
        lines.append(f"- [{tag}] `{q}`")
        lines.append(f"  - 찾던 키워드: {keywords} → {reason}")
    lines.append("")

out_path.write_text(existing + "\n\n---\n\n" + "\n".join(lines), encoding="utf-8")
print(f"결과 저장 완료: {out_path}")
