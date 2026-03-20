"""교수명/과목명 키워드 기반 정확도 평가."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.chroma_search import search

# 청크에서 교수명/과목명 목록 추출
chunks = json.loads(Path("data/processed/chunks/academic/academic_001.json").read_text(encoding="utf-8"))
professors = set(r["professor"] for r in chunks if r.get("type") == "course" and r.get("professor"))
courses = set(r["course_name"] for r in chunks if r.get("type") == "course" and r.get("course_name"))

print(f"교수: {len(professors)}명, 과목: {len(courses)}개")

data = json.loads(Path("data/yongin_univ_questions_1000_student_style.json").read_text(encoding="utf-8"))

results_log = []
prof_total, prof_hit = 0, 0
course_total, course_hit = 0, 0
both_total, both_hit = 0, 0
no_keyword = 0

miss_examples = []

print("평가 시작...")
for i, item in enumerate(data):
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/1000 완료")

    q = item["question"]

    # 질문에서 언급된 교수/과목 찾기
    mentioned_profs = [p for p in professors if p in q]
    mentioned_courses = [c for c in courses if c in q]

    if not mentioned_profs and not mentioned_courses:
        no_keyword += 1
        continue

    results = search(q, n_results=10)
    result_profs = {r["metadata"].get("professor", "") for r in results}
    result_courses = {r["metadata"].get("course_name", "") for r in results}

    prof_hit_flag = any(p in result_profs for p in mentioned_profs)
    course_hit_flag = any(c in result_courses for c in mentioned_courses)

    if mentioned_profs:
        prof_total += 1
        if prof_hit_flag:
            prof_hit += 1

    if mentioned_courses:
        course_total += 1
        if course_hit_flag:
            course_hit += 1

    if mentioned_profs and mentioned_courses:
        both_total += 1
        if prof_hit_flag and course_hit_flag:
            both_hit += 1

    # 미스 예시 수집
    if mentioned_profs and not prof_hit_flag:
        miss_examples.append(("교수 미스", q, mentioned_profs, results[0] if results else None))
    elif mentioned_courses and not course_hit_flag:
        miss_examples.append(("과목 미스", q, mentioned_courses, results[0] if results else None))

evaluable = prof_total + course_total - both_total  # 중복 제거
print(f"\n키워드 없는 질문: {no_keyword}개 (평가 제외)")
print(f"교수명 질문: {prof_total}개 → Hit {prof_hit} ({prof_hit/prof_total*100:.1f}%)")
print(f"과목명 질문: {course_total}개 → Hit {course_hit} ({course_hit/course_total*100:.1f}%)")

# 파일 저장
out_path = Path("docs/할일_정리.md")
existing = out_path.read_text(encoding="utf-8") if out_path.exists() else ""

lines = []
lines.append("# 키워드 기반 정확도 평가 (교수명/과목명)\n")
lines.append(f"- 평가 일자: 2026-03-20")
lines.append(f"- 데이터: yongin_univ_questions_1000_student_style.json (1000개)")
lines.append(f"- 방법: 질문 내 교수명/과목명 → 상위 10개 결과에 포함 여부 체크\n")

lines.append("## 전체 결과\n")
lines.append("| 항목 | 질문 수 | Hit | 정확도 |")
lines.append("|---|---|---|---|")
lines.append(f"| 교수명 포함 질문 | {prof_total} | {prof_hit} | **{prof_hit/prof_total*100:.1f}%** |")
lines.append(f"| 과목명 포함 질문 | {course_total} | {course_hit} | **{course_hit/course_total*100:.1f}%** |")
if both_total:
    lines.append(f"| 교수+과목 둘 다 포함 | {both_total} | {both_hit} | **{both_hit/both_total*100:.1f}%** |")
lines.append(f"| 키워드 없어 평가 불가 | {no_keyword} | - | - |")
lines.append("")

lines.append("## 미스 예시 (Top 15)\n")
for tag, q, keywords, top_r in miss_examples[:15]:
    lines.append(f"- [{tag}] `{q}`")
    lines.append(f"  - 찾던 키워드: {keywords}")
    if top_r:
        m = top_r["metadata"]
        lines.append(f"  - Top-1 반환: {m.get('course_name','')} / {m.get('professor','')} (dist={top_r['distance']:.4f})")
lines.append("")

out_path.write_text(existing + "\n\n---\n\n" + "\n".join(lines), encoding="utf-8")
print(f"결과 저장 완료: {out_path}")
