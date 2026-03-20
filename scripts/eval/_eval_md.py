"""
마크다운 청크 추가 후 ChromaDB 정확도 평가
yongin_univ_questions_1000_student_style.json 사용
키워드(교수명/과목명) 매칭 방식 + Top-10
"""
import json
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.chroma_search import search

Q_FILE    = PROJECT_ROOT / "data" / "yongin_univ_questions_1000_student_style.json"
CHUNK_FILE = PROJECT_ROOT / "data" / "processed" / "chunks" / "academic" / "academic_001.json"
OUT_FILE  = PROJECT_ROOT / "_eval_md_result.json"

N_RESULTS = 10

def load_keywords():
    """DB에 있는 교수명/과목명 추출"""
    data = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    professors = set()
    courses = set()
    for r in data:
        if r.get("professor"):
            professors.add(r["professor"])
        if r.get("course_name"):
            courses.add(r["course_name"])
    return professors, courses

def extract_keywords(question: str, professors: set, courses: set):
    found_profs = [p for p in professors if p in question]
    found_courses = [c for c in courses if c in question]
    return found_profs, found_courses

def main():
    questions = json.loads(Q_FILE.read_text(encoding="utf-8"))
    professors, courses = load_keywords()

    stats = {
        "prof": {"total": 0, "hit": 0},
        "course": {"total": 0, "hit": 0},
        "both": {"total": 0, "hit": 0},
        "none": 0,
    }
    misses = {"prof": [], "course": []}

    for item in questions:
        q = item.get("question", item) if isinstance(item, dict) else item
        found_profs, found_courses = extract_keywords(q, professors, courses)

        has_prof   = len(found_profs) > 0
        has_course = len(found_courses) > 0

        if not has_prof and not has_course:
            stats["none"] += 1
            continue

        results = search(q, n_results=N_RESULTS)
        hit = False

        # Hit 판정: top-N 결과 중 교수명 or 과목명 일치
        for r in results:
            m = r["metadata"]
            result_prof   = m.get("professor", "") or m.get("name", "")
            result_course = m.get("course_name", "")
            if has_prof and any(p in result_prof for p in found_profs):
                hit = True; break
            if has_course and any(c in result_course for c in found_courses):
                hit = True; break

        if has_prof and has_course:
            stats["both"]["total"] += 1
            if hit: stats["both"]["hit"] += 1
        elif has_prof:
            stats["prof"]["total"] += 1
            if hit:
                stats["prof"]["hit"] += 1
            else:
                misses["prof"].append({
                    "q": q,
                    "keywords": found_profs,
                    "top1": results[0]["metadata"].get("course_name", results[0]["metadata"].get("name", "")),
                    "top1_prof": results[0]["metadata"].get("professor", results[0]["metadata"].get("name", "")),
                    "dist": round(results[0]["distance"], 4) if results else 9,
                })
        else:
            stats["course"]["total"] += 1
            if hit:
                stats["course"]["hit"] += 1
            else:
                misses["course"].append({
                    "q": q,
                    "keywords": found_courses,
                    "top1": results[0]["metadata"].get("course_name", ""),
                    "top1_prof": results[0]["metadata"].get("professor", ""),
                    "dist": round(results[0]["distance"], 4) if results else 9,
                })

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
        "professor_misses_sample": misses["prof"][:10],
        "course_misses_sample": misses["course"][:10],
    }
    OUT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("평가 완료 →", OUT_FILE)
    for k, v in list(summary.items())[:4]:
        if isinstance(v, dict):
            print(f"  {k}: {v['hit']}/{v['total']} = {v['accuracy']}")
        else:
            print(f"  {k}: {v}개")

if __name__ == "__main__":
    main()
