"""학생 말투 질문 ChromaDB 검색 평가."""
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.chroma_search import search

data = json.load(open("data/yongin_univ_questions_1000_student_style.json", encoding="utf-8"))

distances = []
type_counts = defaultdict(int)
bucket = {"<0.3": 0, "0.3~0.4": 0, "0.4~0.5": 0, ">0.5": 0}

worst = []  # (dist, question, top_result)
best = []

print("평가 시작 (1000개)...")
for i, item in enumerate(data):
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/1000 완료")

    q = item["question"]
    results = search(q, n_results=10)
    if not results:
        continue

    top = results[0]
    d = top["distance"]
    distances.append(d)
    type_counts[top["metadata"].get("type", "?")] += 1

    if d < 0.3:
        bucket["<0.3"] += 1
    elif d < 0.4:
        bucket["0.3~0.4"] += 1
    elif d < 0.5:
        bucket["0.4~0.5"] += 1
    else:
        bucket[">0.5"] += 1

    worst.append((d, q, top))
    best.append((d, q, top))

worst.sort(reverse=True)
best.sort()

avg = sum(distances) / len(distances)
print(f"\n전체 평균 거리: {avg:.4f}")

# 기존 파일에 추가
out_path = Path("docs/할일_정리.md")
existing = out_path.read_text(encoding="utf-8") if out_path.exists() else ""

lines = []
lines.append("# 학생 말투 질문 검색 정확도 평가\n")
lines.append(f"- 평가 일자: 2026-03-20")
lines.append(f"- 데이터: yongin_univ_questions_1000_student_style.json ({len(distances)}개)")
lines.append(f"- 검색 방식: paraphrase-multilingual-MiniLM-L12-v2, top-10\n")

lines.append("## 전체 결과\n")
lines.append(f"| 항목 | 값 |")
lines.append(f"|---|---|")
lines.append(f"| 평균 Top-1 거리 | {avg:.4f} |")
lines.append(f"| 최솟값 | {min(distances):.4f} |")
lines.append(f"| 최댓값 | {max(distances):.4f} |")
lines.append("")

lines.append("## 거리 분포 (Top-1 기준)\n")
lines.append("| 구간 | 개수 | 비율 | 해석 |")
lines.append("|---|---|---|---|")
lines.append(f"| < 0.3 | {bucket['<0.3']} | {bucket['<0.3']/len(distances)*100:.1f}% | 매우 잘 매칭 |")
lines.append(f"| 0.3 ~ 0.4 | {bucket['0.3~0.4']} | {bucket['0.3~0.4']/len(distances)*100:.1f}% | 잘 매칭 |")
lines.append(f"| 0.4 ~ 0.5 | {bucket['0.4~0.5']} | {bucket['0.4~0.5']/len(distances)*100:.1f}% | 보통 |")
lines.append(f"| > 0.5 | {bucket['>0.5']} | {bucket['>0.5']/len(distances)*100:.1f}% | 매칭 불량 |")
lines.append("")

lines.append("## Top-1 결과 타입 분포\n")
lines.append("| 타입 | 개수 | 비율 |")
lines.append("|---|---|---|")
for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
    lines.append(f"| {t} | {cnt} | {cnt/len(distances)*100:.1f}% |")
lines.append("")

lines.append("## 매칭 잘 된 질문 예시 (Top 10)\n")
for d, q, r in best[:10]:
    t = r["metadata"].get("type")
    if t == "course":
        ans = f"{r['metadata'].get('course_name','')} / {r['metadata'].get('professor','')} / {r['metadata'].get('day','')} {r['metadata'].get('time_range','')}"
    else:
        ans = r["text"][:60]
    lines.append(f"- `{q}`")
    lines.append(f"  - dist={d:.4f} | {ans}")
lines.append("")

lines.append("## 매칭 잘 안 된 질문 예시 (Bottom 10)\n")
for d, q, r in worst[:10]:
    t = r["metadata"].get("type")
    if t == "course":
        ans = f"{r['metadata'].get('course_name','')} / {r['metadata'].get('professor','')}"
    else:
        ans = r["text"][:60]
    lines.append(f"- `{q}`")
    lines.append(f"  - dist={d:.4f} | Top-1: {ans}")
lines.append("")

output = "\n".join(lines)
out_path.write_text(existing + "\n\n---\n\n" + output, encoding="utf-8")
print(f"결과 저장 완료: {out_path}")
