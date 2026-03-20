"""ChromaDB 검색 정확도 평가."""
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.chroma_search import search

AVAILABLE_DOCS = {"academic_001", "calendar_001"}
DOC_NAME_MAP = {
    "학사_2026_1학기_종합강의시간표.pdf": "academic_001",
    "학사_2026_겨울계절학기_개설안내.pdf": "calendar_001",
}

data = json.load(open("data/yongin_univ_questions_1000.json", encoding="utf-8"))

# 카테고리별 집계
cat_stats = defaultdict(lambda: {"total": 0, "hit": 0, "dist_sum": 0.0})
hit_examples = defaultdict(list)
miss_examples = defaultdict(list)

print("평가 시작 (1000개)...")
for i, item in enumerate(data):
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/1000 완료")

    q = item["question"]
    cat = item["category"]
    expected_docs = {DOC_NAME_MAP[d] for d in item.get("source_docs", []) if d in DOC_NAME_MAP}

    results = search(q, n_results=10)
    top_dist = results[0]["distance"] if results else 1.0
    retrieved_docs = {r["metadata"].get("doc_id", "") for r in results}

    hit = bool(expected_docs & retrieved_docs)

    cat_stats[cat]["total"] += 1
    cat_stats[cat]["dist_sum"] += top_dist
    if hit:
        cat_stats[cat]["hit"] += 1
        if len(hit_examples[cat]) < 2:
            hit_examples[cat].append((q, results[0]))
    else:
        if len(miss_examples[cat]) < 2:
            miss_examples[cat].append((q, results[0] if results else None))

# 전체 집계
total = sum(s["total"] for s in cat_stats.values())
total_hit = sum(s["hit"] for s in cat_stats.values())
avg_dist_all = sum(s["dist_sum"] for s in cat_stats.values()) / total

print(f"\n전체: {total_hit}/{total} = {total_hit/total*100:.1f}%  평균거리: {avg_dist_all:.4f}")

# 결과 저장
lines = []
lines.append("# ChromaDB 검색 정확도 평가 결과\n")
lines.append(f"- 평가 일자: 2026-03-20")
lines.append(f"- 총 질문 수: {total}개")
lines.append(f"- 적재 문서: academic_001 (1054청크), calendar_001 (28청크)")
lines.append(f"- 검색 방식: paraphrase-multilingual-MiniLM-L12-v2, top-10")
lines.append(f"- **Hit 정의**: 상위 10개 결과 중 기대 문서(academic_001 / calendar_001)가 1개 이상 포함\n")

lines.append(f"## 전체 결과\n")
lines.append(f"| 항목 | 값 |")
lines.append(f"|---|---|")
lines.append(f"| 전체 Hit율 | {total_hit/total*100:.1f}% ({total_hit}/{total}) |")
lines.append(f"| 평균 Top-1 거리 | {avg_dist_all:.4f} |")
lines.append("")

lines.append("## 카테고리별 Hit율\n")
lines.append("| 카테고리 | 질문 수 | Hit | Hit율 | 평균거리 |")
lines.append("|---|---|---|---|---|")
for cat, s in sorted(cat_stats.items(), key=lambda x: -x[1]["hit"]/x[1]["total"]):
    hit_rate = s["hit"] / s["total"] * 100
    avg_dist = s["dist_sum"] / s["total"]
    lines.append(f"| {cat} | {s['total']} | {s['hit']} | {hit_rate:.1f}% | {avg_dist:.4f} |")
lines.append("")

lines.append("## 분석 및 개선 포인트\n")
low_cats = [(cat, s) for cat, s in cat_stats.items() if s["hit"]/s["total"] < 0.5]
if low_cats:
    lines.append("### Hit율 50% 미만 카테고리 (개선 필요)")
    for cat, s in sorted(low_cats, key=lambda x: x[1]["hit"]/x[1]["total"]):
        lines.append(f"- **{cat}**: {s['hit']}/{s['total']} ({s['hit']/s['total']*100:.1f}%)")
        if miss_examples[cat]:
            q, r = miss_examples[cat][0]
            lines.append(f"  - 예시 미스: \"{q}\"")
            if r:
                lines.append(f"  - Top-1 반환: type={r['metadata'].get('type')}, doc={r['metadata'].get('doc_id')}, dist={r['distance']:.4f}")
    lines.append("")

high_cats = [(cat, s) for cat, s in cat_stats.items() if s["hit"]/s["total"] >= 0.8]
if high_cats:
    lines.append("### Hit율 80% 이상 카테고리 (잘 동작)")
    for cat, s in sorted(high_cats, key=lambda x: -x[1]["hit"]/x[1]["total"]):
        lines.append(f"- **{cat}**: {s['hit']}/{s['total']} ({s['hit']/s['total']*100:.1f}%)")
    lines.append("")

output = "\n".join(lines)
out_path = Path("docs/할일_정리.md")
existing = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
out_path.write_text(existing + "\n\n---\n\n" + output, encoding="utf-8")
print(f"\n결과 저장 완료: {out_path}")
