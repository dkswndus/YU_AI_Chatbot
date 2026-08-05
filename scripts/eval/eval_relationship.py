"""Relationship Query 특화 평가 (Neo4j 활용도 측정).

목적: "Neo4j 를 왜 도입했는가" 의 실증.
      Vector · BM25 만으로는 어려운 다중 홉 관계 질문 정확도를 측정한다.

사용:
  # Neo4j 활성 상태
  python scripts/eval/eval_relationship.py --config with_neo4j

  # Neo4j 비활성 (docker stop yu_neo4j 후)
  python scripts/eval/eval_relationship.py --config no_neo4j

각 정답은 실제 데이터(academic_001.json)에서 수동 큐레이트.
"""
import argparse
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import pipeline as pl

OUT_DIR = PROJECT_ROOT / "data" / "eval"


# ── 큐레이트된 관계 질문 세트 (정답: keyword 리스트, 이 중 하나라도 포함되면 정답)
RELATIONSHIP_TESTS = [
    # (category, question, expected_keywords_in_answer)
    ("prof→courses", "김중헌 교수가 담당하는 과목은?",
     ["태권도", "품새"]),
    ("prof→courses", "강맹수 교수 담당 강의 알려줘",
     ["확률과통계"]),
    ("course→prof",  "확률과통계 담당 교수 누구야",
     ["강맹수"]),
    ("prof+day",     "강맹수 교수 금요일 수업은?",
     ["확률과통계", "금"]),
    ("prof+day",     "김중헌 교수 화요일 수업 뭐 있어",
     ["품새", "화"]),
    ("course→time",  "확률과통계 수업 시간 알려줘",
     ["금", "09:25", "13:35"]),
    ("course→room",  "확률과통계 강의실 어디야",
     ["용오름", "7206", "ITMS"]),
    ("dept→phone",   "물리치료학과 전화번호 알려줘",
     ["031-8020-2769", "물리치료학과"]),
    ("dept→phone",   "경영학과 전화번호",
     ["031-8020-2720", "경영학과"]),
    ("prof→credits", "김중헌 교수 담당 과목 학점 알려줘",
     ["학점", "3", "6"]),
]


def _hit(answer: str, expected: list) -> bool:
    if not answer:
        return False
    return any(kw in answer for kw in expected)


def run(config_name: str) -> dict:
    print(f"[{config_name}] N={len(RELATIONSHIP_TESTS)}")
    per_query = []
    hits_by_cat = {}
    total_hits = 0
    latencies = []

    t_start = time.perf_counter()
    for i, (cat, q, expected) in enumerate(RELATIONSHIP_TESTS, 1):
        try:
            meta = pl.answer_with_metadata(q)
        except Exception as e:
            per_query.append({"q": q, "error": str(e)})
            continue

        answer = meta.get("answer", "")
        latency = meta.get("system_efficiency", {}).get("total_ms", 0)
        latencies.append(latency)
        is_hit = _hit(answer, expected)
        total_hits += 1 if is_hit else 0
        hits_by_cat.setdefault(cat, {"total": 0, "hit": 0})
        hits_by_cat[cat]["total"] += 1
        hits_by_cat[cat]["hit"] += 1 if is_hit else 0

        used = meta.get("usage", {}).get("retrieval_types_used", [])
        empty_neo4j = meta.get("reliability", {}).get("retrieval_empty", {}).get("neo4j")
        print(f"  {i:2d}. [{cat}] {'✓' if is_hit else '✗'} ({latency:.0f}ms, {used}, neo4j_empty={empty_neo4j})")
        print(f"      Q: {q}")
        print(f"      A: {answer[:150]}")

        per_query.append({
            "category":  cat,
            "question":  q,
            "expected":  expected,
            "answer":    answer,
            "hit":       is_hit,
            "latency_ms": latency,
            "retrieval_used": used,
            "neo4j_empty":    empty_neo4j,
        })

    duration = time.perf_counter() - t_start
    n = len(RELATIONSHIP_TESTS)
    result = {
        "config":        config_name,
        "date":          str(date.today()),
        "n_questions":   n,
        "accuracy_pct":  round(total_hits / n * 100, 2),
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "by_category":   {
            cat: {
                "total":   d["total"],
                "hit":     d["hit"],
                "acc_pct": round(d["hit"] / d["total"] * 100, 2),
            }
            for cat, d in hits_by_cat.items()
        },
        "duration_sec":  round(duration, 1),
        "per_query":     per_query,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="예: with_neo4j / no_neo4j")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run(args.config)

    out_path = OUT_DIR / f"relationship_{args.config}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[저장] {out_path}")
    print(f"[전체 정확도] {result['accuracy_pct']}% ({result['duration_sec']}초)")
    print(f"[카테고리별]")
    for cat, d in result["by_category"].items():
        print(f"  {cat}: {d['hit']}/{d['total']} ({d['acc_pct']}%)")


if __name__ == "__main__":
    main()
