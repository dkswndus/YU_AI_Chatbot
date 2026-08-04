"""KPI 자동 집계 평가 스크립트.

파이프라인을 평가셋 전량에 실행하고 4개 카테고리 KPI 를 집계한다.
결과는 docs/EXPERIMENT.md §8 (실험 로그 스키마) 에 맞춘 JSON 으로 저장.

[사용법]
  # 개발 모드 (샘플 20개)
  python scripts/eval/run_kpi.py --config full --dataset student_style --limit 20

  # 전체 실행
  python scripts/eval/run_kpi.py --config full --dataset student_style

  # 자연어 벤치
  python scripts/eval/run_kpi.py --config full --dataset natural_v2

[정답 라벨 부재 대응]
  평가셋에 ground truth chunk_id 가 없으므로 heuristic 사용:
    - 질문에서 매칭된 교수명 · 과목명을 pseudo-truth 로 취급
    - Top-K 후보의 metadata 나 content 에 해당 keyword 가 포함되면 hit
  → 실제 정답률 대비 낙관적 편향 가능성 있음. 절대치보다 조합 간 델타로 해석.
"""
import argparse
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import pipeline as pl

DATASETS: Dict[str, Path] = {
    "student_style": PROJECT_ROOT / "data" / "yongin_univ_questions_1000_student_style.json",
    "natural_v2":    PROJECT_ROOT / "data" / "yongin_univ_questions_1000_natural_v2.json",
}
OUT_DIR = PROJECT_ROOT / "data" / "eval"


# ── Heuristic ground truth ────────────────────────────────────────────
def _extract_truth(question: str, profs: Set[str], courses: Set[str]) -> Dict:
    return {
        "professors": [p for p in profs if p in question],
        "courses":    [c for c in courses if c in question],
    }


def _hit_in_citation(citation: Dict, needles: List[str]) -> bool:
    if not needles:
        return False
    md      = citation.get("metadata", {}) or {}
    content = citation.get("content", "") or ""
    blob    = " ".join([
        str(md.get("professor",   "")),
        str(md.get("course_name", "")),
        str(md.get("name",        "")),
        str(md.get("dept",        "")),
        content,
    ])
    return any(n in blob for n in needles)


def _eval_retrieval(question: str, citations: List[Dict],
                    profs: Set[str], courses: Set[str]) -> Dict:
    truth = _extract_truth(question, profs, courses)

    def _score(needles: List[str]) -> Dict:
        if not needles:
            return {"evaluable": False, "hit_1": False, "hit_5": False, "rr": 0.0}
        for rank, cit in enumerate(citations, 1):
            if _hit_in_citation(cit, needles):
                return {
                    "evaluable": True,
                    "hit_1":     rank == 1,
                    "hit_5":     rank <= 5,
                    "rr":        1.0 / rank,
                }
        return {"evaluable": True, "hit_1": False, "hit_5": False, "rr": 0.0}

    prof   = _score(truth["professors"])
    course = _score(truth["courses"])
    return {"prof": prof, "course": course}


# ── 실행 ──────────────────────────────────────────────────────────────
def run(config_name: str, dataset: str, limit: int = None) -> Dict:
    if dataset not in DATASETS:
        raise SystemExit(f"unknown dataset: {dataset}")

    pl._init_keywords()  # populate _professors / _courses
    profs   = set(pl._professors)
    courses = set(pl._courses)

    questions = json.loads(DATASETS[dataset].read_text(encoding="utf-8"))
    if limit:
        questions = questions[:limit]
    N = len(questions)
    print(f"[{config_name}] dataset={dataset}  N={N}")
    print(f"  known professors: {len(profs)},  known courses: {len(courses)}")

    # 집계 버킷
    latencies: List[float]  = []
    llm_calls: List[int]    = []
    tokens_glossary_hits    = 0
    rewrite_called          = 0
    analyze_fallback        = 0
    analyze_json_success    = 0
    pipeline_errors         = 0
    empty = {"bm25": 0, "chroma": 0, "neo4j": 0}
    called = {"keyword": 0, "semantic": 0, "graph": 0}

    prof_eval = prof_h1 = prof_h5 = 0
    prof_rr_sum = 0.0
    course_eval = course_h1 = course_h5 = 0
    course_rr_sum = 0.0

    per_query: List[Dict] = []

    t_start = time.perf_counter()
    for i, item in enumerate(questions):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{N}  elapsed={time.perf_counter() - t_start:.0f}s")
        try:
            meta = pl.answer_with_metadata(item["question"])
        except Exception as e:
            pipeline_errors += 1
            per_query.append({"id": item.get("id"), "error": str(e)})
            continue

        eff   = meta.get("system_efficiency", {}) or {}
        usage = meta.get("usage", {}) or {}
        rel   = meta.get("reliability", {}) or {}
        cits  = meta.get("citations", []) or []

        latencies.append(eff.get("total_ms") or 0)
        llm_calls.append(eff.get("llm_calls") or 0)
        if (usage.get("glossary_hits") or 0) > 0:
            tokens_glossary_hits += 1
        if usage.get("rewrite_called"):
            rewrite_called += 1
        if rel.get("analyze_used_fallback"):
            analyze_fallback += 1
        if rel.get("analyze_json_success"):
            analyze_json_success += 1
        if rel.get("pipeline_error"):
            pipeline_errors += 1

        for rt in usage.get("retrieval_types_used", []) or []:
            if rt in called:
                called[rt] += 1

        empty_dict = rel.get("retrieval_empty", {}) or {}
        for rt in empty:
            if empty_dict.get(rt) is True:
                empty[rt] += 1

        r = _eval_retrieval(item["question"], cits, profs, courses)
        if r["prof"]["evaluable"]:
            prof_eval   += 1
            prof_h1     += r["prof"]["hit_1"]
            prof_h5     += r["prof"]["hit_5"]
            prof_rr_sum += r["prof"]["rr"]
        if r["course"]["evaluable"]:
            course_eval   += 1
            course_h1     += r["course"]["hit_1"]
            course_h5     += r["course"]["hit_5"]
            course_rr_sum += r["course"]["rr"]

        per_query.append({
            "id":            item.get("id"),
            "question":      item["question"],
            "normalized":    meta.get("normalized_query"),
            "intent":        meta.get("intent"),
            "latency_ms":    eff.get("total_ms"),
            "llm_calls":     eff.get("llm_calls"),
            "retrieval_used": usage.get("retrieval_types_used"),
            "prof":          r["prof"],
            "course":        r["course"],
        })
    duration = time.perf_counter() - t_start

    lat_sorted = sorted(latencies)
    p95 = lat_sorted[int(len(lat_sorted) * 0.95)] if lat_sorted else None
    ran = len(latencies) or 1  # 오류 제외 실측 건수

    def _pct(x: int, denom: int) -> float:
        return round(x / denom * 100, 2) if denom else 0.0

    result: Dict[str, Any] = {
        "experiment_id": f"{config_name}_{dataset}",
        "date":          str(date.today()),
        "config": {
            "components":  [config_name],
            "dataset":     dataset,
            "sample_size": N,
            "n_evaluated": ran,
        },
        "retrieval_quality": {
            "professor_name": {
                "evaluable":   prof_eval,
                "hit_at_1_pct": _pct(prof_h1, prof_eval),
                "hit_at_5_pct": _pct(prof_h5, prof_eval),
                "mrr":         round(prof_rr_sum / prof_eval, 4) if prof_eval else None,
            },
            "course_name": {
                "evaluable":   course_eval,
                "hit_at_1_pct": _pct(course_h1, course_eval),
                "hit_at_5_pct": _pct(course_h5, course_eval),
                "mrr":         round(course_rr_sum / course_eval, 4) if course_eval else None,
            },
        },
        "system_efficiency": {
            "avg_latency_ms":       round(statistics.mean(latencies), 2) if latencies else None,
            "p95_latency_ms":       round(p95, 2) if p95 is not None else None,
            "avg_llm_calls":        round(statistics.mean(llm_calls), 2) if llm_calls else None,
            "rewrite_call_rate":    _pct(rewrite_called, ran),
            "bm25_usage_rate":      _pct(called["keyword"],  ran),
            "graph_usage_rate":     _pct(called["graph"],    ran),
            "semantic_usage_rate":  _pct(called["semantic"], ran),
        },
        "reliability": {
            "json_success_rate":    _pct(analyze_json_success, ran),
            "fallback_rate":        _pct(analyze_fallback,     ran),
            "pipeline_error_rate":  _pct(pipeline_errors,      N),
            "glossary_coverage":    _pct(tokens_glossary_hits, ran),
            "retrieval_empty_rate": {
                "bm25":   _pct(empty["bm25"],   ran),
                "chroma": _pct(empty["chroma"], ran),
                "neo4j":  _pct(empty["neo4j"],  ran),
            },
        },
        "duration_sec": round(duration, 1),
    }
    return result, per_query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default="full", help="실험 구성 이름 (라벨링용)")
    ap.add_argument("--dataset", choices=list(DATASETS.keys()), default="student_style")
    ap.add_argument("--limit",   type=int, default=None, help="샘플 N개만 (개발 모드)")
    ap.add_argument("--out",     default=None, help="출력 JSON 경로")
    ap.add_argument("--save-per-query", action="store_true",
                    help="질문별 상세 결과도 별도 파일로 저장")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result, per_query = run(args.config, args.dataset, limit=args.limit)

    out_path = Path(args.out) if args.out else OUT_DIR / f"{args.config}_{args.dataset}.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[결과 저장] {out_path}")

    if args.save_per_query:
        pq_path = out_path.with_name(out_path.stem + "_per_query.json")
        pq_path.write_text(
            json.dumps(per_query, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[상세 저장] {pq_path}")

    print("\n" + json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
