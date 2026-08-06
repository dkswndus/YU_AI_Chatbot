"""Gold Label 자동 라벨링 (v1 → v1_gold).

EXPERIMENT.md §4.3.2 스펙 준수.

접근:
  - entity·simple·relational: chunk file 에서 metadata 규칙 매칭 (professor·course_name·day 조합)
  - semantic: ChromaDB Top-1 을 초안 gold 로 사용
  - ambiguous·non-existent: gold_chunk_ids = [] · answerable = False

출력: data/eval/stratified_200_gold_v1.json (미검수)
사용자 30개 검수 후 v2 로 확정.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHUNK_FILE = PROJECT_ROOT / "data" / "processed" / "chunks" / "academic" / "academic_001.json"
MD_CHUNK_FILE = PROJECT_ROOT / "data" / "processed" / "chunks" / "academic_md" / "academic_001_md.json"
IN_PATH  = PROJECT_ROOT / "data" / "eval" / "stratified_200_v1.json"
OUT_PATH = PROJECT_ROOT / "data" / "eval" / "stratified_200_gold_v1.json"

# 요일 · 과목 관련 파싱
DAY_TOKENS_RE = re.compile(r"(월|화|수|목|금)요일|(?<![가-힣])([월화수목금]{2,5})(?=[\s,\.\?!]|$)")


def _find_days(text: str) -> List[str]:
    found = []
    for m in DAY_TOKENS_RE.finditer(text):
        d = m.group(1) or ""
        if d and d not in found:
            found.append(d)
        else:
            concat = m.group(2) or ""
            for c in concat:
                if c in "월화수목금" and c not in found:
                    found.append(c)
    return found


def _stable_chunk_id(rec: Dict, fallback_idx: int = 0) -> str:
    """안정된 unique chunk_id 생성.

    course/professor/dept chunk 는 명시적 chunk_id 필드가 없으므로
    identifying field 기반으로 생성. 파이프라인·라벨링에서 동일 로직 사용해야 함.
    """
    if rec.get("chunk_id"):
        return rec["chunk_id"]
    doc = rec.get("doc_id", "unknown")
    t = rec.get("type", "")
    if t == "course" and rec.get("course_number"):
        return f"{doc}_course_{rec['course_number']}"
    if t == "professor" and rec.get("name"):
        return f"{doc}_professor_{rec['name']}"
    if t == "dept_phone" and rec.get("dept"):
        return f"{doc}_dept_{rec['dept']}"
    return f"{doc}_{t or 'unknown'}_{fallback_idx}"


def _load_chunks() -> Tuple[List[Dict], Dict[str, Dict]]:
    """(모든 chunk 리스트, chunk_id → chunk 매핑)."""
    chunks: List[Dict] = []
    for path in (CHUNK_FILE, MD_CHUNK_FILE):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for idx, r in enumerate(data):
            chunk_id = _stable_chunk_id(r, fallback_idx=idx)
            r["_chunk_id"] = chunk_id
            r["_source_file"] = path.name
            chunks.append(r)
    by_id = {c["_chunk_id"]: c for c in chunks}
    return chunks, by_id


def _extract_entities(question: str, professors: List[str], courses: List[str]) -> Dict:
    """질문에서 교수·과목·요일 엔티티 추출."""
    prof_hits = [p for p in professors if p in question]
    course_hits = [c for c in courses if c in question]
    day_hits = _find_days(question)
    return {"professors": prof_hits, "courses": course_hits, "days": day_hits}


def _match_gold_by_entity(chunks: List[Dict], entities: Dict) -> List[str]:
    """엔티티 조건에 맞는 chunk_id 반환."""
    profs = entities["professors"]
    courses = entities["courses"]
    days = entities["days"]

    matched = []
    for c in chunks:
        t = c.get("type")
        if t == "course":
            # course chunk: professor + course_name (+ day) 매칭
            if profs and c.get("professor") not in profs:
                continue
            if courses and c.get("course_name") not in courses:
                continue
            if days and c.get("day") not in days:
                continue
            matched.append(c["_chunk_id"])
        elif t == "professor":
            # professor chunk: 이름 매칭
            if profs and c.get("name") in profs:
                matched.append(c["_chunk_id"])
        elif t == "dept_phone":
            # dept 매칭 (질문에 학과명 있는 경우)
            pass  # 별도 처리
    return matched


def _match_gold_by_chroma(question: str, top_k: int = 1) -> List[str]:
    """ChromaDB Top-K chunk_id 반환 (semantic 질문용).

    ChromaDB 미실행이거나 실패 시 [] 반환.
    반환 chunk_id 는 _stable_chunk_id 규칙과 동일하게 재구성.
    """
    try:
        from app.retrieval.chroma_search import search
        results = search(question, n_results=top_k)
        chunk_ids = []
        for r in results:
            md = r.get("metadata", {}) or {}
            cid = _stable_chunk_id(md)
            if cid:
                chunk_ids.append(cid)
        return chunk_ids
    except Exception as e:
        print(f"  [chroma error] {e}", file=sys.stderr)
        return []


def _match_dept_chunk(question: str, chunks: List[Dict], depts: List[str]) -> List[str]:
    """학과 chunk (dept_phone) 매칭."""
    matched = []
    for c in chunks:
        if c.get("type") != "dept_phone":
            continue
        dept = c.get("dept") or ""
        if dept and dept in question:
            matched.append(c["_chunk_id"])
    return matched


def build_gold(entry: Dict, chunks: List[Dict], by_id: Dict, professors: List[str],
               courses: List[str], depts: List[str], use_chroma: bool = False) -> Dict:
    """단일 엔트리에 gold label 부여."""
    q = entry["question"]
    cat = entry["category"]
    diff = entry.get("difficulty")

    # ambiguous · non-existent: answerable=False, gold=[]
    if cat == "ambiguous" or diff == "non-existent":
        return {
            "gold_chunk_ids": [],
            "required_entities": [],
            "answerable": False,
            "auto_confidence": "high",
            "auto_method": "rule/non-answerable",
        }

    entities = _extract_entities(q, professors, courses)
    required = entities["professors"] + entities["courses"] + entities["days"]

    # 학과 관련 질문
    dept_ids = _match_dept_chunk(q, chunks, depts)
    if dept_ids:
        return {
            "gold_chunk_ids": dept_ids[:3],  # 최대 3개
            "required_entities": required,
            "answerable": True,
            "auto_confidence": "high",
            "auto_method": "rule/dept",
        }

    # 엔티티 기반 매칭
    if entities["professors"] or entities["courses"]:
        matched = _match_gold_by_entity(chunks, entities)
        if matched:
            # 너무 많으면 상위 5개만 (여러 섹션 대응)
            return {
                "gold_chunk_ids": matched[:5],
                "required_entities": required,
                "answerable": True,
                "auto_confidence": "high" if len(matched) <= 3 else "medium",
                "auto_method": "rule/entity",
            }
        # 엔티티 있는데 매칭 실패 → chroma 로 폴백
    # semantic 또는 매칭 실패
    if use_chroma:
        chroma_ids = _match_gold_by_chroma(q, top_k=1)
        if chroma_ids:
            return {
                "gold_chunk_ids": chroma_ids,
                "required_entities": required,
                "answerable": True,
                "auto_confidence": "low",
                "auto_method": "chroma/top-1",
            }

    # 완전 실패
    return {
        "gold_chunk_ids": [],
        "required_entities": required,
        "answerable": None,  # 사람 검수 필요
        "auto_confidence": "none",
        "auto_method": "rule/unmatched",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(IN_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--no-chroma", action="store_true",
                    help="ChromaDB 폴백 사용 안 함 (오프라인 실행)")
    args = ap.parse_args()

    print("[loading] chunks + v1 dataset")
    chunks, by_id = _load_chunks()

    profs = sorted({c.get("professor") for c in chunks if c.get("professor")}, key=len, reverse=True)
    course_names = sorted({c.get("course_name") for c in chunks if c.get("course_name")}, key=len, reverse=True)
    depts = sorted({c.get("dept") for c in chunks if c.get("dept")}, key=len, reverse=True)
    profs = [p for p in profs if p]
    course_names = [c for c in course_names if c]
    depts = [d for d in depts if d]
    print(f"  chunks={len(chunks)}, profs={len(profs)}, courses={len(course_names)}, depts={len(depts)}")

    v1 = json.loads(Path(args.in_path).read_text(encoding="utf-8"))
    print(f"  dataset entries: {len(v1['dataset'])}")

    use_chroma = not args.no_chroma
    print(f"[gold labeling] chroma fallback: {use_chroma}")

    stats = defaultdict(int)
    for entry in v1["dataset"]:
        gold_info = build_gold(entry, chunks, by_id, profs, course_names, depts, use_chroma=use_chroma)
        entry["gold_chunk_ids"] = gold_info["gold_chunk_ids"]
        entry["required_entities"] = gold_info["required_entities"]
        entry["answerable"] = gold_info["answerable"]
        entry["auto_confidence"] = gold_info["auto_confidence"]
        entry["auto_method"] = gold_info["auto_method"]
        stats[gold_info["auto_confidence"]] += 1
        stats[gold_info["auto_method"]] += 1

    print("\n[summary]")
    for k, v in sorted(stats.items()):
        print(f"  {k:20} : {v}")

    # 검수 필요 케이스 (confidence=none/low) 강조
    review_needed = [e for e in v1["dataset"] if e.get("auto_confidence") in ("none", "low")]
    print(f"\n[review needed] {len(review_needed)} 개 (confidence: none/low)")

    v1["meta"]["gold_labeling"] = {
        "date": "2026-08-05",
        "method": "rule-based + chroma-fallback",
        "chunks_used": len(chunks),
        "confidence_counts": {k: v for k, v in stats.items() if k in ("high", "medium", "low", "none")},
        "review_needed": len(review_needed),
        "review_target": 30,
    }

    Path(args.out).write_text(json.dumps(v1, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
