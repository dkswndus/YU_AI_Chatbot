"""Gold labeling 사용자 검수용 표시 스크립트.

30개 층화 표본을 화면에 표시. 사용자는 각 항목이 정답인지 확인.

사용:
  python scripts/eval/gold_review_sheet.py > review_sheet.txt
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IN_PATH = PROJECT_ROOT / "data" / "eval" / "stratified_200_gold_v1.json"
CHUNK_FILE = PROJECT_ROOT / "data" / "processed" / "chunks" / "academic" / "academic_001.json"
MD_CHUNK_FILE = PROJECT_ROOT / "data" / "processed" / "chunks" / "academic_md" / "academic_001_md.json"


def _stable_chunk_id(rec, fallback_idx=0):
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


def _load_chunks():
    by_id = {}
    for path in (CHUNK_FILE, MD_CHUNK_FILE):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for idx, r in enumerate(data):
            cid = _stable_chunk_id(r, fallback_idx=idx)
            by_id[cid] = r
    return by_id


def _summarize_chunk(rec):
    """chunk 를 한 줄로 요약."""
    t = rec.get("type", "?")
    if t == "course":
        return f"[course] {rec.get('professor','?')} 교수 · {rec.get('course_name','?')} · {rec.get('day','?')}요일 {rec.get('time_range','?')} · {rec.get('room','?')}"
    if t == "professor":
        return f"[prof] {rec.get('name','?')} · {rec.get('dept','?')} · 전화 {rec.get('phone','?')} · 연구실 {rec.get('room','?')}"
    if t == "dept_phone":
        return f"[dept] {rec.get('dept','?')} · {rec.get('college','?')} · {rec.get('phone','?')}"
    if t == "info":
        text = rec.get("text", "")[:80].replace("\n", " ")
        return f"[info] {rec.get('section','?')} · {text}"
    return f"[{t}] {str(rec)[:80]}"


def pick_review_samples(dataset, n_per_group):
    """카테고리별 표본 선택."""
    by_key = defaultdict(list)
    for e in dataset:
        if e["category"] == "relational":
            key = f"relational/{e['difficulty']}"
        else:
            key = e["category"]
        by_key[key].append(e)

    picks = []
    for key, n in n_per_group.items():
        picks.extend(by_key[key][:n])
    return picks


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d = json.load(open(IN_PATH, encoding="utf-8"))
    chunks_by_id = _load_chunks()
    dataset = d["dataset"]

    # 카테고리별 표본 수량 (총 30)
    n_per_group = {
        "semantic": 6,
        "entity": 6,
        "simple": 5,
        "ambiguous": 3,
        "relational/single-hop": 3,
        "relational/constraint": 3,
        "relational/multi-hop": 2,
        "relational/non-existent": 2,
    }

    samples = pick_review_samples(dataset, n_per_group)
    print(f"=== Gold Labeling 검수 시트 (N={len(samples)}) ===\n")
    print("각 질문에 대해:")
    print("  1) 답변 가능한 질문인가? (answerable 필드 맞나?)")
    print("  2) gold_chunk_ids 가 실제 정답 근거를 포함하는가?")
    print("  3) 문제 있으면 아래에 표기\n")
    print("=" * 80)

    for i, e in enumerate(samples, 1):
        print(f"\n--- {i}. [{e['question_id']}] {e['category']}/{e.get('difficulty') or '-'} ---")
        print(f"Q: {e['question']}")
        if e.get("notes"):
            print(f"NOTE: {e['notes']}")
        print(f"answerable: {e['answerable']}")
        print(f"required_entities: {e['required_entities']}")
        print(f"auto_confidence: {e['auto_confidence']}  ({e['auto_method']})")
        print(f"gold_chunk_ids ({len(e['gold_chunk_ids'])}):")
        for cid in e["gold_chunk_ids"][:5]:
            rec = chunks_by_id.get(cid)
            if rec:
                print(f"    {cid}")
                print(f"        {_summarize_chunk(rec)}")
            else:
                print(f"    {cid}  (chunk not found)")

    print("\n" + "=" * 80)
    print("[사용자 회신 형식]")
    print("문제 있는 항목만 아래처럼 알려주세요:")
    print("  Q001 : answerable=True → False (실제로 답 없음)")
    print("  Q053 : gold 부정확 (X 강의 chunk 여야 함)")
    print("  Q181 : 이 카테고리는 single-hop 이어야 함")
    print("\n전부 OK 면 '통과' 만.")


if __name__ == "__main__":
    main()
