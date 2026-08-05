"""층화 평가셋 200개 자동 생성.

EXPERIMENT.md §4.3.1 스펙 준수:
  의미·구어체 50 · 교수명·과목명 50 · 관계형 50 (15/15/15/5) · 단순 30 · 모호 20

기존 데이터셋:
  - data/yongin_univ_questions_1000_student_style.json (구어체)
  - data/yongin_univ_questions_1000_natural_v2.json (자연어)

두 셋을 병합 후 규칙 기반 카테고리 자동 분류 → 카테고리별 무작위 샘플링.
결과: data/eval/stratified_200_v1.json (v1 = 미검수, 사용자 검수 후 v2 로 확정)

사용:
  python scripts/eval/build_stratified_dataset.py --seed 42
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR  = DATA_DIR / "eval"
CHUNK_FILE = DATA_DIR / "processed" / "chunks" / "academic" / "academic_001.json"

# 카테고리별 목표 수량
TARGETS = {
    "semantic":   50,
    "entity":     50,
    "relational": 50,
    "simple":     30,
    "ambiguous":  20,
}
RELATIONAL_SUB_TARGETS = {
    "single-hop":  15,
    "constraint":  15,
    "multi-hop":   15,
    "non-existent": 5,
}

# 규칙 사전
DAY_TOKENS_RE = re.compile(r"(월|화|수|목|금)요일|(?<![가-힣])([월화수목금]{2,5})(?=[\s,\.\?!]|$)")
SLANG_TERMS = {
    "확통", "미적", "선대", "이산", "자구", "알고", "컴구", "운체",
    "패논패", "패논팹", "교필", "전필", "교선", "전선",
    "물치과", "컴공", "산공",
}
COLLOQUIAL_MARKERS = {
    "쫌", "좀", "그럼", "야", "함", "심?", "해요?", "언제야",
    "뭐야", "어디야", "있냐", "있어", "되나", "돼요",
}
AMBIGUOUS_WORDS = {
    "엘리베이터", "화장실", "카페", "편의점", "식당", "매점",
    "주차", "동아리", "기숙사", "체육관",  # 학사와 무관
}
SIMPLE_FIELD_MARKERS = {
    "전화번호", "번호", "학점", "몇 학점", "몇학점",
    "언제까지", "어디야", "어디에", "언제 하", "위치",
    "이수구분", "구분", "카데",
}
RELATIONAL_KEYWORDS = {
    "담당", "가르치", "강의", "수업", "가르쳐", "하심", "하나", "맡",
}


def _load_keywords() -> Tuple[List[str], List[str]]:
    """chunk 파일에서 교수명·과목명 로드."""
    data = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    profs = sorted({r["professor"] for r in data if r.get("professor")}, key=len, reverse=True)
    courses = sorted({r["course_name"] for r in data if r.get("course_name")}, key=len, reverse=True)
    return profs, courses


def _load_prof_course_pairs() -> Dict[str, set]:
    """교수 -> 담당 과목 set 매핑."""
    data = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    m: Dict[str, set] = defaultdict(set)
    for r in data:
        p, c = r.get("professor"), r.get("course_name")
        if p and c:
            m[p].add(c)
    return m


# ── 합성용 프리셋 (원본 데이터셋에 없는 카테고리) ────────────────────

# 모호·근거 없음·오류 조합 (학사 무관 · 챗봇이 refusal 해야 하는 것)
AMBIGUOUS_QUESTIONS = [
    "엘리베이터 고장났어",
    "화장실 어디야",
    "카페 위치 알려줘",
    "편의점 몇 시까지 열어",
    "식당 오늘 메뉴 뭐야",
    "매점 위치가 어디지",
    "주차장 이용료 얼마야",
    "동아리방 어디에 있어",
    "기숙사 신청 어디서 해",
    "체육관 이용시간 알려줘",
    "학교 근처 맛집 추천",
    "셔틀버스 시간표",
    "도서관 몇 시까지 여는지",
    "학생회관 위치",
    "강당 사용법",
    "프린터 어디 있어",
    "커피 마실 곳 없나",
    "벤치 있는 데 어디야",
    "흡연구역 위치",
    "자전거 거치대 어디",
]


def _synthesize_non_existent(prof_course_map: Dict[str, set], count: int, seed: int) -> List[Dict]:
    """존재하는 교수·과목이지만 실제 담당 안 하는 조합.

    반환: [{'question': ..., 'notes': '실제로는 X 교수가 담당'}]
    """
    rng = random.Random(seed + 1000)
    profs_with_courses = [p for p, cs in prof_course_map.items() if cs]
    all_courses = set()
    for cs in prof_course_map.values():
        all_courses.update(cs)
    all_courses = sorted(all_courses)

    result = []
    tried = 0
    while len(result) < count and tried < 200:
        tried += 1
        prof = rng.choice(profs_with_courses)
        course = rng.choice(all_courses)
        # 실제로 담당 안 하는 조합만 선택
        if course in prof_course_map[prof]:
            continue
        # 실제 담당 교수 찾기 (notes 용)
        actual_profs = [p for p, cs in prof_course_map.items() if course in cs]
        actual_prof = actual_profs[0] if actual_profs else "?"
        result.append({
            "question": f"{prof} 교수님 {course} 언제야?",
            "notes":    f"실제로는 {actual_prof} 교수 담당 (교수·과목 조합 불일치)",
            "wrong_prof": prof,
            "target_course": course,
            "actual_prof": actual_prof,
        })
    return result


def _find_days(question: str) -> List[str]:
    found = []
    for m in DAY_TOKENS_RE.finditer(question):
        d = (m.group(1) or "")
        if d:
            if d not in found:
                found.append(d)
        else:
            concat = m.group(2) or ""
            for c in concat:
                if c in "월화수목금" and c not in found:
                    found.append(c)
    return found


def _entity_hits(question: str, terms: List[str]) -> List[str]:
    return [t for t in terms if t and t in question]


def _has_any(question: str, words: set) -> bool:
    return any(w in question for w in words)


def _categorize(question: str, profs: List[str], courses: List[str]) -> Tuple[str, Optional[str]]:
    """반환: (category, difficulty_sub)"""
    prof_hits = _entity_hits(question, profs)
    course_hits = _entity_hits(question, courses)
    day_hits = _find_days(question)
    has_slang = _has_any(question, SLANG_TERMS)
    has_colloq = _has_any(question, COLLOQUIAL_MARKERS)
    has_ambig = _has_any(question, AMBIGUOUS_WORDS)
    has_relational_kw = _has_any(question, RELATIONAL_KEYWORDS)
    has_simple_field = _has_any(question, SIMPLE_FIELD_MARKERS)

    # 1) Ambiguous 최상위 우선
    if has_ambig:
        return "ambiguous", None

    # 2) Relational: 엔티티 + 요일 = constraint
    if (prof_hits or course_hits) and day_hits:
        return "relational", "constraint"

    # 3) Relational: 교수+과목 조합 (multi-hop 의심)
    if prof_hits and course_hits:
        return "relational", "multi-hop"

    # 4) Relational: 교수 + 관계 키워드 = single-hop
    if prof_hits and has_relational_kw:
        return "relational", "single-hop"
    if course_hits and has_relational_kw:
        return "relational", "single-hop"

    # 5) Simple: 명시적 필드 조회
    if (prof_hits or course_hits) and has_simple_field:
        return "simple", None

    # 6) Entity: 고유명사 단독
    if prof_hits or course_hits or has_slang:
        # 슬랭도 entity 로 분류 (표준 용어 변환 대상)
        return "entity", None

    # 7) Semantic: 개념·구어체·오픈 질문
    if has_colloq:
        return "semantic", None

    # 8) 기본값
    return "semantic", None


def build(seed: int = 42) -> Dict:
    random.seed(seed)
    profs, courses = _load_keywords()
    print(f"[keywords] professors={len(profs)}, courses={len(courses)}")

    # 데이터 로드
    d1 = json.loads((DATA_DIR / "yongin_univ_questions_1000_student_style.json").read_text(encoding="utf-8"))
    d2 = json.loads((DATA_DIR / "yongin_univ_questions_1000_natural_v2.json").read_text(encoding="utf-8"))
    for r in d1:
        r["_source"] = "student_style"
    for r in d2:
        r["_source"] = "natural_v2"
    pool = d1 + d2
    random.shuffle(pool)
    print(f"[pool] total={len(pool)} (student={len(d1)} + natural={len(d2)})")

    # 자동 분류
    buckets = defaultdict(list)               # category -> list
    sub_buckets = defaultdict(list)           # (relational, sub) -> list
    for item in pool:
        cat, sub = _categorize(item["question"], profs, courses)
        item["_category"] = cat
        item["_difficulty"] = sub
        buckets[cat].append(item)
        if cat == "relational" and sub:
            sub_buckets[sub].append(item)

    # 통계
    print("\n[auto-classification counts]")
    for cat, items in buckets.items():
        print(f"  {cat:12} : {len(items)}")
    print("  ---- relational sub-categories ----")
    for sub, items in sub_buckets.items():
        print(f"  relational/{sub:14} : {len(items)}")

    # 층화 샘플링
    sampled = []
    shortage = {}

    def _sample(candidates: List[Dict], k: int, tag: str) -> List[Dict]:
        if len(candidates) < k:
            shortage[tag] = {"needed": k, "available": len(candidates)}
            return list(candidates)
        return random.sample(candidates, k)

    # semantic, entity, simple 는 그대로 층화
    for cat in ["semantic", "entity", "simple"]:
        picks = _sample(buckets[cat], TARGETS[cat], cat)
        sampled.extend(picks)

    # ambiguous 는 원본 데이터셋에 없어서 합성 프리셋 사용
    ambig_picks = []
    for q in AMBIGUOUS_QUESTIONS[:TARGETS["ambiguous"]]:
        ambig_picks.append({
            "question": q,
            "_source": "synthesized",
            "id": None,
            "_category": "ambiguous",
            "_difficulty": None,
            "_notes": "학사 무관 · refusal 대상",
        })
    sampled.extend(ambig_picks)

    # relational 은 하위 구조로 층화 (non-existent 는 합성)
    used_ids = set()
    for sub, need in RELATIONAL_SUB_TARGETS.items():
        if sub == "non-existent":
            prof_course_map = _load_prof_course_pairs()
            synth = _synthesize_non_existent(prof_course_map, need, seed=42)
            for s in synth:
                sampled.append({
                    "question": s["question"],
                    "_source": "synthesized",
                    "id": None,
                    "_category": "relational",
                    "_difficulty": "non-existent",
                    "_notes": s["notes"],
                })
            continue
        available = [it for it in sub_buckets[sub] if id(it) not in used_ids]
        picks = _sample(available, need, f"relational/{sub}")
        for p in picks:
            used_ids.add(id(p))
        sampled.extend(picks)

    print(f"\n[sampled] total={len(sampled)}")
    if shortage:
        print("[shortage warnings]")
        for tag, info in shortage.items():
            print(f"  {tag}: needed {info['needed']}, available {info['available']}")

    # 최종 스키마화 (Gold 는 다음 스텝에서 라벨링)
    dataset = []
    for i, item in enumerate(sampled, 1):
        entry = {
            "question_id": f"Q{i:03d}",
            "category":    item["_category"],
            "difficulty":  item["_difficulty"],
            "question":    item["question"],
            "source":      item["_source"],
            "original_id": item.get("id"),
            "gold_answer":     None,
            "gold_doc_ids":    [],
            "gold_chunk_ids":  [],
            "required_entities": [],
            "answerable":  False if item["_category"] == "ambiguous" else (
                False if item["_difficulty"] == "non-existent" else None
            ),
            "notes":       item.get("_notes"),
            "reviewed":    False,
        }
        dataset.append(entry)

    # 카테고리별 최종 확인
    final_counts = Counter(e["category"] for e in dataset)
    print("\n[final counts]")
    for cat, cnt in sorted(final_counts.items()):
        print(f"  {cat:12} : {cnt}")
    sub_counts = Counter(e["difficulty"] for e in dataset if e["category"] == "relational")
    print("  ---- relational sub ----")
    for sub, cnt in sorted(sub_counts.items(), key=lambda x: str(x[0])):
        print(f"  {sub!s:15} : {cnt}")

    return {
        "meta": {
            "seed": seed,
            "total": len(dataset),
            "targets": TARGETS,
            "relational_sub_targets": RELATIONAL_SUB_TARGETS,
            "counts": dict(final_counts),
            "relational_sub_counts": {str(k): v for k, v in sub_counts.items()},
            "shortage": shortage,
            "spec_ref": "docs/EXPERIMENT.md §4.3.1",
        },
        "dataset": dataset,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT_DIR / "stratified_200_v1.json"),
                    help="v1 = 미검수 초안. 사용자 검수 후 v2 로 저장")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = build(seed=args.seed)

    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
