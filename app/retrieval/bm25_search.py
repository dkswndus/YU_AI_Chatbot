"""BM25 키워드 검색.

역할:
  Vector 검색(ChromaDB) · 관계 검색(Neo4j)의 사각지대인
  고유명사·희귀어의 exact match 성능을 담당한다.
  특히 교수명 · 과목명처럼 embedding이 잘 잡지 못하는 토큰에 강하다.

인덱싱:
  - data/processed/chunks/academic/academic_001.json (1054 chunks)
  - data/processed/chunks/academic_md/academic_001_md.json (315 chunks)
  두 소스를 병합하여 단일 BM25Okapi 인덱스 구축 (lazy singleton).

토크나이저:
  Konlpy(Okt) 등 형태소 분석기는 Java 의존성이 있어 배포 부담이 크다.
  대신 whitespace + 문장부호 분리 + 한국어 조사 제거 규칙으로
  순수 Python 토크나이저를 구현한다. 후속 튜닝 여지 있음.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "chunks"
_CHUNK_FILES = [
    _DATA_ROOT / "academic"    / "academic_001.json",
    _DATA_ROOT / "academic_md" / "academic_001_md.json",
]

# 접미사 절단용 조사 (안전한 것만).
# 단일 문자 조사 중 "과·이·가·도·만·와·에·로" 는 명사의 일부일 가능성이 높아 제외
# (예: 물리치료학'과', '이'순신, '가'족, '도'서관 등).
_STRIP_PARTICLES = ("에서", "으로", "이나", "까지", "부터", "의", "은", "는", "을", "를")

# stopword: 단독 토큰으로 등장하면 제거. suffix strip 대상은 포함되지 않아도 됨.
_STOPWORDS = set(_STRIP_PARTICLES) | {
    "및", "또는", "그리고", "이러한", "관련",
    "께", "와", "도", "만",  # 단독일 때만 제거
}

_PUNCT_RE = re.compile(r"[,\.\!\?\(\)\[\]\{\}\"\'\:\;\-\+\|/·]")

_documents: List[Dict] = []
_tokenized_corpus: List[List[str]] = []
_bm25: Optional[BM25Okapi] = None


def _tokenize(text: str) -> List[str]:
    """공백 · 문장부호 분리 후 안전한 조사만 절단. 순수 Python."""
    if not text:
        return []
    text = _PUNCT_RE.sub(" ", text)
    out: List[str] = []
    for tok in text.split():
        tok = tok.strip()
        if not tok:
            continue
        # 안전한 조사만 접미사 절단, 스템은 최소 2자 유지
        for p in _STRIP_PARTICLES:
            if len(tok) - len(p) >= 2 and tok.endswith(p):
                tok = tok[: -len(p)]
                break
        if tok and tok not in _STOPWORDS:
            out.append(tok.lower())
    return out


def _build_natural_content(record: Dict) -> str:
    """구조화된 레코드(course/professor/dept)를 자연어 문장으로 조립.

    academic_001.json 의 course chunk 는 text 필드가 비어있고 metadata 로만 구성됨.
    → BM25 인덱싱 시 매칭 토큰 부족 + Cross-Encoder rerank 시 판단 불가.
    이 함수가 metadata → 자연어 변환을 담당하여 두 문제를 동시에 해결한다.
    """
    t = record.get("type", "")

    if t == "course":
        parts: List[str] = []
        if record.get("professor"):
            parts.append(f"{record['professor']} 교수")
        if record.get("course_name"):
            parts.append(record["course_name"])
        if record.get("course_type"):
            parts.append(f"({record['course_type']})")
        if record.get("credits"):
            parts.append(f"{record['credits']}학점")
        if record.get("day") and record.get("time_range"):
            parts.append(f"{record['day']}요일 {record['time_range']}")
        elif record.get("day"):
            parts.append(f"{record['day']}요일")
        if record.get("room"):
            parts.append(record["room"])
        if record.get("course_number"):
            parts.append(f"과목코드 {record['course_number']}")
        return " ".join(parts)

    if t == "professor":
        parts = []
        if record.get("name"):
            parts.append(f"{record['name']} 교수")
        if record.get("dept"):
            parts.append(record["dept"])
        if record.get("day"):
            parts.append(f"연구일 {record['day']}요일")
        if record.get("phone"):
            parts.append(f"전화 {record['phone']}")
        if record.get("room"):
            parts.append(f"연구실 {record['room']}")
        return " ".join(parts)

    if t == "dept_phone":
        parts = []
        if record.get("college"):
            parts.append(record["college"])
        if record.get("dept"):
            parts.append(record["dept"])
        if record.get("phone"):
            parts.append(f"전화 {record['phone']}")
        return " ".join(parts)

    return ""  # info 등: text 필드를 직접 사용


def _build_indexable_text(record: Dict) -> str:
    """BM25 인덱싱용 텍스트. 자연어 조립본 + 원본 text 결합.

    자연어 조립본이 있으면 그것 + text (info chunk 대응).
    없으면 기존 로직 (title, section 등을 앞에 부착 후 text).
    """
    natural = _build_natural_content(record)
    text = record.get("text", "")

    if natural:
        # course/professor/dept: 자연어 우선, text 는 대개 비어있음
        return f"{natural} {text}".strip() if text else natural

    # info 등: title/section 을 앞에 부착
    parts: List[str] = []
    for key in ("title", "section"):
        v = record.get(key)
        if v:
            parts.append(str(v))
    if text:
        parts.append(text)
    return " ".join(parts)


def _load_documents() -> None:
    """text 필드에 저장할 내용 결정:
        - course/professor/dept: 자연어 조립본 (Rerank · LLM 이 소비 가능)
        - info 등: 원본 text
    """
    global _documents
    if _documents:
        return
    for path in _CHUNK_FILES:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for rec in data:
            natural = _build_natural_content(rec)
            content = natural if natural else rec.get("text", "")
            _documents.append({
                "text":     content,
                "metadata": {**rec, "source_file": path.name},
            })


def _build_index() -> None:
    global _bm25, _tokenized_corpus
    if _bm25 is not None:
        return
    _load_documents()
    _tokenized_corpus = [
        _tokenize(_build_indexable_text(d["metadata"])) for d in _documents
    ]
    if _tokenized_corpus:
        _bm25 = BM25Okapi(_tokenized_corpus)


def search(query: str, n_results: int = 10) -> List[Dict[str, Any]]:
    """BM25 검색.

    반환: [{"text": str, "metadata": dict, "score": float}, ...]
    score 높을수록 관련. score == 0 인 결과는 제외.
    """
    _build_index()
    if _bm25 is None:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = _bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    return [
        {
            "text":     _documents[i]["text"],
            "metadata": _documents[i]["metadata"],
            "score":    float(scores[i]),
        }
        for i in ranked
        if scores[i] > 0
    ]


def index_size() -> int:
    _build_index()
    return len(_documents)
