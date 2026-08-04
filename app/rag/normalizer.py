"""사전 기반 정규화: 학생 구어체·줄임말을 표준 학사 용어로 치환.

LLM 호출 없이 O(1) 조회 + 단일 패스 정규식으로 처리한다.
사전은 data/glossary.json 에서 로드하며, 최초 호출 시 1회 캐싱한다.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

_GLOSSARY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "glossary.json"
)

_glossary: Dict[str, str] = {}
_pattern: Optional[re.Pattern] = None


def _load() -> None:
    global _glossary, _pattern
    if _pattern is not None:
        return
    _glossary = json.loads(_GLOSSARY_PATH.read_text(encoding="utf-8"))
    if not _glossary:
        _pattern = re.compile(r"(?!)")  # 절대 매칭되지 않는 패턴
        return
    # 긴 키부터 매칭 (부분 문자열 중복 방지)
    keys = sorted(_glossary.keys(), key=len, reverse=True)
    # 앞뒤가 한글/영문/숫자면 매칭 제외 → "확통계학과"에서 "확통" 오매칭 방지
    _pattern = re.compile(
        r"(?<![가-힣A-Za-z0-9])(" + "|".join(re.escape(k) for k in keys) + r")(?![가-힣A-Za-z0-9])"
    )


def normalize(text: str) -> str:
    """구어체·줄임말 치환. 매칭 없으면 원문 그대로 반환."""
    if not text:
        return text
    _load()
    return _pattern.sub(lambda m: _glossary[m.group(1)], text)


def normalize_with_stats(text: str) -> Dict:
    """정규화 결과 + 계측용 통계.

    반환: {
      "normalized":     str,
      "hits":           int,       # 치환된 총 횟수
      "matched_terms":  List[str], # 실제 매칭된 slang 목록
    }
    """
    if not text:
        return {"normalized": text, "hits": 0, "matched_terms": []}
    _load()
    matched: List[str] = []

    def _sub(m: re.Match) -> str:
        term = m.group(1)
        matched.append(term)
        return _glossary[term]

    normalized = _pattern.sub(_sub, text)
    return {
        "normalized":    normalized,
        "hits":          len(matched),
        "matched_terms": matched,
    }


def glossary_size() -> int:
    _load()
    return len(_glossary)
