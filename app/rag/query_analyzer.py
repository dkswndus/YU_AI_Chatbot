"""Query Analysis: 구조화된 질의 분석.

전략:
  1) Ollama format=json 모드로 LLM 호출
  2) Pydantic 스키마로 검증
  3) 파싱/검증 실패 시 1회 재시도
  4) 최종 실패 시 rule-based fallback (BM25+ChromaDB 라우팅 기본값)

confidence 는 LLM 자기평가 값으로, calibration 이 약함 → 보조 신호로만 사용.
"""
import json
import os
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL",    "exaone3.5:7.8b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

IntentType = Literal[
    "course_schedule",
    "professor_info",
    "department_info",
    "academic_calendar",
    "policy",
    "other",
]

RetrievalType = Literal["graph", "keyword", "semantic"]

# 기존 파이프라인 라우팅 로직(한국어 intent)과의 호환용 매핑
INTENT_EN_TO_KO: Dict[str, str] = {
    "course_schedule":   "강의_시간표",
    "professor_info":    "교수_정보",
    "department_info":   "학과_연락처",
    "academic_calendar": "학사_일정",
    "policy":            "수강_규정",
    "other":             "기타",
}


class QueryAnalysis(BaseModel):
    intent: IntentType = "other"
    entities: Dict[str, str] = Field(default_factory=dict)
    requested_fields: List[str] = Field(default_factory=list)
    retrieval_types: List[RetrievalType] = Field(
        default_factory=lambda: ["keyword", "semantic"]
    )
    needs_rewrite: bool = True
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_korean_intent(self) -> str:
        return INTENT_EN_TO_KO.get(self.intent, "기타")


ANALYZE_PROMPT = """당신은 용인대학교 학사 챗봇의 질의 분석기입니다.
학생의 질문을 아래 JSON 스키마에 맞춰 분석하세요.

레이블 정의:
- intent: 질문의 의도 (다음 중 하나)
  - course_schedule   : 수업 시간, 강의실, 강의 목록
  - professor_info    : 교수 연구실, 전화번호, 연구일
  - department_info   : 학과/학부 전화번호
  - academic_calendar : 수강신청, 성적열람, P/NP 기간
  - policy            : 재수강, 이수구분, 학점, 졸업요건
  - other             : 위에 해당 없음
- entities: 질문에서 추출한 명사, 예: {"professor":"김중헌","course":"확률과통계"}
- requested_fields: 사용자가 알고싶어하는 필드, 예: ["day","time","phone"]
- retrieval_types: 필요한 검색 방식 (여러 개 가능)
  - graph    : 관계 검색 (교수-과목-요일)
  - keyword  : 정확 매칭 (고유명사, 코드)
  - semantic : 의미 검색 (개념, 규정)
- needs_rewrite: 검색 전 재작성 필요 여부 (모호/생략/복수의도이면 true)
- confidence: 분석 신뢰도 0.0~1.0 (자기평가)

반드시 JSON 하나만 출력하세요. 설명, 마크다운 금지.

예시:
질문: "김중헌 교수님 확률과통계 언제야?"
{"intent":"course_schedule","entities":{"professor":"김중헌","course":"확률과통계"},"requested_fields":["day","time"],"retrieval_types":["graph","keyword"],"needs_rewrite":false,"confidence":0.95}

질문: "휴학하려면 어떻게 해?"
{"intent":"policy","entities":{},"requested_fields":["procedure"],"retrieval_types":["semantic"],"needs_rewrite":true,"confidence":0.7}

질문: "그거 언제까지야?"
{"intent":"other","entities":{},"requested_fields":[],"retrieval_types":["keyword","semantic"],"needs_rewrite":true,"confidence":0.4}
"""

_json_llm: Optional[Any] = None


def _get_json_llm():
    """지연 초기화. langchain_ollama 는 실제 호출 시점에만 import."""
    global _json_llm
    if _json_llm is None:
        from langchain_ollama import ChatOllama
        _json_llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            format="json",
            temperature=0.0,
        )
    return _json_llm


def _parse_json_lenient(raw: str) -> Optional[dict]:
    """마크다운 fence·앞뒤 문장 제거 후 JSON 파싱."""
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _rule_based_fallback(
    question: str,
    known_professors: set,
    known_courses: set,
) -> QueryAnalysis:
    """LLM 실패 시 규칙 기반 최소 분석. 안전한 기본 검색 조합(BM25+ChromaDB)."""
    entities: Dict[str, str] = {}
    for p in known_professors:
        if p in question:
            entities["professor"] = p
            break
    for c in known_courses:
        if c in question:
            entities["course"] = c
            break

    retrieval_types: List[RetrievalType] = ["keyword", "semantic"]
    if entities:
        retrieval_types = ["graph", "keyword", "semantic"]

    return QueryAnalysis(
        intent="other",
        entities=entities,
        requested_fields=[],
        retrieval_types=retrieval_types,
        needs_rewrite=True,
        confidence=0.3,
    )


def analyze(
    question: str,
    known_professors: Optional[set] = None,
    known_courses: Optional[set] = None,
    max_retries: int = 1,
) -> QueryAnalysis:
    return analyze_with_stats(
        question, known_professors, known_courses, max_retries
    )["analysis"]


def analyze_with_stats(
    question: str,
    known_professors: Optional[set] = None,
    known_courses: Optional[set] = None,
    max_retries: int = 1,
) -> Dict:
    """분석 결과 + Reliability 계측 신호.

    반환: {
      "analysis":            QueryAnalysis,
      "used_fallback":       bool,   # rule-based로 떨어졌는지
      "json_parse_success":  bool,   # LLM이 valid JSON을 뱉었는지
      "retries":             int,    # 재시도 횟수
    }
    """
    known_professors = known_professors or set()
    known_courses = known_courses or set()

    from langchain_core.messages import HumanMessage, SystemMessage

    msgs = [
        SystemMessage(content=ANALYZE_PROMPT),
        HumanMessage(content=f"질문: {question}"),
    ]

    for attempt in range(max_retries + 1):
        try:
            llm = _get_json_llm()
            raw = llm.invoke(msgs).content.strip()
            data = _parse_json_lenient(raw)
            if data is not None:
                analysis = QueryAnalysis.model_validate(data)
                return {
                    "analysis":           analysis,
                    "used_fallback":      False,
                    "json_parse_success": True,
                    "retries":            attempt,
                }
        except (ValidationError, Exception):
            if attempt >= max_retries:
                break

    return {
        "analysis":           _rule_based_fallback(question, known_professors, known_courses),
        "used_fallback":      True,
        "json_parse_success": False,
        "retries":            max_retries,
    }
