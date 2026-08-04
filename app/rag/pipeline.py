"""하이브리드 RAG 파이프라인 (LangGraph)

[그래프 흐름 — 선형]
START
  → normalize        사전 기반 정규화 (LLM X)
  → analyze          Query Analysis (intent · entities · retrieval_types · needs_rewrite)
  → rewrite          조건부 Query Rewriting (needs_rewrite=True 일 때만 LLM)
  → retrieval        Query Analysis 의 retrieval_types 에 따라 검색기 조건부 호출
                      ├─ keyword  → BM25
                      ├─ semantic → ChromaDB
                      └─ graph    → Neo4j
  → fusion           RRF (Reciprocal Rank Fusion) 로 순위 기반 통합
  → rerank           Cross-Encoder 로 최종 Top-K 선정
  → format           Evidence → LLM context 문자열
  → answer           LLM 최종 답변
  → END

핵심 설계:
  · 저비용·확정성 우선 (Dict → Rule → LLM)
  · 검색기별 결과는 Evidence 공통 스키마로 통일 (app/rag/evidence.py)
  · classify / HyDE 제거 (Query Analysis 로 대체, LLM 호출 감소)
"""
import json
import os
import time
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

# 모델 · 서비스 엔드포인트 (환경변수로 override 가능)
OLLAMA_MODEL     = os.environ.get("OLLAMA_MODEL",     "exaone3.5:7.8b")
OLLAMA_BASE_URL  = os.environ.get("OLLAMA_BASE_URL",  "http://localhost:11434")

from langgraph.graph import StateGraph, END, START

from app.retrieval.chroma_search import search as chroma_search
from app.retrieval.bm25_search  import search as bm25_search
from app.rag.normalizer         import normalize_with_stats
from app.rag.query_analyzer     import analyze_with_stats, QueryAnalysis
from app.rag.evidence           import (
    Evidence, from_chroma, from_bm25, from_neo4j
)
from app.rag.fusion             import rrf_fuse

_llm = None  # lazy init


def _get_llm():
    """지연 초기화. langchain_ollama 는 실제 호출 시점에만 로드."""
    global _llm
    if _llm is None:
        from langchain_ollama import ChatOllama
        _llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    return _llm


def _chat(system: str, user: str) -> str:
    from langchain_core.messages import SystemMessage, HumanMessage
    response = _get_llm().invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content.strip()


# ── 상수 ─────────────────────────────────────────────────────────────
NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "yu_chatbot_2026"

CHUNK_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "chunks" / "academic" / "academic_001.json"
)

# 하위 호환: 외부 코드/테스트에서 참조 가능
INTENT_NEO4J  = {"강의_시간표", "교수_정보", "학과_연락처"}
INTENT_CHROMA = {"학사_일정", "수강_규정"}
VALID_INTENTS = INTENT_NEO4J | INTENT_CHROMA | {"기타"}

FUSION_TOP_N   = 15
RERANK_TOP_K   = 5
RETRIEVAL_TOPK = 10

# ── 프롬프트 ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 용인대학교 학사 안내 챗봇입니다.
학생들의 질문에 친절하고 정확하게 답변해주세요.
제공된 정보(컨텍스트)를 기반으로만 답변하고, 모르는 정보는 "확인이 필요합니다"라고 하세요.
답변은 간결하게 핵심만 전달해주세요."""

REWRITE_PROMPT = """학생의 구어체 질문을 학사 정보 검색에 최적화된 쿼리로 재작성하세요.
규칙:
- 핵심 키워드(과목명, 교수명, 학과명, 시간, 학점 등)를 명확하게 포함
- 구어체/줄임말/비문을 표준어로 변환
- 대명사/생략을 명시화 (예: "그거", "그 교수님")
- 한 줄로만 출력, 설명 없이 재작성된 쿼리만 출력

예시:
학생: 확률과통계 시간 언제야 → 확률과통계 강의 요일 및 시간
학생: 김중헌 교수님 뭐 가르쳐 → 김중헌 교수 담당 강의 목록
학생: 그거 언제까지야 → 문맥에 따라 명시화된 대상의 마감 일정"""

# 하위 호환용 (classify_node 에서 참조)
INTENT_PROMPT = """학생의 질문 의도를 아래 6가지 중 하나로만 분류하세요.
반드시 아래 레이블 중 하나만 출력하세요. 설명 없이 레이블만.

레이블:
- 강의_시간표  : 수업 시간, 강의실, 강의 목록 조회
- 교수_정보    : 교수 연구실, 전화번호, 연구일 조회
- 학과_연락처  : 학과/학부 전화번호 조회
- 학사_일정    : 수강신청, 성적열람, 패논패(P/NP) 기간 등
- 수강_규정    : 재수강, 이수구분, 학점, 졸업요건 등
- 기타         : 위 분류에 해당 없음"""

# ── 싱글톤 ────────────────────────────────────────────────────────────
_cross_encoder = None
_neo4j_driver  = None
_professors: set = set()
_courses: set    = set()


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    return _cross_encoder


def _get_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        try:
            from neo4j import GraphDatabase
            _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
            _neo4j_driver.verify_connectivity()
        except Exception:
            _neo4j_driver = None
    return _neo4j_driver


def _init_keywords():
    global _professors, _courses
    if _professors:
        return
    data = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    _professors = {r["professor"] for r in data if r.get("professor")}
    _courses    = {r["course_name"] for r in data if r.get("course_name")}


# ── LangGraph State ───────────────────────────────────────────────────
class RAGState(TypedDict, total=False):
    question:            str
    original_question:   str
    history:             List[Dict]
    rewritten:           str
    intent:              str
    query_analysis:      Dict
    evidences_by_source: Dict[str, List[Evidence]]  # {bm25, chroma, neo4j}
    fused_evidences:     List[Evidence]              # RRF 이후
    top_evidences:       List[Evidence]              # rerank 이후
    context:             str
    answer:              str
    _timings:            Dict[str, float]
    _metrics:            Dict


# ── metrics helpers ───────────────────────────────────────────────────
def _metrics(state: RAGState) -> Dict:
    m = state.setdefault("_metrics", {})
    m.setdefault("usage", {
        "glossary_hits":            0,
        "glossary_matched_terms":   [],
        "retrieval_types_selected": [],
        "retrieval_types_used":     [],
        "candidates_before_rerank": 0,
        "top_k_final":              0,
        "llm_calls":                0,
        "rewrite_called":           False,
    })
    m.setdefault("reliability", {
        "analyze_used_fallback": False,
        "analyze_json_success":  False,
        "retrieval_empty": {"bm25": None, "chroma": None, "neo4j": None},
        "pipeline_error": None,
    })
    return m


def _bump_llm_calls(state: RAGState, n: int = 1) -> None:
    _metrics(state)["usage"]["llm_calls"] += n


# ── 단계별 지연시간 계측 데코레이터 ───────────────────────────────────
def _timed(name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(state: RAGState) -> RAGState:
            t0 = time.perf_counter()
            result = func(state)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            target = result if result is not None else state
            target.setdefault("_timings", {})[name] = elapsed_ms
            return result
        return wrapper
    return decorator


# ── Neo4j 쿼리 헬퍼 (retrieval_node 에서 재사용) ──────────────────────
#
# v2 정규화 스키마 기반 (scripts/rag/run_ingest_neo4j.py 참고):
#   (Professor)-[:TEACHES]->(Course)
#   (Course)-[:HELD_ON]->(Day)
#   (Course)-[:HAS_TIME]->(Time)
#   (Course)-[:LOCATED_IN]->(Room)
#   (Professor)-[:BELONGS_TO]->(Department)
#
# day / time_range 는 Day / Time 노드에서 alias 로 가져와 v1 과 동일한 shape 반환
# → evidence.from_neo4j() 는 변경 없이 그대로 동작.

import re as _re

_DAY_CHARS      = set("월화수목금")
_DAY_MULTI_RE   = _re.compile(r"(월|화|수|목|금)요일")
# 축약 표기 "월수금" 등: 앞뒤가 한글이 아닐 때만 매칭 → "수업", "교수" 오검지 방지
_DAY_CONCAT_RE  = _re.compile(r"(?<![가-힣])([월화수목금]{2,5})(?=[\s,\.\?!]|$)")


def _find_days(question: str) -> List[str]:
    """질문에서 요일 감지 (단일 문자로 정규화).

    - "화요일" / "월요일에는" → ['화'], ['월']
    - "월수금 수업" → ['월', '수', '금']
    - "수업 시간" → [] (수업의 '수'를 잘못 잡지 않음)
    """
    found: List[str] = []
    for m in _DAY_MULTI_RE.finditer(question):
        d = m.group(1)
        if d not in found:
            found.append(d)
    for m in _DAY_CONCAT_RE.finditer(question):
        for d in m.group(1):
            if d in _DAY_CHARS and d not in found:
                found.append(d)
    return found


def _neo4j_query(question: str) -> List[Dict]:
    _init_keywords()
    driver = _get_driver()
    if not driver:
        return []

    found_profs   = [p for p in _professors if p in question]
    found_courses = [c for c in _courses    if c in question]
    found_days    = _find_days(question)
    results: List[Dict] = []

    with driver.session() as session:
        # 1) 교수 + 요일 교집합 (다중 홉, GraphRAG 특화 쿼리)
        for name in found_profs:
            for day in found_days:
                rows = session.run("""
                    MATCH (p:Professor {name: $name})-[:TEACHES]->(c:Course)-[:HELD_ON]->(d:Day {name: $day})
                    OPTIONAL MATCH (c)-[:HAS_TIME]->(tm:Time)
                    OPTIONAL MATCH (c)-[:LOCATED_IN]->(r:Room)
                    RETURN p.name AS professor, c.course_name AS course_name,
                           c.course_type AS course_type, c.credits AS credits,
                           d.name AS day, tm.range AS time_range, r.name AS room,
                           p.research_day AS research_day, p.phone AS phone, p.office AS office
                """, name=name, day=day).data()
                results.extend(rows)

        # 2) 교수 (요일 없거나 요일 매칭 실패 대비 fallback)
        for name in found_profs:
            rows = session.run("""
                MATCH (p:Professor {name: $name})-[:TEACHES]->(c:Course)
                OPTIONAL MATCH (c)-[:HELD_ON]->(d:Day)
                OPTIONAL MATCH (c)-[:HAS_TIME]->(tm:Time)
                OPTIONAL MATCH (c)-[:LOCATED_IN]->(r:Room)
                RETURN p.name AS professor, c.course_name AS course_name,
                       c.course_type AS course_type, c.credits AS credits,
                       d.name AS day, tm.range AS time_range, r.name AS room,
                       p.research_day AS research_day, p.phone AS phone, p.office AS office
            """, name=name).data()
            results.extend(rows)

        # 3) 과목 → 담당 교수 + 시간
        for cname in found_courses:
            rows = session.run("""
                MATCH (p:Professor)-[:TEACHES]->(c:Course {course_name: $name})
                OPTIONAL MATCH (c)-[:HELD_ON]->(d:Day)
                OPTIONAL MATCH (c)-[:HAS_TIME]->(tm:Time)
                OPTIONAL MATCH (c)-[:LOCATED_IN]->(r:Room)
                RETURN p.name AS professor, c.course_name AS course_name,
                       c.course_type AS course_type, c.credits AS credits,
                       d.name AS day, tm.range AS time_range, r.name AS room
            """, name=cname).data()
            results.extend(rows)

        # 4) 학과 (기존과 동일)
        if any(kw in question for kw in ("학과", "학부", "대학")):
            rows = session.run("""
                MATCH (d:Department)
                WHERE any(word IN $words WHERE d.name CONTAINS word)
                RETURN d.name AS dept, d.phone AS phone, d.college AS college
            """, words=[w for w in question.split() if len(w) >= 3]).data()
            results.extend(rows)

    return results


# ── 노드 함수 ─────────────────────────────────────────────────────────
@_timed("normalize")
def normalize_node(state: RAGState) -> RAGState:
    """0. 사전 기반 정규화 (LLM 호출 없음)"""
    original = state["question"]
    state["original_question"] = original
    stats = normalize_with_stats(original)
    state["question"] = stats["normalized"]
    usage = _metrics(state)["usage"]
    usage["glossary_hits"]          = stats["hits"]
    usage["glossary_matched_terms"] = stats["matched_terms"]
    return state


@_timed("analyze")
def analyze_node(state: RAGState) -> RAGState:
    """1. Query Analysis (JSON 강제 + Pydantic 검증 + rule-based fallback)"""
    _init_keywords()
    result = analyze_with_stats(
        question=state["question"],
        known_professors=_professors,
        known_courses=_courses,
    )
    analysis: QueryAnalysis = result["analysis"]
    state["query_analysis"] = analysis.model_dump()
    state["intent"]         = analysis.to_korean_intent()

    m = _metrics(state)
    m["usage"]["retrieval_types_selected"] = list(analysis.retrieval_types)
    m["reliability"]["analyze_used_fallback"] = result["used_fallback"]
    m["reliability"]["analyze_json_success"]  = result["json_parse_success"]
    if not result["used_fallback"]:
        _bump_llm_calls(state)
    return state


@_timed("rewrite")
def rewrite_node(state: RAGState) -> RAGState:
    """2. 조건부 Rewrite: query_analysis.needs_rewrite=False 이면 스킵."""
    qa = state.get("query_analysis", {})
    needs = qa.get("needs_rewrite", True)
    if not needs:
        state["rewritten"] = state["question"]
        return state

    _metrics(state)["usage"]["rewrite_called"] = True
    question = state["question"]
    try:
        rewritten = _chat(REWRITE_PROMPT, f"학생: {question}")
        if "→" in rewritten:
            rewritten = rewritten.split("→")[-1].strip()
        state["rewritten"] = rewritten if rewritten else question
        _bump_llm_calls(state)
    except Exception:
        state["rewritten"] = question
    return state


@_timed("retrieval")
def retrieval_node(state: RAGState) -> RAGState:
    """3. 조건부 검색. Query Analysis 의 retrieval_types 에 따라 선택적 호출."""
    qa = state.get("query_analysis", {})
    types = set(qa.get("retrieval_types", []))
    # 안전망: 지정 없으면 semantic + keyword (Neo4j 는 명확한 엔티티가 없으면 노이즈)
    if not types:
        types = {"semantic", "keyword"}

    query    = state.get("rewritten") or state["question"]
    question = state["question"]
    m        = _metrics(state)
    used: List[str] = []
    by_source: Dict[str, List[Evidence]] = {}

    if "keyword" in types:
        results = bm25_search(query, n_results=RETRIEVAL_TOPK)
        by_source["bm25"] = [from_bm25(r, i + 1) for i, r in enumerate(results)]
        used.append("keyword")
        m["reliability"]["retrieval_empty"]["bm25"] = (len(results) == 0)

    if "semantic" in types:
        results = chroma_search(query, n_results=RETRIEVAL_TOPK)
        by_source["chroma"] = [from_chroma(r, i + 1) for i, r in enumerate(results)]
        used.append("semantic")
        m["reliability"]["retrieval_empty"]["chroma"] = (len(results) == 0)

    if "graph" in types:
        rows = _neo4j_query(question)
        by_source["neo4j"] = [from_neo4j(r, i + 1) for i, r in enumerate(rows)]
        used.append("graph")
        m["reliability"]["retrieval_empty"]["neo4j"] = (len(rows) == 0)

    state["evidences_by_source"] = by_source
    m["usage"]["retrieval_types_used"] = used
    return state


@_timed("fusion")
def fusion_node(state: RAGState) -> RAGState:
    """4. RRF: 순위 기반 통합. 검색기 간 교집합에 agreement bonus 부여."""
    by_source = state.get("evidences_by_source", {})
    lists = [lst for lst in by_source.values() if lst]
    fused = rrf_fuse(lists, top_n=FUSION_TOP_N) if lists else []
    state["fused_evidences"] = fused
    _metrics(state)["usage"]["candidates_before_rerank"] = len(fused)
    return state


def _rerank_text(ev: Evidence) -> str:
    """Cross-Encoder 입력용 텍스트. content 가 비면 metadata 조립."""
    if ev.content:
        return ev.content
    md = ev.metadata
    parts: List[str] = []
    for k in ("professor", "course_name", "dept", "title", "section"):
        v = md.get(k)
        if v:
            parts.append(str(v))
    return " ".join(parts) or "(no content)"


@_timed("rerank")
def rerank_node(state: RAGState) -> RAGState:
    """5. Cross-Encoder Rerank → Top-K 최종 근거."""
    fused = state.get("fused_evidences", [])
    usage = _metrics(state)["usage"]
    if not fused:
        state["top_evidences"] = []
        usage["top_k_final"] = 0
        return state

    encoder = _get_cross_encoder()
    pairs   = [(state["question"], _rerank_text(ev)) for ev in fused]
    scores  = encoder.predict(pairs)
    ranked  = sorted(zip(scores, fused), key=lambda x: x[0], reverse=True)
    top_k: List[Evidence] = []
    for new_rank, (_, ev) in enumerate(ranked[:RERANK_TOP_K], 1):
        top_k.append(ev.model_copy(update={"rank": new_rank}))
    state["top_evidences"] = top_k
    usage["top_k_final"] = len(top_k)
    return state


_SOURCE_TAG = {"graph": "[Graph]", "keyword": "[Keyword]", "semantic": "[Semantic]"}


@_timed("format")
def format_node(state: RAGState) -> RAGState:
    """6. Evidence 리스트를 LLM context 문자열로 조립. 인용 마커 포함."""
    top = state.get("top_evidences", [])
    if not top:
        state["context"] = "관련 정보를 찾을 수 없습니다."
        return state

    lines: List[str] = []
    for i, ev in enumerate(top, 1):
        content = ev.content or _rerank_text(ev)
        tag     = _SOURCE_TAG.get(ev.source_type, "[?]")
        lines.append(f"[{i}] {tag} ({ev.source}) {content}")
    state["context"] = "\n".join(lines)
    return state


@_timed("answer")
def answer_node(state: RAGState) -> RAGState:
    """7. 최종 답변 생성."""
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    user_content = (
        f"다음 정보를 참고하여 질문에 답변해주세요.\n\n"
        f"{state['context']}\n\n"
        f"질문: {state['question']}"
    )
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    if state.get("history"):
        for m in state["history"]:
            if m["role"] == "user":
                msgs.append(HumanMessage(content=m["content"]))
            else:
                msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=user_content))
    state["answer"] = _get_llm().invoke(msgs).content.strip()
    _bump_llm_calls(state)
    return state


# ── 하위 호환: 외부/테스트에서 import 가능하도록 유지, 그래프에는 미포함 ──
@_timed("classify")
def classify_node(state: RAGState) -> RAGState:
    """[Deprecated] 하위 호환용. 그래프에는 포함되지 않음. analyze_node 가 intent 를 뽑음."""
    question = state["question"]
    try:
        label = _chat(INTENT_PROMPT, f"학생: {question}")
        if "→" in label:
            label = label.split("→")[-1].strip()
        label = label.replace(" ", "_").strip("-").strip()
        state["intent"] = label if label in VALID_INTENTS else "기타"
        _bump_llm_calls(state)
    except Exception:
        state["intent"] = "기타"
    return state


# ── 그래프 빌드 ────────────────────────────────────────────────────────
def _build_graph():
    g = StateGraph(RAGState)

    for name, node in [
        ("normalize", normalize_node),
        ("analyze",   analyze_node),
        ("rewrite",   rewrite_node),
        ("retrieval", retrieval_node),
        ("fusion",    fusion_node),
        ("rerank",    rerank_node),
        ("format",    format_node),
        ("answer",    answer_node),
    ]:
        g.add_node(name, node)

    g.add_edge(START,       "normalize")
    g.add_edge("normalize", "analyze")
    g.add_edge("analyze",   "rewrite")
    g.add_edge("rewrite",   "retrieval")
    g.add_edge("retrieval", "fusion")
    g.add_edge("fusion",    "rerank")
    g.add_edge("rerank",    "format")
    g.add_edge("format",    "answer")
    g.add_edge("answer",    END)

    return g.compile()


_graph = _build_graph()


# ── 공개 API ──────────────────────────────────────────────────────────
def answer_with_metadata(question: str, history: List[Dict] = None) -> Dict:
    """답변 + 완전한 계측 데이터 반환.

    반환 구조는 docs/EXPERIMENT.md §8 (실험 로그 스키마)의 per-query 부분과
    일치하도록 설계됨. 평가 스크립트가 그대로 집계 가능.
    """
    initial_state: RAGState = {
        "question":            question,
        "original_question":   question,
        "history":             history or [],
        "rewritten":           "",
        "intent":              "",
        "query_analysis":      {},
        "evidences_by_source": {},
        "fused_evidences":     [],
        "top_evidences":       [],
        "context":             "",
        "answer":              "",
        "_timings":            {},
        "_metrics":            {},
    }

    error: Optional[str] = None
    try:
        result = _graph.invoke(initial_state)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        result = initial_state

    timings     = result.get("_timings", {})
    metrics     = result.get("_metrics", {})
    usage       = metrics.get("usage", {})
    reliability = metrics.get("reliability", {})
    if error:
        reliability = {**reliability, "pipeline_error": error}

    top: List[Evidence] = result.get("top_evidences", [])
    citations = [
        {
            "evidence_id": ev.evidence_id,
            "source_type": ev.source_type,
            "source":      ev.source,
            "rank":        ev.rank,
            "metadata":    ev.metadata,   # 평가용 (recall/MRR heuristic 판정)
            "content":     ev.content,    # 평가용
        }
        for ev in top
    ]

    return {
        "answer":            result.get("answer", ""),
        "original_question": result.get("original_question", question),
        "normalized_query":  result.get("question", question),
        "rewritten_query":   result.get("rewritten", ""),
        "intent":            result.get("intent", ""),
        "query_analysis":    result.get("query_analysis", {}),
        "citations":         citations,
        "system_efficiency": {
            "total_ms":                 round(sum(timings.values()), 2),
            "stage_latency_ms":         timings,
            "llm_calls":                usage.get("llm_calls", 0),
            "candidates_before_rerank": usage.get("candidates_before_rerank", 0),
            "top_k_final":              usage.get("top_k_final", 0),
        },
        "usage": {
            "glossary_hits":            usage.get("glossary_hits", 0),
            "glossary_matched_terms":   usage.get("glossary_matched_terms", []),
            "retrieval_types_selected": usage.get("retrieval_types_selected", []),
            "retrieval_types_used":     usage.get("retrieval_types_used", []),
            "rewrite_called":           usage.get("rewrite_called", False),
        },
        "reliability": reliability,
    }


def answer(question: str, history: List[Dict] = None) -> str:
    return answer_with_metadata(question, history)["answer"]
