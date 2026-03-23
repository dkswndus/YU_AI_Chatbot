"""
하이브리드 RAG 파이프라인 (LangGraph)

[그래프 흐름]
START
  └→ rewrite (Query Rewriting)
      └→ classify (Intent Classification)
          ├─ 강의_시간표 / 교수_정보 / 학과_연락처 → neo4j_search → chroma_search
          ├─ 학사_일정 / 수강_규정              → hyde        → chroma_search
          └─ 기타                              → neo4j_search → hyde → chroma_search
                                                                   ↓
                                                               rerank → format → answer → END
"""
import json
from pathlib import Path
from typing import List, Dict, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END, START
from neo4j import GraphDatabase
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from app.retrieval.chroma_search import search as chroma_search

# ChatOllama: LangSmith 트레이싱 지원
_llm = ChatOllama(model="exaone3.5:7.8b", base_url="http://localhost:11434")


def _chat(system: str, user: str) -> str:
    """LangChain ChatOllama 호출 (LangSmith 자동 트레이싱)"""
    response = _llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content.strip()

# ── 상수 ─────────────────────────────────────────────────────────────
NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "yu_chatbot_2026"

CHUNK_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "chunks" / "academic" / "academic_001.json"
)

INTENT_NEO4J  = {"강의_시간표", "교수_정보", "학과_연락처"}
INTENT_CHROMA = {"학사_일정", "수강_규정"}
VALID_INTENTS = INTENT_NEO4J | INTENT_CHROMA | {"기타"}

# ── 프롬프트 ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 용인대학교 학사 안내 챗봇입니다.
학생들의 질문에 친절하고 정확하게 답변해주세요.
제공된 정보(컨텍스트)를 기반으로만 답변하고, 모르는 정보는 "확인이 필요합니다"라고 하세요.
답변은 간결하게 핵심만 전달해주세요."""

REWRITE_PROMPT = """학생의 구어체 질문을 학사 정보 검색에 최적화된 쿼리로 재작성하세요.
규칙:
- 핵심 키워드(과목명, 교수명, 학과명, 시간, 학점 등)를 명확하게 포함
- 구어체/줄임말/비문을 표준어로 변환
- 패논패 = P/NP (Pass/None Pass) 성적처리 방식
- 한 줄로만 출력, 설명 없이 재작성된 쿼리만 출력

예시:
학생: 확률과통계 시간 언제야 → 확률과통계 강의 요일 및 시간
학생: 김중헌 교수님 뭐 가르쳐 → 김중헌 교수 담당 강의 목록
학생: 물치과 전화번호 알아? → 물리치료학과 전화번호
학생: 패논패 언제까지야 → P/NP(Pass/None Pass) 성적처리 수강신청 기간"""

INTENT_PROMPT = """학생의 질문 의도를 아래 6가지 중 하나로만 분류하세요.
반드시 아래 레이블 중 하나만 출력하세요. 설명 없이 레이블만.

레이블:
- 강의_시간표  : 수업 시간, 강의실, 강의 목록 조회
- 교수_정보    : 교수 연구실, 전화번호, 연구일 조회
- 학과_연락처  : 학과/학부 전화번호 조회
- 학사_일정    : 수강신청, 성적열람, 패논패(P/NP) 기간 등
- 수강_규정    : 재수강, 이수구분, 학점, 졸업요건 등
- 기타         : 위 분류에 해당 없음

예시:
학생: 확률과통계 시간 언제야 → 강의_시간표
학생: 김중헌 연구실 어디야 → 교수_정보
학생: 물리치료학과 전화번호 → 학과_연락처
학생: 패논패 언제까지야 → 학사_일정
학생: 재수강 되냐 → 수강_규정
학생: 엘리베이터 고장났어 → 기타"""

HYDE_PROMPT = """당신은 용인대학교 학사 안내 전문가입니다.
아래 학생 질문에 대해 실제 답변처럼 자세하게 가상의 답변을 작성하세요.
이 답변은 검색에 활용되므로 관련 키워드를 풍부하게 포함해야 합니다.
3~5문장으로 작성하세요.

참고 용어:
- 패논패 = P/NP (Pass/None Pass) 성적처리 방식, 수강신청 시 선택 가능
- 교필 = 교양필수, 전필 = 전공필수, 전선 = 전공선택, 교선 = 교양선택"""

# ── 싱글톤 ────────────────────────────────────────────────────────────
_cross_encoder = None
_neo4j_driver  = None
_professors: set = set()
_courses: set    = set()


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    return _cross_encoder


def _get_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        try:
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
class RAGState(TypedDict):
    question:      str
    history:       List[Dict]
    rewritten:     str
    intent:        str
    hyde_doc:      str
    neo4j_results: List[Dict]
    chroma_results: List[Dict]
    context:       str
    answer:        str


# ── 노드 함수 ─────────────────────────────────────────────────────────
def rewrite_node(state: RAGState) -> RAGState:
    """0단계: Query Rewriting"""
    question = state["question"]
    try:
        rewritten = _chat(REWRITE_PROMPT, f"학생: {question}")
        if "→" in rewritten:
            rewritten = rewritten.split("→")[-1].strip()
        state["rewritten"] = rewritten if rewritten else question
    except Exception:
        state["rewritten"] = question
    return state


def classify_node(state: RAGState) -> RAGState:
    """1단계: Intent Classification"""
    question = state["question"]
    try:
        label = _chat(INTENT_PROMPT, f"학생: {question}")
        if "→" in label:
            label = label.split("→")[-1].strip()
        label = label.replace(" ", "_").strip("-").strip()
        state["intent"] = label if label in VALID_INTENTS else "기타"
    except Exception:
        state["intent"] = "기타"
    return state


def neo4j_node(state: RAGState) -> RAGState:
    """Neo4j 키워드 검색"""
    question = state["question"]
    _init_keywords()
    driver = _get_driver()
    if not driver:
        state["neo4j_results"] = []
        return state

    found_profs   = [p for p in _professors if p in question]
    found_courses = [c for c in _courses    if c in question]
    results = []

    with driver.session() as session:
        for name in found_profs:
            rows = session.run("""
                MATCH (p:Professor {name: $name})-[t:TEACHES]->(c:Course)
                RETURN p.name AS professor, c.course_name AS course_name,
                       c.course_type AS course_type, c.credits AS credits,
                       t.day AS day, t.time_range AS time_range,
                       p.research_day AS research_day, p.phone AS phone, p.office AS office
            """, name=name).data()
            results.extend(rows)

        for cname in found_courses:
            rows = session.run("""
                MATCH (p:Professor)-[t:TEACHES]->(c:Course {course_name: $name})
                RETURN p.name AS professor, c.course_name AS course_name,
                       c.course_type AS course_type, c.credits AS credits,
                       t.day AS day, t.time_range AS time_range
            """, name=cname).data()
            results.extend(rows)

        dept_keywords = ["학과", "학부", "대학"]
        if any(kw in question for kw in dept_keywords):
            rows = session.run("""
                MATCH (d:Department)
                WHERE any(word IN $words WHERE d.name CONTAINS word)
                RETURN d.name AS dept, d.phone AS phone, d.college AS college
            """, words=[w for w in question.split() if len(w) >= 3]).data()
            results.extend(rows)

    state["neo4j_results"] = results
    return state


def hyde_node(state: RAGState) -> RAGState:
    """2단계: HyDE - 가상 답변 생성"""
    try:
        state["hyde_doc"] = _chat(HYDE_PROMPT, state["rewritten"])
    except Exception:
        state["hyde_doc"] = state["rewritten"]
    return state


def chroma_node(state: RAGState) -> RAGState:
    """ChromaDB 벡터 검색 (HyDE 또는 rewritten 쿼리 사용)"""
    query = state.get("hyde_doc") or state["rewritten"]
    state["chroma_results"] = chroma_search(query, n_results=10)
    return state


def rerank_node(state: RAGState) -> RAGState:
    """3단계: Cross-Encoder Re-ranking"""
    chroma_results = state["chroma_results"]
    if not chroma_results:
        return state

    encoder = _get_cross_encoder()
    docs    = [r.get("document", "") for r in chroma_results]
    pairs   = [(state["question"], doc) for doc in docs]
    scores  = encoder.predict(pairs)
    ranked  = sorted(zip(scores, chroma_results), key=lambda x: x[0], reverse=True)
    state["chroma_results"] = [r for _, r in ranked[:5]]
    return state


def format_node(state: RAGState) -> RAGState:
    """컨텍스트 조합"""
    neo4j_ctx  = _format_neo4j(state.get("neo4j_results", []))
    chroma_ctx = _format_chroma(state.get("chroma_results", []))
    parts = [p for p in [neo4j_ctx, chroma_ctx] if p]
    state["context"] = "\n\n".join(parts) if parts else "관련 정보를 찾을 수 없습니다."
    return state


def answer_node(state: RAGState) -> RAGState:
    """최종 답변 생성"""
    user_content = f"다음 정보를 참고하여 질문에 답변해주세요.\n\n{state['context']}\n\n질문: {state['question']}"
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    if state.get("history"):
        for m in state["history"]:
            if m["role"] == "user":
                msgs.append(HumanMessage(content=m["content"]))
            else:
                from langchain_core.messages import AIMessage
                msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=user_content))
    state["answer"] = _llm.invoke(msgs).content.strip()
    return state


# ── 조건부 라우팅 ──────────────────────────────────────────────────────
def route_by_intent(state: RAGState) -> str:
    intent = state["intent"]
    if intent in INTENT_NEO4J:
        return "neo4j"          # Neo4j → ChromaDB
    elif intent in INTENT_CHROMA:
        return "hyde"           # HyDE → ChromaDB
    else:
        return "neo4j_then_hyde"  # Neo4j + HyDE → ChromaDB


def route_after_neo4j(state: RAGState) -> str:
    if state["intent"] == "기타":
        return "hyde"
    return "chroma"


# ── 포매터 ────────────────────────────────────────────────────────────
def _format_neo4j(results: List[Dict]) -> str:
    if not results:
        return ""
    lines = ["[강의/교수 정보]"]
    seen = set()
    for r in results:
        key = str(r)
        if key in seen:
            continue
        seen.add(key)
        if "course_name" in r:
            parts = [f"과목: {r.get('course_name', '')}"]
            if r.get("professor"):    parts.append(f"교수: {r['professor']}")
            if r.get("course_type"):  parts.append(f"이수구분: {r['course_type']}")
            if r.get("credits"):      parts.append(f"학점: {r['credits']}")
            if r.get("day") and r.get("time_range"):
                parts.append(f"시간: {r['day']}요일 {r['time_range']}")
            if r.get("research_day"): parts.append(f"연구일: {r['research_day']}요일")
            if r.get("phone"):        parts.append(f"전화: {r['phone']}")
            if r.get("office"):       parts.append(f"연구실: {r['office']}")
            lines.append("- " + ", ".join(parts))
        elif "dept" in r:
            lines.append(f"- {r.get('college', '')} {r['dept']}: {r.get('phone', '')}")
    return "\n".join(lines)


def _format_chroma(results: List[Dict]) -> str:
    if not results:
        return ""
    lines = ["[관련 학사 정보]"]
    for r in results:
        m = r["metadata"]
        t = m.get("type", "")
        if t == "course":
            lines.append(f"- {m.get('course_name','')} ({m.get('professor','')}) "
                         f"{m.get('day','')}요일 {m.get('time_range','')}")
        elif t == "professor":
            lines.append(f"- {m.get('name','')} 교수 연구일: {m.get('day','')} "
                         f"전화: {m.get('phone','')} 연구실: {m.get('room','')}")
        elif t == "dept_phone":
            lines.append(f"- {m.get('dept','')} 전화: {m.get('phone','')}")
        else:
            doc = r.get("document", "")
            if doc:
                lines.append(f"- {doc[:200]}")
    return "\n".join(lines)


# ── 그래프 빌드 ────────────────────────────────────────────────────────
def _build_graph():
    g = StateGraph(RAGState)

    # 노드 등록
    g.add_node("rewrite",  rewrite_node)
    g.add_node("classify", classify_node)
    g.add_node("neo4j",    neo4j_node)
    g.add_node("hyde",     hyde_node)
    g.add_node("chroma",   chroma_node)
    g.add_node("rerank",   rerank_node)
    g.add_node("format",   format_node)
    g.add_node("answer",   answer_node)

    # 엣지 연결
    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "classify")

    # classify 이후 의도별 분기
    g.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "neo4j":          "neo4j",
            "hyde":           "hyde",
            "neo4j_then_hyde": "neo4j",
        },
    )

    # neo4j 이후: 기타면 hyde로, 아니면 바로 chroma
    g.add_conditional_edges(
        "neo4j",
        route_after_neo4j,
        {
            "hyde":   "hyde",
            "chroma": "chroma",
        },
    )

    g.add_edge("hyde",   "chroma")
    g.add_edge("chroma", "rerank")
    g.add_edge("rerank", "format")
    g.add_edge("format", "answer")
    g.add_edge("answer", END)

    return g.compile()


_graph = _build_graph()


# ── 공개 API ──────────────────────────────────────────────────────────
def answer(question: str, history: List[Dict] = None) -> str:
    initial_state: RAGState = {
        "question":      question,
        "history":       history or [],
        "rewritten":     "",
        "intent":        "",
        "hyde_doc":      "",
        "neo4j_results": [],
        "chroma_results": [],
        "context":       "",
        "answer":        "",
    }
    result = _graph.invoke(initial_state)
    return result["answer"]
