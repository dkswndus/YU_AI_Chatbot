"""
하이브리드 RAG 파이프라인
0. Query Rewriting (LLM으로 질문 재작성)
1. Neo4j 키워드 검색 (교수명/과목명)
2. ChromaDB 벡터 검색 (재작성된 쿼리로)
3. 컨텍스트 조합 → EXAONE LLM 답변 생성
"""
import json
from pathlib import Path
from typing import List, Dict

from neo4j import GraphDatabase
from app.retrieval.chroma_search import search as chroma_search
from app.llm.ollama_client import chat, is_available

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "yu_chatbot_2026"

CHUNK_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "chunks" / "academic" / "academic_001.json"

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

# 키워드 DB (시작 시 1회 로드)
_professors: set = set()
_courses: set = set()
_neo4j_driver = None


def _init_keywords():
    global _professors, _courses
    if _professors:
        return
    data = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    _professors = {r["professor"] for r in data if r.get("professor")}
    _courses    = {r["course_name"] for r in data if r.get("course_name")}


def _get_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        try:
            _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
            _neo4j_driver.verify_connectivity()
        except Exception:
            _neo4j_driver = None
    return _neo4j_driver


INTENT_NEO4J   = {"강의_시간표", "교수_정보", "학과_연락처"}
INTENT_HYDE    = {"학사_일정", "수강_규정", "기타"}  # HyDE 적용 대상
INTENT_CHROMA  = {"학사_일정", "수강_규정"}
VALID_INTENTS  = INTENT_NEO4J | INTENT_CHROMA | {"기타"}


def _hyde_query(question: str) -> str:
    """HyDE: 가상 답변을 생성해 ChromaDB 검색 쿼리로 사용"""
    try:
        messages = [
            {"role": "system", "content": HYDE_PROMPT},
            {"role": "user", "content": question},
        ]
        return chat(messages).strip()
    except Exception:
        return question


def _classify_intent(question: str) -> str:
    """LLM으로 질문 의도 분류"""
    try:
        messages = [
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": f"학생: {question}"},
        ]
        label = chat(messages).strip()
        # 레이블만 추출 (화살표 있으면 뒤만)
        if "→" in label:
            label = label.split("→")[-1].strip()
        # 공백/특수문자 제거
        label = label.replace(" ", "_").strip("-").strip()
        return label if label in VALID_INTENTS else "기타"
    except Exception:
        return "기타"


def _rewrite_query(question: str) -> str:
    """LLM으로 질문을 검색 최적화 쿼리로 재작성"""
    try:
        messages = [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": f"학생: {question}"},
        ]
        rewritten = chat(messages).strip()
        # 화살표 이후 텍스트만 추출 (예시 형식 그대로 반환하는 경우 방지)
        if "→" in rewritten:
            rewritten = rewritten.split("→")[-1].strip()
        return rewritten if rewritten else question
    except Exception:
        return question


def _neo4j_search(question: str) -> List[Dict]:
    """교수명/과목명 키워드로 Neo4j 검색"""
    _init_keywords()
    driver = _get_driver()
    if not driver:
        return []

    found_profs   = [p for p in _professors if p in question]
    found_courses = [c for c in _courses if c in question]
    results = []

    with driver.session() as session:
        # 교수 수업 조회
        for name in found_profs:
            rows = session.run("""
                MATCH (p:Professor {name: $name})-[t:TEACHES]->(c:Course)
                RETURN p.name AS professor, c.course_name AS course_name,
                       c.course_type AS course_type, c.credits AS credits,
                       t.day AS day, t.time_range AS time_range,
                       p.research_day AS research_day, p.phone AS phone, p.office AS office
            """, name=name).data()
            results.extend(rows)

        # 과목 조회
        for cname in found_courses:
            rows = session.run("""
                MATCH (p:Professor)-[t:TEACHES]->(c:Course {course_name: $name})
                RETURN p.name AS professor, c.course_name AS course_name,
                       c.course_type AS course_type, c.credits AS credits,
                       t.day AS day, t.time_range AS time_range
            """, name=cname).data()
            results.extend(rows)

        # 학과 전화번호 조회
        for name in found_profs:
            rows = session.run("""
                MATCH (d:Department) WHERE d.name CONTAINS $keyword
                RETURN d.name AS dept, d.phone AS phone, d.college AS college
            """, keyword=name[:2]).data()  # 교수명 앞 2글자로 학과 매칭은 부정확 → 직접 검색

        # 학과명이 질문에 있으면 전화번호 조회
        dept_keywords = ["학과", "학부", "대학"]
        if any(kw in question for kw in dept_keywords):
            rows = session.run("""
                MATCH (d:Department)
                WHERE any(word IN $words WHERE d.name CONTAINS word)
                RETURN d.name AS dept, d.phone AS phone, d.college AS college
            """, words=[w for w in question.split() if len(w) >= 3]).data()
            results.extend(rows)

    return results


def _format_neo4j_context(results: List[Dict]) -> str:
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
            parts = [f"과목: {r.get('course_name','')}"]
            if r.get("professor"): parts.append(f"교수: {r['professor']}")
            if r.get("course_type"): parts.append(f"이수구분: {r['course_type']}")
            if r.get("credits"): parts.append(f"학점: {r['credits']}")
            if r.get("day") and r.get("time_range"): parts.append(f"시간: {r['day']}요일 {r['time_range']}")
            if r.get("research_day"): parts.append(f"연구일: {r['research_day']}요일")
            if r.get("phone"): parts.append(f"전화: {r['phone']}")
            if r.get("office"): parts.append(f"연구실: {r['office']}")
            lines.append("- " + ", ".join(parts))
        elif "dept" in r:
            lines.append(f"- {r.get('college','')} {r['dept']}: {r.get('phone','')}")
    return "\n".join(lines)


def _format_chroma_context(results: List[Dict]) -> str:
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


def answer(question: str, history: List[Dict] = None) -> str:
    """
    질문에 대한 답변 생성
    history: [{'role': 'user'|'assistant', 'content': '...'}]
    """
    # 0. Query Rewriting + Intent Classification
    rewritten = _rewrite_query(question)
    intent    = _classify_intent(question)

    # 1~2. 의도에 따라 검색 전략 분기
    if intent in INTENT_NEO4J:
        # 강의/교수/학과 → Neo4j 우선, ChromaDB 보조 (HyDE 불필요)
        neo4j_results  = _neo4j_search(question)
        chroma_results = chroma_search(rewritten, n_results=3)
    else:
        # 학사일정/수강규정/기타 → HyDE로 ChromaDB 검색
        neo4j_results  = [] if intent in INTENT_CHROMA else _neo4j_search(question)
        hyde_doc       = _hyde_query(rewritten)
        n              = 7 if intent in INTENT_CHROMA else 5
        chroma_results = chroma_search(hyde_doc, n_results=n)

    # 3. 컨텍스트 조합
    context_parts = []
    neo4j_ctx = _format_neo4j_context(neo4j_results)
    chroma_ctx = _format_chroma_context(chroma_results)
    if neo4j_ctx:
        context_parts.append(neo4j_ctx)
    if chroma_ctx:
        context_parts.append(chroma_ctx)

    context = "\n\n".join(context_parts) if context_parts else "관련 정보를 찾을 수 없습니다."

    # 4. 프롬프트 구성
    user_content = f"""다음 정보를 참고하여 질문에 답변해주세요.

{context}

질문: {question}"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    # 5. LLM 호출
    return chat(messages)
