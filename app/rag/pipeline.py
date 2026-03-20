"""
하이브리드 RAG 파이프라인
1. Neo4j 키워드 검색 (교수명/과목명)
2. ChromaDB 벡터 검색 (의미 기반)
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
    # 1. Neo4j 키워드 검색
    neo4j_results = _neo4j_search(question)

    # 2. ChromaDB 벡터 검색
    chroma_results = chroma_search(question, n_results=5)

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
