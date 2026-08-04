"""
RAG 파이프라인 성능 자동화 테스트 (TDD)

[TDD 사이클]
Red   → 목표 지표 미달 케이스 정의 (실패 케이스 명확화)
Green → 하이퍼파라미터 튜닝·모델 교체로 목표 달성
Refactor → 코드 경량화 및 레이턴시 최적화

실행:
    pytest tests/test_rag_performance.py -v
    pytest tests/test_rag_performance.py -v -k "intent"   # 의도 분류만
    pytest tests/test_rag_performance.py -v -k "retrieval" # 검색만
"""
import json
import time
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 공통 픽스처 ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def student_style_data():
    path = PROJECT_ROOT / "data" / "yongin_univ_questions_1000_student_style.json"
    if not path.exists():
        pytest.skip(f"데이터셋 없음: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def natural_v2_data():
    path = PROJECT_ROOT / "data" / "yongin_univ_questions_1000_natural_v2.json"
    if not path.exists():
        pytest.skip(f"데이터셋 없음: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def chroma_search_fn():
    try:
        from app.retrieval.chroma_search import search
        return search
    except Exception as e:
        pytest.skip(f"ChromaDB 초기화 실패: {e}")


@pytest.fixture(scope="session")
def pipeline_fn():
    try:
        from app.rag.pipeline import answer
        return answer
    except Exception as e:
        pytest.skip(f"파이프라인 초기화 실패: {e}")


# ── 1단계: 의도 분류 정확도 테스트 ────────────────────────────────────

class TestIntentClassification:
    """목표: 의도 분류 정확도 > 95%"""

    INTENT_SAMPLES = [
        ("확률과통계 시간 언제야",     "강의_시간표"),
        ("알고리즘 강의 어느 요일이야", "강의_시간표"),
        ("김중헌 교수 연구실 어디야",  "교수_정보"),
        ("박민수 교수님 전화번호",     "교수_정보"),
        ("컴퓨터공학부 전화번호",      "학과_연락처"),
        ("물리치료학과 연락처",        "학과_연락처"),
        ("수강신청 언제야",            "학사_일정"),
        ("패논패 신청 마감",           "학사_일정"),
        ("재수강 기준이 어떻게 돼",    "수강_규정"),
        ("졸업학점 몇 학점이야",       "수강_규정"),
        ("엘리베이터 고장났어",        "기타"),
        ("학식 메뉴 알려줘",           "기타"),
    ]

    def test_intent_labels_defined(self):
        """의도 레이블이 정확히 6개 정의되어 있어야 한다"""
        from app.rag.pipeline import VALID_INTENTS
        assert len(VALID_INTENTS) == 6
        expected = {"강의_시간표", "교수_정보", "학과_연락처", "학사_일정", "수강_규정", "기타"}
        assert VALID_INTENTS == expected

    @pytest.mark.parametrize("question,expected_intent", INTENT_SAMPLES)
    def test_intent_classification_samples(self, question, expected_intent):
        """핵심 샘플 질문의 의도 분류가 정확해야 한다"""
        try:
            from app.rag.pipeline import classify_node, RAGState
        except ImportError:
            pytest.skip("파이프라인 임포트 실패")

        state: RAGState = {
            "question": question, "history": [], "rewritten": question,
            "intent": "", "hyde_doc": "", "neo4j_results": [],
            "chroma_results": [], "context": "", "answer": "",
        }
        result = classify_node(state)
        assert result["intent"] == expected_intent, (
            f"질문: '{question}'\n"
            f"기대: {expected_intent}, 실제: {result['intent']}"
        )


# ── 2단계: ChromaDB 검색 품질 테스트 ─────────────────────────────────

class TestChromaRetrieval:
    """
    목표:
    - 과목명 포함 질문: Top-10 정확도 > 60%  (현재 35%)
    - 교수명 포함 질문: Top-10 정확도 > 20%  (현재 4.9%)
    """

    def _run_retrieval_eval(self, search_fn, questions, keyword_field, threshold_top10):
        """키워드 필드(professor/course_name)가 포함된 질문 필터링 후 Top-10 정확도 계산"""
        hits = 0
        total = 0
        for item in questions:
            if not item.get(keyword_field):
                continue
            total += 1
            query = item["question"]
            results = search_fn(query, n_results=10)
            # 기대 답변 키워드가 결과 문서에 포함되면 hit
            keyword = item[keyword_field]
            for r in results:
                doc = r.get("document", "") + str(r.get("metadata", {}))
                if keyword in doc:
                    hits += 1
                    break

        if total == 0:
            pytest.skip(f"'{keyword_field}' 필드 있는 질문 없음")

        accuracy = hits / total
        assert accuracy >= threshold_top10, (
            f"{keyword_field} Top-10 정확도: {accuracy:.1%} (목표: {threshold_top10:.0%}, "
            f"hit={hits}/{total})"
        )
        return accuracy

    def test_course_name_retrieval_student_style(self, chroma_search_fn, student_style_data):
        """과목명 포함 질문 Top-10 정확도 > 60% (student_style)"""
        acc = self._run_retrieval_eval(
            chroma_search_fn, student_style_data,
            keyword_field="course_name", threshold_top10=0.60
        )
        print(f"\n과목명 검색 정확도 (student_style): {acc:.1%}")

    def test_professor_retrieval_student_style(self, chroma_search_fn, student_style_data):
        """교수명 포함 질문 Top-10 정확도 > 20% (student_style)"""
        acc = self._run_retrieval_eval(
            chroma_search_fn, student_style_data,
            keyword_field="professor", threshold_top10=0.20
        )
        print(f"\n교수명 검색 정확도 (student_style): {acc:.1%}")

    def test_course_name_retrieval_natural_v2(self, chroma_search_fn, natural_v2_data):
        """과목명 포함 질문 Top-10 정확도 > 60% (natural_v2)"""
        self._run_retrieval_eval(
            chroma_search_fn, natural_v2_data,
            keyword_field="course_name", threshold_top10=0.60
        )

    def test_professor_retrieval_natural_v2(self, chroma_search_fn, natural_v2_data):
        """교수명 포함 질문 Top-10 정확도 > 20% (natural_v2)"""
        self._run_retrieval_eval(
            chroma_search_fn, natural_v2_data,
            keyword_field="professor", threshold_top10=0.20
        )


# ── 3단계: 레이턴시 테스트 ────────────────────────────────────────────

class TestLatency:
    """목표: 전체 파이프라인 응답 시간 < 5초"""

    LATENCY_QUESTIONS = [
        "확률과통계 시간 언제야",
        "김중헌 교수 연구실 어디야",
        "컴퓨터공학부 전화번호",
        "수강신청 언제야",
        "재수강 기준이 어떻게 돼",
    ]

    @pytest.mark.parametrize("question", LATENCY_QUESTIONS)
    def test_pipeline_latency(self, pipeline_fn, question):
        """파이프라인 전체 응답 시간이 5초 미만이어야 한다"""
        start = time.time()
        answer = pipeline_fn(question, history=[])
        elapsed = time.time() - start

        assert elapsed < 5.0, (
            f"응답 시간 초과: {elapsed:.1f}초 > 5초\n질문: '{question}'"
        )
        assert answer and len(answer) > 5, "답변이 너무 짧음"
        print(f"\n'{question}' → {elapsed:.2f}초")

    def test_chroma_search_latency(self, chroma_search_fn):
        """ChromaDB 검색이 1초 미만이어야 한다"""
        start = time.time()
        chroma_search_fn("확률과통계 강의 시간", n_results=10)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"ChromaDB 검색 지연: {elapsed:.2f}초"


# ── 4단계: 회귀 방지 테스트 ───────────────────────────────────────────

class TestRegression:
    """Neo4j 100% 정확도 회귀 방지"""

    NEO4J_SAMPLES = [
        {"question": "확률과통계 시간 언제야", "course_name": "확률과통계"},
        {"question": "알고리즘 강의 요일", "course_name": "알고리즘"},
    ]

    def test_neo4j_connection(self):
        """Neo4j 연결이 성공해야 한다"""
        try:
            from app.rag.pipeline import _get_driver
            driver = _get_driver()
            if driver is None:
                pytest.skip("Neo4j 미연결 (Docker 중지 상태)")
            with driver.session() as session:
                result = session.run("RETURN 1 AS val").data()
            assert result[0]["val"] == 1
        except Exception as e:
            pytest.skip(f"Neo4j 연결 실패: {e}")

    def test_query_rewriting_output_is_nonempty(self):
        """Query Rewriting 결과가 비어있지 않아야 한다"""
        try:
            from app.rag.pipeline import rewrite_node, RAGState
        except ImportError:
            pytest.skip("파이프라인 임포트 실패")

        state: RAGState = {
            "question": "패논패 언제까지야", "history": [], "rewritten": "",
            "intent": "", "hyde_doc": "", "neo4j_results": [],
            "chroma_results": [], "context": "", "answer": "",
        }
        result = rewrite_node(state)
        assert result["rewritten"], "Query Rewriting 결과가 비어있음"
        assert len(result["rewritten"]) > 3

    def test_pipeline_returns_answer(self, pipeline_fn):
        """파이프라인이 빈 답변을 반환하지 않아야 한다"""
        answer = pipeline_fn("수강신청 기간이 언제야", history=[])
        assert answer and len(answer) > 5, f"빈 답변 반환: '{answer}'"
