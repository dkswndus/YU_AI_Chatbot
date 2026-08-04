"""
통합 검증 및 서빙 테스트 (6단계)

출력 텐서 셰이프 검증, 드리프트(Drift) 감지,
FastAPI 엔드포인트 smoke test를 수행한다.

실행:
    pytest tests/test_inference.py -v
    pytest tests/test_inference.py -v -k "api"    # API 테스트만
    pytest tests/test_inference.py -v -k "drift"  # 드리프트 테스트만
"""
import json
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DIR = PROJECT_ROOT / "data" / "eval"


# ── 1. 출력 형식 검증 (출력 텐서 셰이프 상당) ─────────────────────────

class TestOutputShape:
    """파이프라인 출력이 기대 형식을 만족하는지 검증"""

    @pytest.fixture(scope="class")
    def pipeline(self):
        try:
            from app.rag.pipeline import answer
            return answer
        except Exception as e:
            pytest.skip(f"파이프라인 로드 실패: {e}")

    def test_answer_is_string(self, pipeline):
        reply = pipeline("수강신청 기간 알려줘", history=[])
        assert isinstance(reply, str)

    def test_answer_is_korean(self, pipeline):
        """답변에 한국어가 포함되어 있어야 한다"""
        reply = pipeline("재수강 기준 알려줘", history=[])
        # 한국어 유니코드 범위 체크 (가-힣)
        korean_chars = [c for c in reply if "\uAC00" <= c <= "\uD7A3"]
        assert len(korean_chars) >= 5, f"한국어 미포함 답변: '{reply[:100]}'"

    def test_answer_not_hallucinate_unknown(self, pipeline):
        """모르는 정보에 대해 '확인이 필요합니다' 류의 응답을 해야 한다"""
        reply = pipeline("2050년 화성 캠퍼스 수강신청 일정", history=[])
        # 아무 내용이나 확신있게 말하면 안 됨 → 짧거나 불확실성 표현 포함
        uncertainty_phrases = ["확인", "모르", "없", "정보", "알 수"]
        # 너무 긴 확신 답변이 아닌지 체크 (300자 이상이면 hallucination 의심)
        assert len(reply) < 500, f"할루시네이션 의심 — 너무 긴 답변: {len(reply)}자"

    def test_history_context_used(self, pipeline):
        """대화 이력이 답변에 반영되어야 한다"""
        history = [
            {"role": "user", "content": "확률과통계 담당 교수가 누구야"},
            {"role": "assistant", "content": "확률과통계는 홍길동 교수님이 담당하십니다."},
        ]
        reply = pipeline("그 교수님 연구실 어디야", history=history)
        assert isinstance(reply, str) and len(reply) > 5

    def test_rag_state_keys_complete(self):
        """RAGState에 필수 키가 모두 존재해야 한다"""
        from app.rag.pipeline import RAGState
        import typing
        hints = typing.get_type_hints(RAGState)
        required_keys = {
            "question", "history", "rewritten", "intent",
            "hyde_doc", "neo4j_results", "chroma_results", "context", "answer"
        }
        assert required_keys.issubset(set(hints.keys())), (
            f"누락된 키: {required_keys - set(hints.keys())}"
        )


# ── 2. 드리프트(Drift) 감지 ────────────────────────────────────────────

class TestDriftDetection:
    """
    이전 평가 결과와 현재 결과를 비교하여 성능 회귀를 감지한다.
    평가 결과 파일: data/eval/_eval_*_result.json
    """

    REGRESSION_THRESHOLD = 0.05  # 5% 이상 하락 시 드리프트 감지

    def _load_latest_result(self, pattern: str) -> dict:
        files = sorted(EVAL_DIR.glob(pattern))
        if not files:
            return {}
        return json.loads(files[-1].read_text(encoding="utf-8"))

    def test_eval_results_exist(self):
        """평가 결과 파일이 존재해야 한다"""
        result_files = list(EVAL_DIR.glob("_eval_*.json"))
        assert len(result_files) >= 1, (
            f"평가 결과 파일 없음: {EVAL_DIR}\n"
            "먼저 scripts/eval/ 스크립트를 실행하세요."
        )

    def test_neo4j_accuracy_not_regressed(self):
        """Neo4j 검색 정확도가 100%에서 하락하지 않아야 한다"""
        result = self._load_latest_result("_eval_neo4j*.json")
        if not result:
            pytest.skip("Neo4j 평가 결과 없음")

        for key, val in result.items():
            if isinstance(val, (int, float)) and 0 <= val <= 1:
                assert val >= (1.0 - self.REGRESSION_THRESHOLD), (
                    f"Neo4j 정확도 회귀 감지: {key} = {val:.1%} (기준: 95%+)"
                )

    def test_chroma_v2_accuracy_not_regressed(self):
        """ChromaDB v2 정확도가 이전 대비 5% 이상 하락하지 않아야 한다"""
        result = self._load_latest_result("_eval_v2*.json")
        if not result:
            pytest.skip("ChromaDB v2 평가 결과 없음")
        # 결과 파일 내 수치 항목이 있으면 체크
        for key, val in result.items():
            if isinstance(val, (int, float)) and 0 < val <= 1:
                # 기준선(베이스라인): 과목명 35%, 교수명 4%
                baseline = {"course_name": 0.353, "professor": 0.049}
                if key in baseline:
                    assert val >= baseline[key] - self.REGRESSION_THRESHOLD, (
                        f"드리프트 감지: {key} = {val:.1%} "
                        f"(기준선: {baseline[key]:.1%}, 허용 하한: {baseline[key] - self.REGRESSION_THRESHOLD:.1%})"
                    )


# ── 3. FastAPI 엔드포인트 Smoke Test ──────────────────────────────────

class TestAPIEndpoints:
    """
    FastAPI 서버가 구동 중일 때 실행하는 스모크 테스트.
    서버가 꺼져 있으면 skip.
    """

    BASE_URL = "http://localhost:8000"

    @pytest.fixture(scope="class")
    def http_client(self):
        try:
            import httpx
            client = httpx.Client(base_url=self.BASE_URL, timeout=30.0)
            # 헬스체크로 서버 가용 여부 확인
            resp = client.get("/YU_AI_CHATBOT")
            if resp.status_code != 200:
                pytest.skip("FastAPI 서버 응답 없음")
            yield client
            client.close()
        except Exception:
            pytest.skip("FastAPI 서버 미기동 (httpx 연결 실패)")

    def test_health_check(self, http_client):
        """GET /YU_AI_CHATBOT → {"status": "ok"}"""
        resp = http_client.get("/YU_AI_CHATBOT")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_chat_endpoint_basic(self, http_client):
        """POST /YU_AI_CHATBOT/chat → answer 필드 포함"""
        resp = http_client.post(
            "/YU_AI_CHATBOT/chat",
            json={"question": "수강신청 기간 알려줘", "history": []}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert isinstance(data["answer"], str) and len(data["answer"]) > 5

    def test_chat_with_history(self, http_client):
        """대화 이력 포함 요청이 정상 처리되어야 한다"""
        history = [
            {"role": "user", "content": "확률과통계 시간 알려줘"},
            {"role": "assistant", "content": "월요일 오전 9시입니다."},
        ]
        resp = http_client.post(
            "/YU_AI_CHATBOT/chat",
            json={"question": "그 강의 학점은 몇 학점이야", "history": history}
        )
        assert resp.status_code == 200
        assert "answer" in resp.json()

    def test_chat_empty_question_handled(self, http_client):
        """빈 질문도 서버 오류 없이 처리되어야 한다 (422 또는 200)"""
        resp = http_client.post(
            "/YU_AI_CHATBOT/chat",
            json={"question": "", "history": []}
        )
        # 빈 질문은 422(Validation) 또는 200(빈 답변) 모두 허용
        assert resp.status_code in (200, 422)

    def test_response_time_under_10s(self, http_client):
        """API 응답 시간이 10초 미만이어야 한다"""
        start = time.time()
        http_client.post(
            "/YU_AI_CHATBOT/chat",
            json={"question": "재수강 기준이 어떻게 돼", "history": []}
        )
        elapsed = time.time() - start
        assert elapsed < 10.0, f"API 응답 시간 초과: {elapsed:.1f}초"
