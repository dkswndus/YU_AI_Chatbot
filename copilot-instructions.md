# GitHub Copilot / Claude Code 협업 지침
# 용인대학교 AI 챗봇 프로젝트

## 프로젝트 개요

용인대학교 학사 정보(강의 시간표, 교수 정보, 학과 연락처, 학사 일정, 수강 규정)를
자연어로 질의응답하는 RAG 기반 챗봇.

## 핵심 아키텍처

```
사용자 질문
  → Query Rewriting  (구어체 → 검색 최적화)
  → Intent Classification (6개 의도 분류)
  → 분기:
      강의_시간표 / 교수_정보 / 학과_연락처 → Neo4j → ChromaDB
      학사_일정 / 수강_규정               → HyDE → ChromaDB
      기타                               → Neo4j → HyDE → ChromaDB
  → Cross-Encoder Re-ranking
  → LLM 답변 생성 (EXAONE 3.5 7.8B, Ollama)
```

## 코드 작성 규칙

1. **RAG 파이프라인 수정**: `app/rag/pipeline.py`의 `RAGState` TypedDict와 노드 함수 패턴 유지
2. **새 검색 전략 추가 시**: `route_by_intent()` 조건 분기에 case 추가
3. **프롬프트 수정**: 파일 상단의 상수(`REWRITE_PROMPT`, `INTENT_PROMPT` 등) 직접 수정
4. **Neo4j 쿼리**: Cypher 문법, 기존 노드 레이블(`Professor`, `Course`, `Department`) 유지
5. **평가**: `scripts/eval/`의 기존 스크립트 참조, `data/eval/` JSON 포맷 유지

## 의도 레이블 (변경 금지)

`강의_시간표` / `교수_정보` / `학과_연락처` / `학사_일정` / `수강_규정` / `기타`

## 주요 경로

| 경로 | 역할 |
|---|---|
| `app/rag/pipeline.py` | LangGraph RAG 파이프라인 |
| `app/api/main.py` | FastAPI 엔드포인트 |
| `app/retrieval/chroma_search.py` | ChromaDB 벡터 검색 |
| `scripts/eval/` | 검색 성능 평가 스크립트 |
| `data/eval/` | 평가 결과 JSON |
| `data/yongin_univ_questions_*.json` | 평가 질문 데이터셋 |

## 금지 사항

- `RAGState`의 기존 키 삭제/이름 변경
- 의도 레이블 한글 → 영어 변경
- Neo4j 노드 레이블(`Professor`, `Course`, `Department`) 변경
- `docker-compose.yml` 임의 수정
