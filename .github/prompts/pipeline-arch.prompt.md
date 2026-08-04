---
mode: "agent"
description: "RAG 파이프라인 구조를 분석하고 아키텍처 개선안을 제안합니다."
---

# RAG 파이프라인 아키텍처 분석 및 개선 제안

## 컨텍스트

- 파이프라인 파일: `app/rag/pipeline.py`
- 현재 구조: LangGraph StateGraph (rewrite → classify → [neo4j/hyde] → chroma → rerank → format → answer)
- 현재 미완료 항목:
  - 임베딩 모델 개선 (한국어 강화)
  - Few-shot 예시 추가
  - 답변 형식 제어
  - Fine-tuning (EXAONE)

## 태스크

1. `app/rag/pipeline.py` 전체를 읽어라
2. 다음 관점에서 분석하라:
   - **검색 품질**: 각 노드의 입출력 데이터 품질
   - **레이턴시 병목**: 가장 느린 노드 식별
   - **오류 처리**: graceful degradation 미비 구간
   - **확장성**: 새 의도 추가 시 수정 범위

3. 개선안을 우선순위별로 제안하라:
   ```
   [P0] 즉시 적용 가능 (코드 수정만)
   [P1] 단기 개선 (모델/DB 변경)
   [P2] 중장기 (Fine-tuning, 아키텍처 변경)
   ```

4. 각 제안에 대해 구체적인 코드 변경 예시를 포함하라

## 제약 조건

- `RAGState` TypedDict 키 구조 유지
- 의도 레이블 (강의_시간표 등) 한글 유지
- Ollama 로컬 서버 의존성 유지 (외부 API 사용 금지)
- Neo4j + ChromaDB 하이브리드 구조 유지
