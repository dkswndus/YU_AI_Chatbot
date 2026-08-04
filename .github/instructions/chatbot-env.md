---
applyTo: "**"
---

# 용인대학교 AI 챗봇 — 환경 표준화 지침

## 핵심 스택

| 컴포넌트 | 버전/모델 | 역할 |
|---|---|---|
| Python | 3.10+ | 런타임 |
| Ollama | exaone3.5:7.8b | LLM (로컬) |
| ChromaDB | 0.4+ | 벡터 DB |
| Neo4j | 5.x | 그래프 DB (키워드 검색) |
| sentence-transformers | CrossEncoder mmarco-mMiniLMv2-L12-H384-v1 | Re-ranking |
| LangGraph | 1.1+ | RAG 파이프라인 오케스트레이션 |
| FastAPI | 0.110+ | REST API 서버 |

## 의존성 관리 규칙

- 모든 패키지는 `requirements.txt`에 버전 범위로 고정
- 모델 파일(`.gguf`, `.bin`)은 `.gitignore`에 포함, 별도 공유
- ChromaDB 데이터: `data/chroma/` — git 제외
- Neo4j 데이터: Docker volume으로 관리 (`docker-compose.yml` 참조)

## 서비스 기동 순서

```bash
# 1. Neo4j 시작
docker-compose up -d neo4j

# 2. Ollama 시작 (GPU 자동 할당)
ollama serve
ollama pull exaone3.5:7.8b

# 3. FastAPI 서버 시작
python scripts/server/run_server.py
```

## 환경변수 (.env)

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASS=yu_chatbot_2026
OLLAMA_BASE_URL=http://localhost:11434
LANGSMITH_API_KEY=<선택>
LANGSMITH_PROJECT=yu-chatbot
```

## 코드 스타일

- 모든 노드 함수는 `RAGState → RAGState` 시그니처 유지
- 싱글톤 패턴: `_get_cross_encoder()`, `_get_driver()` 방식 사용
- 외부 의존성 초기화 실패 시 빈 결과 반환 (graceful degradation)
- 한국어 로그/주석 유지
