# YU AI Chatbot

**챗봇 바로가기**: https://chatbot.yongin.ac.kr/

용인대학교 학사·장학 안내 챗봇입니다.
학생들이 자주 묻는 강의 시간표, 교수 정보, 학사 일정, 수강 규정 등을 LLM 기반 RAG 파이프라인으로 빠르게 답변합니다.

---

## 주요 기능

- **하이브리드 RAG 파이프라인** (LangGraph 기반)
  - Query Rewriting: 구어체 질문 → 검색 최적화 쿼리
  - Intent Classification: 질문 의도에 따라 검색 전략 자동 분기
  - HyDE: 가상 답변 생성 후 임베딩 검색
  - Neo4j 그래프 DB: 교수·과목·학과 관계 키워드 검색
  - ChromaDB 벡터 검색: 학사 문서 의미 기반 검색
  - Re-ranking: Cross-Encoder로 검색 결과 정밀 재정렬
- **LLM**: ChatOllama + EXAONE 3.5 7.8B (로컬 실행)
- **웹 UI**: FastAPI + HTML/CSS/JS 채팅 인터페이스
- **LangSmith 트레이싱**: 파이프라인 실행 흐름 모니터링

---

## 파이프라인 구조

```
사용자 질문
    ↓
[1] Query Rewriting       구어체 → 검색 최적화 쿼리
    ↓
[2] Intent Classification 의도 분류 (강의시간표 / 교수정보 / 학과연락처 / 학사일정 / 수강규정 / 기타)
    ↓
[3] 검색 전략 분기
    ├─ 강의시간표·교수정보·학과연락처 → Neo4j 검색 + ChromaDB 검색
    ├─ 학사일정·수강규정             → HyDE + ChromaDB 검색
    └─ 기타                          → Neo4j + HyDE + ChromaDB 검색
    ↓
[4] Re-ranking            Cross-Encoder로 결과 재정렬 (상위 5개)
    ↓
[5] 답변 생성             EXAONE 3.5 7.8B
```

---

## 기술 스택

| 영역 | 기술 |
|---|---|
| LLM | ChatOllama + EXAONE 3.5 7.8B |
| 파이프라인 | LangGraph, LangChain |
| 벡터 DB | ChromaDB |
| 그래프 DB | Neo4j 5.18 (Docker) |
| 임베딩 | paraphrase-multilingual-MiniLM-L12-v2 |
| Re-ranking | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 |
| 백엔드 | FastAPI, Uvicorn |
| 트레이싱 | LangSmith |

---

## 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일 생성:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=YU_AI_Chatbot
OPENBLAS_NUM_THREADS=1
```

### 3. Neo4j 실행

```bash
docker compose up -d
```

브라우저: `http://localhost:7474` (ID: `neo4j` / PW: `yu_chatbot_2026`)

### 4. Ollama + EXAONE 실행

```bash
ollama run exaone3.5:7.8b
```

### 5. 서버 실행

```bash
python scripts/server/run_server.py
```

웹 UI: `http://localhost:8000`

---

## 프로젝트 구조

```
YU_AI_Chatbot/
├── app/
│   ├── api/               FastAPI 백엔드 + 웹 UI
│   │   └── static/        index.html, logo.png
│   ├── llm/               Ollama 클라이언트
│   ├── rag/               LangGraph 파이프라인
│   └── retrieval/         ChromaDB 검색
├── data/
│   ├── raw/               원본 PDF·문서
│   └── processed/         청킹된 JSON, 마크다운
├── scripts/
│   ├── chat/              터미널 챗봇
│   ├── pipeline/          문서 처리 스크립트
│   └── server/            서버 실행 스크립트
├── docs/                  프로젝트 문서
├── docker-compose.yml
└── requirements.txt
```
