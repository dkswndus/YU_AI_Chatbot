# YU_AI_Chatbot

용인대학교 학사·장학 문서를 수집·정제·청킹하고, ChromaDB로 벡터 검색할 수 있게 만든 RAG(Retrieval-Augmented Generation)용 파이프라인 프로젝트입니다.

## 주요 기능

- **PDF / URL 텍스트 추출**: `pdf_list.csv`, `url_list.csv` 기준으로 본문 추출
- **텍스트 정제**: 메뉴·푸터·개인정보 문구 등 제거, 줄바꿈 정리
- **답변 단위 청킹**: 질문에 답하기 좋은 크기로 분할, 카테고리별 JSON 저장
- **ChromaDB 적재**: 멀티링구얼 임베딩으로 벡터 DB 저장 후 유사도 검색

## 프로젝트 구조

```
YU_AI_Chatbot/
├── data/
│   ├── raw/           # PDF 원본, pdf_list.csv, url_list.csv
│   ├── processed/     # 추출·정제 txt, 청크 JSON (chunks/{category}/)
│   └── chroma_db/     # ChromaDB 저장 (로컬, .gitignore 대상)
├── app/
│   ├── loaders/       # PDF·URL 추출
│   ├── preprocess/    # 텍스트 정제
│   ├── ingest/        # 청킹
│   └── retrieval/     # ChromaDB 클라이언트·적재·검색
├── docs/              # 할일_정리.md, 메타데이터 스키마 등
├── run_extract.py     # 텍스트 추출
├── run_clean.py       # 정제 → *_clean.txt
├── run_chunk.py       # 청킹 → chunks/{category}/{doc_id}.json
├── run_ingest_chroma.py  # ChromaDB 적재
├── run_verify_chroma.py  # ChromaDB 적재 확인
├── main.py
└── requirements.txt
```

## 환경 설정

```bash
git clone https://github.com/dkswndus/YU_AI_Chatbot.git
cd YU_AI_Chatbot
pip install -r requirements.txt
```

## 실행 순서

1. **텍스트 추출** (PDF·URL 목록 전체)  
   `python run_extract.py`

2. **텍스트 정제**  
   `python run_clean.py`

3. **청킹** (답변 단위로 분할)  
   `python run_chunk.py`

4. **ChromaDB 적재** (첫 실행 시 임베딩 모델 다운로드 가능)  
   `python run_ingest_chroma.py`

5. **적재 확인**  
   `python run_verify_chroma.py`

## 검색 사용 예시

```python
from app.retrieval.chroma_search import search

# 전체에서 검색
results = search("장학금 중복 수혜 가능한가요?", n_results=5)

# 카테고리 지정
results = search("수강신청 기간이 언제예요?", n_results=3, category="academic")
```

## 카테고리

| 코드        | 설명          |
|-------------|---------------|
| scholarship | 장학          |
| academic    | 학사          |
| faq         | FAQ           |
| calendar    | 일정/캘린더   |

## 문서

- [할 일 정리 및 파이프라인 설명](docs/할일_정리.md)
- [메타데이터 스키마](docs/metadata_schema.md)

## 저장소

[https://github.com/dkswndus/YU_AI_Chatbot](https://github.com/dkswndus/YU_AI_Chatbot)
