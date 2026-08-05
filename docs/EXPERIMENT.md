# 실험 · 평가 프레임워크 (KPI-First)
> 용인대학교 AI 챗봇 — Hybrid GraphRAG 파이프라인

**설계 원칙:** 측정할 KPI를 코드보다 먼저 정의한다.
각 컴포넌트는 *왜 넣었는지 + 어떤 지표로 검증할지 + 언제 뺄지*까지 확정한 상태로 구현한다.
숫자가 아직 없는 항목은 "측정 예정"으로 명시하고, 절대 임의값으로 채우지 않는다.

---

## 1. KPI 프레임워크

파이프라인 성능은 4개 축으로 측정한다.

### 1.1 Retrieval Quality — 검색 품질

| 지표 | 정의 | 측정 방법 | 목표 |
|---|---|---|---|
| Hit@5 (교수명, heuristic) | 정답 교수명이 상위 5개에 포함된 비율 | 평가셋 1,000개 × pipeline | 목표 ≥ 30% · 실측 **65.3% / 67.9%** ✅ (2배 초과) |
| Hit@5 (과목명, heuristic) | 정답 과목명이 상위 5개에 포함된 비율 | 동일 | 목표 ≥ 70% · 실측 **54.1% / 44.7%** ⚠️ 미달 |
| MRR (교수명 / 과목명) | 정답 chunk의 역순위 평균 (Mean Reciprocal Rank) | 동일 | 실측 prof **0.49 / 0.60**, course **0.46 / 0.39** |
| Hit@1 (교수명 / 과목명) | 정답이 1위로 나온 비율 | 동일 | 실측 prof **40.2% / 55.5%**, course **40.3% / 36.0%** |
| Relationship Query Accuracy | 관계형 질문(교수-과목-요일)의 정확도 | Neo4j 특화 평가셋 | 목표 ≥ 90% · 측정 예정 (Neo4j 재활성화 후) |

### 1.2 Generation Quality — 답변 품질

| 지표 | 정의 | 측정 방법 | 목표 |
|---|---|---|---|
| Answer Correctness | 답변이 정답과 의미상 일치 | LLM-as-Judge 또는 human eval | 측정 예정 |
| Groundedness | 답변이 검색 context에 근거함 | Context ↔ Answer entailment | 측정 예정 |
| Citation Accuracy | 출처 인용이 실제 근거 chunk와 일치 | metadata 매칭 | 측정 예정 |
| Refusal Rate | 근거 부족 시 "확인이 필요합니다" 반환 비율 | keyword 매칭 | 측정 예정 |

### 1.3 System Efficiency — 시스템 효율

| 지표 | 정의 | 측정 방법 | 목표 |
|---|---|---|---|
| Average Latency | 질문당 평균 응답시간 | pipeline `_timings` 집계 | 목표 < 5초 · 실측 **3.63s / 3.58s** ✅ (Colab T4 GPU) |
| P95 Latency | 상위 5% 응답시간 (최악 UX) | 동일 | 목표 < 8초 · 실측 **6.27s / 5.97s** ✅ |
| Stage Latency Breakdown | 단계별(normalize/analyze/rewrite/retrieval/rerank/answer) 평균 | `_timings` 세부 | 병목 식별용 |
| LLM Calls per Query | 질문당 평균 LLM 호출 수 | metrics.reliability | 목표 ≤ 3회 · 실측 **2.29 / 2.31** ✅ |
| Average Tokens per Query | 질문당 평균 프롬프트 토큰 | tokenizer 집계 | 측정 예정 |
| Rewrite Call Rate | 전체 질문 중 Rewrite가 발동한 비율 | metrics.usage | 목표 < 40% · 실측 **29.5% / 32.4%** (student/natural) ✅ |
| BM25 Usage Rate | BM25가 실제 호출된 질문 비율 | metrics.usage | 실측 **73.8% / 67.8%** |
| Graph Usage Rate | Neo4j가 실제 호출된 질문 비율 | metrics.usage | 실측 **5.7% / 8.7%** (Neo4j 미실행 상태) |

### 1.4 Reliability — 안정성

| 지표 | 정의 | 측정 방법 | 목표 |
|---|---|---|---|
| Query Analysis JSON Success Rate | LLM이 valid JSON을 뱉은 비율 | try/except 카운터 | 목표 ≥ 95% · 실측 **99.3% / 98.8%** ✅ |
| Fallback Trigger Rate | rule-based fallback으로 떨어진 비율 | metrics.reliability | 목표 < 5% · 실측 **0.7% / 1.2%** ✅ |
| Retrieval Empty Rate | 검색기별 빈 결과 반환 비율 | metrics.reliability | 실측 bm25 **0.0/0.6%** · chroma **0.0%** · neo4j **5.7/8.7%** |
| Pipeline Error Rate | 파이프라인 예외 발생률 | 로깅 | 목표 < 1% · 실측 **0.1% / 0.0%** ✅ |
| Glossary Coverage | 평가셋 질문 중 사전 치환이 발생한 비율 | normalizer 카운터 | 실측 **0.9% / 4.7%** (사전 확장 필요) |

---

## 2. 컴포넌트 선택 근거 (Vector vs BM25 vs Graph)

각 검색기의 도입 이유를 "무엇을 못 해서 도입했는가" 관점에서 정리한다.
포트폴리오 서술의 근간이 되는 섹션이다.

### 2.1 왜 Vector(ChromaDB) 단독으로는 부족한가

- **강점:** 의미적 유사도 검색, 표현이 달라도 매칭 (예: "휴학" ↔ "학교 쉬기")
- **한계:** 고유명사·희귀어의 exact match가 약함
- **실측 근거:** ChromaDB 단독 시 교수명 Recall@10 **4.9%**, 과목명 Recall@10 **35.3%**

### 2.2 왜 BM25를 추가하는가

- **목적:** Vector가 놓치는 고유명사(교수명·과목명·학과명·과목코드)의 exact match 확보
- **대체 후보와 비교:**
  - Elasticsearch: 오버킬. 인덱스 관리·클러스터 운영 부담이 프로젝트 규모에 비해 과함
  - 단순 substring: score 계산 없음 → 랭킹 불가
- **선택 이유:** `rank_bm25` 순수 Python. 인덱스 관리 부담 없음. 배포 단순화.
- **검증 KPI:** 교수명 Recall@5 (Vector Only 대비)

### 2.3 왜 Neo4j를 추가하는가 (SQL이 아니라)

**기존 방식의 한계:**

- **Vector RAG:** 문서 유사도만 본다. "김중헌 교수가 담당하는 과목 중 화요일에 진행되는 과목은?"처럼 여러 엔티티의 관계를 정확히 조합해야 하는 질문은 관련 chunk를 찾더라도 관계 조합이 부정확할 수 있음.
- **SQL:** 시간표 단건 조회는 충분하지만 다음 상황에서 확장성 한계
  - 교수·과목·학기·전공·요일·강의실 등 관계가 계속 확장됨
  - 질문마다 탐색해야 하는 관계 경로가 다름 (교수→과목, 학과→교수, 요일→과목, 과목→선수과목→담당교수)
  - 다단계 관계 표현이 JOIN 중첩으로 장황해짐

**Neo4j를 선택한 근거:**

- 가변적인 다중 관계 탐색을 Cypher로 명료하게 표현 가능
- 문서 검색 결과와 별도로 **구조화된 관계 근거(Graph Evidence)** 를 확보
- 스키마 확장(Day/Time/Classroom 노드 분리, Roadmap Step 6)에 따라 표현력 극대화

**효과가 기대되는 질문 예시:**

| 질문 | 필요한 관계 경로 |
|---|---|
| 김중헌 교수님이 담당하는 과목은? | Professor → TEACHES → Course |
| 김중헌 교수님의 화요일 수업은? | Professor → TEACHES → Course → HELD_ON → Day |
| 확률과통계 담당 교수와 강의실은? | Course ← TEACHES ← Professor, Course → LOCATED_IN → Classroom |
| AI학과 3학년 과목 중 화요일 오전 수업은? | Department → 소속 Course → HELD_ON → Day |

**검증 계획 (실측 후 채움):**

| 지표 | Vector Only | + Neo4j | 개선폭 |
|---|:---:|:---:|:---:|
| Relationship Query Accuracy | 측정 예정 | 측정 예정 | 측정 예정 |

**정직한 회고 원칙:**
효과가 없거나 미미해도 그대로 기록한다.
예상 대안 서술:
> "단순 시간표 조회에서는 SQL과 성능 차이가 크지 않았지만, 교수·과목·학기·시간처럼 2단계 이상의 관계를 탐색하는 질문에서 정확도가 개선됐다."

### 2.4 왜 이 3가지 조합인가

세 검색기는 서로 다른 사각지대를 커버한다.

| 검색기 | 강한 사각지대 | 대표 질문 |
|---|---|---|
| BM25 | 고유명사·희귀어 exact match | "김중헌 교수" |
| ChromaDB (Vector) | 표현 다양성, 의미 유사성 | "학교 쉬려면 어떻게 해?" |
| Neo4j (Graph) | 다단계 엔티티 관계 | "김중헌 교수 화요일 수업" |

**핵심:** 항상 3개 다 실행하지 않는다. Query Analysis가 뽑은 `retrieval_types`에 따라 조건부로 호출한다 (Roadmap Step 7).

**한 줄 요약 (포트폴리오용):**
> 교수·과목·학기·시간 간 다단계 관계를 문서 유사도가 아닌 명시적 관계 경로로 탐색하기 위해 Neo4j를 사용했고, 고유명사 exact match는 BM25로, 표현 다양성은 Vector로 분담했다.

---

## 3. 설계 의도 → KPI 매핑

각 컴포넌트는 명확한 목적, 검증 지표, 실패 시 결정 기준을 갖는다.

| 컴포넌트 | 왜 넣었는가 | 검증 KPI | 실패 시 신호 (버릴/재설계 기준) |
|---|---|---|---|
| Dictionary Normalization | 확정 약어를 LLM 없이 처리 → 비용·지연 절감 | Glossary Coverage, Average Tokens per Query | 커버리지 < 5% → 사전 대폭 확장 필요 |
| Query Analysis | 구조화된 라우팅 근거 확보 | JSON Success Rate, Confidence 분포 | Success Rate < 90% → 프롬프트/모델 재설계 |
| Conditional Rewrite | 모호 질문만 재작성 → 불필요 호출 방지 | Rewrite Call Rate, Recall@5 유지 여부 | Rewrite 후 Recall 하락 → 트리거 조건 재검토 |
| BM25 | 고유명사(교수명·과목명) Recall 보완 | 교수명 Recall@5 (Vector Only 대비) | Vector 대비 개선 < 5%p → 인덱싱/토크나이저 재검토 |
| Neo4j (Graph) | 다중 홉 관계 질의 해결 | Relationship Query Accuracy | 관계 질문 정확도 < 70% → 스키마 재설계 |
| Conditional Retrieval | 필요한 검색기만 호출 → Latency·Cost 최적화 | BM25/Graph Usage Rate, Latency | 조건부 도입 후 Recall 하락 → 라우팅 로직 완화 |
| RRF Fusion | 서로 다른 점수 척도의 결과를 순위 기반 통합 | MRR, Hit@1 (단일 검색기 대비) | 단일 검색기 대비 MRR 하락 → 가중치/k 파라미터 튜닝 |
| Cross-Encoder Reranker | 관련도 재정렬 | Answer Correctness, Groundedness | Latency +500ms 대비 품질 향상 없음 → 제거 검토 |

---

## 4. Ablation Study Matrix

파이프라인 컴포넌트를 순차 추가하며 KPI 기여도를 측정한다.
숫자는 실측 후 채운다. 순서는 **저비용·확정성 우선** 원칙을 따른다.

| # | 구성 | Recall@5 | MRR | Answer Correctness | Avg Latency | LLM Calls | 목적 |
|---|---|---|---|---|---|---|---|
| 0 | ChromaDB Only (Baseline) | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 최소 baseline |
| 1 | + Dictionary Normalization | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 정규화 효과 |
| 2 | + BM25 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 고유명사 개선 검증 |
| 3 | + Conditional Routing (Query Analysis) | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 라우팅 효율화 |
| 4 | + Neo4j (Graph) | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 관계 질의 해결 |
| 5 | + RRF Fusion | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 결과 통합 품질 |
| 6 | + Cross-Encoder Reranker (Final) | 아래 §4.1 참조 | 아래 참조 | 아래 참조 | 아래 참조 | 아래 참조 | 관련도 재정렬 |

### 4.1 실측 결과 — Full Pipeline (2026-08-05)

Roadmap Step 6까지 완성된 파이프라인을 두 평가셋 전량(N=1,000)에 실행한 결과.
**Neo4j 미실행 상태 실측** (Graph Usage 5~9%만 반영). 완전한 Graph 기여도는 Neo4j 활성화 후 재측정 필요.

**환경:** Colab T4 GPU · exaone3.5:2.4b · N=1,000 × 2 datasets

| 카테고리 | 지표 | student_style | natural_v2 | 목표 대비 |
|---|---|:---:|:---:|---|
| **Retrieval Quality** | 교수명 Hit@1 | 40.17% | 55.47% | — |
| | 교수명 Hit@5 | **65.27%** | **67.88%** | 🎯 목표 30% **2배 초과** |
| | 교수명 MRR | 0.4899 | 0.5978 | 준수 |
| | 과목명 Hit@1 | 40.34% | 36.01% | — |
| | 과목명 Hit@5 | 54.14% | 44.72% | ⚠️ 목표 70% 미달 |
| | 과목명 MRR | 0.4586 | 0.3896 | — |
| | 평가 대상(prof/course) | 239 / 290 | 137 / 436 | — |
| **System Efficiency** | Avg Latency | 3,627 ms | 3,579 ms | 🎯 목표 <5,000 달성 |
| | P95 Latency | 6,268 ms | 5,971 ms | 🎯 목표 <8,000 여유 |
| | Avg LLM Calls | 2.29 | 2.31 | 🎯 목표 ≤3 달성 |
| | Rewrite Call Rate | 29.5% | 32.4% | 🎯 목표 <40% 달성 (LLM 70% 절감) |
| | BM25 Usage | 73.8% | 67.8% | — |
| | Semantic Usage | 49.3% | 62.1% | — |
| | Graph Usage | 5.7% | 8.7% | ⚠️ Neo4j 미실행 반영 |
| **Reliability** | JSON Success Rate | **99.3%** | **98.8%** | 🎯 목표 ≥95% 달성 |
| | Fallback Rate | 0.7% | 1.2% | 🎯 목표 <5% 달성 |
| | Pipeline Error Rate | 0.1% | 0.0% | 🎯 목표 <1% 달성 |
| | Glossary Coverage | 0.9% | 4.7% | ⚠️ 사전 확장 필요 |

**해석 및 결정 근거:**

- **교수명 검색 압도적 개선**: Vector 단독 시 Recall@10 4.9% → Hybrid Pipeline Hit@5 65~68% (약 **13배 개선**). BM25의 고유명사 exact match가 예상대로 핵심 기여.
- **과목명 Hit@5 44~54%**: 목표(70%) 미달. 원인 후보:
  - `academic_001.json`의 course chunk 는 `text` 필드가 빈 경우 다수 → BM25 문서 길이가 짧아 랭킹 하방 편향
  - Cross-Encoder 는 자연어에 최적화 → metadata-only chunk 의 relevance 판단 약함
  - **후속 개선안**: BM25 인덱싱 시 course chunk 는 metadata 를 자연어 문장으로 조립하여 저장 (Evidence.from_neo4j 스타일)
- **Rewrite Call Rate 29~32%**: 조건부 라우팅으로 **70%의 질문에서 Rewrite LLM 호출 회피** — 설계 의도대로 작동.
- **Glossary Coverage 0.9~4.7%**: 사전이 너무 작음 → 확장 지표. 평가셋 실패 케이스에서 자주 나오는 구어체를 사전에 추가하여 개선 여지.
- **Neo4j 부분 활성 (5~9%)**: Colab 에 Neo4j 미배포. 로컬 재적재 후 재측정 시 교수·과목·요일 관계 질의 정확도 상승 기대.

**예상 효과 방향** (검증 전 가설, 실측으로 확인):

| 단계 | Recall | Latency | LLM Calls | Answer Correctness |
|---|:---:|:---:|:---:|:---:|
| + Dict Normalization | ↑ | ≈ | ↓ | ≈ |
| + BM25 | ↑↑ (교수명) | ↑ | ≈ | ↑ |
| + Conditional Routing | ≈ | ↓ | ↓↓ | ≈ |
| + Neo4j | ↑↑ (관계) | ↑ | ≈ | ↑↑ |
| + RRF | ↑ | ≈ | ≈ | ↑ |
| + Reranker | ↑ | ↑ | ≈ | ↑↑ |

---

## 5. 평가 데이터셋

| 파일 | 수량 | 특징 | 용도 |
|---|---|---|---|
| `yongin_univ_questions_1000_student_style.json` | 1,000 | 구어체, 줄임말, 오타 | 실제 사용자 시뮬레이션 |
| `yongin_univ_questions_1000_natural_v2.json` | 1,000 | 자연어, 문어체 | 표준 벤치마크 |
| Neo4j Relationship 특화셋 (신규 구축) | 200 | 교수-과목-요일 다중 홉 | Graph 성능 검증 |

**과적합 방지:**
- 평가셋은 학습/튜닝에 사용하지 않는다.
- 프롬프트/사전 튜닝은 `student_style`로 진행하고, 검증은 `natural_v2`로 수행한다.
- 임베딩 파인튜닝 시 train / val / test = 70 / 15 / 15 분리.

---

## 6. 모델 · 하이퍼파라미터 스펙

### 6.1 LLM

| 항목 | 값 | 비고 |
|---|---|---|
| 모델 | `exaone3.5:7.8b` | Ollama 로컬 |
| 온도 (일반) | ChatOllama 기본값 | 답변 생성 |
| 온도 (Query Analysis) | `0.0` | 결정성 확보 |
| Format | `json` (Query Analysis 한정) | JSON 강제 모드 |
| 대화 이력 | 최근 6턴 | |

### 6.2 임베딩 (실험 대상)

| 코드명 | 모델 | 차원 |
|---|---|---|
| baseline | `paraphrase-multilingual-MiniLM-L12-v2` | 384 |
| exp-A | `jhgan/ko-sroberta-multitask` | 768 |
| exp-B | `BAAI/bge-m3` | 1024 |

### 6.3 Retriever

| 검색기 | 구현 | 후보 수 (top-K) |
|---|---|---|
| BM25 | `rank_bm25` + Okt 형태소 (or whitespace) | 10 |
| ChromaDB | HNSW, cosine | 10 |
| Neo4j | Cypher 관계 매칭 | 그래프 반환 |
| RRF | k=60 (표준) | 통합 후 15 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 최종 5 |

### 6.4 Neo4j 스키마 (정규화 목표)

**목표 스키마 (Roadmap Step 6에서 리팩터):**

```
(:Professor {name, phone, office, research_day})
(:Course {course_name, course_type, credits})
(:Department {name, phone, college})
(:Day {name})           ← 정규화 (기존 TEACHES 속성에서 분리)
(:Time {range})         ← 정규화
(:Classroom {name})     ← 정규화

관계:
(Professor)-[:TEACHES]->(Course)
(Course)-[:HELD_ON]->(Day)
(Course)-[:HAS_TIME]->(Time)
(Course)-[:LOCATED_IN]->(Classroom)
(Professor)-[:BELONGS_TO]->(Department)
```

**현재 스키마:** `(Professor)-[:TEACHES {day, time_range}]->(Course)` — day/time이 관계 속성으로 저장.
**전환 이유:** GraphRAG의 다중 홉 표현력 확보, "월요일에 강의 있는 교수는?" 같은 역방향 질의 지원.

---

## 7. 평가 프로토콜

1. **베이스라인 확보** — ChromaDB Only(구성 #0)로 전체 KPI 최초 측정.
2. **컴포넌트 순차 추가** — Ablation Matrix 순서대로 하나씩 활성화, 각 단계에서 전 KPI 재측정.
3. **회귀 방지** — 새 컴포넌트 도입 후 특정 KPI가 하락하면 원인 규명 후에만 진행.
4. **최종 파이프라인** — 모든 컴포넌트 조합에서 `student_style` + `natural_v2` 전량 재실행.
5. **결과 아카이빙** — 각 실행의 KPI를 `data/eval/` 하위에 스키마 준수 JSON으로 저장.

### 7.1 실행 명령

```bash
# 사전 준비: Ollama · ChromaDB · Neo4j 모두 실행 상태
docker-compose up -d
ollama serve  # 별도 창

# 개발 모드 (20개 샘플로 파이프라인 sanity check)
python scripts/eval/run_kpi.py --config full --dataset student_style --limit 20

# 전체 실행 (1000개, 약 60~90분)
python scripts/eval/run_kpi.py --config full --dataset student_style
python scripts/eval/run_kpi.py --config full --dataset natural_v2

# 상세 per-query 로그도 저장
python scripts/eval/run_kpi.py --config full --dataset student_style --save-per-query
```

결과는 `data/eval/{config}_{dataset}.json` 에 §8 스키마로 저장됨.

### 7.2 Retrieval Quality Heuristic

평가셋에 ground truth chunk_id 라벨이 없으므로 다음 heuristic 사용:

- 질문에서 매칭된 **교수명 · 과목명** (`_professors`, `_courses` 집합) 을 pseudo-truth 로 취급
- Top-K 후보의 `metadata` 나 `content` 에 해당 keyword 가 포함되면 hit

**주의:** 실제 정답률 대비 낙관적 편향 가능. **절대치보다 조합 간 델타** 로 해석할 것.
정확한 절대 recall 이 필요하면 라벨링 데이터셋 별도 구축 필요.

---

## 8. 실험 로그 스키마

`data/eval/{experiment_id}.json`:

```json
{
  "experiment_id": "ablation_02_bm25",
  "date": "2026-04-15",
  "config": {
    "components": ["dict_norm", "bm25", "chroma"],
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "reranker": null,
    "dataset": "student_style"
  },
  "retrieval_quality": {
    "recall_at_5":  null,
    "recall_at_10": null,
    "mrr":          null,
    "hit_at_1":     null,
    "by_type": {
      "professor_name": { "recall_at_10": null },
      "course_name":    { "recall_at_10": null }
    }
  },
  "generation_quality": {
    "answer_correctness": null,
    "groundedness":       null,
    "citation_accuracy":  null,
    "refusal_rate":       null
  },
  "system_efficiency": {
    "avg_latency_ms":    null,
    "p95_latency_ms":    null,
    "avg_llm_calls":     null,
    "avg_tokens":        null,
    "rewrite_call_rate": null,
    "bm25_usage_rate":   null,
    "graph_usage_rate":  null,
    "stage_latency_ms": {
      "normalize": null,
      "analyze":   null,
      "rewrite":   null,
      "retrieval": null,
      "rerank":    null,
      "answer":    null
    }
  },
  "reliability": {
    "json_success_rate":    null,
    "fallback_rate":        null,
    "pipeline_error_rate":  null,
    "glossary_coverage":    null,
    "retrieval_empty_rate": {
      "bm25":   null,
      "chroma": null,
      "neo4j":  null
    }
  },
  "notes": ""
}
```

---

## 9. 기술적 제약 조건

| 제약 | 내용 |
|---|---|
| 하드웨어 | GPU VRAM 8GB 이하 가정 |
| 응답 시간 | 전체 파이프라인 P95 < 8초, 평균 < 5초 |
| 외부 API | 금지 (Ollama 로컬만 허용) |
| 언어 | 한국어 질문/답변 전용 |
| 데이터 추가 | 기존 `academic_001.json` 포맷 유지 |

---

## 10. 구현 진행 상태

| # | 컴포넌트 | 상태 |
|---|---|---|
| 1 | Dictionary Normalization | ✅ 완료 |
| 2 | Query Analysis + Pydantic + fallback | ✅ 완료 |
| 3 | Metrics 자동 수집 인프라 (Reliability + Usage) | ✅ 완료 |
| 4 | BM25 검색기 (모듈 단위) | ✅ 완료 |
| 5 | Evidence 공통 스키마 | ✅ 완료 |
| 6 | Neo4j 스키마 정규화 (Day/Time/Classroom 분리) | ✅ 완료 (코드) — 배포 시 `--wipe` 재적재 필요 |
| 7 | 조건부 Retrieval Routing (analyze.retrieval_types 소비) | ✅ 완료 (파이프라인 통합) |
| 8 | RRF Fusion | ✅ 완료 (파이프라인 통합) |
| 9 | Reranker (Evidence 포맷 대응) | ✅ 완료 (파이프라인 통합) |
| 10 | 평가 스크립트 (KPI 자동 집계) | ✅ 완료 — `scripts/eval/run_kpi.py` |
| 11 | Full Pipeline 실측 (N=1,000 × 2 datasets, Colab T4 GPU) | ✅ 완료 — §4.1 참조 |
| 12 | Ablation 각 조합 실측 (Vector Only → +BM25 → …) | ⏳ 향후 (config 토글 로직 추가 필요) |
| 13 | Neo4j 활성 상태 재측정 (Relationship Query Accuracy 확보) | ⏳ 로컬 Docker 환경 |
