---
mode: "agent"
description: "RAG 검색 성능 평가 리포트를 자동 생성합니다."
---

# RAG 평가 리포트 자동 생성

## 컨텍스트

- 평가 데이터셋: `data/yongin_univ_questions_1000_student_style.json`, `data/yongin_univ_questions_1000_natural_v2.json`
- 평가 결과: `data/eval/_eval_*.json`
- 평가 스크립트: `scripts/eval/_eval_accuracy.py`
- 임베딩 모델: `paraphrase-multilingual-MiniLM-L12-v2`

## 태스크

1. `data/eval/` 디렉터리의 최신 JSON 결과 파일들을 읽어라
2. 각 결과를 의도 유형별(강의_시간표, 교수_정보, 학과_연락처, 학사_일정, 수강_규정)로 분류하라
3. 아래 형식으로 마크다운 리포트를 생성하라:

```markdown
## RAG 검색 성능 평가 리포트
> 평가일: {오늘날짜} · 임베딩: paraphrase-multilingual-MiniLM-L12-v2

### 데이터셋별 Top-K 정확도
| 데이터셋 | Top-1 | Top-3 | Top-5 | Top-10 |
|---|---|---|---|---|

### 의도별 정확도
| 의도 | 검색 전략 | 정확도 |
|---|---|---|

### 실패 케이스 분석 (상위 5개)
...

### 개선 제안
...
```

4. 이전 평가 대비 회귀(regression)가 있으면 빨간색 경고로 표시하라
5. 결과를 `docs/eval_report_{날짜}.md`로 저장하라
