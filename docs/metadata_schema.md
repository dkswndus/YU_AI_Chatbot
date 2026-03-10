# 메타데이터 스키마

수집·저장 시 사용하는 공통 메타데이터 항목입니다.

| 항목 | 필드명 | 타입 | 설명 |
|------|--------|------|------|
| ID | id | string | 카테고리_번호 (예: academic_001, scholarship_003) |
| 제목 | title | string | 문서/자료 제목 |
| 카테고리 | category | string | scholarship / academic / faq / calendar |
| 출처 유형 | source_type | string | pdf / url |
| 출처 | source | string | PDF 경로(raw/pdf/...) 또는 URL |
| 수정일 | updated_at | string | YYYY-MM-DD |

## 레코드 예시

```
academic_001 | 2025 2학기 종합강의시간표 | academic | pdf | raw/pdf/학사_2025_2학기_종합강의시간표.pdf | 2025-10-17
scholarship_003 | 국가장학금 신청안내 | scholarship | url | https://... | 2026-03-01
```

## 카테고리 값

- **scholarship** – 장학
- **academic** – 학사
- **faq** – FAQ
- **calendar** – 일정/캘린더
