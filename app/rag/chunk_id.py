"""안정된 unique chunk_id 생성.

course/professor/dept chunk 는 명시적 chunk_id 필드가 없어 doc_id 로 fallback 되면
서로 다른 chunk 가 같은 ID 로 collapse 되는 문제 발생.

이 모듈이 파이프라인 · 라벨링 · 평가 스크립트에서 공통 사용되어야 함
(그렇지 않으면 gold_chunk_ids 와 retrieval 반환 ID 가 매칭 실패).
"""
from typing import Dict


def stable_chunk_id(rec: Dict, fallback_idx: int = 0) -> str:
    """레코드에서 안정된 unique chunk_id 생성.

    우선순위:
      1) explicit chunk_id (info 타입 등)
      2) course:  {doc_id}_course_{course_number}
      3) professor: {doc_id}_professor_{name}
      4) dept_phone: {doc_id}_dept_{dept}
      5) fallback: {doc_id}_{type}_{fallback_idx}
    """
    if rec.get("chunk_id"):
        return rec["chunk_id"]
    doc = rec.get("doc_id", "unknown")
    t = rec.get("type", "")
    if t == "course" and rec.get("course_number"):
        return f"{doc}_course_{rec['course_number']}"
    if t == "professor" and rec.get("name"):
        return f"{doc}_professor_{rec['name']}"
    if t == "dept_phone" and rec.get("dept"):
        return f"{doc}_dept_{rec['dept']}"
    return f"{doc}_{t or 'unknown'}_{fallback_idx}"
