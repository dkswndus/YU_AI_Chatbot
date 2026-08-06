"""Evidence 공통 스키마.

BM25 · ChromaDB · Neo4j 세 검색기의 결과를 하나의 Evidence 포맷으로 통일한다.
이후 RRF Fusion, Cross-Encoder Reranking, LLM Answer Generation이
동일한 인터페이스로 소비할 수 있게 한다.

포트폴리오 서술:
    "각 검색기의 결과는 raw score 척도가 서로 달라 직접 합산이 불가능하다.
     Evidence 공통 스키마로 정규화하여 순위 기반 RRF Fusion과 Cross-Encoder
     Reranking을 동일한 인터페이스로 처리했다."
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.rag.chunk_id import stable_chunk_id

SourceType = Literal["graph", "keyword", "semantic"]


class Evidence(BaseModel):
    """단일 근거 (하나의 chunk 또는 그래프 사실)."""
    evidence_id: str                         # 전역 고유 식별자 "{prefix}:{local_id}"
    source_type: SourceType                  # graph / keyword / semantic
    content: str                             # LLM · reranker가 소비할 자연어 근거
    source: str                              # 원본 인용 (파일명 · 노드 식별)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    rank: int                                # 해당 검색기 내에서의 순위 (1-indexed)
    raw_score: Optional[float] = None        # 원 점수 (bm25 score, cosine distance 등)


# ── 어댑터 (검색기 결과 → Evidence) ──────────────────────────────────

def from_chroma(result: Dict, rank: int) -> Evidence:
    """ChromaDB 결과 → Evidence.

    입력: {"text": str, "metadata": dict, "distance": float}
    chunk_id 는 stable_chunk_id 로 재구성 (gold labeling 과 매칭 위해).
    """
    md = result.get("metadata", {}) or {}
    local_id = stable_chunk_id(md, fallback_idx=rank)
    text = result.get("text") or result.get("document", "")
    return Evidence(
        evidence_id=f"chroma:{local_id}",
        source_type="semantic",
        content=text,
        source=md.get("source_file") or md.get("doc_id", "chroma"),
        metadata=md,
        rank=rank,
        raw_score=result.get("distance"),  # cosine distance: 낮을수록 유사
    )


def from_bm25(result: Dict, rank: int) -> Evidence:
    """BM25 결과 → Evidence.

    입력: {"text": str, "metadata": dict, "score": float}
    """
    md = result.get("metadata", {}) or {}
    local_id = stable_chunk_id(md, fallback_idx=rank)
    text = result.get("text") or ""
    return Evidence(
        evidence_id=f"bm25:{local_id}",
        source_type="keyword",
        content=text,
        source=md.get("source_file") or md.get("doc_id", "bm25"),
        metadata=md,
        rank=rank,
        raw_score=result.get("score"),
    )


def from_neo4j(row: Dict, rank: int) -> Evidence:
    """Neo4j 결과(구조화된 행) → 자연어로 serialize된 Evidence.

    강의/교수/학과 형태를 각각 다르게 문장화한다.
    """
    content = _serialize_graph_row(row)
    key_parts: List[str] = []
    for k in ("professor", "course_name", "dept"):
        v = row.get(k)
        if v:
            key_parts.append(str(v))
    local_id = "_".join(key_parts) if key_parts else f"unranked_{rank}"
    return Evidence(
        evidence_id=f"graph:{local_id}",
        source_type="graph",
        content=content,
        source="Neo4j",
        metadata=dict(row),
        rank=rank,
        raw_score=None,
    )


def _serialize_graph_row(row: Dict) -> str:
    """Neo4j 행을 사람 · LLM이 읽기 좋은 한국어 문장으로."""
    if row.get("course_name"):
        parts: List[str] = []
        if row.get("professor"):
            parts.append(f"{row['professor']} 교수")
        parts.append(row["course_name"])
        if row.get("course_type"):
            parts.append(f"({row['course_type']})")
        if row.get("day") and row.get("time_range"):
            parts.append(f"{row['day']}요일 {row['time_range']}")
        if row.get("credits"):
            parts.append(f"{row['credits']}학점")
        if row.get("research_day"):
            parts.append(f"연구일 {row['research_day']}요일")
        if row.get("phone"):
            parts.append(f"전화 {row['phone']}")
        if row.get("office"):
            parts.append(f"연구실 {row['office']}")
        return " ".join(parts)
    if row.get("dept"):
        parts = [row.get("college", ""), row["dept"]]
        if row.get("phone"):
            parts.append(f"전화 {row['phone']}")
        return " ".join(p for p in parts if p)
    return str(row)


# ── 유틸 ──────────────────────────────────────────────────────────────

def dedupe_by_id(evidences: List[Evidence]) -> List[Evidence]:
    """evidence_id 기준 중복 제거. 첫 등장 우선."""
    seen = set()
    out: List[Evidence] = []
    for ev in evidences:
        if ev.evidence_id in seen:
            continue
        seen.add(ev.evidence_id)
        out.append(ev)
    return out
