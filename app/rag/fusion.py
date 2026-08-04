"""RRF (Reciprocal Rank Fusion).

여러 검색기(BM25 · ChromaDB · Neo4j)의 결과를 순위 기반으로 통합한다.
각 검색기는 서로 다른 점수 척도(BM25 절댓값, cosine distance, 없음)를 사용하므로
원점수를 직접 합산할 수 없다. RRF는 순위만 사용해 이 문제를 해결한다.

공식:
    RRF_score(d) = Σ (1 / (k + rank_i(d)))
    - rank_i(d): 검색기 i에서의 순위 (1-indexed)
    - k: 완충 상수 (표준값 60). 낮으면 상위 순위 가중치 ↑, 높으면 평탄화.

Cross-retrieval matching:
    Chunk 기반 결과(chroma / bm25)는 local_id(prefix 제거)로 매칭 → 여러 검색기에서
    같은 chunk를 찾으면 점수가 합산되어 자연스러운 agreement bonus 발생.
    Graph 결과는 evidence_id 그대로 사용 → 각 그래프 fact는 독립적.
"""
from typing import Callable, Dict, List, Optional

from app.rag.evidence import Evidence

DEFAULT_K = 60


def _default_fusion_key(ev: Evidence) -> str:
    """검색기 간 매칭 규칙.

    - chunk 기반(semantic·keyword): prefix 제거 후 local_id 로 매칭
      (같은 chunk를 여러 검색기가 찾으면 합산)
    - graph: evidence_id 그대로 유지 (각 fact 독립)
    """
    if ev.source_type == "graph":
        return ev.evidence_id
    _, _, local = ev.evidence_id.partition(":")
    return local or ev.evidence_id


def rrf_fuse(
    evidence_lists: List[List[Evidence]],
    k: int = DEFAULT_K,
    top_n: Optional[int] = None,
    key_fn: Callable[[Evidence], str] = _default_fusion_key,
) -> List[Evidence]:
    """여러 Evidence 리스트를 RRF 점수로 통합.

    Args:
        evidence_lists: 검색기별 결과 리스트. 각 리스트는 이미 rank 순으로 정렬돼 있어야 함.
        k: RRF 완충 상수 (기본 60).
        top_n: 상위 N개만 반환. None 이면 전체.
        key_fn: 매칭 키 추출 함수. 기본은 chunk/graph 구분 매칭.

    Returns:
        fused rank로 재정렬된 Evidence 리스트.
        각 Evidence의 rank는 fused 순위, raw_score는 fused RRF score로 교체됨.
        원본 검색기별 점수는 metadata["_original_scores"] 에 보존.
    """
    scores: Dict[str, float] = {}
    representatives: Dict[str, Evidence] = {}
    original_scores: Dict[str, List[Dict]] = {}

    for ev_list in evidence_lists:
        for ev in ev_list:
            key = key_fn(ev)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + ev.rank)
            original_scores.setdefault(key, []).append({
                "source_type": ev.source_type,
                "rank":        ev.rank,
                "raw_score":   ev.raw_score,
            })
            # 대표(representative) 선정: content가 비어있으면 non-empty 로 승격
            existing = representatives.get(key)
            if existing is None or (not existing.content and ev.content):
                representatives[key] = ev

    ranked_keys = sorted(scores.keys(), key=lambda kk: scores[kk], reverse=True)
    if top_n is not None:
        ranked_keys = ranked_keys[:top_n]

    fused: List[Evidence] = []
    for new_rank, key in enumerate(ranked_keys, 1):
        rep = representatives[key]
        fused_ev = rep.model_copy(update={
            "rank":      new_rank,
            "raw_score": round(scores[key], 6),
            "metadata":  {
                **rep.metadata,
                "_original_scores": original_scores[key],
                "_fusion_hits":     len(original_scores[key]),
            },
        })
        fused.append(fused_ev)
    return fused
