"""Ablation 실험 조건 설정 (환경변수 기반).

EXPERIMENT.md §4.3.5 스펙 준수. 7 조건 + 별표 실험 지원.

사용:
    # env var로 preset 지정
    RAG_ABLATION=vector_only python scripts/eval/run_ablation.py

    # 또는 개별 flag
    RAG_DISABLE_BM25=1 RAG_DISABLE_NEO4J=1 python ...

Preset 정의:
    vector_only     : ChromaDB 만 사용
    bm25_only       : BM25 만 사용
    bm25_vector     : BM25 + ChromaDB (union, RRF 없음, rerank 없음)
    +rrf            : BM25 + ChromaDB + RRF
    +reranker       : BM25 + ChromaDB + RRF + Cross-Encoder
    +cond_rewrite   : +reranker + 조건부 Rewrite (§4.3.4 규칙)
    +cond_neo4j     : +cond_rewrite + 조건부 Neo4j (Query Analysis 판단) = FULL

    # Neo4j 순수 기여 별표 실험
    neo4j_off       : full pipeline + Neo4j 완전 off
    neo4j_always    : full pipeline + Neo4j 항상 호출 (강제)

    # Rewrite 실험 별표
    no_rewrite      : full pipeline + Rewrite off
    force_rewrite   : full pipeline + Rewrite 항상 (RAG_FORCE_REWRITE=1)
"""
import os
from typing import Dict


# ── Preset 정의 ──────────────────────────────────────────────────────
# 각 preset 은 다음 flag 조합:
#   use_bm25: bool
#   use_chroma: bool
#   neo4j_mode: 'off' | 'conditional' | 'always'
#   use_fusion: bool  (RRF)
#   use_rerank: bool
#   rewrite_mode: 'off' | 'conditional' | 'force'

PRESETS: Dict[str, Dict] = {
    # 단일 검색기 기준선
    "vector_only": {
        "use_bm25": False, "use_chroma": True, "neo4j_mode": "off",
        "use_fusion": False, "use_rerank": False, "rewrite_mode": "off",
    },
    "bm25_only": {
        "use_bm25": True, "use_chroma": False, "neo4j_mode": "off",
        "use_fusion": False, "use_rerank": False, "rewrite_mode": "off",
    },
    # 누적 Ablation
    "bm25_vector": {
        "use_bm25": True, "use_chroma": True, "neo4j_mode": "off",
        "use_fusion": False, "use_rerank": False, "rewrite_mode": "off",
    },
    "+rrf": {
        "use_bm25": True, "use_chroma": True, "neo4j_mode": "off",
        "use_fusion": True, "use_rerank": False, "rewrite_mode": "off",
    },
    "+reranker": {
        "use_bm25": True, "use_chroma": True, "neo4j_mode": "off",
        "use_fusion": True, "use_rerank": True, "rewrite_mode": "off",
    },
    "+cond_rewrite": {
        "use_bm25": True, "use_chroma": True, "neo4j_mode": "off",
        "use_fusion": True, "use_rerank": True, "rewrite_mode": "conditional",
    },
    "+cond_neo4j": {  # = FULL
        "use_bm25": True, "use_chroma": True, "neo4j_mode": "conditional",
        "use_fusion": True, "use_rerank": True, "rewrite_mode": "conditional",
    },
    # Neo4j 순수 기여 별표
    "neo4j_off": {
        "use_bm25": True, "use_chroma": True, "neo4j_mode": "off",
        "use_fusion": True, "use_rerank": True, "rewrite_mode": "conditional",
    },
    "neo4j_always": {
        "use_bm25": True, "use_chroma": True, "neo4j_mode": "always",
        "use_fusion": True, "use_rerank": True, "rewrite_mode": "conditional",
    },
    # Rewrite 별표
    "no_rewrite": {
        "use_bm25": True, "use_chroma": True, "neo4j_mode": "conditional",
        "use_fusion": True, "use_rerank": True, "rewrite_mode": "off",
    },
    "force_rewrite": {
        "use_bm25": True, "use_chroma": True, "neo4j_mode": "conditional",
        "use_fusion": True, "use_rerank": True, "rewrite_mode": "force",
    },
}

# 기본값 (환경변수 없을 때 = 원래 full pipeline)
DEFAULT_CONFIG = PRESETS["+cond_neo4j"]


def get_ablation_config() -> Dict:
    """환경변수에서 실험 조건 읽기.

    우선순위:
      1) RAG_ABLATION=preset_name (있으면 preset 사용)
      2) 개별 env flag 로 override (RAG_USE_BM25 등)
      3) 기본값 (full pipeline)
    """
    preset_name = os.environ.get("RAG_ABLATION", "").strip()
    if preset_name and preset_name in PRESETS:
        cfg = dict(PRESETS[preset_name])
        cfg["_preset"] = preset_name
    else:
        cfg = dict(DEFAULT_CONFIG)
        cfg["_preset"] = "default (full pipeline)"

    # 개별 env override (있으면)
    _env_bool_override(cfg, "use_bm25", "RAG_USE_BM25")
    _env_bool_override(cfg, "use_chroma", "RAG_USE_CHROMA")
    _env_bool_override(cfg, "use_fusion", "RAG_USE_FUSION")
    _env_bool_override(cfg, "use_rerank", "RAG_USE_RERANK")

    if "RAG_NEO4J_MODE" in os.environ:
        v = os.environ["RAG_NEO4J_MODE"].strip().lower()
        if v in ("off", "conditional", "always"):
            cfg["neo4j_mode"] = v

    if "RAG_REWRITE_MODE" in os.environ:
        v = os.environ["RAG_REWRITE_MODE"].strip().lower()
        if v in ("off", "conditional", "force"):
            cfg["rewrite_mode"] = v

    return cfg


def _env_bool_override(cfg: Dict, key: str, env_name: str) -> None:
    if env_name in os.environ:
        v = os.environ[env_name].strip().lower()
        cfg[key] = v in ("1", "true", "yes", "on")


def describe(cfg: Dict) -> str:
    """설정 요약 문자열."""
    return (
        f"preset={cfg.get('_preset','?')} "
        f"bm25={cfg['use_bm25']} chroma={cfg['use_chroma']} "
        f"fusion={cfg['use_fusion']} rerank={cfg['use_rerank']} "
        f"neo4j={cfg['neo4j_mode']} rewrite={cfg['rewrite_mode']}"
    )
