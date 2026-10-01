"""
Hybrid retrieval: fuse vector search + BM25 using Reciprocal Rank Fusion (RRF).
"""
from typing import List, Dict, Any
import numpy as np

RRF_K = 60  # RRF constant


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    rrf_k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    Returns merged list sorted by RRF score descending.
    """
    scores: Dict[str, float] = {}
    docs: Dict[str, Dict[str, Any]] = {}

    for rank, result in enumerate(vector_results):
        key = _result_key(result)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        docs[key] = result

    for rank, result in enumerate(bm25_results):
        key = _result_key(result)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        docs[key] = result

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    merged = []
    for key in sorted_keys:
        doc = docs[key].copy()
        doc["rrf_score"] = scores[key]
        doc["source"] = "hybrid"
        merged.append(doc)

    return merged


def _result_key(result: Dict[str, Any]) -> str:
    """Unique key for deduplication: use text hash."""
    return str(hash(result["text"][:200]))
