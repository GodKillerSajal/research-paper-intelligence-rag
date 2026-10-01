"""
Cross-encoder reranker using a lightweight sentence-transformers cross-encoder.
Falls back to score-based ordering if model unavailable.
"""
from typing import List, Dict, Any

try:
    from sentence_transformers import CrossEncoder
    _RERANKER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    _RERANKER_AVAILABLE = True
except Exception:
    _RERANKER_AVAILABLE = False
    _RERANKER_MODEL = None


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rerank candidate chunks using cross-encoder.
    Returns top_k most relevant chunks.
    """
    if not candidates:
        return []

    if not _RERANKER_AVAILABLE or _RERANKER_MODEL is None:
        # Fallback: return top_k by existing score
        return sorted(
            candidates,
            key=lambda x: x.get("rrf_score", x.get("score", 0)),
            reverse=True,
        )[:top_k]

    pairs = [(query, c["text"]) for c in candidates]
    scores = _RERANKER_MODEL.predict(pairs)

    scored = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )[:top_k]

    results = []
    for score, candidate in scored:
        doc = candidate.copy()
        doc["rerank_score"] = float(score)
        results.append(doc)

    return results
