from typing import List, Dict, Any


def recall_at_k(
    retrieved: List[Dict[str, Any]],
    relevant_document: str,
    relevant_page: int,
    k: int = 5,
) -> float:
    """
    Compute Recall@K: was the relevant document/page retrieved in top K results?
    Returns 1.0 if found, 0.0 otherwise.
    """
    top_k = retrieved[:k]
    for result in top_k:
        meta = result.get("metadata", {})
        if (
            relevant_document.lower() in meta.get("filename", "").lower()
            and abs(meta.get("page", -1) - relevant_page) <= 1  # ±1 page tolerance
        ):
            return 1.0
    return 0.0


def mean_reciprocal_rank(
    retrieved: List[Dict[str, Any]],
    relevant_document: str,
    relevant_page: int,
) -> float:
    """
    Compute MRR: reciprocal of the rank of the first relevant result.
    """
    for rank, result in enumerate(retrieved, start=1):
        meta = result.get("metadata", {})
        if (
            relevant_document.lower() in meta.get("filename", "").lower()
            and abs(meta.get("page", -1) - relevant_page) <= 1
        ):
            return 1.0 / rank
    return 0.0
