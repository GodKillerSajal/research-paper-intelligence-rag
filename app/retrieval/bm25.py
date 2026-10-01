"""
BM25 keyword retrieval using rank_bm25.
We store a per-collection BM25 index in memory (rebuilt from ChromaDB on first use).
"""
from typing import List, Dict, Any
import re

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

_indexes: Dict[str, Any] = {}  # collection_name -> {"bm25": BM25Okapi, "docs": list}


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())


def build_bm25_index(collection_name: str, docs: List[Dict[str, Any]]) -> None:
    """
    Build (or rebuild) BM25 index from a list of doc dicts.
    Each doc dict must have 'text' and 'metadata' keys.
    """
    if not _BM25_AVAILABLE:
        return

    corpus = [_tokenize(doc["text"]) for doc in docs]
    bm25 = BM25Okapi(corpus)
    _indexes[collection_name] = {"bm25": bm25, "docs": docs}


def bm25_search(
    collection_name: str,
    query: str,
    n_results: int = 20,
) -> List[Dict[str, Any]]:
    """
    Run BM25 keyword search.
    Returns list of result dicts sorted by BM25 score desc.
    """
    if not _BM25_AVAILABLE or collection_name not in _indexes:
        return []

    index_data = _indexes[collection_name]
    bm25 = index_data["bm25"]
    docs = index_data["docs"]

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Pair with docs and sort
    scored = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )[:n_results]

    results = []
    for score, doc in scored:
        if score > 0:
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(score),
                "source": "bm25",
            })
    return results


def get_index(collection_name: str) -> bool:
    return collection_name in _indexes
