import chromadb
import re
from typing import List, Dict, Any, Optional
import numpy as np

_clients: Dict[str, chromadb.PersistentClient] = {}
_collections: Dict[str, Any] = {}

CHROMA_PATH = "./data/chroma"


def _sanitize_name(name: str) -> str:
    """
    ChromaDB requires names matching [a-zA-Z0-9._-], 3-512 chars,
    starting and ending with alphanumeric. Sanitize by replacing
    spaces and invalid chars with hyphens, then strip leading/trailing hyphens.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "-", name)
    sanitized = sanitized.strip("-")
    if len(sanitized) < 3:
        sanitized = sanitized + "-col"
    return sanitized[:512]


def _get_collection(collection_name: str):
    collection_name = _sanitize_name(collection_name)
    if collection_name not in _collections:
        if "default" not in _clients:
            _clients["default"] = chromadb.PersistentClient(path=CHROMA_PATH)
        client = _clients["default"]
        _collections[collection_name] = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    return _collections[collection_name]


def get_chroma_client():
    if "default" not in _clients:
        _clients["default"] = chromadb.PersistentClient(path=CHROMA_PATH)
    return _clients["default"]


def store_chunks(
    collection_name: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
) -> None:
    """Upsert chunks with embeddings into ChromaDB."""
    col = _get_collection(collection_name)

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "document_id": chunk["document_id"],
            "filename": chunk["filename"],
            "page": chunk["page"],
            "section": chunk["section"],
            "chunk_size": chunk["chunk_size"],
        }
        for chunk in chunks
    ]
    embeds = [e.tolist() for e in embeddings]

    col.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeds,
        metadatas=metadatas,
    )


def vector_search(
    collection_name: str,
    query_embedding: np.ndarray,
    n_results: int = 20,
    where: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic vector search.
    Returns list of result dicts with keys: text, metadata, distance.
    """
    col = _get_collection(collection_name)

    kwargs = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": min(n_results, col.count() or 1),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)

    output = []
    if results["documents"] and results["documents"][0]:
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": text,
                "metadata": meta,
                "score": 1.0 - dist,  # cosine similarity
                "source": "vector",
            })
    return output


def delete_collection(collection_name: str) -> None:
    client = get_chroma_client()
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    _collections.pop(collection_name, None)


def list_collections() -> List[str]:
    client = get_chroma_client()
    return [c.name for c in client.list_collections()]


def collection_count(collection_name: str) -> int:
    try:
        col = _get_collection(collection_name)
        return col.count()
    except Exception:
        return 0
