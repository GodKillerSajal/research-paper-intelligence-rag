from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model: SentenceTransformer = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of text strings. Returns (N, D) numpy array."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns (D,) numpy array."""
    model = _get_model()
    return model.encode(
        query,
        normalize_embeddings=True,
        prompt_name="query",  # BGE instruction prefix for queries
    )
