"""
Orchestration layer: ties together ingestion and query pipelines.
"""
import time
from typing import List, Dict, Any, Tuple

from app.ingestion.parser import extract_pages
from app.ingestion.chunker import chunk_pages
from app.retrieval.embeddings import embed_texts, embed_query
from app.retrieval.vector_search import store_chunks, vector_search, collection_count
from app.retrieval.bm25 import build_bm25_index, bm25_search, get_index
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.generation.llm import generate_answer, is_insufficient
from app.core.collections import add_document_to_collection

CONFIDENCE_THRESHOLD = 0.25  # min rerank/rrf score for "high" confidence


# ─────────────────────────────── INGESTION ────────────────────────────────── #

def ingest_document(
    pdf_path: str,
    collection_name: str = "default",
    chunk_size: int = 512,
    overlap: int = 50,
) -> Dict[str, Any]:
    """
    Full ingestion pipeline:
    PDF → parse → chunk → embed → store in ChromaDB + build BM25 index.
    Returns metadata dict.
    """
    pages = extract_pages(pdf_path)
    if not pages:
        raise ValueError(f"No text extracted from {pdf_path}")

    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError(f"No chunks generated from {pdf_path}")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    store_chunks(collection_name, chunks, embeddings)

    # Build BM25 index from all docs currently in this collection
    _rebuild_bm25(collection_name, chunks)

    document_id = pages[0]["document_id"]
    filename = pages[0]["filename"]
    num_pages = pages[0]["num_pages"]

    doc_info = {
        "document_id": document_id,
        "filename": filename,
        "num_pages": num_pages,
        "num_chunks": len(chunks),
    }
    add_document_to_collection(collection_name, doc_info)

    return doc_info


def _rebuild_bm25(collection_name: str, new_chunks: List[Dict[str, Any]]) -> None:
    """Append new chunks to BM25 index (in-memory rebuild)."""
    existing_docs = _BM25_DOCS.get(collection_name, [])
    new_docs = [
        {"text": c["text"], "metadata": {
            "filename": c["filename"],
            "page": c["page"],
            "section": c["section"],
        }}
        for c in new_chunks
    ]
    all_docs = existing_docs + new_docs
    _BM25_DOCS[collection_name] = all_docs
    build_bm25_index(collection_name, all_docs)


_BM25_DOCS: Dict[str, List] = {}  # in-memory store for BM25 corpus


def warm_up_bm25() -> None:
    """
    Called on server startup: rebuild BM25 indexes from all ChromaDB collections.
    This ensures BM25 survives server restarts (ChromaDB is persistent; BM25 is not).
    """
    from app.core.collections import list_collections
    from app.retrieval.vector_search import _get_collection
    import logging
    log = logging.getLogger("uvicorn")

    collections = list_collections()
    for col_meta in collections:
        col_name = col_meta["name"]
        try:
            chroma_col = _get_collection(col_name)
            count = chroma_col.count()
            if count == 0:
                continue
            result = chroma_col.get(include=["documents", "metadatas"])
            docs = []
            for text, meta in zip(result["documents"], result["metadatas"]):
                docs.append({"text": text, "metadata": meta or {}})
            _BM25_DOCS[col_name] = docs
            build_bm25_index(col_name, docs)
            log.info(f"BM25 warmed up: '{col_name}' ({len(docs)} docs)")
        except Exception as e:
            log.warning(f"BM25 warm-up failed for '{col_name}': {e}")


# ─────────────────────────────── QUERY ────────────────────────────────────── #

def query(
    question: str,
    collection_name: str = "default",
    top_k: int = 5,
    retrieval_mode: str = "hybrid",  # "vector", "bm25", "hybrid"
    candidate_k: int = 20,
) -> Dict[str, Any]:
    """
    Full query pipeline:
    question → embed → retrieve (vector/bm25/hybrid) → rerank → LLM → answer.
    Returns full response dict with sources, latencies, confidence.
    """
    t0 = time.perf_counter()

    # 1. Embed query
    q_embedding = embed_query(question)

    # 2. Retrieval
    candidates = _retrieve(
        question=question,
        collection_name=collection_name,
        q_embedding=q_embedding,
        mode=retrieval_mode,
        candidate_k=candidate_k,
    )

    retrieval_latency_ms = (time.perf_counter() - t0) * 1000

    # 3. Rerank
    final_chunks = rerank(question, candidates, top_k=top_k)

    # 4. Confidence check
    confidence = _compute_confidence(final_chunks)

    # 5. Generate answer
    t_gen = time.perf_counter()
    answer, gen_latency_ms = generate_answer(question, final_chunks)
    total_latency_ms = (time.perf_counter() - t0) * 1000

    # 6. Build citations from metadata
    sources = _build_citations(final_chunks)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieval_latency_ms": retrieval_latency_ms,
        "generation_latency_ms": gen_latency_ms,
        "total_latency_ms": total_latency_ms,
        "confidence": confidence,
        "candidates": candidates,  # kept for evaluation
    }


def _retrieve(
    question: str,
    collection_name: str,
    q_embedding,
    mode: str,
    candidate_k: int,
) -> List[Dict[str, Any]]:
    if mode == "vector":
        return vector_search(collection_name, q_embedding, n_results=candidate_k)
    elif mode == "bm25":
        return bm25_search(collection_name, question, n_results=candidate_k)
    else:  # hybrid
        vec_results = vector_search(collection_name, q_embedding, n_results=candidate_k)
        bm25_results = bm25_search(collection_name, question, n_results=candidate_k)
        return reciprocal_rank_fusion(vec_results, bm25_results)


def _compute_confidence(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "insufficient"
    top_score = chunks[0].get("rerank_score", chunks[0].get("rrf_score", chunks[0].get("score", 0)))
    if top_score >= 0.5:
        return "high"
    elif top_score >= CONFIDENCE_THRESHOLD:
        return "medium"
    else:
        return "low"


def _build_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        citations.append({
            "index": i,
            "document": meta.get("filename", "Unknown"),
            "page": meta.get("page", 0),
            "section": meta.get("section", "Unknown"),
            "text_snippet": chunk["text"][:200],
        })
    return citations
