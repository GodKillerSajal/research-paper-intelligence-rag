from fastapi import APIRouter, HTTPException

from app.models.schemas import QuestionRequest, QuestionResponse, CitationSource
from app.core.pipeline import query as run_query
from app.core.collections import get_collection

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    # Validate collection exists
    col = get_collection(request.collection)
    if col is None:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{request.collection}' not found. Upload documents first."
        )

    result = run_query(
        question=request.question,
        collection_name=request.collection,
        top_k=request.top_k,
        retrieval_mode=request.retrieval_mode,
    )

    sources = [
        CitationSource(
            index=s["index"],
            document=s["document"],
            page=s["page"],
            section=s["section"],
            text_snippet=s["text_snippet"],
        )
        for s in result["sources"]
    ]

    return QuestionResponse(
        success=True,
        question=result["question"],
        answer=result["answer"],
        sources=sources,
        retrieval_latency_ms=result["retrieval_latency_ms"],
        generation_latency_ms=result["generation_latency_ms"],
        total_latency_ms=result["total_latency_ms"],
        confidence=result["confidence"],
    )
