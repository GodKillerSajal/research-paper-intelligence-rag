from fastapi import APIRouter, HTTPException
from typing import List
import time

from app.models.schemas import EvaluationRequest, EvaluationResult
from app.evaluation.dataset import load_dataset
from app.evaluation.retrieval import recall_at_k, mean_reciprocal_rank
from app.core.pipeline import query as run_query

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationResult)
def run_evaluation(request: EvaluationRequest):
    dataset = load_dataset(request.dataset_path)
    if not dataset:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation dataset found at {request.dataset_path}"
        )

    recalls = []
    mrrs = []
    retrieval_latencies = []
    total_latencies = []

    for item in dataset:
        result = run_query(
            question=item["question"],
            collection_name=request.collection,
            top_k=5,
            retrieval_mode="hybrid",
        )

        candidates = result.get("candidates", [])
        r_at_5 = recall_at_k(
            candidates,
            relevant_document=item.get("relevant_document", ""),
            relevant_page=item.get("relevant_page", -1),
            k=5,
        )
        mrr = mean_reciprocal_rank(
            candidates,
            relevant_document=item.get("relevant_document", ""),
            relevant_page=item.get("relevant_page", -1),
        )
        recalls.append(r_at_5)
        mrrs.append(mrr)
        retrieval_latencies.append(result["retrieval_latency_ms"])
        total_latencies.append(result["total_latency_ms"])

    n = len(dataset)
    return EvaluationResult(
        recall_at_5=sum(recalls) / n,
        mrr=sum(mrrs) / n,
        avg_retrieval_latency_ms=sum(retrieval_latencies) / n,
        avg_total_latency_ms=sum(total_latencies) / n,
        num_questions=n,
    )
