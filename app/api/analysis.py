from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

from app.core.collections import get_documents, get_collection
from app.analysis.pipeline import analyse_document

router = APIRouter(prefix="/analysis", tags=["analysis"])

UPLOADS_DIR = "./uploads"


class AnalysisRequest(BaseModel):
    collection: str
    filename: str


@router.post("/integrity")
def run_integrity_analysis(request: AnalysisRequest):
    col = get_collection(request.collection)
    if col is None:
        raise HTTPException(status_code=404, detail=f"Collection '{request.collection}' not found")

    docs = get_documents(request.collection)
    filenames = [d['filename'] for d in docs]
    if request.filename not in filenames:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.filename}' not found. Available: {filenames}"
        )

    pdf_path = os.path.join(UPLOADS_DIR, request.filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF file not found on disk: {pdf_path}")

    try:
        report = analyse_document(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return report


@router.get("/documents/{collection}")
def list_analysable_documents(collection: str):
    """List documents available for integrity analysis in a collection."""
    col = get_collection(collection)
    if col is None:
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")
    docs = get_documents(collection)
    return [{"filename": d["filename"], "num_pages": d["num_pages"]} for d in docs]
