import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List

from app.models.schemas import DocumentResponse
from app.core.pipeline import ingest_document
from app.core.collections import get_documents, get_collection, create_collection

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOADS_DIR = "./uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)


@router.post("/upload", response_model=DocumentResponse)
def upload(
    file: UploadFile = File(...),
    collection: str = Form(default="default"),
    chunk_size: int = Form(default=512),
):
    # Ensure collection exists
    if not get_collection(collection):
        create_collection(collection)

    # Save the file
    safe_name = os.path.basename(file.filename)
    file_path = os.path.join(UPLOADS_DIR, safe_name)
    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # Ingest
    try:
        doc_info = ingest_document(
            pdf_path=file_path,
            collection_name=collection,
            chunk_size=chunk_size,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return DocumentResponse(
        document_id=doc_info["document_id"],
        filename=doc_info["filename"],
        collection=collection,
        num_pages=doc_info["num_pages"],
        num_chunks=doc_info["num_chunks"],
    )


@router.get("/{collection}", response_model=List[dict])
def list_documents(collection: str):
    docs = get_documents(collection)
    return docs
