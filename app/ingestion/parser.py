import fitz  # PyMuPDF
import uuid
import os
from typing import List, Dict, Any


def extract_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF with per-page metadata.

    Returns a list of page dicts:
    {
        "document_id": str,
        "filename": str,
        "page": int,
        "text": str,
        "num_pages": int
    }
    """
    document_id = str(uuid.uuid4())
    filename = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    num_pages = len(doc)

    pages = []
    for page_num, page in enumerate(doc, start=1):
        raw_text = page.get_text()
        cleaned = _clean_text(raw_text)
        if not cleaned.strip():
            continue
        pages.append({
            "document_id": document_id,
            "filename": filename,
            "page": page_num,
            "num_pages": num_pages,
            "text": cleaned,
        })

    doc.close()
    return pages


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize line breaks."""
    import re
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()
