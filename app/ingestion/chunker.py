from typing import List, Dict, Any
import re

DEFAULT_CHUNK_SIZE = 300   # Reduced from 512: tighter chunks = more precise retrieval
DEFAULT_OVERLAP = 50


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Structure-aware chunking: chunk within each page, preserving metadata.

    Args:
        pages: list of page dicts from parser.extract_pages()
        chunk_size: approximate number of words per chunk
        overlap: number of words to overlap between consecutive chunks

    Returns:
        List of chunk dicts with full metadata.
    """
    chunks = []
    global_chunk_id = 0

    for page in pages:
        words = page["text"].split()
        if not words:
            continue

        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i: i + chunk_size]
            chunk_text = " ".join(chunk_words)

            if len(chunk_words) < 15:  # skip tiny tail chunks
                continue

            section = _detect_section(chunk_text)

            chunks.append({
                "chunk_id": f"{page['document_id']}_{global_chunk_id}",
                "document_id": page["document_id"],
                "filename": page["filename"],
                "page": page["page"],
                "section": section,
                "text": chunk_text,
                "chunk_size": chunk_size,
            })
            global_chunk_id += 1

    return chunks


def _detect_section(text: str) -> str:
    """
    Improved section detection: scans ALL lines in the chunk (not just
    the first) for academic paper section headers, including numbered
    sections like '1. Introduction', '2.1 Data', 'III. Results'.
    """
    numbered = re.compile(
        r'^(?:\d+\.?\d*\.?\s+|[IVXivx]+\.\s+)'
        r'(abstract|introduction|background|related work|literature|'
        r'methodology|method|experiment|result|discussion|conclusion|'
        r'reference|appendix|dataset|data|model|training|evaluation|'
        r'analysis|findings|approach|framework|future work|limitation)',
        re.IGNORECASE,
    )
    plain = re.compile(
        r'^(abstract|introduction|background|related work|literature review|'
        r'methodology|method|experimental setup|results?|discussion|'
        r'conclusion|references?|appendix|dataset|data|model|training|'
        r'evaluation|analysis|findings|future work|limitations?)\s*$',
        re.IGNORECASE,
    )

    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped or len(stripped) > 80:   # headers are short lines
            continue
        if numbered.match(stripped) or plain.match(stripped):
            return stripped.title()[:60]

    return "Body"
