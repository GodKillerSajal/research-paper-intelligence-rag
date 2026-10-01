import fitz


def extract_text(pdf_path: str) -> dict:

    doc = fitz.open(pdf_path)

    pages = []

    for page in doc:
        pages.append(page.get_text())

    return {
        "filename": pdf_path,
        "num_pages": len(doc),
        "text": "\n".join(pages)
    }