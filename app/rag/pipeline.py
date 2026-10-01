from app.parser.pdf_parser import extract_text
from app.chunking.chunker import chunk_text
from app.embeddings.embedder import (
    create_embeddings,
    create_query_embedding
)
from app.vectordb.chroma_store import (
    store_chunks,
    search
)
from app.llm.generator import (
    generate_answer
)

def ingest_document(
    pdf_path: str
):

    paper = extract_text(
        pdf_path
    )

    chunks = chunk_text(
        paper["text"]
    )

    embeddings = create_embeddings(
        chunks
    )

    store_chunks(
        chunks,
        embeddings
    )

    return len(chunks)

def ask_question(
    query: str
):
    query_embedding = create_query_embedding(
        query
    )

    results = search(
    query_embedding
)

    print("\nRETRIEVED CHUNKS:\n")

    for i, doc in enumerate(
        results["documents"][0]
    ):

        print(
            f"\nChunk {i+1}"
        )

        print(
            doc[:300]
        )

        print(
            "\n" + "="*50
        )
    contexts =  results["documents"][0]

    answer = generate_answer(
    query,
    contexts
    )
    return answer