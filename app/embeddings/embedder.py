from sentence_transformers import (
    SentenceTransformer
)

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)
def create_embeddings(
    chunks
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts
    )

    return embeddings
def create_query_embedding(
    query: str
    ):

        return model.encode(query)