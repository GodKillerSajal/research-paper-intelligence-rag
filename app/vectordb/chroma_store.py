import chromadb

client = chromadb.PersistentClient(
    path="./data/chroma"
)

collection = client.get_or_create_collection(
    name="research_papers"
)
def store_chunks(
    chunks,
    embeddings
):

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        collection.add(
            ids=[
                str(chunk["chunk_id"])
            ],

            documents=[
                chunk["text"]
            ],

            embeddings=[
                embedding.tolist()
            ],

            metadatas=[
                {
                    "chunk_id":
                    chunk["chunk_id"]
                }
            ]
        )
def search(
    query_embedding,
    n_results=8
    ):

        results = collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],

            n_results=n_results
        )

        return results
