def chunk_text(
    text: str,
    chunk_size: int = 250
):

    words = text.split()

    chunks = []

    chunk_id = 0

    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk
            }
        )

        chunk_id += 1

    return chunks