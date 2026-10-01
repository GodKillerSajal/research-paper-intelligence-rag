import polars as pl


def save_chunks(
    chunks,
    path="data/chunks.parquet"
):

    df = pl.DataFrame(chunks)

    df.write_parquet(path)

    return df