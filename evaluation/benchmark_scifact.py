"""Evaluate the project's retrieval stack on the held-out BEIR SciFact qrels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval.bm25 import _tokenize
from app.retrieval.hybrid import reciprocal_rank_fusion


DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DATASET_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def ensure_dataset(root: Path) -> Path:
    dataset = root / "scifact"
    archive = root / "scifact.zip"
    if not (dataset / "corpus.jsonl").exists():
        root.mkdir(parents=True, exist_ok=True)
        if not archive.exists() or hashlib.md5(archive.read_bytes()).hexdigest() != DATASET_MD5:
            request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "portfolio-rag-evaluation/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, archive.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        digest = hashlib.md5(archive.read_bytes()).hexdigest()
        if digest != DATASET_MD5:
            raise ValueError(f"SciFact archive checksum mismatch: {digest}")
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = (root / member.filename).resolve()
                if root.resolve() not in target.parents and target != root.resolve():
                    raise ValueError(f"Unsafe path in SciFact archive: {member.filename}")
            source.extractall(root)
    return dataset


def read_jsonl(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as file:
        return {str(row["_id"]): row for row in (json.loads(line) for line in file if line.strip())}


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            query_id = row.get("query-id") or row.get("query_id")
            doc_id = row.get("corpus-id") or row.get("corpus_id")
            score = row.get("score") or row.get("relevance")
            if query_id and doc_id and score is not None and int(score) > 0:
                result.setdefault(str(query_id), {})[str(doc_id)] = int(score)
    if not result:
        raise ValueError(f"No relevance judgments found in {path}")
    return result


def metrics(ranking: list[str], relevant: dict[str, int], k: int) -> dict[str, float]:
    hits = [doc_id for doc_id in ranking[:k] if doc_id in relevant]
    reciprocal_rank = 1.0 / (ranking.index(hits[0]) + 1) if hits else 0.0
    dcg = sum((2 ** relevant[doc_id] - 1) / math.log2(rank + 2)
              for rank, doc_id in enumerate(ranking[:k]) if doc_id in relevant)
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2 ** score - 1) / math.log2(rank + 2) for rank, score in enumerate(ideal))
    return {
        f"recall@{k}": len(hits) / max(1, len(relevant)),
        f"mrr@{k}": reciprocal_rank,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-root", type=Path, default=project.parents[1] / "data" / "benchmarks")
    parser.add_argument("--output-dir", type=Path, default=project / "outputs" / "scifact")
    parser.add_argument("--max-queries", type=int, default=0, help="Debug cap; 0 evaluates the full test qrels")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--rerank-k", type=int, default=20)
    args = parser.parse_args()
    dataset_dir = ensure_dataset(args.data_root.resolve())
    corpus = read_jsonl(dataset_dir / "corpus.jsonl")
    queries = read_jsonl(dataset_dir / "queries.jsonl")
    qrels = read_qrels(dataset_dir / "qrels" / "test.tsv")
    query_ids = sorted(query_id for query_id in qrels if query_id in queries)
    if args.max_queries:
        query_ids = query_ids[: args.max_queries]
    if not corpus or not query_ids:
        raise RuntimeError("SciFact corpus or held-out query split is empty")

    corpus_ids = list(corpus)
    corpus_texts = [" ".join(part for part in (corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")) if part)
                    for doc_id in corpus_ids]
    query_texts = [str(queries[query_id]["text"]) for query_id in query_ids]
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    encoder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    started = time.perf_counter()
    doc_vectors = encoder.encode(
        corpus_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
    )
    try:
        query_vectors = encoder.encode(
            query_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True,
            convert_to_numpy=True, prompt_name="query",
        )
        query_prompt = "sentence-transformers registered query prompt"
    except (KeyError, ValueError):
        prefixed = ["Represent this sentence for searching relevant passages: " + query for query in query_texts]
        query_vectors = encoder.encode(
            prefixed, batch_size=32, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
        )
        query_prompt = "BGE recommended retrieval instruction"
    embedding_seconds = time.perf_counter() - started
    dense_scores = query_vectors @ doc_vectors.T

    tokenized_corpus = [_tokenize(text) for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    cross_encoder = CrossEncoder(RERANKER_MODEL, device=device, max_length=512)
    metric_sums: dict[str, dict[str, float]] = {}
    per_query_rows = []
    method_times = {name: 0.0 for name in ("bm25", "dense", "hybrid_rrf", "hybrid_rrf_reranked")}

    for position, (query_id, question) in enumerate(zip(query_ids, query_texts), start=1):
        relevant = qrels[query_id]
        started = time.perf_counter()
        keyword_scores = bm25.get_scores(_tokenize(question))
        keyword_positions = np.argsort(keyword_scores)[::-1][: args.candidate_k]
        bm25_rank = [corpus_ids[index] for index in keyword_positions]
        method_times["bm25"] += time.perf_counter() - started

        started = time.perf_counter()
        dense_positions = np.argsort(dense_scores[position - 1])[::-1][: args.candidate_k]
        dense_rank = [corpus_ids[index] for index in dense_positions]
        method_times["dense"] += time.perf_counter() - started

        def as_docs(ids: list[str], source: str) -> list[dict]:
            return [
                {
                    "text": " ".join(part for part in (corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")) if part),
                    "metadata": {"doc_id": doc_id},
                    "score": float(len(ids) - rank),
                    "source": source,
                }
                for rank, doc_id in enumerate(ids)
            ]

        started = time.perf_counter()
        fused = reciprocal_rank_fusion(as_docs(dense_rank, "dense"), as_docs(bm25_rank, "bm25"))
        hybrid_rank = [str(row["metadata"]["doc_id"]) for row in fused[: args.candidate_k]]
        method_times["hybrid_rrf"] += time.perf_counter() - started

        started = time.perf_counter()
        candidates = fused[: args.rerank_k]
        rerank_scores = cross_encoder.predict(
            [(question, row["text"]) for row in candidates], batch_size=32, show_progress_bar=False
        )
        reranked = [str(row["metadata"]["doc_id"]) for _score, row in sorted(
            zip(rerank_scores, candidates), key=lambda pair: float(pair[0]), reverse=True
        )]
        method_times["hybrid_rrf_reranked"] += time.perf_counter() - started

        rankings = {
            "bm25": bm25_rank,
            "dense": dense_rank,
            "hybrid_rrf": hybrid_rank,
            "hybrid_rrf_reranked": reranked + [doc_id for doc_id in hybrid_rank if doc_id not in reranked],
        }
        row: dict[str, object] = {"query_id": query_id, "query": question, "relevant_document_ids": sorted(relevant)}
        for method, ranking in rankings.items():
            values = metrics(ranking, relevant, 10)
            row[f"{method}_top10"] = ranking[:10]
            if method not in metric_sums:
                metric_sums[method] = {key: 0.0 for key in values}
            for key, value in values.items():
                metric_sums[method][key] += value
        per_query_rows.append(row)
        if position % 25 == 0 or position == len(query_ids):
            print(f"Evaluated {position}/{len(query_ids)} SciFact test queries", flush=True)

    results = {
        "benchmark": "BEIR SciFact test qrels",
        "dataset_url": DATASET_URL,
        "dataset_archive_md5": DATASET_MD5,
        "license": "CC-BY-SA-4.0; preserve attribution and share-alike terms when redistributing derived dataset material.",
        "corpus_documents": len(corpus),
        "test_queries": len(query_ids),
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "query_prompt": query_prompt,
        "candidate_k": args.candidate_k,
        "rerank_k": args.rerank_k,
        "device": device,
        "embedding_runtime_seconds": embedding_seconds,
        "mean_query_runtime_seconds": {name: method_times[name] / len(query_ids) for name in method_times},
        "metrics": {method: {key: value / len(query_ids) for key, value in scores.items()}
                    for method, scores in metric_sums.items()},
        "limitations": [
            "This is retrieval-only evaluation; it does not measure generated-answer correctness or faithfulness.",
            "SciFact is scientific claim-to-abstract retrieval and does not cover every paper-question format.",
            "BEIR qrels are incomplete relevance judgments; unjudged documents are not necessarily irrelevant.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (args.output_dir / "per_query_results.jsonl").open("w", encoding="utf-8") as output:
        for row in per_query_rows:
            output.write(json.dumps(row) + "\n")
    try:
        import mlflow

        tracking = f"sqlite:///{(args.output_dir / 'mlflow.db').resolve().as_posix()}"
        mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment("research-rag-beir-scifact")
        with mlflow.start_run(run_name="hybrid-retrieval-comparison"):
            mlflow.log_params({"benchmark": "BEIR SciFact test", "encoder": EMBEDDING_MODEL,
                               "reranker": RERANKER_MODEL, "candidate_k": args.candidate_k,
                               "rerank_k": args.rerank_k, "test_queries": len(query_ids), "device": device})
            for method, values in results["metrics"].items():
                for key, value in values.items():
                    # MLflow metric names do not allow '@' even though it is
                    # conventional in retrieval metric labels.
                    mlflow.log_metric(f"{method}.{key.replace('@', '_at_')}", value)
            mlflow.log_artifact(str(args.output_dir / "metrics.json"))
    except ImportError:
        pass
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
