# Research Paper Intelligence System

A production-grade **Retrieval-Augmented Generation (RAG)** system for research papers — built to demonstrate every layer of an AI/ML engineering pipeline.

---

## Architecture

```
Streamlit UI  ──HTTP──►  FastAPI Backend
                              │
              ┌───────────────┴──────────────────┐
              │                                  │
      Ingestion Pipeline               Query Pipeline
              │                                  │
         PDF Parser                     Query Embedding
              │                                  │
       Text Chunker                    Hybrid Retrieval
      (structure-aware)              (Vector + BM25 + RRF)
              │                                  │
         BGE Embedder                Cross-Encoder Reranker
              │                                  │
         ChromaDB  ◄──────────────────────────── │
                                       Context Builder
                                                 │
                                           Gemini LLM
                                                 │
                                    Answer + Citations + Metrics
```

---

## Features

| Feature | Status |
|---|---|
| PDF parsing (PyMuPDF) | ✅ |
| Structure-aware chunking (configurable size) | ✅ |
| BGE embeddings (`BAAI/bge-small-en-v1.5`) | ✅ |
| ChromaDB vector store | ✅ |
| BM25 keyword retrieval | ✅ |
| Hybrid retrieval (RRF fusion) | ✅ |
| Cross-encoder reranker | ✅ |
| Multi-document collections | ✅ |
| Grounded answers with citations | ✅ |
| Hallucination protection (confidence threshold) | ✅ |
| Evaluation framework (Recall@K, MRR) | ✅ |
| FastAPI REST backend | ✅ |
| Streamlit frontend | ✅ |
| Docker + docker-compose | ✅ |

---

## Project Structure

```
retrieval-rag/
│
├── app/
│   ├── main.py                  # FastAPI app entry point
│   │
│   ├── api/                     # Route handlers
│   │   ├── health.py            # GET /health
│   │   ├── collections.py       # CRUD /collections
│   │   ├── documents.py         # POST /documents/upload
│   │   ├── query.py             # POST /query
│   │   └── evaluation.py        # POST /evaluation/run
│   │
│   ├── ingestion/               # Document ingestion pipeline
│   │   ├── parser.py            # PyMuPDF page extraction
│   │   ├── cleaner.py           # Text normalization
│   │   └── chunker.py           # Structure-aware sliding window
│   │
│   ├── retrieval/               # Retrieval stack
│   │   ├── embeddings.py        # BGE embedder
│   │   ├── vector_search.py     # ChromaDB CRUD + semantic search
│   │   ├── bm25.py              # BM25 keyword index
│   │   ├── hybrid.py            # RRF fusion
│   │   └── reranker.py          # Cross-encoder reranker
│   │
│   ├── generation/              # Answer generation
│   │   ├── prompts.py           # Prompt templates + context builder
│   │   └── llm.py               # Gemini client (replaceable)
│   │
│   ├── core/                    # Orchestration
│   │   ├── pipeline.py          # Full ingest + query pipelines
│   │   └── collections.py       # Collection metadata store
│   │
│   ├── evaluation/              # RAG evaluation
│   │   ├── dataset.py           # Load labeled Q&A datasets
│   │   ├── retrieval.py         # Recall@K, MRR
│   │   └── generation.py        # Answer relevance, faithfulness
│   │
│   └── models/
│       └── schemas.py           # Pydantic request/response models
│
├── frontend/
│   └── streamlit_app.py         # 3-page Streamlit UI
│
├── evaluation/
│   └── questions.json           # Labeled evaluation dataset
│
├── uploads/                     # PDF storage
├── data/                        # ChromaDB + collection metadata
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env
```

---

## Quick Start

### 1. Clone & activate environment

```bash
# Using the existing venv
.venv-1\Scripts\activate        # Windows
source .venv-1/bin/activate     # Linux/Mac
```

### 2. Set your Gemini API key

```bash
# .env
GEMINI_API_KEY=your_key_here
```

> You need a free Gemini API key from [https://aistudio.google.com](https://aistudio.google.com)

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the backend

```bash
uvicorn app.main:app --reload
```

Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 5. Start the frontend

```bash
streamlit run frontend/streamlit_app.py
```

UI: [http://localhost:8501](http://localhost:8501)

### 6. Docker (optional)

```bash
docker compose up
```

---

## API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/collections` | List all collections |
| `POST` | `/collections` | Create collection |
| `DELETE` | `/collections/{name}` | Delete collection |
| `POST` | `/documents/upload` | Upload + ingest a PDF |
| `GET` | `/documents/{collection}` | List documents in collection |
| `POST` | `/query` | Ask a question |
| `POST` | `/evaluation/run` | Run retrieval evaluation |

### Example: Upload

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@paper.pdf" \
  -F "collection=transformers" \
  -F "chunk_size=512"
```

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What attention mechanism was proposed?",
    "collection": "transformers",
    "top_k": 5,
    "retrieval_mode": "hybrid"
  }'
```

Response includes `answer`, `sources` (with page + section citations), `confidence`, and latency metrics.

---

## Retrieval Modes

| Mode | Description |
|---|---|
| `vector` | Semantic similarity only (BGE embeddings + ChromaDB) |
| `bm25` | Keyword-based only (BM25Okapi) |
| `hybrid` | **Reciprocal Rank Fusion** of vector + BM25, then cross-encoder reranker |

Hybrid + reranker consistently outperforms either alone for research paper Q&A.

---

## Evaluation

Create a labeled dataset at `evaluation/questions.json`:

```json
[
  {
    "question": "What was the sample size?",
    "expected_answer": "36 patients in group A.",
    "relevant_document": "study.pdf",
    "relevant_page": 4
  }
]
```

Then run via API or Streamlit **Evaluation** page. Metrics returned:
- **Recall@5** — was the relevant chunk in the top 5?
- **MRR** — Mean Reciprocal Rank
- **Avg Retrieval Latency** (ms)
- **Avg Total Latency** (ms)

---

## Interview Discussion Points

This project demonstrates:

- **Ingestion**: page-level parsing, structure-aware chunking, configurable chunk size/overlap
- **Retrieval**: why BM25 beats vectors for exact-match queries (learning rate, sample size), why hybrid wins overall
- **Reranking**: cross-encoders are slower than bi-encoders but more accurate — used as a final filter on a small candidate set
- **Hallucination mitigation**: confidence thresholding, grounding policy in prompt, `INSUFFICIENT_EVIDENCE` fallback
- **Citations**: metadata-driven, not LLM-generated
- **Evaluation**: quantitative Recall@K and MRR rather than vibes-based testing
- **Scalability**: can add Pinecone/Weaviate instead of ChromaDB, swap Gemini for any LLM via `llm.py`