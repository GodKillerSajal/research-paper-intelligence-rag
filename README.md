# Research Paper Intelligence & Integrity System

A production-grade **Retrieval-Augmented Generation (RAG)** and **Research Integrity Screening** platform for academic literature — designed as an end-to-end AI/ML engineering portfolio system.

---

## 🏗️ Architecture

```text
                      ┌──────────────────────────────────────────────┐
                      │             Streamlit Frontend UI            │
                      │                                              │
                      │  🏠 Query  │  📂 Collections  │  📊 Evaluation│
                      │         🔍 Research Integrity Analysis       │
                      └──────────────────────┬───────────────────────┘
                                             │ HTTP REST
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │              FastAPI Backend                 │
                      │                                              │
                      │  /query  /collections  /documents  /analysis │
                      └──────────────┬───────────────────────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
┌───────────────────────┐                           ┌─────────────────────┐
│  Ingestion Pipeline   │                           │   Query Pipeline    │
├───────────────────────┤                           ├─────────────────────┤
│ • PyMuPDF Page Parser │                           │ • Query Embedding   │
│ • Structure Chunking  │                           │ • Hybrid Retrieval  │
│   (300w, 50w overlap) │                           │   - BGE Vector      │
│ • Section Detection   │                           │   - BM25 Keyword    │
│ • BGE Embedding       │                           │ • Reciprocal Rank   │
│ • ChromaDB Upsert     │                           │   Fusion (RRF)      │
│ • BM25 Index Rebuild  │                           │ • Cross-Encoder     │
└──────────┬────────────┘                           │   Reranker          │
           │                                        │ • Grounded Context  │
           ▼                                        │ • Gemini LLM        │
    ┌─────────────┐                                 │ • Verified Citation │
    │  ChromaDB   │◄────────────────────────────────┤   & Confidence      │
    └─────────────┘                                 └─────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               🔬 Research Integrity Analysis Pipeline                   │
├─────────────────────────────────────────────────────────────────────────┤
│ • P-Value Clustering: Detects p-hacking around α = 0.05                 │
│ • Significance Rate: Detects outcome publication bias                   │
│ • Multiple Comparisons: Flags unadjusted multi-hypothesis testing       │
│ • Benford's Law Analysis: χ² goodness-of-fit test for fabricated data   │
│ • HARKing Detector: Identifies post-hoc hypotheses presented a priori   │
│ • Selective Outcome Reporting: Compares Methods vs Results variables    │
│ • Data Transparency: Scans for missing data handling & open datasets   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Capabilities

| Feature | Description | Status |
|---|---|:---:|
| **Structure-Aware Chunking** | Dynamic sliding-window chunking (default 300 words, 50 overlap) with section detection | ✅ |
| **Hybrid Search (Vector + BM25)** | Combines semantic BGE embeddings with BM25Okapi using Reciprocal Rank Fusion ($k=60$) | ✅ |
| **Cross-Encoder Reranking** | Re-scores top candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2` | ✅ |
| **Multi-Document Collections** | Manage discrete research paper collections with persistent ChromaDB storage | ✅ |
| **Grounded Citations** | Citations linked to real document metadata (filename, page, section, text snippet) | ✅ |
| **Hallucination Shield** | Strict prompt grounding + confidence scoring + `INSUFFICIENT_EVIDENCE` fallback | ✅ |
| **BM25 Persistence** | Rebuilds in-memory BM25 index across all ChromaDB collections automatically on startup | ✅ |
| **Research Integrity Analysis** | Automated screening for p-hacking, HARKing, selective reporting & Benford's Law | ✅ |
| **RAG Evaluation Suite** | Computes quantitative Recall@5, MRR, latency metrics on test sets | ✅ |
| **Containerization** | Production-ready `Dockerfile` and `docker-compose.yml` | ✅ |

---

## 📁 Project Structure

```text
retrieval-rag/
├── app/
│   ├── main.py                     # FastAPI application & lifespan management
│   │
│   ├── api/                        # REST API endpoints
│   │   ├── health.py               # GET /health & /
│   │   ├── collections.py          # CRUD /collections
│   │   ├── documents.py            # POST /documents/upload, GET /documents/{col}
│   │   ├── query.py                # POST /query
│   │   ├── evaluation.py           # POST /evaluation/run
│   │   └── analysis.py             # POST /analysis/integrity
│   │
│   ├── ingestion/                  # Document parsing and chunking
│   │   ├── parser.py               # PyMuPDF text & page extractor
│   │   ├── cleaner.py              # Text normalizer & cleaner
│   │   └── chunker.py              # Structure-aware chunking with section detector
│   │
│   ├── retrieval/                  # Hybrid retrieval stack
│   │   ├── embeddings.py           # BAAI/bge-small-en-v1.5 embedding wrapper
│   │   ├── vector_search.py        # ChromaDB client & semantic search
│   │   ├── bm25.py                 # BM25Okapi index & tokenization
│   │   ├── hybrid.py               # Reciprocal Rank Fusion (RRF)
│   │   └── reranker.py             # Cross-encoder reranker
│   │
│   ├── generation/                 # LLM generation & prompt formatting
│   │   ├── prompts.py              # Strict citation prompts & context blocks
│   │   └── llm.py                  # Gemini API wrapper with latency tracking
│   │
│   ├── analysis/                   # Paper integrity & data-manipulation detection
│   │   ├── extractor.py            # Regex extractors for p-values, N, & digits
│   │   ├── flags.py                # Statistical checks: p-hacking, Benford's law
│   │   ├── llm_checks.py           # Gemini checks: HARKing & selective reporting
│   │   └── pipeline.py             # Full document integrity orchestrator
│   │
│   ├── core/                       # Core orchestration
│   │   ├── collections.py          # JSON metadata storage for collections
│   │   └── pipeline.py             # Ingestion, query, and BM25 warm-up orchestration
│   │
│   ├── evaluation/                 # Retrieval benchmarking
│   │   ├── dataset.py              # Labeled Q&A dataset loader
│   │   ├── retrieval.py            # Recall@K and MRR computation
│   │   └── generation.py           # Heuristic relevance & faithfulness
│   │
│   └── models/
│       └── schemas.py              # Pydantic request/response models
│
├── frontend/
│   └── streamlit_app.py            # 4-page Streamlit application
│
├── evaluation/
│   └── questions.json              # Benchmark evaluation dataset
│
├── uploads/                        # Uploaded PDF document storage
├── data/                           # ChromaDB vector index & metadata
├── main.py                         # Root entrypoint
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Multi-container orchestration
├── requirements.txt                # Python package dependencies
└── .env                            # Environment variables (API keys)
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup

Clone the repository and create a Python 3.10+ virtual environment:

```bash
git clone https://github.com/GodKillerSajal/research-paper-intelligence-rag.git
cd research-paper-intelligence-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

> Get a free API key from [Google AI Studio](https://aistudio.google.com).

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Locally

**Start the FastAPI Backend (Port 8000):**
```bash
uvicorn app.main:app --reload --port 8000
```
Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Start the Streamlit UI (Port 8501):**
```bash
streamlit run frontend/streamlit_app.py --server.port 8501
```
Open in browser: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Docker Deployment

To spin up both the FastAPI backend and Streamlit frontend in isolated containers:

```bash
docker compose up --build
```

- FastAPI: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`

---

## ☁️ Cloud Deployment Options

### Option A: Streamlit Community Cloud (Frontend) + Render / Railway (Backend)
1. **Deploy Backend to Render or Railway**:
   - Create a Web Service pointing to your GitHub repo.
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Add Environment Variable: `GEMINI_API_KEY`
2. **Deploy Frontend to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io).
   - Select your repo and point to `frontend/streamlit_app.py`.
   - Update `API_URL` in `frontend/streamlit_app.py` (or set via `st.secrets`) to point to your deployed backend URL.

### Option B: Hugging Face Spaces (Docker)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) with SDK: **Docker**.
2. Push this repo to the Space.
3. Configure `GEMINI_API_KEY` under Space Settings -> Variables and Secrets.

---

## 🔌 API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status & uptime |
| `GET` | `/collections` | List all collections & document counts |
| `POST` | `/collections` | Create a new paper collection |
| `DELETE` | `/collections/{name}` | Delete collection & ChromaDB index |
| `POST` | `/documents/upload` | Upload & ingest a PDF document |
| `GET` | `/documents/{col}` | List ingested papers in a collection |
| `POST` | `/query` | Execute hybrid search & generate cited answer |
| `POST` | `/analysis/integrity` | Run statistical & AI research integrity audit |
| `GET` | `/analysis/documents/{col}`| List documents eligible for integrity audit |
| `POST` | `/evaluation/run` | Benchmark retrieval metrics (Recall@5, MRR) |

---

## 🎓 Technical Interview Highlights

When discussing this architecture in ML engineering and applied AI interviews:

1. **Why Hybrid Retrieval?**
   - Pure semantic search often struggles with specific entity names, formulas, sample sizes, and acronyms.
   - BM25 provides exact token matching, while BGE provides semantic understanding.
   - Reciprocal Rank Fusion (RRF) standardizes rankings without needing calibrated score distributions.

2. **Why Cross-Encoder Reranking?**
   - Bi-encoders embed query and passages independently (efficient candidate generation).
   - Cross-encoders evaluate full query-document cross-attention (higher fidelity, applied only on top 20 candidates for minimal latency impact).

3. **Hallucination Prevention**:
   - Explicit prompt bounding prevents external training memory leaks.
   - Per-chunk verified source attribution metadata ensures citations cannot be made up by the LLM.
   - Confidence thresholding enables graceful `INSUFFICIENT_EVIDENCE` degradation.

4. **Integrity Screening Pipeline**:
   - Combines statistical anomaly detection (p-value distributions, Benford's Law on empirical measurements) with LLM-based structural comprehension (detecting disparities between Methods and Results).