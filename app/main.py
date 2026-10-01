from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.collections import router as collections_router
from app.api.documents import router as documents_router
from app.api.query import router as query_router
from app.api.evaluation import router as evaluation_router
from app.api.analysis import router as analysis_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: rebuild BM25 indexes from persisted ChromaDB data."""
    from app.core.pipeline import warm_up_bm25
    warm_up_bm25()
    yield


app = FastAPI(
    title="Research Paper Intelligence System",
    description="Production-grade RAG system for research papers.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(collections_router)
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(evaluation_router)
app.include_router(analysis_router)
