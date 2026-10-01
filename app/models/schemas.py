from pydantic import BaseModel
from typing import Optional, List


class CollectionCreate(BaseModel):
    name: str
    description: Optional[str] = ""


class CollectionResponse(BaseModel):
    name: str
    description: str
    document_count: int


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    collection: str
    num_pages: int
    num_chunks: int


class QuestionRequest(BaseModel):
    question: str
    collection: str
    top_k: int = 5
    retrieval_mode: str = "hybrid"  # "vector", "bm25", "hybrid"


class CitationSource(BaseModel):
    index: int
    document: str
    page: int
    section: str
    text_snippet: str


class QuestionResponse(BaseModel):
    success: bool
    question: str
    answer: str
    sources: List[CitationSource]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    confidence: str  # "high", "medium", "low", "insufficient"


class EvaluationRequest(BaseModel):
    collection: str
    dataset_path: str = "evaluation/questions.json"


class EvaluationResult(BaseModel):
    recall_at_5: float
    mrr: float
    avg_retrieval_latency_ms: float
    avg_total_latency_ms: float
    num_questions: int
