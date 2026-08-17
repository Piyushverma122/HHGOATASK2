from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str = Field(description="Deterministic chunk ID from retrieval")
    source_passage_id: Optional[str] = Field(default=None, description="MSMARCO-XI passage ID")
    relevance_score: Optional[float] = Field(default=None, description="Reranker or retrieval relevance score")
    snippet: Optional[str] = Field(default=None, description="Exact passage snippet supporting the claim")


class AnswerResponse(BaseModel):
    answer: str = Field(description="Factual answer grounded in retrieved context")
    language: str = Field(default="hi", description="Language of generated answer")
    grounded: bool = Field(default=True, description="Whether answer passed grounding verification")
    confidence: float = Field(default=0.90, description="Overall confidence score (0.0 to 1.0)")
    citations: List[Citation] = Field(default_factory=list, description="Grounding source citations")
    abstained: bool = Field(default=False, description="Whether the system abstained from answering")
    abstention_reason: Optional[str] = Field(default=None, description="Reason if abstained")


class RAGLatencyBreakdown(BaseModel):
    stt_ms: Optional[float] = Field(default=None, description="Speech-to-Text latency if voice query")
    normalization_ms: float = Field(default=0.0, description="Query NFC normalization latency")
    analysis_ms: float = Field(default=0.0, description="Query linguistic analysis latency")
    guardrail_pre_ms: float = Field(default=0.0, description="Input and context pre-guardrail check latency")
    dense_retrieval_ms: float = Field(default=0.0, description="Dense FAISS vector search latency")
    bm25_retrieval_ms: float = Field(default=0.0, description="Okapi BM25 sparse search latency")
    reranking_ms: float = Field(default=0.0, description="Cross-Encoder reranking latency")
    retrieval_total_ms: float = Field(default=0.0, description="Total retrieval pipeline latency")
    context_prep_ms: float = Field(default=0.0, description="Context budgeting and formatting latency")
    generation_ms: float = Field(default=0.0, description="LLM inference latency")
    verification_ms: float = Field(default=0.0, description="Grounding and citation verification latency")
    total_ms: float = Field(default=0.0, description="Complete end-to-end RAG latency")


class RAGQueryRequest(BaseModel):
    query: str = Field(description="Search or question query string")
    strategy: str = Field(default="adaptive", description="Chunking strategy to retrieve from")
    top_k: int = Field(default=5, description="Number of context chunks to supply to LLM")
    enable_reranking: bool = Field(default=True, description="Enable Cross-Encoder reranking")


class RAGQueryResponse(BaseModel):
    query: str
    normalized_query: str
    detected_language: str
    strategy: str
    answer: str
    grounded: bool
    confidence: float
    citations: List[Citation]
    abstained: bool
    abstention_reason: Optional[str] = None
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    latency: RAGLatencyBreakdown
    request_id: Optional[str] = None
