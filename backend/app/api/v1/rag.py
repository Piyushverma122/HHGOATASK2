import logging
from typing import Optional
from fastapi import APIRouter, Request
from app.utils.response import success_response
from generation.schemas import RAGQueryRequest, RAGQueryResponse
from generation.service import get_rag_harness
from app.core.config import settings

logger = logging.getLogger("voice_rag.api.rag")
router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/info", summary="Get RAG & LLM Provider Information")
async def get_rag_info(request: Request):
    """Return active LLM provider configuration, context budget, and guardrails status."""
    request_id = getattr(request.state, "request_id", None)
    harness = get_rag_harness()
    info = harness.llm_provider.get_model_info()

    return success_response(
        data={
            "provider_info": info,
            "max_context_chunks": settings.MAX_CONTEXT_CHUNKS,
            "max_context_tokens": settings.MAX_CONTEXT_TOKENS,
            "relevance_threshold": settings.RELEVANCE_SCORE_THRESHOLD,
            "grounding_threshold": settings.GROUNDING_SCORE_THRESHOLD,
            "guardrails_enabled": settings.ENABLE_GUARDRAILS,
        },
        request_id=request_id,
    )


@router.post("/query", summary="Grounded RAG Answer Generation (Text Query)")
async def rag_query(
    request: Request,
    payload: RAGQueryRequest,
):
    """
    Executes full Grounded RAG lifecycle:
    Query -> Pre-Guardrail -> Hybrid Retrieval -> Cross-Encoder -> Context Budget ->
    LLM Generation -> Grounding Verification -> Post-Guardrail -> Response.
    """
    request_id = getattr(request.state, "request_id", None)
    harness = get_rag_harness()

    result: RAGQueryResponse = harness.process_rag_query(
        query=payload.query,
        strategy=payload.strategy,
        top_k=payload.top_k,
        enable_reranking=payload.enable_reranking,
        request_id=request_id,
    )

    return success_response(
        data=result.model_dump(),
        message="RAG query executed successfully",
        request_id=request_id,
    )


@router.post("/inspect", summary="Inspect Hybrid Retrieval & Cross-Encoder Candidate Pipelines")
async def rag_inspect(
    request: Request,
    payload: RAGQueryRequest,
):
    """
    Executes full hybrid retrieval and reranking pipeline returning transparent intermediate candidate stages:
    Dense FAISS candidates, BM25 candidates, RRF fused rankings, Cross-Encoder scores, and final context.
    """
    request_id = getattr(request.state, "request_id", None)
    harness = get_rag_harness()

    retrieval_out = harness.retrieval_pipeline.retrieve(
        query=payload.query,
        strategy=payload.strategy,
        dense_k=20,
        bm25_k=20,
        hybrid_k=20,
        rerank_top_k=payload.top_k or 5,
        enable_reranking=payload.enable_reranking,
        parallel=True,
    )

    return success_response(
        data=retrieval_out,
        message="Retrieval candidates inspected successfully",
        request_id=request_id,
    )
