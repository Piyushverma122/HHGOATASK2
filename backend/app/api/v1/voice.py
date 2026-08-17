import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, Request, status
from pydantic import BaseModel, Field

from app.utils.response import success_response
from voice.stt.service import get_stt_service
from generation.service import get_rag_harness
from retrieval.query.normalize import normalize_query
from retrieval.query.analyze import analyze_query

logger = logging.getLogger("voice_rag.api.voice")
router = APIRouter(prefix="/voice", tags=["voice"])


class TextQueryRequest(BaseModel):
    query: str = Field(description="Text search query")
    strategy: str = Field(default="adaptive", description="Chunking strategy index to search")
    top_k: int = Field(default=5, description="Number of final context chunks to retrieve")
    enable_reranking: bool = Field(default=True, description="Enable cross-encoder reranking")


@router.get("/info", summary="Get Voice & STT Provider Information")
async def get_voice_info(request: Request):
    """Return active STT provider configuration, constraints, and supported formats."""
    request_id = getattr(request.state, "request_id", None)
    stt_service = get_stt_service()
    info = stt_service.provider.get_provider_info()
    return success_response(
        data={
            "provider_info": info,
            "supported_mime_types": [
                "audio/wav", "audio/webm", "audio/mp3", "audio/ogg", "audio/flac"
            ],
            "max_file_size_mb": 10,
            "max_duration_seconds": 30,
        },
        request_id=request_id,
    )


@router.post("/transcribe", summary="Transcribe Audio to Text (Sarvam STT)")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Optional BCP-47 language code (e.g. hi-IN, en-IN)"),
    model: Optional[str] = Form(None, description="Optional STT model override"),
):
    """
    Accepts multipart/form-data audio file, validates, and returns transcription via Sarvam AI.
    """
    request_id = getattr(request.state, "request_id", None)
    audio_bytes = await file.read()

    stt_service = get_stt_service()
    result = stt_service.transcribe_audio_bytes(
        audio_bytes=audio_bytes,
        filename=file.filename or "recording.wav",
        mime_type=file.content_type or "audio/wav",
        language_code=language,
        model=model,
        request_id=request_id,
    )

    return success_response(
        data=result,
        message="Audio transcribed successfully",
        request_id=request_id,
    )


@router.post("/query", summary="End-to-End Voice Query Ingestion, Retrieval & Answer Generation")
async def voice_query(
    request: Request,
    file: UploadFile = File(..., description="Audio query recording"),
    strategy: str = Form("adaptive", description="Chunking strategy index"),
    language: Optional[str] = Form(None, description="Optional audio language code"),
    top_k: int = Form(5, description="Number of final context chunks to retrieve"),
    enable_reranking: bool = Form(True, description="Enable cross-encoder reranking"),
):
    """
    Executes full Voice Grounded RAG pipeline:
    Audio -> Sarvam STT -> NFC Normalization -> Query Analysis -> Pre-Guardrail ->
    Hybrid Retrieval & Cross-Encoder -> LLM Generation -> Grounding Verification -> Response.
    """
    request_id = getattr(request.state, "request_id", None)
    audio_bytes = await file.read()

    # Step 1: Transcribe via STT Service
    stt_service = get_stt_service()
    stt_res = stt_service.transcribe_audio_bytes(
        audio_bytes=audio_bytes,
        filename=file.filename or "recording.wav",
        mime_type=file.content_type or "audio/wav",
        language_code=language,
        request_id=request_id,
    )

    transcript = stt_res["transcript"]

    # Step 2: Execute Grounded RAG Harness
    rag_harness = get_rag_harness()
    rag_res = rag_harness.process_rag_query(
        query=transcript,
        strategy=strategy,
        top_k=top_k,
        enable_reranking=enable_reranking,
        request_id=request_id,
    )

    # Attach STT latency to breakdown
    lat = rag_res.latency
    lat.stt_ms = stt_res.get("latency", {}).get("stt_ms", 0.0)
    lat.total_ms = round(lat.total_ms + (lat.stt_ms or 0.0), 3)

    return success_response(
        data={
            "transcript": transcript,
            "stt_language": stt_res.get("language_code"),
            "stt_provider": stt_res.get("provider"),
            "query": rag_res.query,
            "normalized_query": rag_res.normalized_query,
            "detected_language": rag_res.detected_language,
            "strategy": rag_res.strategy,
            "answer": rag_res.answer,
            "grounded": rag_res.grounded,
            "confidence": rag_res.confidence,
            "citations": [c.model_dump() for c in rag_res.citations],
            "abstained": rag_res.abstained,
            "abstention_reason": rag_res.abstention_reason,
            "retrieved_chunks": rag_res.retrieved_chunks,
            "final_context": rag_res.retrieved_chunks,
            "latency": lat.model_dump(),
        },
        message="Voice query processed, transcribed, and answered successfully",
        request_id=request_id,
    )


@router.post("/text-query", summary="Text Query Fallback for Voice RAG")
async def text_query_fallback(
    request: Request,
    payload: TextQueryRequest,
):
    """
    Text search fallback executing through the complete Grounded RAG harness.
    """
    request_id = getattr(request.state, "request_id", None)
    rag_harness = get_rag_harness()

    rag_res = rag_harness.process_rag_query(
        query=payload.query,
        strategy=payload.strategy,
        top_k=payload.top_k,
        enable_reranking=payload.enable_reranking,
        request_id=request_id,
    )

    data = rag_res.model_dump()
    data["final_context"] = rag_res.retrieved_chunks

    return success_response(
        data=data,
        message="Text query answered successfully",
        request_id=request_id,
    )
