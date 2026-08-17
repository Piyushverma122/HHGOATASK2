import time
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path

from voice.stt.service import STTService, get_stt_service
from voice.stt.models import VoicePipelineResult, STTLatencyBreakdown
from voice.errors import EmptyTranscriptError
from retrieval.query.normalize import normalize_query
from retrieval.query.analyze import analyze_query
from retrieval.pipeline import RetrievalPipeline

logger = logging.getLogger("voice_rag.voice.pipeline")


class VoicePipeline:
    """
    End-to-End Voice Input to Hybrid Retrieval Pipeline.
    Architecture:
        Audio -> Validation -> Preprocessing -> Sarvam STT -> Validate Transcript ->
        Query Normalization (NFC) -> Query Analysis -> Module 5 Hybrid Retrieval & Cross-Encoder
    """

    def __init__(
        self,
        stt_service: Optional[STTService] = None,
        retrieval_pipeline: Optional[RetrievalPipeline] = None,
    ):
        self.stt_service = stt_service or get_stt_service()
        self.retrieval_pipeline = retrieval_pipeline or RetrievalPipeline(strategy="adaptive")

    def process_voice_query(
        self,
        audio_bytes: bytes,
        filename: str = "recording.wav",
        mime_type: str = "audio/wav",
        language_code: Optional[str] = None,
        strategy: str = "adaptive",
        dense_k: int = 20,
        bm25_k: int = 20,
        hybrid_k: int = 20,
        rerank_top_k: int = 8,
        enable_reranking: bool = True,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute full Voice -> STT -> Normalization -> Analysis -> Retrieval pipeline.
        """
        pipeline_start = time.perf_counter()

        # 1. Audio Ingestion & STT Transcription
        stt_res = self.stt_service.transcribe_audio_bytes(
            audio_bytes=audio_bytes,
            filename=filename,
            mime_type=mime_type,
            language_code=language_code,
            request_id=request_id,
        )

        transcript = stt_res["transcript"]
        if not transcript or not transcript.strip():
            raise EmptyTranscriptError("Speech-to-Text produced an empty transcript.")

        # 2. Query Normalization (Reuses Module 5 NFC normalizer)
        norm_start = time.perf_counter()
        normalized = normalize_query(transcript)
        norm_ms = (time.perf_counter() - norm_start) * 1000.0

        # 3. Query Linguistic Analysis (Reuses Module 5 analyzer)
        analysis_start = time.perf_counter()
        analysis = analyze_query(normalized)
        analysis_ms = (time.perf_counter() - analysis_start) * 1000.0

        # 4. Hybrid Retrieval Pipeline Execution
        retrieval_start = time.perf_counter()
        retrieval_out = self.retrieval_pipeline.retrieve(
            query=normalized,
            strategy=strategy,
            dense_k=dense_k,
            bm25_k=bm25_k,
            hybrid_k=hybrid_k,
            rerank_top_k=rerank_top_k,
            enable_reranking=enable_reranking,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        total_ms = (time.perf_counter() - pipeline_start) * 1000.0

        # Assemble full latency telemetry
        stt_lat = stt_res["latency"]
        combined_latencies = {
            "validation_ms": stt_lat.get("validation_ms", 0.0),
            "preprocessing_ms": stt_lat.get("preprocessing_ms", 0.0),
            "stt_ms": stt_lat.get("stt_ms", 0.0),
            "normalization_ms": round(norm_ms, 3),
            "analysis_ms": round(analysis_ms, 3),
            "dense_ms": retrieval_out["latency"].get("dense_ms", 0.0),
            "bm25_ms": retrieval_out["latency"].get("bm25_ms", 0.0),
            "fusion_ms": retrieval_out["latency"].get("fusion_ms", 0.0),
            "rerank_ms": retrieval_out["latency"].get("rerank_ms", 0.0),
            "retrieval_total_ms": round(retrieval_ms, 3),
            "total_ms": round(total_ms, 3),
        }

        return {
            "transcript": transcript,
            "normalized_query": normalized,
            "detected_language": stt_res.get("language_code") or analysis.language,
            "stt_provider": stt_res.get("provider", "sarvam"),
            "stt_model": stt_res.get("model", "saaras:v3"),
            "strategy": strategy,
            "query_analysis": analysis.model_dump(),
            "dense_candidates": retrieval_out["dense_candidates"],
            "bm25_candidates": retrieval_out["bm25_candidates"],
            "fused_candidates": retrieval_out["fused_candidates"],
            "reranked_results": retrieval_out["reranked_results"],
            "final_context": retrieval_out["final_context"],
            "latency": combined_latencies,
            "request_id": request_id or stt_res.get("request_id", ""),
        }
