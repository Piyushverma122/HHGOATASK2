from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class STTLatencyBreakdown(BaseModel):
    """Detailed timing metrics across the voice & STT pipeline stages."""
    validation_ms: float = Field(default=0.0, description="Audio validation time in milliseconds")
    preprocessing_ms: float = Field(default=0.0, description="Audio preprocessing time in milliseconds")
    network_ms: float = Field(default=0.0, description="HTTP network transit time in milliseconds")
    stt_ms: float = Field(default=0.0, description="STT inference and response parsing time in milliseconds")
    normalization_ms: float = Field(default=0.0, description="Query text normalization time in milliseconds")
    analysis_ms: float = Field(default=0.0, description="Query linguistic analysis time in milliseconds")
    total_ms: float = Field(default=0.0, description="Total end-to-end voice pipeline time in milliseconds")


class STTResponse(BaseModel):
    """Standardized response envelope from Speech-to-Text providers."""
    transcript: str = Field(description="Transcribed text output")
    language_code: Optional[str] = Field(default=None, description="BCP-47 language code (e.g. hi-IN, en-IN)")
    provider: str = Field(description="STT provider name (e.g. sarvam)")
    model: str = Field(description="STT model identifier (e.g. saaras:v3)")
    duration_ms: float = Field(default=0.0, description="Audio file duration in milliseconds")
    stt_latency_ms: float = Field(default=0.0, description="Provider inference latency in milliseconds")
    request_id: Optional[str] = Field(default=None, description="Request tracking ID")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Raw provider metadata")


class VoicePipelineResult(BaseModel):
    """Result of voice query ingestion including transcript and retrieval candidates."""
    query: str = Field(description="Original transcription text")
    normalized_query: str = Field(description="NFC normalized query string")
    detected_language: str = Field(description="Detected language code")
    query_analysis: Dict[str, Any] = Field(description="Linguistic taxonomy and entities")
    retrieval_results: Optional[Dict[str, Any]] = Field(default=None, description="Module 5 retrieval output")
    latency: STTLatencyBreakdown = Field(description="Granular latency breakdown across all pipeline stages")
    request_id: str = Field(description="X-Request-ID identifier")
