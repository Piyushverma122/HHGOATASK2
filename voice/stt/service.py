import time
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path

from voice.stt.base import SpeechToTextProvider
from voice.stt.sarvam import SarvamSTTProvider
from voice.stt.models import STTResponse, STTLatencyBreakdown
from voice.audio.validator import AudioValidator
from voice.audio.preprocess import AudioPreprocessor

logger = logging.getLogger("voice_rag.stt.service")

_GLOBAL_STT_SERVICE: Optional["STTService"] = None


class STTService:
    """
    Centralized Speech-to-Text Service.
    Coordinates audio validation, preprocessing, provider execution, and latency metrics.
    """

    def __init__(
        self,
        provider: Optional[SpeechToTextProvider] = None,
        validator: Optional[AudioValidator] = None,
        preprocessor: Optional[AudioPreprocessor] = None,
    ):
        self.provider = provider or SarvamSTTProvider()
        self.validator = validator or AudioValidator()
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.provider.load()

    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "recording.wav",
        mime_type: str = "audio/wav",
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full STT execution lifecycle with granular timing breakdown.
        """
        total_start = time.perf_counter()

        # 1. Audio Validation
        val_start = time.perf_counter()
        validation_info = self.validator.validate_bytes(
            audio_bytes=audio_bytes,
            filename=filename,
            mime_type=mime_type,
        )
        val_ms = (time.perf_counter() - val_start) * 1000.0

        # 2. Audio Preprocessing
        prep_start = time.perf_counter()
        processed_bytes, prep_meta = self.preprocessor.preprocess_audio(audio_bytes, filename=filename)
        prep_ms = (time.perf_counter() - prep_start) * 1000.0

        # 3. STT Provider Execution
        stt_start = time.perf_counter()
        stt_response = self.provider.transcribe_bytes(
            audio_bytes=processed_bytes,
            filename=filename,
            mime_type=mime_type,
            language_code=language_code,
            model=model,
            request_id=request_id,
        )
        stt_ms = (time.perf_counter() - stt_start) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0

        latencies = STTLatencyBreakdown(
            validation_ms=round(val_ms, 3),
            preprocessing_ms=round(prep_ms, 3),
            network_ms=round(stt_response.stt_latency_ms, 3),
            stt_ms=round(stt_ms, 3),
            total_ms=round(total_ms, 3),
        )

        return {
            "transcript": stt_response.transcript,
            "language_code": stt_response.language_code,
            "provider": stt_response.provider,
            "model": stt_response.model,
            "duration_ms": validation_info.get("duration_seconds", 0.0) * 1000.0 if validation_info.get("duration_seconds") else 0.0,
            "audio_metadata": validation_info,
            "latency": latencies.model_dump(),
            "request_id": stt_response.request_id or request_id,
        }

    def transcribe_file(
        self,
        file_path: Union[str, Path],
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience method for transcribing audio file on disk."""
        path = Path(file_path)
        with open(path, "rb") as f:
            audio_bytes = f.read()
        return self.transcribe_audio_bytes(
            audio_bytes=audio_bytes,
            filename=path.name,
            mime_type=f"audio/{path.suffix.lstrip('.')}",
            language_code=language_code,
            model=model,
            request_id=request_id,
        )


def get_stt_service(api_key: Optional[str] = None) -> STTService:
    """Singleton getter for the global STT service."""
    global _GLOBAL_STT_SERVICE
    if _GLOBAL_STT_SERVICE is None:
        provider = SarvamSTTProvider(api_key=api_key)
        _GLOBAL_STT_SERVICE = STTService(provider=provider)
    return _GLOBAL_STT_SERVICE
