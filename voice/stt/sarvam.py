import os
import time
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any
from dotenv import load_dotenv

import httpx

# Ensure environment variables are loaded
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()

from voice.stt.base import SpeechToTextProvider
from voice.stt.models import STTResponse
from voice.errors import (
    STTAuthenticationError,
    STTTimeoutError,
    STTRateLimitError,
    STTProviderError,
    EmptyTranscriptError,
    AudioValidationError,
)

logger = logging.getLogger("voice_rag.stt.sarvam")

# Mapping of standard language codes to Sarvam BCP-47 identifiers
SARVAM_LANGUAGE_MAP = {
    "hi": "hi-IN",
    "hindi": "hi-IN",
    "en": "en-IN",
    "english": "en-IN",
    "hinglish": "hi-IN",
    "hi-latn": "hi-IN",
    "bn": "bn-IN",
    "bengali": "bn-IN",
    "ta": "ta-IN",
    "tamil": "ta-IN",
    "te": "te-IN",
    "telugu": "te-IN",
    "mr": "mr-IN",
    "marathi": "mr-IN",
    "gu": "gu-IN",
    "gujarati": "gu-IN",
    "kn": "kn-IN",
    "kannada": "kn-IN",
    "ml": "ml-IN",
    "malayalam": "ml-IN",
    "pa": "pa-IN",
    "punjabi": "pa-IN",
    "od": "od-IN",
    "odia": "od-IN",
    "auto": "unknown",
    "unknown": "unknown",
}


class SarvamSTTProvider(SpeechToTextProvider):
    """
    Production Speech-to-Text provider for Sarvam AI Saaras v3 REST API.
    Endpoint: POST https://api.sarvam.ai/speech-to-text
    Headers: api-subscription-key: <KEY>
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.sarvam.ai",
        model: str = "saaras:v3",
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        mock_mode: bool = False,
    ):
        self._api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self._base_url = (base_url or "https://api.sarvam.ai").rstrip("/")
        self._model = model or "saaras:v3"
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.mock_mode = mock_mode or (not self._api_key)

    @property
    def provider_name(self) -> str:
        return "sarvam"

    @property
    def default_model(self) -> str:
        return self._model

    def load(self) -> None:
        """Verify API key and configuration readiness."""
        if not self._api_key and not self.mock_mode:
            logger.warning("SARVAM_API_KEY is not set. Operating in fallback mock mode.")
            self.mock_mode = True

    def is_available(self) -> bool:
        """Check if provider credentials exist or mock mode is active."""
        return bool(self._api_key) or self.mock_mode

    def get_provider_info(self) -> Dict[str, Any]:
        """Return provider configuration details."""
        return {
            "provider": self.provider_name,
            "model": self._model,
            "base_url": self._base_url,
            "endpoint": f"{self._base_url}/speech-to-text",
            "has_api_key": bool(self._api_key),
            "mock_mode": self.mock_mode,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }

    def _resolve_language_code(self, lang: Optional[str]) -> str:
        """Normalize user language parameter to Sarvam BCP-47 identifier."""
        if not lang:
            return "unknown"
        clean = lang.strip().lower()
        return SARVAM_LANGUAGE_MAP.get(clean, lang)

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> STTResponse:
        """Transcribe audio file from disk."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        with open(path, "rb") as f:
            audio_bytes = f.read()

        return self.transcribe_bytes(
            audio_bytes=audio_bytes,
            filename=path.name,
            mime_type=f"audio/{path.suffix.lstrip('.')}",
            language_code=language_code,
            model=model,
            request_id=request_id,
        )

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        mime_type: str = "audio/wav",
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> STTResponse:
        """
        Transcribe audio bytes using Sarvam REST API with retries and structured error handling.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            raise AudioValidationError("Audio bytes are empty", code="EMPTY_AUDIO")

        target_model = model or self._model
        resolved_lang = self._resolve_language_code(language_code)

        # Handle Mock Mode (for tests and offline development)
        if self.mock_mode:
            return self._execute_mock_transcription(
                audio_bytes=audio_bytes,
                filename=filename,
                language_code=resolved_lang,
                model=target_model,
                request_id=request_id,
            )

        endpoint_url = f"{self._base_url}/speech-to-text"
        headers = {
            "api-subscription-key": self._api_key,
        }
        if request_id:
            headers["X-Request-ID"] = request_id

        data = {
            "model": target_model,
            "language_code": resolved_lang,
            "with_timestamps": "false",
        }

        clean_mime = (mime_type or "audio/wav").split(";")[0].strip().lower()
        clean_filename = filename or "recording.wav"
        if "webm" in clean_mime:
            clean_mime = "audio/webm"
            if not clean_filename.endswith(".webm"):
                clean_filename = "recording.webm"
        elif "wav" in clean_mime:
            clean_mime = "audio/wav"
            if not clean_filename.endswith(".wav"):
                clean_filename = "recording.wav"
        elif "mp3" in clean_mime or "mpeg" in clean_mime:
            clean_mime = "audio/mp3"
            if not clean_filename.endswith(".mp3"):
                clean_filename = "recording.mp3"
        elif "ogg" in clean_mime:
            clean_mime = "audio/ogg"
            if not clean_filename.endswith(".ogg"):
                clean_filename = "recording.ogg"
        elif "flac" in clean_mime:
            clean_mime = "audio/flac"
            if not clean_filename.endswith(".flac"):
                clean_filename = "recording.flac"
        else:
            clean_mime = "audio/wav"

        files = {
            "file": (clean_filename, audio_bytes, clean_mime),
        }

        # Execute HTTP POST with exponential backoff for transient failures
        last_exception: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(endpoint_url, headers=headers, data=data, files=files)

                stt_latency_ms = (time.perf_counter() - t0) * 1000.0

                # 1. Successful 200 OK Response
                if response.status_code == 200:
                    resp_json = response.json()
                    transcript = resp_json.get("transcript", "").strip()
                    detected_lang = resp_json.get("language_code") or resolved_lang

                    if not transcript:
                        raise EmptyTranscriptError(
                            "Sarvam STT returned an empty transcript",
                            details={"raw_response": resp_json, "request_id": request_id},
                        )

                    return STTResponse(
                        transcript=transcript,
                        language_code=detected_lang,
                        provider="sarvam",
                        model=target_model,
                        duration_ms=0.0,
                        stt_latency_ms=round(stt_latency_ms, 2),
                        request_id=resp_json.get("request_id") or request_id,
                        raw_response=resp_json,
                    )

                # 2. Non-retryable Client Errors (4xx)
                if response.status_code in [401, 403]:
                    raise STTAuthenticationError(
                        f"Sarvam authentication failed (HTTP {response.status_code}). Check SARVAM_API_KEY.",
                        details={"status_code": response.status_code, "body": response.text},
                    )

                if response.status_code == 429:
                    raise STTRateLimitError(
                        "Sarvam API rate limit exceeded (HTTP 429).",
                        details={"status_code": response.status_code, "body": response.text},
                    )

                if 400 <= response.status_code < 500:
                    raise STTProviderError(
                        f"Sarvam rejected request (HTTP {response.status_code}): {response.text}",
                        code="STT_CLIENT_ERROR",
                        details={"status_code": response.status_code, "body": response.text},
                    )

                # 3. Retryable Server Errors (5xx)
                logger.warning(
                    f"Sarvam server error (HTTP {response.status_code}) on attempt {attempt}/{self.max_retries}."
                )
                last_exception = STTProviderError(
                    f"Sarvam server error (HTTP {response.status_code})",
                    code="STT_SERVER_ERROR",
                    details={"status_code": response.status_code, "body": response.text},
                )

            except (httpx.TimeoutException, httpx.ConnectTimeout) as e:
                logger.warning(f"Sarvam STT timeout on attempt {attempt}/{self.max_retries}: {e}")
                last_exception = STTTimeoutError(f"Sarvam STT request timed out after {self.timeout_seconds}s")
            except (httpx.NetworkError, httpx.RemoteProtocolError) as e:
                logger.warning(f"Sarvam STT network error on attempt {attempt}/{self.max_retries}: {e}")
                last_exception = STTProviderError(f"Sarvam network connection failed: {e}")
            except (EmptyTranscriptError, STTAuthenticationError, STTRateLimitError, AudioValidationError):
                raise
            except Exception as e:
                logger.error(f"Unexpected error calling Sarvam STT: {e}")
                last_exception = STTProviderError(f"Unexpected STT error: {e}")

            # Apply backoff before retry if retries remain
            if attempt < self.max_retries:
                sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                time.sleep(sleep_time)

        # If all retries exhausted, raise last exception
        if last_exception:
            raise last_exception
        raise STTProviderError("Failed to transcribe audio after retries")

    def _execute_mock_transcription(
        self,
        audio_bytes: bytes,
        filename: str,
        language_code: str,
        model: str,
        request_id: Optional[str],
    ) -> STTResponse:
        """Deterministic mock transcription for testing without live API keys."""
        time.sleep(0.015)  # Simulate 15ms network latency
        # Generate representative mock text based on language
        mock_texts = {
            "hi-IN": "भारत की राजधानी नई दिल्ली है।",
            "en-IN": "What is the capital of India?",
            "bn-IN": "ভারতের রাজধানী কী?",
            "ta-IN": "இந்தியாவின் தலைநகரம் எது?",
            "te-IN": "భారతదేశ రాజధాని ఏది?",
            "mr-IN": "भारताची राजधानी कोणती आहे?",
            "unknown": "भारत की राजधानी क्या है?",
        }
        transcript = mock_texts.get(language_code, "भारत की राजधानी क्या है?")
        return STTResponse(
            transcript=transcript,
            language_code=language_code if language_code != "unknown" else "hi-IN",
            provider="sarvam_mock",
            model=model,
            duration_ms=round(len(audio_bytes) / 32.0, 2),  # Approx for 16kHz 16-bit
            stt_latency_ms=15.2,
            request_id=request_id or "mock_req_123",
            raw_response={"mock": True, "note": "SARVAM_API_KEY not configured"},
        )
