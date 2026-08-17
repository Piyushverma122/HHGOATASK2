import io
import wave
import struct
import logging
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
from voice.errors import AudioValidationError

logger = logging.getLogger("voice_rag.audio.validator")

# Supported audio MIME types and extensions
SUPPORTED_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/webm",
    "audio/webm;codecs=opus",
    "audio/mp3",
    "audio/mpeg",
    "audio/ogg",
    "audio/ogg;codecs=opus",
    "audio/flac",
    "audio/x-m4a",
    "audio/m4a",
    "audio/mp4",
}

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".webm",
    ".mp3",
    ".ogg",
    ".flac",
    ".m4a",
}


class AudioValidator:
    """
    Validates audio files and byte buffers before transmission to STT services.
    Enforces format, size, duration, and stream integrity constraints.
    """

    def __init__(
        self,
        max_size_bytes: int = 10 * 1024 * 1024,  # 10 MB
        max_duration_seconds: float = 30.0,      # 30 seconds for REST STT
        min_duration_seconds: float = 0.2,       # Minimum audible audio
    ):
        self.max_size_bytes = max_size_bytes
        self.max_duration_seconds = max_duration_seconds
        self.min_duration_seconds = min_duration_seconds

    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate an audio file on disk."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise AudioValidationError(f"Audio file does not exist: {path}", code="FILE_NOT_FOUND")

        size_bytes = path.stat().st_size
        if size_bytes == 0:
            raise AudioValidationError("Audio file is empty (0 bytes)", code="EMPTY_AUDIO")

        if size_bytes > self.max_size_bytes:
            raise AudioValidationError(
                f"Audio file size ({size_bytes} bytes) exceeds maximum allowed ({self.max_size_bytes} bytes)",
                code="AUDIO_TOO_LARGE",
                details={"size_bytes": size_bytes, "max_size_bytes": self.max_size_bytes},
            )

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise AudioValidationError(
                f"Audio extension '{ext}' is not supported. Supported: {list(SUPPORTED_EXTENSIONS)}",
                code="UNSUPPORTED_AUDIO_FORMAT",
                details={"extension": ext, "supported": list(SUPPORTED_EXTENSIONS)},
            )

        with open(path, "rb") as f:
            audio_bytes = f.read()

        return self.validate_bytes(audio_bytes=audio_bytes, filename=path.name)

    def validate_bytes(
        self,
        audio_bytes: bytes,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate an in-memory audio byte buffer."""
        if not audio_bytes or len(audio_bytes) == 0:
            raise AudioValidationError("Audio buffer is empty (0 bytes)", code="EMPTY_AUDIO")

        size_bytes = len(audio_bytes)
        if size_bytes > self.max_size_bytes:
            raise AudioValidationError(
                f"Audio payload size ({size_bytes} bytes) exceeds limit ({self.max_size_bytes} bytes)",
                code="AUDIO_TOO_LARGE",
                details={"size_bytes": size_bytes, "max_size_bytes": self.max_size_bytes},
            )

        # Validate MIME type if provided
        if mime_type:
            cleaned_mime = mime_type.split(";")[0].strip().lower()
            if cleaned_mime not in [m.split(";")[0].strip().lower() for m in SUPPORTED_MIME_TYPES]:
                raise AudioValidationError(
                    f"MIME type '{mime_type}' is not supported.",
                    code="UNSUPPORTED_MIME_TYPE",
                    details={"mime_type": mime_type},
                )

        # Validate extension if filename provided
        ext = Path(filename).suffix.lower() if filename else ".wav"
        if ext and ext not in SUPPORTED_EXTENSIONS:
            raise AudioValidationError(
                f"Audio format '{ext}' is not supported.",
                code="UNSUPPORTED_AUDIO_FORMAT",
                details={"extension": ext},
            )

        # Inspect WAV header properties if WAV format
        duration_seconds, sample_rate, channels = self._inspect_wav_if_applicable(audio_bytes)

        if duration_seconds is not None:
            if duration_seconds > self.max_duration_seconds:
                raise AudioValidationError(
                    f"Audio duration ({duration_seconds:.2f}s) exceeds maximum allowed ({self.max_duration_seconds}s)",
                    code="AUDIO_TOO_LONG",
                    details={"duration_seconds": duration_seconds, "max_duration_seconds": self.max_duration_seconds},
                )
            if duration_seconds < self.min_duration_seconds:
                raise AudioValidationError(
                    f"Audio duration ({duration_seconds:.2f}s) is too short (min {self.min_duration_seconds}s)",
                    code="AUDIO_TOO_SHORT",
                    details={"duration_seconds": duration_seconds, "min_duration_seconds": self.min_duration_seconds},
                )

        return {
            "valid": True,
            "filename": filename or "recording.wav",
            "size_bytes": size_bytes,
            "duration_seconds": duration_seconds,
            "sample_rate": sample_rate,
            "channels": channels,
            "mime_type": mime_type or "audio/wav",
            "format": ext.lstrip("."),
        }

    def _inspect_wav_if_applicable(self, audio_bytes: bytes) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        """Safely parse standard WAV header without third-party dependencies."""
        if len(audio_bytes) < 44 or not audio_bytes.startswith(b"RIFF"):
            return None, None, None

        try:
            with io.BytesIO(audio_bytes) as bio:
                with wave.open(bio, "rb") as wf:
                    channels = wf.getnchannels()
                    sample_rate = wf.getframerate()
                    frames = wf.getnframes()
                    duration = frames / float(sample_rate) if sample_rate > 0 else 0.0
                    return round(duration, 3), sample_rate, channels
        except Exception as e:
            logger.debug(f"WAV header inspection fallback: {e}")
            return None, None, None
