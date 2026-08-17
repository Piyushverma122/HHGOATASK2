from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional, Dict, Any
from voice.stt.models import STTResponse


class SpeechToTextProvider(ABC):
    """
    Abstract Base Class for Speech-to-Text Providers.
    All STT implementations (Sarvam, Mock, Future Providers) must implement this interface.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the STT provider."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model identifier."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Initialize provider connections, clients, or credentials."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider credentials and network endpoints are accessible."""
        pass

    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Return provider metadata, model name, and configuration."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> STTResponse:
        """Transcribe an audio file on disk."""
        pass

    @abstractmethod
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        mime_type: str = "audio/wav",
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> STTResponse:
        """Transcribe in-memory audio byte buffer."""
        pass
