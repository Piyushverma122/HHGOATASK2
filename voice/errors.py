from typing import Optional, Dict, Any


class VoiceBaseError(Exception):
    """Base exception for all voice and STT errors."""

    def __init__(self, message: str, code: str = "VOICE_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class AudioValidationError(VoiceBaseError):
    """Raised when uploaded audio fails format, size, or duration validation."""

    def __init__(self, message: str, code: str = "INVALID_AUDIO", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=code, details=details)


class STTAuthenticationError(VoiceBaseError):
    """Raised when the STT provider rejects the API key or authentication credentials."""

    def __init__(self, message: str = "Invalid or missing STT API key", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code="STT_AUTH_ERROR", details=details)


class STTTimeoutError(VoiceBaseError):
    """Raised when the STT API call times out."""

    def __init__(self, message: str = "STT provider request timed out", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code="STT_TIMEOUT", details=details)


class STTRateLimitError(VoiceBaseError):
    """Raised when the STT provider rate limits requests (HTTP 429)."""

    def __init__(self, message: str = "STT provider rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code="STT_RATE_LIMIT", details=details)


class EmptyTranscriptError(VoiceBaseError):
    """Raised when STT succeeds but returns an empty or whitespace-only transcript."""

    def __init__(self, message: str = "STT generated an empty transcript", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code="EMPTY_TRANSCRIPT", details=details)


class STTProviderError(VoiceBaseError):
    """Raised when an unrecoverable 5xx or external provider error occurs."""

    def __init__(self, message: str, code: str = "STT_PROVIDER_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code=code, details=details)
