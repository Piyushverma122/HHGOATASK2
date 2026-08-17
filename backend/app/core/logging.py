import logging
import sys
import json
from datetime import datetime, timezone
from typing import Any, Dict


class StructuredJsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    Ensures no secrets (api_key, password, token, authorization) are leaked.
    """

    REDACTED_KEYS = {
        "api_key",
        "apikey",
        "authorization",
        "secret",
        "password",
        "token",
        "access_token",
        "sarvam_api_key",
        "llm_api_key",
    }

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: "[REDACTED]" if str(k).lower() in self.REDACTED_KEYS else self._sanitize(v)
                for k, v in value.items()
            }
        elif isinstance(value, list):
            return [self._sanitize(v) for v in value]
        return value

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include structured context if attached to record
        for attr in ("request_id", "method", "path", "status_code", "duration_ms"):
            val = getattr(record, attr, None)
            if val is not None:
                log_data[attr] = val

        # Include custom extra fields if present
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_data["extra"] = self._sanitize(record.extra_data)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("voice_rag")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False

    return logger


logger = setup_logging()
