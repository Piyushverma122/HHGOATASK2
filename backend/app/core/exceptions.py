import logging
from typing import Any, Optional
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.utils.response import error_response

logger = logging.getLogger("voice_rag")


class AppException(Exception):
    """Base application exception."""

    def __init__(
        self,
        message: str,
        code: str = "BAD_REQUEST",
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[Any] = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details
        super().__init__(message)


def register_exception_handlers(app: FastAPI) -> None:
    """Register all custom exception handlers to the FastAPI app."""

    from voice.errors import (
        VoiceBaseError,
        AudioValidationError,
        STTAuthenticationError,
        STTRateLimitError,
        STTTimeoutError,
        EmptyTranscriptError,
    )

    @app.exception_handler(VoiceBaseError)
    async def voice_exception_handler(request: Request, exc: VoiceBaseError):
        request_id = getattr(request.state, "request_id", None)
        status_code = status.HTTP_400_BAD_REQUEST
        if isinstance(exc, STTAuthenticationError):
            status_code = status.HTTP_401_UNAUTHORIZED
        elif isinstance(exc, STTRateLimitError):
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
        elif isinstance(exc, STTTimeoutError):
            status_code = status.HTTP_504_GATEWAY_TIMEOUT
        elif isinstance(exc, EmptyTranscriptError):
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

        return error_response(
            code=exc.code,
            message=exc.message,
            status_code=status_code,
            request_id=request_id,
            details=exc.details,
        )

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        request_id = getattr(request.state, "request_id", None)
        return error_response(
            code=exc.code,
            message=exc.message,
            status_code=exc.status_code,
            request_id=request_id,
            details=exc.details,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        request_id = getattr(request.state, "request_id", None)
        code_map = {
            status.HTTP_404_NOT_FOUND: "NOT_FOUND",
            status.HTTP_401_UNAUTHORIZED: "UNAUTHORIZED",
            status.HTTP_403_FORBIDDEN: "FORBIDDEN",
            status.HTTP_405_METHOD_NOT_ALLOWED: "METHOD_NOT_ALLOWED",
            status.HTTP_429_TOO_MANY_REQUESTS: "RATE_LIMITED",
        }
        code = code_map.get(exc.status_code, "HTTP_ERROR")
        message = exc.detail if isinstance(exc.detail, str) else "HTTP Exception"

        return error_response(
            code=code,
            message=message,
            status_code=exc.status_code,
            request_id=request_id,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", None)
        # Format validation errors cleanly without leaking sensitive fields
        clean_errors = []
        for error in exc.errors():
            loc = " -> ".join([str(x) for x in error.get("loc", [])])
            clean_errors.append({"field": loc, "message": error.get("msg")})

        return error_response(
            code="VALIDATION_ERROR",
            message="Invalid request parameters",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            request_id=request_id,
            details=clean_errors,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.error(
            f"Unhandled internal server error on {request.url.path}: {str(exc)}",
            extra={"request_id": request_id, "path": request.url.path},
            exc_info=True,
        )
        # Never expose Python tracebacks to the client
        return error_response(
            code="INTERNAL_ERROR",
            message="An unexpected internal server error occurred",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request_id=request_id,
        )
