import time
import uuid
import logging
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("voice_rag")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to:
    1. Assign or extract X-Request-ID for every incoming request.
    2. Attach request_id to request.state.
    3. Measure request duration.
    4. Log structured request and response metrics.
    5. Append X-Request-ID and X-Process-Time headers to the response.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID")
        if not request_id or not request_id.strip():
            request_id = str(uuid.uuid4())

        request.state.request_id = request_id
        start_time = time.perf_counter()

        response: Response
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                f"Unhandled error processing {request.method} {request.url.path}: {str(exc)}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "duration_ms": duration_ms,
                },
                exc_info=True,
            )
            raise exc

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration_ms}ms"

        # Structured request logging
        logger.info(
            f"{request.method} {request.url.path} {response.status_code} - {duration_ms}ms",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )

        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    In-memory token-bucket / sliding window rate limiting for public endpoints.
    Protects against runaway external API costs while remaining transparent to tests.
    """

    def __init__(self, app, max_requests_per_minute: int = 120):
        super().__init__(app)
        self.max_requests_per_minute = max_requests_per_minute
        self._request_history: dict[str, list[float]] = {}
        self._protected_endpoints = {
            "/api/v1/voice/query",
            "/api/v1/voice/transcribe",
            "/api/v1/rag/query",
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        from app.core.config import settings
        from starlette.responses import JSONResponse

        # Skip rate limiting if disabled or in TEST mode
        if not settings.RATE_LIMIT_ENABLED or settings.is_test_mode():
            return await call_next(request)

        path = request.url.path
        if path in self._protected_endpoints and request.method == "POST":
            client_ip = request.client.host if request.client else "unknown"
            now = time.time()
            window_start = now - 60.0

            # Prune old timestamps
            history = self._request_history.setdefault(client_ip, [])
            self._request_history[client_ip] = [t for t in history if t > window_start]

            if len(self._request_history[client_ip]) >= self.max_requests_per_minute:
                request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
                logger.warning(
                    f"Rate limit exceeded for IP {client_ip} on {path}",
                    extra={"request_id": request_id, "client_ip": client_ip, "path": path},
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "code": "RATE_LIMIT_EXCEEDED",
                            "message": f"Rate limit exceeded ({self.max_requests_per_minute} requests/min). Please wait before retrying.",
                            "request_id": request_id,
                        }
                    },
                    headers={"Retry-After": "60", "X-Request-ID": request_id},
                )

            self._request_history[client_ip].append(now)

        return await call_next(request)
