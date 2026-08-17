import sys
from pathlib import Path

# Ensure root workspace and backend directories are in sys.path
_root_dir = str(Path(__file__).resolve().parent.parent.parent)
_backend_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.core.middleware import RequestContextMiddleware
from app.core.exceptions import register_exception_handlers
from app.api.v1.router import api_v1_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(
        f"Starting {settings.APP_NAME} v{settings.APP_VERSION} [{settings.ENVIRONMENT}]",
        extra={"environment": settings.ENVIRONMENT, "version": settings.APP_VERSION},
    )
    try:
        from generation.service import get_rag_harness
        harness = get_rag_harness()
        _ = harness.process_rag_query("warmup")
        logger.info("Startup model and index pre-warming completed successfully.")
    except Exception as e:
        logger.warning(f"Startup model pre-warming skipped: {e}")
    yield
    # Shutdown
    logger.info(f"Shutting down {settings.APP_NAME}")


def create_app() -> FastAPI:
    """Application factory for Voice RAG FastAPI backend."""
    # Setup structured logger
    setup_logging(settings.LOG_LEVEL)

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Production-grade Voice-Enabled RAG Backend API",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else [settings.CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )

    # 2. Request ID & Structured Logging Middleware
    app.add_middleware(RequestContextMiddleware)

    # 3. Rate Limiting Middleware
    from app.core.middleware import RateLimitingMiddleware
    app.add_middleware(RateLimitingMiddleware, max_requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE)

    # 4. Centralized Exception Handlers
    register_exception_handlers(app)

    # 4. API Routers
    app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
