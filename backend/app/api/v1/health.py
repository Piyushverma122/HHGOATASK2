from typing import Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import settings

router = APIRouter(tags=["Health"])


class ProviderStatus(BaseModel):
    configured: bool = Field(..., description="Whether provider credentials are present")
    model: str = Field(..., description="Configured model identifier")
    mock_mode: bool = Field(False, description="Whether mock fallback is active")


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Overall health status")
    service: str = Field(..., description="Service identifier")
    version: str = Field(..., description="Service semantic version")
    environment: str = Field(..., description="Active environment mode (development/test/demo/production)")
    providers: Dict[str, Any] = Field(..., description="Sanitized provider configurations without secrets")
    guardrails: Dict[str, Any] = Field(..., description="Guardrail status")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check & Provider Status",
    description="Returns the health status, active environment mode, and sanitized provider configurations. Never reveals credentials or API keys.",
)
async def get_health():
    sarvam_configured = bool(settings.SARVAM_API_KEY and len(settings.SARVAM_API_KEY.strip()) > 0)
    llm_configured = bool(settings.LLM_API_KEY and len(settings.LLM_API_KEY.strip()) > 0)

    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "providers": {
            "sarvam": {
                "configured": sarvam_configured,
                "model": settings.SARVAM_STT_MODEL,
                "mock_mode": not sarvam_configured or settings.is_test_mode(),
            },
            "llm": {
                "configured": llm_configured,
                "provider": settings.LLM_PROVIDER,
                "model": settings.LLM_MODEL if llm_configured else "MockLLMProvider",
            },
        },
        "guardrails": {
            "enabled": settings.ENABLE_GUARDRAILS,
            "prompt_injection_defense": settings.ENABLE_PROMPT_INJECTION_DEFENSE,
            "max_context_chunks": settings.MAX_CONTEXT_CHUNKS,
        },
    }
