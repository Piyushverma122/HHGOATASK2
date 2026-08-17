import os
import logging
from typing import Optional
from generation.base import LLMProvider
from generation.model import OpenAICompatibleProvider, MockLLMProvider
try:
    from app.core.config import settings
except ImportError:
    import sys
    from pathlib import Path
    backend_dir = Path(__file__).resolve().parent.parent / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from app.core.config import settings

logger = logging.getLogger("voice_rag.generation.provider")

_GLOBAL_PROVIDER: Optional[LLMProvider] = None


def get_llm_provider(
    provider_type: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMProvider:
    """
    Factory function for obtaining configured LLM Provider instance.
    """
    global _GLOBAL_PROVIDER
    if _GLOBAL_PROVIDER is not None and not any([provider_type, api_key, model_name, base_url]):
        return _GLOBAL_PROVIDER

    p_type = provider_type or settings.LLM_PROVIDER
    p_key = api_key or settings.LLM_API_KEY or os.getenv("LLM_API_KEY", "")
    p_model = model_name or settings.LLM_MODEL
    p_base_url = base_url or settings.LLM_BASE_URL

    if p_key:
        logger.info(f"Initializing OpenAI-compatible LLM Provider with model: {p_model} at {p_base_url}")
        provider = OpenAICompatibleProvider(
            api_key=p_key,
            base_url=p_base_url,
            model=p_model,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
        )
    else:
        logger.info("No LLM_API_KEY found in configuration. Operating with deterministic MockLLMProvider.")
        provider = MockLLMProvider(model_name=p_model or "mock-grounded-rag-v1")

    if not any([provider_type, api_key, model_name, base_url]):
        _GLOBAL_PROVIDER = provider
    return provider
