import os
from pathlib import Path
from typing import List, Union
from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load workspace root .env if present
_root_env = Path(__file__).resolve().parent.parent.parent.parent / ".env"
if _root_env.exists():
    load_dotenv(dotenv_path=_root_env)
else:
    load_dotenv()


class Settings(BaseSettings):
    APP_NAME: str = "voice-rag"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_PREFIX: str = "/api/v1"

    # CORS configuration
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Sarvam AI STT configuration
    SARVAM_API_KEY: str = ""
    SARVAM_BASE_URL: str = "https://api.sarvam.ai"
    SARVAM_STT_MODEL: str = "saaras:v3"
    STT_TIMEOUT_SECONDS: float = 15.0
    STT_MAX_RETRIES: int = 3
    STT_RETRY_BACKOFF_FACTOR: float = 0.5
    DEFAULT_STT_LANGUAGE: str = "hi-IN"

    # Audio validation constraints
    MAX_AUDIO_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    MAX_AUDIO_DURATION_SECONDS: float = 30.0      # 30s limit for REST STT
    MIN_AUDIO_DURATION_SECONDS: float = 0.2

    # LLM and Generation configuration
    LLM_PROVIDER: str = "openai_compatible"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "meta-llama/llama-3.3-70b-instruct"
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    LLM_TIMEOUT_SECONDS: float = 20.0
    LLM_MAX_RETRIES: int = 2

    # Context & Grounding Budget
    MAX_CONTEXT_CHUNKS: int = 5
    MAX_CONTEXT_TOKENS: int = 2048
    RELEVANCE_SCORE_THRESHOLD: float = 0.01
    GROUNDING_SCORE_THRESHOLD: float = 0.65
    MAX_REGENERATION_ATTEMPTS: int = 1

    # Guardrails
    ENABLE_GUARDRAILS: bool = True
    MAX_QUERY_CHARS: int = 500
    ENABLE_PROMPT_INJECTION_DEFENSE: bool = True
    ENABLE_GENERATION_CACHE: bool = True

    # Vector & Retrieval
    VECTOR_INDEX_PATH: str = "./indexes"
    TOP_K: int = 10
    RERANK_K: int = 5

    # Rate limiting & Security
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 120

    # Logging
    LOG_LEVEL: str = "INFO"

    def is_test_mode(self) -> bool:
        return self.ENVIRONMENT.lower() == "test"

    def is_demo_mode(self) -> bool:
        return self.ENVIRONMENT.lower() == "demo"

    def is_production_mode(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    def is_development_mode(self) -> bool:
        return self.ENVIRONMENT.lower() in ("development", "dev")

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )


settings = Settings()
