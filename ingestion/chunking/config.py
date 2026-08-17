from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StrategyConfig(BaseModel):
    chunk_size: int = 256
    overlap: int = 0
    min_chunk_tokens: int = 64
    max_chunk_tokens: int = 384
    target_chunk_tokens: int = 256
    semantic_threshold: float = 0.65


class ChunkingSettings(BaseSettings):
    DEFAULT_CHUNK_SIZE: int = 256
    DEFAULT_OVERLAP: int = 32
    MIN_CHUNK_TOKENS: int = 64
    MAX_CHUNK_TOKENS: int = 384
    TARGET_CHUNK_TOKENS: int = 256
    SEMANTIC_THRESHOLD: float = 0.65

    # Directory Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    CHUNKS_DIR: Path = DATA_DIR / "chunks"
    STATISTICS_DIR: Path = DATA_DIR / "statistics"

    # Strategy-specific configuration dictionaries
    STRATEGIES: Dict[str, Dict[str, Any]] = {
        "fixed": {
            "chunk_size": 256,
            "overlap": 0,
        },
        "overlap": {
            "chunk_size": 256,
            "overlap": 32,
        },
        "sentence": {
            "target_chunk_tokens": 256,
            "max_chunk_tokens": 384,
            "min_chunk_tokens": 32,
        },
        "paragraph": {
            "target_chunk_tokens": 256,
            "max_chunk_tokens": 384,
            "min_chunk_tokens": 32,
        },
        "semantic": {
            "target_chunk_tokens": 256,
            "min_chunk_tokens": 64,
            "max_chunk_tokens": 384,
            "semantic_threshold": 0.65,
        },
        "metadata": {
            "default_chunk_size": 256,
            "numeric_chunk_size": 128,
            "entity_chunk_size": 192,
            "description_chunk_size": 256,
        },
        "adaptive": {
            "target_chunk_tokens": 256,
            "min_chunk_tokens": 64,
            "max_chunk_tokens": 384,
            "short_passage_threshold": 64,
            "long_passage_threshold": 256,
        },
    }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


chunking_settings = ChunkingSettings()

# Ensure chunk output directories exist
for strategy_name in chunking_settings.STRATEGIES.keys():
    (chunking_settings.CHUNKS_DIR / strategy_name).mkdir(parents=True, exist_ok=True)
