import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseSettings):
    DATASET_NAME: str = "ai4bharat/MSMARCO-XI"
    DATASET_LANGUAGE: str = "hi"
    DATASET_SPLIT: str = "train"
    DEFAULT_SAMPLE_SIZE: int = 1000
    BATCH_SIZE: int = 500

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    STATISTICS_DIR: Path = DATA_DIR / "statistics"
    SAMPLES_DIR: Path = DATA_DIR / "samples"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


ingestion_settings = IngestionSettings()

# Ensure directories exist
for directory in [
    ingestion_settings.RAW_DIR,
    ingestion_settings.PROCESSED_DIR,
    ingestion_settings.STATISTICS_DIR,
    ingestion_settings.SAMPLES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
