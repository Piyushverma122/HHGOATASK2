import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Generator, Optional
import httpx
import pyarrow.parquet as pq
from ingestion.config import ingestion_settings
from ingestion.normalize import normalize_text

logger = logging.getLogger("voice_rag.ingestion.loader")

LANGUAGE_CODE_MAP = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd",
}

LANGUAGE_NAMES = {
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}


def get_hf_filename(language: str, split: str) -> str:
    """Resolve HF repository relative filename based on language and split."""
    lang_key = language.lower()
    if lang_key not in LANGUAGE_CODE_MAP:
        raise ValueError(
            f"Unsupported language '{language}'. Supported languages: {list(LANGUAGE_CODE_MAP.keys())}"
        )

    code3 = LANGUAGE_CODE_MAP[lang_key]
    split_key = split.lower()

    if split_key in ("train", "training"):
        return f"train/{code3}train.parquet"
    elif split_key in ("val", "validation", "dev"):
        return f"validation/{code3}val.parquet"
    else:
        raise ValueError(f"Unsupported split '{split}'. Supported splits: ['train', 'validation']")


def download_dataset_file(language: str, split: str) -> Path:
    """
    Downloads dataset file from Hugging Face directly into data/raw/
    with streaming chunks and progress tracking.
    """
    rel_filename = get_hf_filename(language, split)
    local_target = ingestion_settings.RAW_DIR / Path(rel_filename).name

    if local_target.exists() and local_target.stat().st_size > 1024 * 1024:
        logger.info(f"Using existing cached raw dataset file: {local_target} ({local_target.stat().st_size / (1024*1024):.2f} MB)")
        return local_target

    url = f"https://huggingface.co/datasets/{ingestion_settings.DATASET_NAME}/resolve/main/{rel_filename}"
    logger.info(f"Downloading {rel_filename} from {url} to {local_target}...")

    temp_target = local_target.with_suffix(".parquet.tmp")

    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to fetch dataset from {url}: HTTP {response.status_code}"
                )

            total_bytes = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024 * 2  # 2MB chunks

            with open(temp_target, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)

    if temp_target.exists():
        temp_target.rename(local_target)

    logger.info(f"Download complete: {local_target} ({local_target.stat().st_size / (1024*1024):.2f} MB)")
    return local_target


def stream_raw_records(
    language: str = "hi",
    split: str = "train",
    sample_size: Optional[int] = None,
    batch_size: int = 500,
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream records iteratively from the Parquet dataset using PyArrow.
    Loads data in small bounded batches of `batch_size` to maintain strict memory safety.
    """
    local_parquet_path = download_dataset_file(language, split)
    parquet_file = pq.ParquetFile(local_parquet_path)

    total_yielded = 0

    # Iteratively stream row batches
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        records = batch.to_pylist()
        for raw_item in records:
            yield raw_item
            total_yielded += 1
            if sample_size is not None and total_yielded >= sample_size:
                return


def raw_to_canonical(raw_record: Dict[str, Any], language: str) -> Dict[str, Any]:
    """
    Transform a raw MSMARCO-XI record into the canonical normalized internal schema.
    Extracts nested passages and generates deterministic stable IDs.
    """
    query_id = raw_record.get("query_id")
    query_raw = raw_record.get("query", "")
    answer_raw = raw_record.get("Answer", "")
    query_type = raw_record.get("query_type", "standard")
    source_lang = raw_record.get("source_lang", "eng_Latn")
    target_lang = raw_record.get("target_lang", f"{language}_Indic")

    # Normalize texts
    normalized_query = normalize_text(query_raw)
    normalized_answer = normalize_text(answer_raw) if answer_raw else ""

    # Passages
    raw_passages = raw_record.get("passages", {})
    trans_passages = raw_passages.get("Translated_passages", []) or []
    eng_passages = raw_passages.get("English_passages", []) or []
    is_selected_list = raw_passages.get("is_selected", []) or []

    canonical_passages = []
    num_passages = max(len(trans_passages), len(eng_passages), len(is_selected_list))

    for idx in range(num_passages):
        trans_p = trans_passages[idx] if idx < len(trans_passages) else ""
        eng_p = eng_passages[idx] if idx < len(eng_passages) else ""
        selected = bool(is_selected_list[idx]) if idx < len(is_selected_list) else False

        # Deterministic stable passage ID
        passage_id = f"{query_id}_{language}_{idx}"

        canonical_passages.append({
            "passage_id": passage_id,
            "passage_index": idx,
            "text": normalize_text(trans_p),
            "english_text": normalize_text(eng_p),
            "is_selected": selected,
        })

    record_id = f"{query_id}_{language}"

    return {
        "record_id": record_id,
        "query_id": query_id,
        "query": normalized_query,
        "answer": normalized_answer,
        "query_type": query_type,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "passages": canonical_passages,
        "original": {
            "eng_query": normalize_text(raw_record.get("Eng_Query", "")),
            "eng_answer": normalize_text(raw_record.get("Eng_Answer", "")),
        },
        "metadata": raw_record.get("meta", {}) if isinstance(raw_record.get("meta"), dict) else {},
    }
