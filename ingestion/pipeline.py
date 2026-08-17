import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import json
import logging
from typing import Optional, List, Dict, Any

from ingestion.config import ingestion_settings
from ingestion.dataset_loader import (
    stream_raw_records,
    raw_to_canonical,
    LANGUAGE_CODE_MAP,
    LANGUAGE_NAMES,
)
from ingestion.inspect_dataset import inspect_dataset_info
from ingestion.validate import validate_canonical_record
from ingestion.deduplicate import Deduplicator
from ingestion.statistics import IngestionStatsCollector
from ingestion.export import ParquetBatchWriter, export_jsonl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("voice_rag.ingestion.pipeline")


def run_ingestion_pipeline(
    language: str = "hi",
    split: str = "train",
    sample_size: Optional[int] = None,
    batch_size: int = 500,
    save_raw_sample: bool = True,
    output_parquet_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute end-to-end memory-efficient ingestion pipeline:
    1. Stream raw records in memory-safe batches.
    2. Normalize text and map to canonical schema.
    3. Validate record structure and field contents.
    4. Perform context-aware deduplication.
    5. Aggregate exact statistical distributions.
    6. Incrementally write Parquet and JSONL artifacts.
    """
    logger.info(
        f"Starting Ingestion Pipeline | Dataset: {ingestion_settings.DATASET_NAME} | "
        f"Language: {language} ({LANGUAGE_NAMES.get(language, 'Unknown')}) | "
        f"Split: {split} | Sample Limit: {sample_size or 'Full Dataset'}"
    )

    stats_collector = IngestionStatsCollector(
        dataset_name=ingestion_settings.DATASET_NAME,
        language=language,
        split=split,
        sample_size=sample_size,
    )
    deduplicator = Deduplicator()

    # Determine output destinations
    parquet_path = (
        output_parquet_path
        or ingestion_settings.PROCESSED_DIR / f"msmarco_xi_{language}_{split}.parquet"
    )
    raw_sample_path = (
        ingestion_settings.SAMPLES_DIR / f"raw_{language}_{split}.jsonl"
    )
    processed_sample_path = (
        ingestion_settings.PROCESSED_DIR / f"sample_{language}_{split}.jsonl"
    )
    validation_errors_path = (
        ingestion_settings.STATISTICS_DIR / "validation_errors.json"
    )

    parquet_writer = ParquetBatchWriter(parquet_path)

    raw_sample_records: List[Dict[str, Any]] = []
    processed_sample_records: List[Dict[str, Any]] = []
    all_validation_errors: List[Dict[str, Any]] = []
    current_batch: List[Dict[str, Any]] = []

    try:
        raw_stream = stream_raw_records(
            language=language,
            split=split,
            sample_size=sample_size,
            batch_size=batch_size,
        )

        for raw_item in raw_stream:
            # 1. Capture small raw sample for development inspection (up to 100)
            if save_raw_sample and len(raw_sample_records) < 100:
                raw_sample_records.append(raw_item)

            # 2. Canonical mapping & Normalization
            canonical_rec = raw_to_canonical(raw_item, language)

            # 3. Validation
            is_valid, validation_errors = validate_canonical_record(canonical_rec)
            if not is_valid:
                for err in validation_errors:
                    all_validation_errors.append(err.to_dict())

            # 4. Deduplication
            is_unique, dup_reason = deduplicator.process_record(canonical_rec)

            # 5. Record Statistics
            stats_collector.record_row(
                is_valid=is_valid,
                is_duplicate=not is_unique,
                record=canonical_rec if (is_valid and is_unique) else None,
                errors=validation_errors if not is_valid else None,
            )

            # 6. Accumulate valid & unique records for batch writing
            if is_valid and is_unique:
                current_batch.append(canonical_rec)
                if len(processed_sample_records) < 100:
                    processed_sample_records.append(canonical_rec)

                if len(current_batch) >= batch_size:
                    parquet_writer.write_batch(current_batch)
                    current_batch.clear()

        # Flush any remaining records in batch
        if current_batch:
            parquet_writer.write_batch(current_batch)
            current_batch.clear()

    finally:
        parquet_writer.close()

    # Export raw and processed samples
    if raw_sample_records:
        export_jsonl(raw_sample_records, raw_sample_path)
        logger.info(f"Saved raw sample ({len(raw_sample_records)} rows) -> {raw_sample_path}")

    if processed_sample_records:
        export_jsonl(processed_sample_records, processed_sample_path)
        logger.info(f"Saved processed sample ({len(processed_sample_records)} rows) -> {processed_sample_path}")

    # Export validation errors
    if all_validation_errors:
        with open(validation_errors_path, "w", encoding="utf-8") as f:
            json.dump(all_validation_errors, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved validation errors ({len(all_validation_errors)} events) -> {validation_errors_path}")

    # Export statistics reports
    json_stats, md_stats = stats_collector.export_reports(ingestion_settings.STATISTICS_DIR)
    logger.info(f"Saved JSON statistics -> {json_stats}")
    logger.info(f"Saved Markdown statistics -> {md_stats}")

    summary = stats_collector.finalize()
    logger.info(
        f"Pipeline Complete | Processed: {summary['rows']['total_processed']} | "
        f"Valid: {summary['rows']['valid']} | Duplicates: {summary['rows']['duplicates']} | "
        f"Passages: {summary['passages']['total_passages']} | "
        f"Throughput: {summary['metadata']['throughput_records_per_sec']} rec/s"
    )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="MSMARCO-XI Indic Dataset Ingestion & Preprocessing Pipeline"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect dataset schema, features, configurations, and remote splits",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="hi",
        help=f"Target Indic language code (default: hi). Supported: {list(LANGUAGE_CODE_MAP.keys())}",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "val"],
        help="Target split (default: train)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit number of records to ingest (e.g. 100, 10000). If omitted, processes all records.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Use memory-efficient streaming mode (default: True)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for incremental Parquet writing (default: 500)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output Parquet file path",
    )

    args = parser.parse_args()

    if args.inspect:
        inspect_dataset_info(
            dataset_name=ingestion_settings.DATASET_NAME,
            sample_lang=args.language,
        )
        return

    output_path = Path(args.output) if args.output else None
    run_ingestion_pipeline(
        language=args.language,
        split=args.split,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        output_parquet_path=output_path,
    )


if __name__ == "__main__":
    main()
