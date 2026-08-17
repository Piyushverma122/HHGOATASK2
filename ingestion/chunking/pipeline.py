import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
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
from typing import List, Dict, Any, Optional
import pyarrow as pa
import pyarrow.parquet as pq

from ingestion.chunking.config import chunking_settings
from ingestion.chunking.factory import ChunkerFactory
from ingestion.chunking.models import Chunk
from ingestion.chunking.statistics import ChunkStatsCollector, generate_comparison_report
from ingestion.chunking.validate_quality import validate_chunk_quality

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("voice_rag.chunking.pipeline")

# PyArrow Schema for Chunks
CHUNK_PYARROW_SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("record_id", pa.string()),
    ("query_id", pa.int64()),
    ("passage_id", pa.string()),
    ("text", pa.string()),
    ("strategy", pa.string()),
    ("language", pa.string()),
    ("source_lang", pa.string()),
    ("target_lang", pa.string()),
    ("query_type", pa.string()),
    ("chunk_index", pa.int32()),
    ("start_position", pa.int32()),
    ("end_position", pa.int32()),
    ("token_count", pa.int32()),
    ("character_count", pa.int32()),
    ("is_selected_passage", pa.bool_()),
    ("metadata_json", pa.string()),
])


def write_chunks_to_parquet(chunks: List[Chunk], output_path: Path):
    """Write list of Chunk objects to Parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not chunks:
        return

    data = {
        "chunk_id": [c.chunk_id for c in chunks],
        "record_id": [c.record_id for c in chunks],
        "query_id": [c.query_id for c in chunks],
        "passage_id": [c.passage_id for c in chunks],
        "text": [c.text for c in chunks],
        "strategy": [c.strategy for c in chunks],
        "language": [c.language for c in chunks],
        "source_lang": [c.source_lang for c in chunks],
        "target_lang": [c.target_lang for c in chunks],
        "query_type": [c.query_type for c in chunks],
        "chunk_index": [c.chunk_index for c in chunks],
        "start_position": [c.start_position for c in chunks],
        "end_position": [c.end_position for c in chunks],
        "token_count": [c.token_count for c in chunks],
        "character_count": [c.character_count for c in chunks],
        "is_selected_passage": [c.is_selected_passage for c in chunks],
        "metadata_json": [json.dumps(c.metadata, ensure_ascii=False) for c in chunks],
    }

    table = pa.Table.from_pydict(data, schema=CHUNK_PYARROW_SCHEMA)
    pq.write_table(table, str(output_path), compression="snappy")


def read_records_from_parquet(
    parquet_path: Path,
    sample_size: Optional[int] = None,
    batch_size: int = 500,
) -> List[Dict[str, Any]]:
    """Load canonical records from processed Parquet file up to sample_size."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input dataset not found at {parquet_path}")

    logger.info(f"Reading dataset: {parquet_path} (Sample limit: {sample_size or 'Full'})")
    parquet_file = pq.ParquetFile(parquet_path)

    records: List[Dict[str, Any]] = []
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        pylist = batch.to_pylist()
        for item in pylist:
            records.append(item)
            if sample_size is not None and len(records) >= sample_size:
                return records

    return records


def run_strategy_chunking(
    strategy_name: str,
    records: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
) -> tuple[List[Chunk], Dict[str, Any]]:
    """
    Run a single chunking strategy across a list of canonical query records.
    Returns generated chunks and performance/token statistics.
    """
    chunker = ChunkerFactory.create(strategy_name)
    stats_collector = ChunkStatsCollector(strategy_name, sample_size=len(records))

    all_chunks: List[Chunk] = []

    for rec in records:
        passages = rec.get("passages", [])
        query_chunk_count = 0

        for p in passages:
            p_chunks = chunker.chunk_passage(p, rec)
            stats_collector.record_passage_result(p_chunks)
            all_chunks.extend(p_chunks)
            query_chunk_count += len(p_chunks)

        stats_collector.record_query_result(query_chunk_count)

    stats = stats_collector.finalize()

    # Save Parquet & JSONL artifacts
    target_dir = output_dir or (chunking_settings.CHUNKS_DIR / strategy_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    parquet_out = target_dir / f"chunks_hi_validation.parquet"
    write_chunks_to_parquet(all_chunks, parquet_out)

    # Save sample JSONL
    sample_jsonl = target_dir / "sample_chunks.jsonl"
    with open(sample_jsonl, "w", encoding="utf-8") as f:
        for c in all_chunks[:100]:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    logger.info(
        f"Strategy: {strategy_name.upper():<10} | Records: {stats['records_processed']} | "
        f"Chunks: {stats['chunks_generated']:,} | Avg Tokens: {stats['tokens']['average']} | "
        f"Time: {stats['performance']['elapsed_seconds']}s | Output: {parquet_out.name}"
    )

    return all_chunks, stats


def main():
    parser = argparse.ArgumentParser(description="Advanced Multi-Strategy Chunking Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default=str(chunking_settings.PROCESSED_DIR / "msmarco_xi_hi_validation.parquet"),
        help="Input processed Parquet dataset path",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=ChunkerFactory.get_available_strategies(),
        default="adaptive",
        help="Chunking strategy to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run and benchmark all 7 chunking strategies sequentially",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Sample size of records to chunk (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for dataset reading",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    records = read_records_from_parquet(
        parquet_path=input_path,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
    )

    if args.all:
        logger.info(f"Executing ALL 7 Chunking Strategies on {len(records)} records...")
        all_stats: List[Dict[str, Any]] = []
        quality_reports: Dict[str, Any] = {}

        for strat in ChunkerFactory.get_available_strategies():
            chunks, stats = run_strategy_chunking(strat, records)
            all_stats.append(stats)

            # Quality validation
            is_valid, q_rep = validate_chunk_quality(chunks)
            quality_reports[strat] = q_rep

        # Save comparison and quality reports
        json_rep, md_rep = generate_comparison_report(
            all_stats,
            chunking_settings.STATISTICS_DIR,
        )
        logger.info(f"Saved chunk comparison report -> {json_rep}")
        logger.info(f"Saved chunk comparison markdown -> {md_rep}")

        quality_json = chunking_settings.STATISTICS_DIR / "chunk_quality_report.json"
        with open(quality_json, "w", encoding="utf-8") as f:
            json.dump(quality_reports, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved chunk quality report -> {quality_json}")

    else:
        chunks, stats = run_strategy_chunking(args.strategy, records)
        is_valid, q_rep = validate_chunk_quality(chunks)
        logger.info(f"Quality Check for '{args.strategy}': Valid = {is_valid} ({len(chunks)} chunks verified)")


if __name__ == "__main__":
    main()
