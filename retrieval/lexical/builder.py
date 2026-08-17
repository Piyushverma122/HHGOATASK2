import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import json
import argparse
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from retrieval.lexical.bm25 import BM25Index
from retrieval.lexical.tokenizer import MultilingualTokenizer
from retrieval.faiss.persistence import METADATA_SCHEMA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("voice_rag.lexical.builder")

PRIMARY_STRATEGIES = ["fixed", "sentence", "adaptive"]
ALL_STRATEGIES = ["fixed", "overlap", "sentence", "paragraph", "semantic", "metadata", "adaptive"]


def build_bm25_strategy_index(
    strategy: str,
    chunks_parquet_path: Optional[Path] = None,
    output_index_dir: Optional[Path] = None,
    k1: float = 1.5,
    b: float = 0.75,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Builds and persists Okapi BM25 Index for a chunking strategy.
    """
    chunks_path = chunks_parquet_path or (BASE_DIR / "data" / "chunks" / strategy / "chunks_hi_validation.parquet")
    out_dir = output_index_dir or (BASE_DIR / "indexes" / "bm25" / strategy)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "bm25_index.pkl"
    meta_path = out_dir / "metadata.parquet"
    config_path = out_dir / "config.json"
    manifest_path = out_dir / "manifest.json"

    if index_path.exists() and not overwrite:
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            logger.info(f"BM25 index for '{strategy}' already exists ({manifest.get('total_documents', 0)} docs). Use --overwrite to rebuild.")
            return manifest

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks parquet not found at: {chunks_path}")

    logger.info(f"Loading chunks for BM25 from: {chunks_path}")
    chunks_df = pd.read_parquet(chunks_path)
    total_docs = len(chunks_df)

    start_time = time.perf_counter()
    tokenizer = MultilingualTokenizer(use_subwords=True)
    bm25 = BM25Index(k1=k1, b=b, tokenizer=tokenizer)

    texts = chunks_df["text"].tolist()
    doc_ids = list(range(total_docs))

    # Fit BM25
    bm25.fit(texts, doc_ids=doc_ids)

    # Save BM25 binary
    bm25.save(index_path)

    # Prepare and save Metadata Parquet
    metadata_records: List[Dict[str, Any]] = []
    for idx, row in chunks_df.iterrows():
        metadata_records.append({
            "vector_id": idx,
            "chunk_id": str(row.get("chunk_id", f"{strategy}_{idx}")),
            "record_id": str(row.get("record_id", "")),
            "query_id": int(row.get("query_id", 0)),
            "passage_id": str(row.get("passage_id", "")),
            "language": str(row.get("language", "hi")),
            "strategy": str(row.get("strategy", strategy)),
            "query_type": str(row.get("query_type", "standard")),
            "is_selected": bool(row.get("is_selected_passage", False)),
            "token_count": int(row.get("token_count", 0)),
            "text": str(row.get("text", "")),
            "metadata_json": str(row.get("metadata_json", "{}")),
        })

    meta_df = pd.DataFrame(metadata_records)
    table = pa.Table.from_pandas(meta_df, schema=METADATA_SCHEMA)
    pq.write_table(table, str(meta_path), compression="snappy")

    elapsed = round(time.perf_counter() - start_time, 3)
    rate = round(total_docs / elapsed, 1) if elapsed > 0 else 0.0

    # Save config
    config = {
        "strategy": strategy,
        "algorithm": "BM25Okapi",
        "k1": k1,
        "b": b,
        "tokenizer": "MultilingualTokenizer_subwords",
        "source_file": str(chunks_path),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Save manifest
    manifest = {
        "strategy": strategy,
        "algorithm": "BM25Okapi",
        "k1": k1,
        "b": b,
        "total_documents": total_docs,
        "total_vocabulary_terms": len(bm25.inverted_index),
        "avg_doc_len": round(bm25.avg_doc_len, 2),
        "build_time_seconds": elapsed,
        "docs_per_sec": rate,
        "source_file": str(chunks_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Successfully built BM25 Index for '{strategy.upper()}': {total_docs:,} docs in {elapsed}s ({rate} docs/s) -> {out_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="BM25 Lexical Index Builder")
    parser.add_argument(
        "--strategy",
        type=str,
        default="adaptive",
        choices=ALL_STRATEGIES,
        help="Chunking strategy to index",
    )
    parser.add_argument(
        "--all-primary",
        action="store_true",
        help="Build BM25 indexes for primary strategies (fixed, sentence, adaptive)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build BM25 indexes for ALL 7 strategies",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter (default: 1.5)",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter (default: 0.75)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing index",
    )

    args = parser.parse_args()

    if args.all:
        logger.info("Building BM25 Indexes for ALL 7 strategies...")
        for s in ALL_STRATEGIES:
            build_bm25_strategy_index(strategy=s, k1=args.k1, b=args.b, overwrite=args.overwrite)
    elif args.all_primary:
        logger.info(f"Building BM25 Indexes for PRIMARY strategies: {PRIMARY_STRATEGIES}...")
        for s in PRIMARY_STRATEGIES:
            build_bm25_strategy_index(strategy=s, k1=args.k1, b=args.b, overwrite=args.overwrite)
    else:
        build_bm25_strategy_index(strategy=args.strategy, k1=args.k1, b=args.b, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
