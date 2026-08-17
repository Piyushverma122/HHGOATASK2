import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import time
import json
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

from retrieval.embeddings.provider import EmbeddingProviderFactory
from retrieval.faiss.index import FaissVectorStore
from retrieval.faiss.persistence import IndexPersistenceManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("voice_rag.faiss.builder")

PRIMARY_STRATEGIES = ["fixed", "sentence", "adaptive"]
ALL_STRATEGIES = ["fixed", "overlap", "sentence", "paragraph", "semantic", "metadata", "adaptive"]


def build_strategy_index(
    strategy: str,
    chunks_parquet_path: Optional[Path] = None,
    output_index_dir: Optional[Path] = None,
    batch_size: int = 32,
    index_type: str = "flat",
    model_name: str = "multilingual-dense-e5",
    dimension: int = 384,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Build a complete FAISS vector index and metadata store for a chunking strategy.
    """
    chunks_path = chunks_parquet_path or (BASE_DIR / "data" / "chunks" / strategy / "chunks_hi_validation.parquet")
    out_dir = output_index_dir or (BASE_DIR / "indexes" / strategy)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks parquet not found at: {chunks_path}")

    persistence = IndexPersistenceManager(out_dir)

    if persistence.faiss_path.exists() and not overwrite:
        manifest = persistence.load_manifest()
        if manifest:
            logger.info(f"Index for '{strategy}' already exists ({manifest.get('total_vectors', 0)} vectors). Use --overwrite to rebuild.")
            return manifest

    logger.info(f"Loading chunks from: {chunks_path}")
    chunks_df = pd.read_parquet(chunks_path)
    total_chunks = len(chunks_df)
    logger.info(f"Total chunks to index for '{strategy}': {total_chunks:,}")

    # Initialize Embedder & Vector Store
    embedder = EmbeddingProviderFactory.get_provider(model_name=model_name, dimension=dimension)
    vector_store = FaissVectorStore(dimension=dimension, index_type=index_type)

    start_time = time.perf_counter()
    metadata_records: List[Dict[str, Any]] = []

    # Batch embedding and FAISS insertion
    for start_idx in range(0, total_chunks, batch_size):
        end_idx = min(start_idx + batch_size, total_chunks)
        batch_df = chunks_df.iloc[start_idx:end_idx]

        texts = batch_df["text"].tolist()
        # Batch generate L2-normalized embeddings
        vectors = embedder.embed_batch(texts, batch_size=batch_size)

        # Generate sequential 64-bit integer vector IDs: [0, 1, 2, ...]
        vector_ids = np.arange(start_idx, end_idx, dtype=np.int64)

        vector_store.add_vectors(vectors, vector_ids)

        for local_i, (_, row) in enumerate(batch_df.iterrows()):
            v_id = start_idx + local_i
            metadata_records.append({
                "vector_id": v_id,
                "chunk_id": str(row.get("chunk_id", f"{strategy}_{v_id}")),
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

        if (end_idx % 2000 == 0) or (end_idx == total_chunks):
            elapsed = time.perf_counter() - start_time
            rate = end_idx / elapsed if elapsed > 0 else 0
            logger.info(f"Progress: [{end_idx:,}/{total_chunks:,}] ({end_idx/total_chunks*100:.1f}%) | {rate:.1f} vectors/sec")

    elapsed_total = round(time.perf_counter() - start_time, 3)
    throughput = round(total_chunks / elapsed_total, 1) if elapsed_total > 0 else 0.0

    # Save to disk
    config = {
        "strategy": strategy,
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "index_type": index_type,
        "batch_size": batch_size,
        "source_file": str(chunks_path),
    }

    manifest = {
        "strategy": strategy,
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "index_type": index_type,
        "metric": "cosine_inner_product",
        "total_vectors": total_chunks,
        "build_time_seconds": elapsed_total,
        "vectors_per_sec": throughput,
        "source_file": str(chunks_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "normalized": True,
    }

    persistence.save(vector_store, metadata_records, config, manifest)
    logger.info(f"Successfully built & saved FAISS index for '{strategy.upper()}': {total_chunks:,} vectors in {elapsed_total}s ({throughput} vec/s) -> {out_dir}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="FAISS Vector Index Builder")
    parser.add_argument(
        "--strategy",
        type=str,
        default="adaptive",
        choices=ALL_STRATEGIES,
        help="Chunking strategy to build index for",
    )
    parser.add_argument(
        "--all-primary",
        action="store_true",
        help="Build indexes for primary strategies: fixed, sentence, adaptive",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build indexes for ALL 7 chunking strategies",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Custom input chunks Parquet file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output index directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding & FAISS insertion batch size (default: 64)",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        choices=["flat", "hnsw"],
        default="flat",
        help="FAISS index type: 'flat' (exact IndexFlatIP) or 'hnsw' (IndexHNSWFlat)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="multilingual-dense-e5",
        help="Embedding model identifier",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing index files",
    )

    args = parser.parse_args()

    if args.all:
        logger.info(f"Building FAISS Indexes for ALL 7 strategies...")
        for strat in ALL_STRATEGIES:
            build_strategy_index(
                strategy=strat,
                batch_size=args.batch_size,
                index_type=args.index_type,
                model_name=args.model,
                overwrite=args.overwrite,
            )
    elif args.all_primary:
        logger.info(f"Building FAISS Indexes for PRIMARY strategies: {PRIMARY_STRATEGIES}...")
        for strat in PRIMARY_STRATEGIES:
            build_strategy_index(
                strategy=strat,
                batch_size=args.batch_size,
                index_type=args.index_type,
                model_name=args.model,
                overwrite=args.overwrite,
            )
    else:
        build_strategy_index(
            strategy=args.strategy,
            chunks_parquet_path=Path(args.input) if args.input else None,
            output_index_dir=Path(args.output) if args.output else None,
            batch_size=args.batch_size,
            index_type=args.index_type,
            model_name=args.model,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
