import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from retrieval.faiss.index import FaissVectorStore

METADATA_SCHEMA = pa.schema([
    ("vector_id", pa.int64()),
    ("chunk_id", pa.string()),
    ("record_id", pa.string()),
    ("query_id", pa.int64()),
    ("passage_id", pa.string()),
    ("language", pa.string()),
    ("strategy", pa.string()),
    ("query_type", pa.string()),
    ("is_selected", pa.bool_()),
    ("token_count", pa.int32()),
    ("text", pa.string()),
    ("metadata_json", pa.string()),
])


class IndexPersistenceManager:
    """
    Manages complete persistence lifecycle for a strategy index:
    - index.faiss (Binary FAISS Index)
    - metadata.parquet (Vector ID to Chunk & Provenance Mapping)
    - config.json (Index & Model hyperparameters)
    - manifest.json (Diagnostic manifest metadata)
    """

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_path = self.index_dir / "index.faiss"
        self.metadata_path = self.index_dir / "metadata.parquet"
        self.config_path = self.index_dir / "config.json"
        self.manifest_path = self.index_dir / "manifest.json"

    def save(
        self,
        vector_store: FaissVectorStore,
        metadata_records: List[Dict[str, Any]],
        config: Dict[str, Any],
        manifest: Optional[Dict[str, Any]] = None,
    ):
        # 1. Save FAISS binary index
        vector_store.save(self.faiss_path)

        # 2. Save metadata to Parquet
        if metadata_records:
            df = pd.DataFrame(metadata_records)
            table = pa.Table.from_pandas(df, schema=METADATA_SCHEMA)
            pq.write_table(table, str(self.metadata_path), compression="snappy")

        # 3. Save config
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # 4. Save manifest
        manifest_payload = manifest or {
            "strategy": config.get("strategy", "unknown"),
            "embedding_model": config.get("embedding_model", "multilingual-dense-e5"),
            "embedding_dimension": vector_store.dimension,
            "index_type": vector_store.index_type,
            "metric": "cosine_inner_product",
            "total_vectors": vector_store.size(),
            "source_file": config.get("source_file", ""),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "normalized": True,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)

    def load_index(self) -> FaissVectorStore:
        config = self.load_config()
        return FaissVectorStore.load(self.faiss_path, config=config)

    def load_metadata_lookup(self) -> Dict[int, Dict[str, Any]]:
        """Load metadata Parquet table into fast memory lookup indexed by vector_id."""
        if not self.metadata_path.exists():
            return {}

        df = pd.read_parquet(self.metadata_path)
        lookup = {}
        for _, row in df.iterrows():
            v_id = int(row["vector_id"])
            lookup[v_id] = {
                "vector_id": v_id,
                "chunk_id": row["chunk_id"],
                "record_id": row["record_id"],
                "query_id": int(row["query_id"]),
                "passage_id": row["passage_id"],
                "language": row["language"],
                "strategy": row["strategy"],
                "query_type": row["query_type"],
                "is_selected": bool(row["is_selected"]),
                "token_count": int(row["token_count"]),
                "text": row["text"],
                "metadata": json.loads(row["metadata_json"]) if row.get("metadata_json") else {},
            }
        return lookup

    def load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
