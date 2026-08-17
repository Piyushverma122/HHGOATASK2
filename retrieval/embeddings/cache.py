import hashlib
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import numpy as np


class EmbeddingCache:
    """
    Persistent local embedding cache backed by SQLite.
    Key: SHA-256(model_name + model_version + text)
    Value: Binary serialized float32 numpy array.
    """

    def __init__(self, db_path: Optional[Path] = None, model_name: str = "default", model_version: str = "1.0"):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            cache_dir = base_dir / "data" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = cache_dir / "embedding_cache.sqlite3"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.model_version = model_version
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON embeddings (model_name, model_version)")
            conn.commit()

    def compute_key(self, text: str) -> str:
        """
        Deterministic SHA-256 cache key based on model identifier, version, and text content.
        """
        payload = f"{self.model_name}::{self.model_version}::{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding for single text."""
        key = self.compute_key(text)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT vector, dimension FROM embeddings WHERE cache_key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if row:
                raw_bytes, dim = row
                vec = np.frombuffer(raw_bytes, dtype=np.float32).copy()
                if len(vec) == dim:
                    return vec
        return None

    def get_batch(self, texts: List[str]) -> Tuple[Dict[int, np.ndarray], List[int]]:
        """
        Retrieve cached embeddings for a batch of texts.
        Returns:
            (cached_map: {index: vector}, missing_indices: [index, ...])
        """
        if not texts:
            return {}, []

        key_to_idx = {self.compute_key(t): i for i, t in enumerate(texts)}
        keys = list(key_to_idx.keys())

        cached: Dict[int, np.ndarray] = {}
        found_indices: Set[int] = set()

        # Batch lookup in SQLite chunks of 500
        chunk_size = 500
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for start in range(0, len(keys), chunk_size):
                sub_keys = keys[start:start + chunk_size]
                placeholders = ",".join(["?"] * len(sub_keys))
                cursor.execute(
                    f"SELECT cache_key, vector, dimension FROM embeddings WHERE cache_key IN ({placeholders})",
                    sub_keys,
                )
                rows = cursor.fetchall()
                for k, raw_bytes, dim in rows:
                    idx = key_to_idx.get(k)
                    if idx is not None:
                        vec = np.frombuffer(raw_bytes, dtype=np.float32).copy()
                        if len(vec) == dim:
                            cached[idx] = vec
                            found_indices.add(idx)

        missing = [i for i in range(len(texts)) if i not in found_indices]
        return cached, missing

    def set(self, text: str, vector: np.ndarray):
        """Store embedding vector in cache."""
        key = self.compute_key(text)
        vec_bytes = vector.astype(np.float32).tobytes()
        dim = len(vector)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings (cache_key, model_name, model_version, dimension, vector)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, self.model_name, self.model_version, dim, vec_bytes),
            )
            conn.commit()

    def set_batch(self, texts: List[str], vectors: np.ndarray):
        """Store a batch of embedding vectors in cache."""
        if len(texts) == 0 or len(vectors) == 0:
            return
        records = []
        for i, text in enumerate(texts):
            vec = vectors[i].astype(np.float32)
            key = self.compute_key(text)
            records.append((key, self.model_name, self.model_version, len(vec), vec.tobytes()))

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings (cache_key, model_name, model_version, dimension, vector)
                VALUES (?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()

    def count(self) -> int:
        """Total number of cached vectors for this model."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_name = ? AND model_version = ?",
                (self.model_name, self.model_version),
            )
            return cursor.fetchone()[0]

    def clear(self):
        """Clear all entries for this model."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM embeddings WHERE model_name = ? AND model_version = ?",
                (self.model_name, self.model_version),
            )
            conn.commit()
