import hashlib
import sqlite3
from pathlib import Path
from typing import Optional


class RerankerCache:
    """
    Persistent SQLite cache for cross-encoder inference pairs.
    Deterministic cache key: SHA-256(model_name + "::" + model_version + "::" + query + "::" + chunk_id)
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        model_version: str = "v1.0",
    ):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            cache_dir = base_dir / "data" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = cache_dir / "reranker_cache.sqlite3"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.model_version = model_version
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reranker_scores (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL DEFAULT 'v1.0',
                    score REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(reranker_scores)")
            columns = [row[1] for row in cursor.fetchall()]
            if "model_version" not in columns:
                conn.execute("ALTER TABLE reranker_scores ADD COLUMN model_version TEXT NOT NULL DEFAULT 'v1.0'")
            conn.commit()

    def compute_key(self, query: str, chunk_id: str) -> str:
        """
        Compute deterministic SHA-256 cache key including model name, model version,
        normalized query, and unique chunk_id.
        """
        payload = f"{self.model_name}::{self.model_version}::{query.strip()}::{chunk_id.strip()}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, query: str, chunk_id: str) -> Optional[float]:
        key = self.compute_key(query, chunk_id)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT score FROM reranker_scores WHERE cache_key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return float(row[0])
        return None

    def set(self, query: str, chunk_id: str, score: float):
        key = self.compute_key(query, chunk_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reranker_scores (cache_key, model_name, model_version, score) VALUES (?, ?, ?, ?)",
                (key, self.model_name, self.model_version, float(score)),
            )
            conn.commit()
