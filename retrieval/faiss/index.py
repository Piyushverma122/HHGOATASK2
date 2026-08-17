import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import faiss


class FaissVectorStore:
    """
    Production-grade FAISS vector store with support for exact (FlatIP) and approximate (HNSW) search.
    Utilizes IndexIDMap2 for deterministic 64-bit vector ID mapping.
    """

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "flat",  # "flat" or "hnsw"
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 64,
        hnsw_ef_search: int = 32,
    ):
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        self.index: Optional[faiss.Index] = None
        self._build_index()

    def _build_index(self):
        if self.index_type == "hnsw":
            # HNSW Flat with Inner Product metric (cosine similarity on L2-normalized vectors)
            base_index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            base_index.hnsw.efConstruction = self.hnsw_ef_construction
            base_index.hnsw.efSearch = self.hnsw_ef_search
            self.index = faiss.IndexIDMap2(base_index)
        else:
            # Default: Exact Flat Inner Product index
            base_index = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap2(base_index)

    def size(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def add_vectors(self, vectors: np.ndarray, ids: np.ndarray) -> int:
        """
        Add batch of vectors with explicit deterministic integer IDs.
        Vectors must be float32 and shape (N, dimension).
        IDs must be int64 and shape (N,).
        """
        if len(vectors) == 0:
            return 0

        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        id_arr = np.ascontiguousarray(ids, dtype=np.int64)

        if vecs.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vecs.shape[1]}")
        if len(vecs) != len(id_arr):
            raise ValueError(f"Length mismatch: {len(vecs)} vectors vs {len(id_arr)} ids")

        self.index.add_with_ids(vecs, id_arr)
        return len(vecs)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Search for top_k nearest neighbors.
        Returns:
            scores: 1D array of cosine similarity scores
            ids: 1D array of matched vector IDs
            latency_ms: time taken in milliseconds
        """
        if self.index is None or self.index.ntotal == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64), 0.0

        q_vec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        k = min(top_k, self.index.ntotal)

        start = time.perf_counter()
        scores, ids = self.index.search(q_vec, k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return scores[0], ids[0], round(elapsed_ms, 3)

    def save(self, index_file: Path):
        """Save FAISS binary index to disk."""
        index_file = Path(index_file)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_file))

    @classmethod
    def load(cls, index_file: Path, config: Optional[Dict[str, Any]] = None) -> "FaissVectorStore":
        """Load FAISS binary index from disk."""
        index_file = Path(index_file)
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_file}")

        loaded_index = faiss.read_index(str(index_file))
        cfg = config or {}
        store = cls(
            dimension=loaded_index.d,
            index_type=cfg.get("index_type", "flat"),
            hnsw_m=cfg.get("hnsw_m", 32),
            hnsw_ef_construction=cfg.get("hnsw_ef_construction", 64),
            hnsw_ef_search=cfg.get("hnsw_ef_search", 32),
        )
        store.index = loaded_index
        return store
