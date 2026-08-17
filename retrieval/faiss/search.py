import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from retrieval.embeddings.provider import EmbeddingProviderFactory, get_default_embedder
from retrieval.faiss.index import FaissVectorStore
from retrieval.faiss.persistence import IndexPersistenceManager


class StrategyVectorSearcher:
    """
    Search engine for a specific chunking strategy's FAISS index.
    Caches loaded index and metadata mapping in memory.
    """

    def __init__(
        self,
        strategy: str = "adaptive",
        index_dir: Optional[Path] = None,
        embedder=None,
    ):
        self.strategy = strategy
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.index_dir = Path(index_dir) if index_dir else base_dir / "indexes" / strategy
        self.persistence = IndexPersistenceManager(self.index_dir)

        self.embedder = embedder or get_default_embedder()
        self.vector_store: Optional[FaissVectorStore] = None
        self.metadata_lookup: Dict[int, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.persistence.faiss_path.exists():
            self.vector_store = self.persistence.load_index()
            self.metadata_lookup = self.persistence.load_metadata_lookup()

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute full vector search pipeline for a query:
        1. Query Embedding (with L2 normalization)
        2. FAISS Vector Search
        3. Metadata & Chunk Provenance Resolution
        4. Latency breakdown
        """
        if not self.vector_store or self.vector_store.size() == 0:
            return {
                "query": query,
                "strategy": self.strategy,
                "results": [],
                "latencies": {"query_embed_ms": 0.0, "faiss_search_ms": 0.0, "metadata_lookup_ms": 0.0, "total_ms": 0.0},
                "total_vectors_in_index": 0,
            }

        total_start = time.perf_counter()

        # 1. Embed query
        embed_start = time.perf_counter()
        query_vector = self.embedder.embed_query(query)
        embed_ms = (time.perf_counter() - embed_start) * 1000.0

        # 2. Search FAISS index
        scores, vector_ids, faiss_search_ms = self.vector_store.search(query_vector, top_k=top_k)

        # 3. Resolve metadata
        meta_start = time.perf_counter()
        results: List[Dict[str, Any]] = []

        for rank, (score, vid) in enumerate(zip(scores, vector_ids), start=1):
            meta = self.metadata_lookup.get(int(vid), {})
            results.append({
                "rank": rank,
                "vector_id": int(vid),
                "score": round(float(score), 4),
                "chunk_id": meta.get("chunk_id", f"unknown_chunk_{vid}"),
                "record_id": meta.get("record_id", ""),
                "query_id": meta.get("query_id", 0),
                "passage_id": meta.get("passage_id", ""),
                "language": meta.get("language", "hi"),
                "strategy": meta.get("strategy", self.strategy),
                "query_type": meta.get("query_type", "standard"),
                "is_selected": meta.get("is_selected", False),
                "token_count": meta.get("token_count", 0),
                "text": meta.get("text", ""),
                "metadata": meta.get("metadata", {}),
            })

        meta_ms = (time.perf_counter() - meta_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        return {
            "query": query,
            "strategy": self.strategy,
            "results": results,
            "latencies": {
                "query_embed_ms": round(embed_ms, 3),
                "faiss_search_ms": round(faiss_search_ms, 3),
                "metadata_lookup_ms": round(meta_ms, 3),
                "total_ms": round(total_ms, 3),
            },
            "total_vectors_in_index": self.vector_store.size(),
        }
