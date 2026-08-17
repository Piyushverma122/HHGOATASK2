import time
import hashlib
import json
import logging
from typing import Dict, Any, Optional
from collections import OrderedDict

logger = logging.getLogger("voice_rag.cache.query")


class QueryCache:
    """
    LRU Memory Cache with TTL for Hybrid Retrieval and RAG Query Results.
    Keys on: (normalized_query, strategy, dense_k, bm25_k, rerank_top_k, enable_reranking).
    """

    def __init__(self, max_capacity: int = 1000, ttl_seconds: int = 300):
        self.max_capacity = max_capacity
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _compute_key(
        self,
        query: str,
        strategy: str,
        dense_k: int = 20,
        bm25_k: int = 20,
        rerank_top_k: int = 5,
        enable_reranking: bool = True,
    ) -> str:
        payload = f"{query.strip().lower()}|{strategy}|{dense_k}|{bm25_k}|{rerank_top_k}|{enable_reranking}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(
        self,
        query: str,
        strategy: str,
        dense_k: int = 20,
        bm25_k: int = 20,
        rerank_top_k: int = 5,
        enable_reranking: bool = True,
    ) -> Optional[Dict[str, Any]]:
        key = self._compute_key(query, strategy, dense_k, bm25_k, rerank_top_k, enable_reranking)
        entry = self._cache.get(key)
        if not entry:
            self.misses += 1
            return None

        # Check TTL
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            self._cache.pop(key, None)
            self.misses += 1
            return None

        # Move to end (LRU)
        self._cache.move_to_end(key)
        self.hits += 1
        return entry["result"]

    def set(
        self,
        query: str,
        strategy: str,
        result: Dict[str, Any],
        dense_k: int = 20,
        bm25_k: int = 20,
        rerank_top_k: int = 5,
        enable_reranking: bool = True,
    ) -> None:
        key = self._compute_key(query, strategy, dense_k, bm25_k, rerank_top_k, enable_reranking)
        if len(self._cache) >= self.max_capacity:
            self._cache.popitem(last=False)

        self._cache[key] = {
            "result": result,
            "timestamp": time.time(),
        }

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100.0) if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_capacity": self.max_capacity,
            "ttl_seconds": self.ttl_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 2),
        }
