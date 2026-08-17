import hashlib
import json
import logging
from typing import Optional, List, Dict, Any
from generation.schemas import AnswerResponse

logger = logging.getLogger("voice_rag.generation.cache")


class GenerationCache:
    """
    In-memory / Persistent Generation Cache.
    Keyed by: (model_name, normalized_query, context_chunk_ids, temperature).
    Ensures that identical queries with different retrieved contexts never share a stale cache entry.
    """

    def __init__(self, max_size: int = 1000, enabled: bool = True):
        self._cache: Dict[str, AnswerResponse] = {}
        self.max_size = max_size
        self.enabled = enabled

    def _compute_key(
        self,
        model_name: str,
        query: str,
        chunk_ids: List[str],
        temperature: float = 0.1,
    ) -> str:
        sorted_ids = sorted(chunk_ids)
        key_raw = f"{model_name}|{query.strip().lower()}|{','.join(sorted_ids)}|{temperature:.2f}"
        return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

    def get(
        self,
        model_name: str,
        query: str,
        chunk_ids: List[str],
        temperature: float = 0.1,
    ) -> Optional[AnswerResponse]:
        if not self.enabled:
            return None
        key = self._compute_key(model_name, query, chunk_ids, temperature)
        hit = self._cache.get(key)
        if hit:
            logger.debug(f"Generation cache hit for key: {key[:12]}...")
        return hit

    def put(
        self,
        model_name: str,
        query: str,
        chunk_ids: List[str],
        response: AnswerResponse,
        temperature: float = 0.1,
    ) -> None:
        if not self.enabled:
            return
        if len(self._cache) >= self.max_size:
            # Evict oldest entry
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        key = self._compute_key(model_name, query, chunk_ids, temperature)
        self._cache[key] = response

    def clear(self) -> None:
        self._cache.clear()
