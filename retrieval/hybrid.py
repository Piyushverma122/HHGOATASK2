import time
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from retrieval.dense.retriever import DenseRetriever
from retrieval.lexical.bm25 import BM25Retriever
from retrieval.fusion.dedup import deduplicate_candidates
from retrieval.fusion.rrf import reciprocal_rank_fusion

# Thread pool for concurrent dense + BM25 retrieval execution
_THREAD_POOL = ThreadPoolExecutor(max_workers=4)


class HybridRetriever:
    """
    Hybrid Retriever combining Multilingual Dense FAISS vector retrieval and BM25 Lexical retrieval.
    Applies candidate deduplication and Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        strategy: str = "adaptive",
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
    ):
        self.strategy = strategy
        self.dense_retriever = dense_retriever or DenseRetriever(strategy=strategy)
        self.bm25_retriever = bm25_retriever or BM25Retriever(strategy=strategy)

    def search_sequential(
        self,
        query: str,
        strategy: Optional[str] = None,
        dense_k: int = 20,
        bm25_k: int = 20,
        final_k: int = 20,
        rrf_k: int = 60,
    ) -> Dict[str, Any]:
        """Execute dense and BM25 search sequentially."""
        strat = strategy or self.strategy
        total_start = time.perf_counter()

        # 1. Dense retrieval
        dense_start = time.perf_counter()
        dense_candidates = self.dense_retriever.search(query=query, strategy=strat, top_k=dense_k)
        dense_ms = (time.perf_counter() - dense_start) * 1000.0

        # 2. BM25 retrieval
        bm25_start = time.perf_counter()
        bm25_candidates = self.bm25_retriever.search(query=query, top_k=bm25_k)
        bm25_ms = (time.perf_counter() - bm25_start) * 1000.0

        # 3. Candidate Deduplication & Fusion
        fusion_start = time.perf_counter()
        deduped = deduplicate_candidates(dense_candidates, bm25_candidates)
        fused = reciprocal_rank_fusion(deduped, rrf_k=rrf_k, top_k=final_k)
        fusion_ms = (time.perf_counter() - fusion_start) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0

        return {
            "query": query,
            "strategy": strat,
            "dense_candidates": dense_candidates,
            "bm25_candidates": bm25_candidates,
            "fused_candidates": fused,
            "total_candidates_before_dedup": len(dense_candidates) + len(bm25_candidates),
            "total_candidates_after_dedup": len(deduped),
            "latencies": {
                "dense_ms": round(dense_ms, 3),
                "bm25_ms": round(bm25_ms, 3),
                "fusion_ms": round(fusion_ms, 3),
                "total_hybrid_ms": round(total_ms, 3),
            },
        }

    def search_parallel(
        self,
        query: str,
        strategy: Optional[str] = None,
        dense_k: int = 20,
        bm25_k: int = 20,
        final_k: int = 20,
        rrf_k: int = 60,
    ) -> Dict[str, Any]:
        """Execute dense and BM25 search concurrently using thread executor."""
        strat = strategy or self.strategy
        total_start = time.perf_counter()

        def _run_dense():
            t0 = time.perf_counter()
            res = self.dense_retriever.search(query=query, strategy=strat, top_k=dense_k)
            return res, (time.perf_counter() - t0) * 1000.0

        def _run_bm25():
            t0 = time.perf_counter()
            res = self.bm25_retriever.search(query=query, top_k=bm25_k)
            return res, (time.perf_counter() - t0) * 1000.0

        future_dense = _THREAD_POOL.submit(_run_dense)
        future_bm25 = _THREAD_POOL.submit(_run_bm25)

        dense_candidates, dense_ms = future_dense.result()
        bm25_candidates, bm25_ms = future_bm25.result()

        fusion_start = time.perf_counter()
        deduped = deduplicate_candidates(dense_candidates, bm25_candidates)
        fused = reciprocal_rank_fusion(deduped, rrf_k=rrf_k, top_k=final_k)
        fusion_ms = (time.perf_counter() - fusion_start) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0

        return {
            "query": query,
            "strategy": strat,
            "dense_candidates": dense_candidates,
            "bm25_candidates": bm25_candidates,
            "fused_candidates": fused,
            "total_candidates_before_dedup": len(dense_candidates) + len(bm25_candidates),
            "total_candidates_after_dedup": len(deduped),
            "latencies": {
                "dense_ms": round(dense_ms, 3),
                "bm25_ms": round(bm25_ms, 3),
                "fusion_ms": round(fusion_ms, 3),
                "total_hybrid_ms": round(total_ms, 3),
            },
        }

    def search(
        self,
        query: str,
        strategy: Optional[str] = None,
        dense_k: int = 20,
        bm25_k: int = 20,
        final_k: int = 20,
        rrf_k: int = 60,
        parallel: bool = False,
    ) -> Dict[str, Any]:
        """Default search method."""
        if parallel:
            return self.search_parallel(query, strategy, dense_k, bm25_k, final_k, rrf_k)
        return self.search_sequential(query, strategy, dense_k, bm25_k, final_k, rrf_k)
