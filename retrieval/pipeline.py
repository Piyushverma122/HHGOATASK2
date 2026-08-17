import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from retrieval.query.normalize import normalize_query
from retrieval.query.analyze import analyze_query, QueryAnalysis
from retrieval.dense.retriever import DenseRetriever
from retrieval.lexical.bm25 import BM25Retriever, get_bm25_retriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranking.reranker import RerankerService
from retrieval.reranking.model import MultilingualCrossEncoderReranker


class RetrievalPipeline:
    """
    End-to-End Hybrid Retrieval and Reranking Pipeline.
    Architecture:
        Query -> Normalize -> Analyze -> Parallel(Dense, BM25) -> Dedup -> RRF -> Rerank -> Final Top-K Context
    Preserves all intermediate candidate collections for frontend inspection & debugging.
    """

    def __init__(
        self,
        strategy: str = "adaptive",
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        reranker_service: Optional[RerankerService] = None,
    ):
        self.strategy = strategy
        self.dense_retriever = dense_retriever or DenseRetriever(strategy=strategy)
        self.bm25_retriever = bm25_retriever or get_bm25_retriever(strategy=strategy)
        self.hybrid_retriever = HybridRetriever(
            strategy=strategy,
            dense_retriever=self.dense_retriever,
            bm25_retriever=self.bm25_retriever,
        )
        self.reranker_service = reranker_service or RerankerService(default_top_k=8)

    def retrieve(
        self,
        query: str,
        strategy: Optional[str] = None,
        dense_k: int = 15,
        bm25_k: int = 15,
        hybrid_k: int = 15,
        rerank_top_k: int = 8,
        rrf_k: int = 60,
        enable_reranking: bool = True,
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full retrieval and reranking pipeline with end-to-end timing.
        """
        pipeline_start = time.perf_counter()
        strat = strategy or self.strategy

        # 1. Query Normalization
        norm_start = time.perf_counter()
        normalized = normalize_query(query)
        norm_ms = (time.perf_counter() - norm_start) * 1000.0

        # 2. Query Analysis
        analysis_start = time.perf_counter()
        analysis = analyze_query(query)
        analysis_ms = (time.perf_counter() - analysis_start) * 1000.0

        query_proc_ms = norm_ms + analysis_ms

        # 3. Hybrid Retrieval (Dense + BM25 + Deduplication + RRF)
        hybrid_out = self.hybrid_retriever.search(
            query=normalized,
            strategy=strat,
            dense_k=dense_k,
            bm25_k=bm25_k,
            final_k=hybrid_k,
            rrf_k=rrf_k,
            parallel=parallel,
        )

        dense_candidates = hybrid_out["dense_candidates"]
        bm25_candidates = hybrid_out["bm25_candidates"]
        fused_candidates = hybrid_out["fused_candidates"]
        hybrid_lats = hybrid_out["latencies"]

        # 4. Reranking (top 5 candidates from RRF for sub-80ms CPU inference)
        if enable_reranking and fused_candidates:
            rerank_candidates_pool = fused_candidates[:5]
            rerank_out = self.reranker_service.rerank_candidates(
                query=normalized,
                candidates=rerank_candidates_pool,
                top_k=rerank_top_k,
            )
            reranked_results = rerank_out["reranked_candidates"]
            rerank_ms = rerank_out["latencies"]["total_rerank_ms"]
            final_context = reranked_results
        else:
            reranked_results = []
            rerank_ms = 0.0
            final_context = fused_candidates[:rerank_top_k]

        total_ms = (time.perf_counter() - pipeline_start) * 1000.0

        return {
            "query": query,
            "normalized_query": normalized,
            "language": analysis.language,
            "query_analysis": analysis.model_dump(),
            "strategy": strat,
            "dense_candidates": dense_candidates,
            "bm25_candidates": bm25_candidates,
            "fused_candidates": fused_candidates,
            "reranked_results": reranked_results,
            "final_context": final_context,
            "latency": {
                "query_processing_ms": round(query_proc_ms, 3),
                "dense_ms": hybrid_lats.get("dense_ms", 0.0),
                "bm25_ms": hybrid_lats.get("bm25_ms", 0.0),
                "fusion_ms": hybrid_lats.get("fusion_ms", 0.0),
                "rerank_ms": round(rerank_ms, 3),
                "total_ms": round(total_ms, 3),
            },
        }
