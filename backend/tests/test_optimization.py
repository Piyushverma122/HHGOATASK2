import time
import pytest
from unittest.mock import patch, MagicMock

from retrieval.pipeline import RetrievalPipeline
from retrieval.hybrid import HybridRetriever
from retrieval.warmup import warmup_system, get_warmup_status
from retrieval.cache.query_cache import QueryCache
from retrieval.embeddings.provider import get_embedding_provider
from retrieval.reranking.reranker import RerankerService
from generation.harness import RAGHarness


class TestParallelRetrieval:
    """Test suite for concurrent Dense FAISS + Sparse BM25 execution."""

    def test_parallel_vs_sequential_results_match(self):
        retriever = HybridRetriever(strategy="adaptive")
        query = "भारत की राजधानी क्या है?"

        seq_out = retriever.search_sequential(query=query, final_k=5)
        par_out = retriever.search_parallel(query=query, final_k=5)

        assert len(seq_out["fused_candidates"]) == len(par_out["fused_candidates"])
        # Check that top candidate chunk_ids are identical
        seq_ids = [c["chunk_id"] for c in seq_out["fused_candidates"]]
        par_ids = [c["chunk_id"] for c in par_out["fused_candidates"]]
        assert seq_ids == par_ids

    def test_parallel_retrieval_latencies(self):
        retriever = HybridRetriever(strategy="adaptive")
        out = retriever.search_parallel(query="कंप्यूटर क्या है?", final_k=5)

        lats = out["latencies"]
        assert "dense_ms" in lats
        assert "bm25_ms" in lats
        assert "fusion_ms" in lats
        assert "total_hybrid_ms" in lats
        assert lats["total_hybrid_ms"] > 0


class TestModelWarmupLifecycle:
    """Test suite for warmup_system() and cold vs warm tracking."""

    def test_warmup_system_execution(self):
        status = warmup_system(verbose=False)
        assert status["status"] == "WARMED"
        assert "timings" in status
        assert status["cold_start_total_ms"] > 0
        assert status["warm_request_ms"] > 0
        assert get_warmup_status() is not None


class TestQueryCache:
    """Test suite for TTL-based QueryCache."""

    def test_query_cache_hit_and_miss(self):
        cache = QueryCache(max_capacity=10, ttl_seconds=60)
        query = "भारत की राजधानी क्या है?"
        dummy_res = {"answer": "नई दिल्ली", "grounded": True}

        # Initial miss
        assert cache.get(query, "adaptive") is None
        assert cache.misses == 1
        assert cache.hits == 0

        # Store entry
        cache.set(query, "adaptive", dummy_res)

        # Cache hit
        hit_res = cache.get(query, "adaptive")
        assert hit_res == dummy_res
        assert cache.hits == 1

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hit_rate_pct"] == 50.0

    def test_query_cache_ttl_expiration(self):
        cache = QueryCache(max_capacity=10, ttl_seconds=0.01)
        cache.set("test", "adaptive", {"data": 123})
        time.sleep(0.02)
        assert cache.get("test", "adaptive") is None


class TestRerankerBatching:
    """Test suite for cross-encoder reranker batching."""

    def test_batch_reranking_accuracy(self):
        reranker = RerankerService(default_top_k=5)
        candidates = [
            {"chunk_id": f"c_{i}", "passage_id": f"p_{i}", "text": f"Passage {i} text for scoring.", "rrf_score": 1.0 / (i + 1)}
            for i in range(12)
        ]
        out = reranker.rerank_candidates("test query", candidates=candidates, top_k=5)
        assert len(out["reranked_candidates"]) == 5
        assert out["latencies"]["total_rerank_ms"] > 0
