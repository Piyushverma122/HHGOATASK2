import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch

from retrieval.reranking.base import BaseReranker
from retrieval.reranking.cache import RerankerCache
from retrieval.reranking.model import CrossEncoderReranker, CustomReranker
from retrieval.reranking.reranker import RerankerService, get_reranker_service


class TestRerankerCache:
    """Test suite for persistent deterministic SQLite reranker cache."""

    def test_cache_key_generation(self):
        cache = RerankerCache(model_name="test-model", model_version="v1.0")
        key1 = cache.compute_key("भारत की राजधानी", "chunk_101")
        key2 = cache.compute_key("भारत की राजधानी", "chunk_101")
        key3 = cache.compute_key("भारत की राजधानी", "chunk_102")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 64  # SHA-256 hex string

    def test_cache_sqlite_persistence(self):
        temp_dir = tempfile.mkdtemp()
        try:
            db_file = Path(temp_dir) / "test_rerank.sqlite3"
            cache = RerankerCache(db_path=db_file, model_name="test-model", model_version="v1.0")

            assert cache.get("q1", "c1") is None
            cache.set("q1", "c1", 0.9876)
            assert cache.get("q1", "c1") == 0.9876

            # Reload cache instance from disk
            cache_reloaded = RerankerCache(db_path=db_file, model_name="test-model", model_version="v1.0")
            assert cache_reloaded.get("q1", "c1") == 0.9876
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestCrossEncoderRerankerUnit:
    """Test suite for CrossEncoderReranker lifecycle, batching, and inference."""

    def test_reranker_lazy_load_and_metadata(self):
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            lazy_load=True,
            use_cache=False,
        )
        assert reranker.is_loaded() is False
        info = reranker.get_model_info()
        assert info["model"] == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        assert info["loaded"] is False
        assert info["device"] in ["cpu", "cuda"]

    def test_reranker_warmup_and_inference(self):
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            lazy_load=True,
            use_cache=False,
        )
        reranker.warmup()
        assert reranker.is_loaded() is True

        query = "भारत की राजधानी क्या है?"
        passages = [
            "नई दिल्ली भारत की आधिकारिक राजधानी है।",
            "सेब एक पौष्टिक फल है।",
        ]
        scores = reranker.score(query, passages)
        assert len(scores) == 2
        assert 0.0 <= scores[0] <= 1.0
        assert 0.0 <= scores[1] <= 1.0
        assert scores[0] > scores[1]  # Relevant passage scores strictly higher

    def test_batch_inference_and_top_k(self):
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            batch_size=4,
            use_cache=False,
        )
        query = "सूर्य की दूरी"
        candidates = [
            {"chunk_id": f"c_{i}", "text": f"यह एक सामान्य अनुच्छेद {i} है।"} for i in range(10)
        ]
        candidates.append({"chunk_id": "c_rel", "text": "पृथ्वी से सूर्य की दूरी लगभग 149.6 मिलियन किमी है।"})

        reranked = reranker.rerank(query, candidates, top_k=5)
        assert len(reranked) == 5
        assert reranked[0]["chunk_id"] == "c_rel"
        assert reranked[0]["rerank_rank"] == 1
        assert "reranker_score" in reranked[0]

    def test_empty_candidates_handling(self):
        reranker = CrossEncoderReranker(use_cache=False)
        assert reranker.rerank("query", [], top_k=5) == []
        assert reranker.score("query", []) == []

    def test_cpu_cuda_detection(self):
        reranker_auto = CrossEncoderReranker(use_cache=False)
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert reranker_auto.device == expected_device

        # Explicit CPU override
        reranker_cpu = CrossEncoderReranker(device="cpu", use_cache=False)
        assert reranker_cpu.device == "cpu"


class TestRerankerService:
    """Test suite for RerankerService management and singleton lifecycle."""

    def test_service_lifecycle_and_timing(self):
        service = RerankerService(auto_load=True)
        assert service.is_loaded() is True

        query = "कंप्यूटर क्या है?"
        candidates = [
            {"chunk_id": "c1", "text": "कंप्यूटर एक इलेक्ट्रॉनिक उपकरण है।"},
            {"chunk_id": "c2", "text": "आम का पेड़ हरा होता है।"},
        ]
        res = service.rerank_candidates(query, candidates, top_k=2)
        assert "reranked_candidates" in res
        assert "latencies" in res
        assert res["latencies"]["total_rerank_ms"] > 0
        assert len(res["reranked_candidates"]) == 2
        assert res["reranked_candidates"][0]["chunk_id"] == "c1"

    def test_global_singleton_getter(self):
        s1 = get_reranker_service("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        s2 = get_reranker_service("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        assert s1 is s2
