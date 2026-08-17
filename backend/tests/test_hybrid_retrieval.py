import pytest
from pathlib import Path
import tempfile
import shutil

from retrieval.query.normalize import normalize_query
from retrieval.query.analyze import analyze_query, QueryAnalysis
from retrieval.lexical.tokenizer import MultilingualTokenizer
from retrieval.lexical.bm25 import BM25Index, BM25Retriever
from retrieval.dense.retriever import DenseRetriever
from retrieval.fusion.dedup import deduplicate_candidates
from retrieval.fusion.rrf import reciprocal_rank_fusion
from retrieval.reranking.model import MultilingualCrossEncoderReranker
from retrieval.reranking.reranker import RerankerService
from retrieval.pipeline import RetrievalPipeline
from retrieval.evaluation.evaluator import RetrievalEvaluator
from retrieval.evaluation.failures import FailureAnalyzer


class TestQueryNormalizationAndAnalysis:
    """Test suite for Query Normalization and Linguistic Analysis."""

    def test_normalize_query_indic_nfc(self):
        raw = "   भारत   की   राजधानी क्या    है ?  \u200b "
        norm = normalize_query(raw)
        assert norm == "भारत की राजधानी क्या है ?"
        assert "\u200b" not in norm

    def test_normalize_query_preserves_matras_and_numbers(self):
        raw = "२०२६ में सूर्य का व्यास 1,392,700 किमी है।"
        norm = normalize_query(raw)
        assert "२०२६" in norm
        assert "सूर्य" in norm
        assert "1,392,700" in norm

    def test_analyze_query_hindi(self):
        query = "भारत का प्रधानमंत्री कौन है?"
        analysis = analyze_query(query)
        assert analysis.language == "hi"
        assert analysis.question_type == "PERSON_OR_ORG"
        assert analysis.char_count == len(query.strip())
        assert analysis.token_count > 3

    def test_analyze_query_numeric_why(self):
        query = "२०२४ में तापमान क्यों बढ़ रहा है?"
        analysis = analyze_query(query)
        assert analysis.has_numbers is True
        assert analysis.question_type == "WHY"
        assert analysis.query_type == "numeric"

    def test_analyze_query_english_hinglish(self):
        query = "kya bharat ek vikasasheel desh hai?"
        analysis = analyze_query(query)
        assert analysis.language in ["hi-Latn", "en", "hinglish"]


class TestLexicalBM25:
    """Test suite for Multilingual Tokenizer and BM25 index."""

    def test_multilingual_tokenizer_indic_subwords(self):
        tok = MultilingualTokenizer(use_subwords=True)
        tokens = tok.tokenize("शक्तिशाली")
        assert "शक्तिशाली" in tokens
        # Check subword character n-grams
        subwords = [t for t in tokens if t.startswith("#")]
        assert len(subwords) > 0

    def test_bm25_index_in_memory_and_persistence(self):
        temp_dir = tempfile.mkdtemp()
        try:
            tok = MultilingualTokenizer(use_subwords=False)
            idx = BM25Index(tokenizer=tok)
            docs = [
                "दिल्ली भारत की राजधानी है।",
                "मुंबई भारत की आर्थिक राजधानी है।",
                "क्रिकेट भारत में एक लोकप्रिय खेल है।",
            ]
            idx.fit(docs)
            assert idx.num_docs == 3

            results = idx.search("राजधानी", top_k=2)
            assert len(results) == 2
            assert results[0][0] in [0, 1]

            # Test save and load
            save_path = Path(temp_dir) / "test_bm25.pkl"
            idx.save(save_path)
            loaded_idx = BM25Index.load(save_path)
            assert loaded_idx.num_docs == 3
            loaded_results = loaded_idx.search("राजधानी", top_k=2)
            assert len(loaded_results) == 2
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestCandidateFusionAndRRF:
    """Test suite for Candidate Deduplication and Reciprocal Rank Fusion."""

    def test_deduplicate_candidates(self):
        dense_cand = [
            {"chunk_id": "c1", "rank": 1, "score": 0.95, "text": "doc1", "is_selected": True},
            {"chunk_id": "c2", "rank": 2, "score": 0.85, "text": "doc2"},
        ]
        bm25_cand = [
            {"chunk_id": "c2", "rank": 1, "score": 8.5, "text": "doc2"},
            {"chunk_id": "c3", "rank": 2, "score": 7.0, "text": "doc3"},
        ]

        merged = deduplicate_candidates(dense_cand, bm25_cand)
        assert len(merged) == 3

        c2_entry = next(item for item in merged if item["chunk_id"] == "c2")
        assert c2_entry["dense_rank"] == 2
        assert c2_entry["bm25_rank"] == 1
        assert c2_entry["dense_score"] == 0.85
        assert c2_entry["bm25_score"] == 8.5

    def test_reciprocal_rank_fusion_scoring(self):
        deduped = [
            {"chunk_id": "c1", "dense_rank": 1, "bm25_rank": None},
            {"chunk_id": "c2", "dense_rank": 2, "bm25_rank": 1},
            {"chunk_id": "c3", "dense_rank": None, "bm25_rank": 2},
        ]
        # For c2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        # For c1: 1/(60+1) = 0.016393
        # For c3: 1/(60+2) = 0.016129
        fused = reciprocal_rank_fusion(deduped, rrf_k=60, top_k=3)
        assert len(fused) == 3
        assert fused[0]["chunk_id"] == "c2"
        assert fused[0]["fused_rank"] == 1
        assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


class TestRerankerAndPipeline:
    """Test suite for Cross-Encoder Reranker and End-to-End Pipeline."""

    def test_reranker_scoring_and_cache(self):
        temp_dir = tempfile.mkdtemp()
        try:
            db_path = Path(temp_dir) / "test_cache.sqlite3"
            reranker = MultilingualCrossEncoderReranker(cache_db_path=db_path, use_cache=True)

            query = "भारत की राजधानी"
            doc1 = "नई दिल्ली भारत की आधिकारिक राजधानी है।"
            doc2 = "सेब एक बहुत ही स्वास्थ्यवर्धक फल है।"

            score1 = reranker.score_pair(query, doc1)
            score2 = reranker.score_pair(query, doc2)

            assert 0.0 <= score1 <= 1.0
            assert 0.0 <= score2 <= 1.0
            assert score1 > score2  # Relevant doc scores higher

            # Test batch rerank
            candidates = [
                {"chunk_id": "doc2", "text": doc2},
                {"chunk_id": "doc1", "text": doc1},
            ]
            reranked = reranker.rerank(query, candidates, top_k=2)
            assert reranked[0]["chunk_id"] == "doc1"
            assert reranked[0]["rerank_rank"] == 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_retrieval_pipeline_e2e(self):
        pipeline = RetrievalPipeline(strategy="fixed")
        query = "भारत का संविधान कब लागू हुआ था?"

        result = pipeline.retrieve(
            query=query,
            strategy="fixed",
            dense_k=5,
            bm25_k=5,
            hybrid_k=5,
            rerank_top_k=3,
            enable_reranking=True,
        )

        assert "query" in result
        assert "normalized_query" in result
        assert "query_analysis" in result
        assert len(result["dense_candidates"]) > 0
        assert len(result["bm25_candidates"]) > 0
        assert len(result["fused_candidates"]) > 0
        assert len(result["reranked_results"]) > 0
        assert len(result["final_context"]) <= 3
        assert "latency" in result
        assert result["latency"]["total_ms"] > 0

    def test_retrieval_evaluator_metrics(self):
        evaluator = RetrievalEvaluator()
        retrieved = [
            {"chunk_id": "c1", "passage_id": "p100", "is_selected": False},
            {"chunk_id": "c2", "passage_id": "p200", "is_selected": True},
        ]
        ground_truth = {"p200"}
        metrics = evaluator.evaluate_query(retrieved, ground_truth, query_id=1, ks=[1, 3, 5])
        assert metrics["recall@1"] == 0.0
        assert metrics["recall@3"] == 1.0
        assert metrics["recall@5"] == 1.0
        assert metrics["reciprocal_rank"] == 0.5
