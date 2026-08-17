import pytest
from unittest.mock import patch, MagicMock
import httpx

from benchmark.fixtures import load_voice_fixtures, get_fixture_by_language
from benchmark.latency import compute_latency_percentiles
from benchmark.evaluation import compute_retrieval_metrics
from benchmark.stress import run_stress_test


class TestBenchmarkFixtures:
    """Test suite for offline voice fixture loader."""

    def test_voice_fixtures_loading(self):
        fixtures = load_voice_fixtures()
        assert len(fixtures) >= 7

        languages = {f["language"] for f in fixtures}
        assert "hi" in languages
        assert "en" in languages
        assert "hinglish" in languages
        assert "bn" in languages
        assert "ta" in languages
        assert "te" in languages
        assert "mr" in languages

    def test_get_fixture_by_language(self):
        hi_fix = get_fixture_by_language("hi")
        assert hi_fix is not None
        assert "भारत" in hi_fix["transcript"]


class TestLatencyPercentileCalculator:
    """Test suite for P50-P100 latency percentiles and <200ms compliance."""

    def test_percentiles_calculation(self):
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        pcts = compute_latency_percentiles(latencies)

        assert pcts["p50"] == 55.0
        assert pcts["p100"] == 100.0
        assert pcts["mean"] == 55.0
        assert pcts["under_200ms"]["overall_compliant"] is True

    def test_over_200ms_compliance_detection(self):
        latencies = [50.0, 150.0, 250.0]
        pcts = compute_latency_percentiles(latencies)

        assert pcts["under_200ms"]["p50"] is True
        assert pcts["under_200ms"]["p100"] is False
        assert pcts["under_200ms"]["overall_compliant"] is False


class TestIRMetricsEvaluation:
    """Test suite for Recall@1, 5, 10 and MRR."""

    def test_perfect_retrieval_metrics(self):
        records = [
            {"retrieved_passage_ids": ["p1", "p2", "p3"], "relevant_passage_ids": ["p1"]},
            {"retrieved_passage_ids": ["p4", "p5", "p6"], "relevant_passage_ids": ["p4"]},
        ]
        metrics = compute_retrieval_metrics(records)
        assert metrics["recall@1"] == 100.0
        assert metrics["mrr"] == 1.0


class TestSarvamMockIsolation:
    """
    CRITICAL TEST: Ensures that benchmark code never calls the real Sarvam API.
    Fails if any benchmark evaluation triggers live network requests to Sarvam.
    """

    def test_benchmark_never_calls_real_sarvam_api(self):
        with patch.object(httpx.Client, "post") as mock_http:
            # Load fixtures
            fixtures = load_voice_fixtures()
            assert len(fixtures) > 0

            # Run benchmark stress test
            queries = ["भारत की राजधानी क्या है?", "What is machine learning?"]
            res = run_stress_test(queries=queries, concurrency_levels=[2])
            assert "concurrency_2" in res

            # Verify that HTTP POST to Sarvam was NEVER called
            sarvam_calls = [
                call for call in mock_http.call_args_list
                if "sarvam.ai" in str(call)
            ]
            assert len(sarvam_calls) == 0, "CRITICAL ERROR: Benchmark code made live Sarvam API calls!"
