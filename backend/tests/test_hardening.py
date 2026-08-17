import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings


@pytest.fixture
def client():
    return TestClient(app)


class TestHardenedHealthEndpoint:
    """Test suite for sanitized health check telemetry."""

    def test_health_check_sanitized_status(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "ok"
        assert "service" in data
        assert "environment" in data
        assert "providers" in data

        # Verify Sarvam status does NOT expose API key
        sarvam = data["providers"]["sarvam"]
        assert "configured" in sarvam
        assert "model" in sarvam
        assert "SARVAM_API_KEY" not in str(data)
        assert "sk_" not in str(data)

        # Verify LLM status does NOT expose API key
        llm = data["providers"]["llm"]
        assert "configured" in llm
        assert "LLM_API_KEY" not in str(data)


class TestRetrievalInspectionEndpoint:
    """Test suite for /api/v1/rag/inspect transparent candidate pipeline."""

    def test_retrieval_inspect_pipeline(self, client):
        payload = {
            "query": "भारत की राजधानी क्या है?",
            "strategy": "adaptive",
            "top_k": 3,
            "enable_reranking": True,
        }
        resp = client.post("/api/v1/rag/inspect", json=payload)
        assert resp.status_code == 200
        json_data = resp.json()

        assert json_data["success"] is True
        data = json_data["data"]
        assert "dense_candidates" in data
        assert "bm25_candidates" in data
        assert "fused_candidates" in data
        assert "reranked_results" in data
        assert "final_context" in data
        assert len(data["final_context"]) <= 3


class TestErrorHandlingSecurity:
    """Test suite ensuring zero stack trace leakage on bad input."""

    def test_invalid_rag_query_payload(self, client):
        resp = client.post("/api/v1/rag/query", json={})
        assert resp.status_code == 422
        data = resp.json()
        # Verify clean structured validation error without Python traceback
        assert "Traceback" not in str(data)
        assert "File " not in str(data)


class TestRateLimitingMiddleware:
    """Test suite for in-memory rate limiting."""

    def test_rate_limiter_in_test_environment(self, client):
        # In test mode, rate limiter should be bypassed
        for _ in range(5):
            resp = client.post(
                "/api/v1/rag/query",
                json={"query": "भारत की राजधानी", "strategy": "adaptive", "top_k": 2},
            )
            assert resp.status_code == 200
