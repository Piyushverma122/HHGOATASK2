import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_request_id_generated():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0
        assert "X-Process-Time" in response.headers


@pytest.mark.asyncio
async def test_request_id_preserved_when_passed():
    custom_request_id = "test-req-id-12345"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/v1/health",
            headers={"X-Request-ID": custom_request_id},
        )
        assert response.status_code == 200
        assert response.headers.get("X-Request-ID") == custom_request_id
