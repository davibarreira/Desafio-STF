from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

@pytest.mark.asyncio
async def test_multiple_requests():
    # Create 5 concurrent requests using TestClient
    tasks = [
        client.post(f"/api/pecas/{i}", json={"texto": f"Texto {i}"})
        for i in range(5)
    ]
    
    # Verify all responses
    for i, response in enumerate(tasks):
        assert response.status_code == 200
        assert response.json()["id"] == i
        assert "ramo_direito" in response.json()
        assert isinstance(response.json()["ramo_direito"], list)
        assert len(response.json()["ramo_direito"]) > 0 