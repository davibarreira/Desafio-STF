from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_inferir_ramo():
    response = client.post(
        "/api/pecas/1",
        json={"texto": "Texto de exemplo"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "id": 1,
        "ramo_direito": ["Direito Civil"]
    }
