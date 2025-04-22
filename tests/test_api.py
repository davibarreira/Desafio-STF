from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "endpoints" in response.json()

def test_inferir_ramo_success():
    response = client.post(
        "/api/pecas/1",
        json={"texto": "Texto de exemplo"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "id": 1,
        "ramo_direito": ["Direito Civil"]
    }

def test_inferir_ramo_invalid_id():
    response = client.post(
        "/api/pecas/abc",  # ID inválido
        json={"texto": "Texto de exemplo"}
    )
    assert response.status_code == 422  # Validation error

def test_inferir_ramo_missing_text():
    response = client.post(
        "/api/pecas/1",
        json={}  # Texto faltando
    )
    assert response.status_code == 422  # Validation error

def test_inferir_ramo_empty_text():
    response = client.post(
        "/api/pecas/1",
        json={"texto": ""}  # Texto vazio
    )
    assert response.status_code == 200
    assert response.json()["id"] == 1
    assert isinstance(response.json()["ramo_direito"], list)