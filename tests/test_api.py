import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "endpoints" in response.json()


def test_inferir_ramo_success():
    response = client.post("/api/pecas/1", json={"texto": "Texto de exemplo"})
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["id"] == 1
    assert isinstance(response_json["ramo_direito"], list)
    assert all(isinstance(ramo, str) for ramo in response_json["ramo_direito"])


def test_inferir_ramo_invalid_id():
    response = client.post(
        "/api/pecas/abc", json={"texto": "Texto de exemplo"}  # ID inválido
    )
    assert response.status_code == 422  # Validation error


def test_inferir_ramo_missing_text():
    response = client.post("/api/pecas/1", json={})  # Texto faltando
    assert response.status_code == 422  # Validation error


def test_inferir_ramo_empty_text():
    response = client.post("/api/pecas/1", json={"texto": ""})  # Texto vazio
    assert response.status_code == 200
    assert response.json()["id"] == 1
    assert isinstance(response.json()["ramo_direito"], list)
