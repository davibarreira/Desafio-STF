import pandas as pd
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


# Load sample data
SAMPLE_DATA_PATH = "tests/sample_data/sample_data.parquet"


def test_inferir_ramo_sample_data():
    df = pd.read_parquet(SAMPLE_DATA_PATH)
    for index, row in df.iterrows():
        response = client.post(
            f"/api/pecas/{index}", json={"texto": row["texto_bruto"]}
        )
        assert response.status_code == 200
        assert response.json()["id"] == index
        assert isinstance(response.json()["ramo_direito"], list)

        # Compute accuracy by comparing predicted labels with actual labels
        predicted_labels = set(response.json()["ramo_direito"])
        actual_labels = set(row["ramo_direito"])

        # Calculate intersection and union
        intersection = predicted_labels.intersection(actual_labels)
        union = predicted_labels.union(actual_labels)

        # Calculate accuracy as intersection over union
        accuracy = len(intersection) / len(union) if len(union) > 0 else 0
        assert accuracy >= 0 and accuracy <= 1
