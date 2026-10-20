
from fastapi.testclient import TestClient

from src.mlops_heart.api import app


VALID_PATIENT = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 1,
    "ca": 0,
    "thal": 6,
}


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_prediction_endpoint():
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PATIENT)

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in (0, 1)
    assert result["label"] in (
        "Heart disease present",
        "Heart disease absent",
    )
    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["probability_presence"] <= 1
    assert "not medical diagnosis" in result["disclaimer"]


def test_invalid_age_is_rejected():
    patient = {**VALID_PATIENT, "age": 0}

    with TestClient(app) as client:
        response = client.post("/predict", json=patient)

    assert response.status_code == 422


def test_unexpected_feature_is_rejected():
    patient = {**VALID_PATIENT, "unexpected": 123}

    with TestClient(app) as client:
        response = client.post("/predict", json=patient)

    assert response.status_code == 422
