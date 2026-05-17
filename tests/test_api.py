"""API endpoint tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model_loaded" in data


def test_home_page():
    """Test home page renders."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Flight Arrival Delay Predictor" in response.content


def test_data_page():
    """Test data page renders."""
    response = client.get("/data")
    assert response.status_code == 200


def test_predict_valid_input():
    """Test prediction with valid input."""
    payload = {
        "departure_time": 900.0,
        "departure_delay": 15.0,
        "scheduled_time": 120.0,
        "arrival_time": 1020.0,
    }
    response = client.post("/api/predict", json=payload)
    
    # If model is not loaded, expect 503
    if response.status_code == 503:
        assert "Model not loaded" in response.json()["detail"]
    else:
        assert response.status_code == 200
        data = response.json()
        assert "predicted_arrival_delay" in data
        assert "model_score" in data
        assert isinstance(data["predicted_arrival_delay"], (int, float))


def test_predict_invalid_input():
    """Test prediction with invalid input."""
    payload = {
        "departure_time": 2000.0,  # Invalid: > 1439
        "departure_delay": 15.0,
        "scheduled_time": 120.0,
        "arrival_time": 1020.0,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_missing_fields():
    """Test prediction with missing fields."""
    payload = {
        "departure_time": 900.0,
        # Missing other required fields
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_model_info():
    """Test model info endpoint."""
    response = client.get("/api/model-info")
    
    # If model is not loaded, expect 503
    if response.status_code == 503:
        assert "Model not loaded" in response.json()["detail"]
    else:
        assert response.status_code == 200
        data = response.json()
        assert "loaded" in data

# Made with Bob
