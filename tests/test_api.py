import pytest
from fastapi.testclient import TestClient
import numpy as np
import json
from pathlib import Path
import torch

from src.api.main import app, ModelService

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)

@pytest.fixture
def sample_request():
    """Create a sample prediction request."""
    return {
        "sequence": [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(10)],
        "prediction_horizon": 4
    }

@pytest.fixture
def model_service(tmp_path):
    """Create a ModelService instance with a mock model."""
    service = ModelService()
    
    # Create mock model and config
    input_size = 5
    mock_config = {
        "input_size": input_size,
        "batch_size": 32,
        "learning_rate": 0.001
    }
    
    # Save mock config
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(mock_config, f)
    
    # Create and save mock model
    from src.models.model import EnsembleModel
    mock_model = EnsembleModel(input_size=input_size)
    model_path = tmp_path / "model.pth"
    torch.save(mock_model.state_dict(), model_path)
    
    # Load model
    service.load_model(str(model_path), str(config_path))
    return service

def test_health_endpoint(test_client):
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint(test_client, sample_request):
    """Test the prediction endpoint."""
    response = test_client.post("/predict", json=sample_request)
    assert response.status_code == 200
    
    result = response.json()
    assert "predictions" in result
    assert "confidence_intervals" in result
    assert len(result["predictions"]) == sample_request["prediction_horizon"]
    assert "lower_95" in result["confidence_intervals"]
    assert "upper_95" in result["confidence_intervals"]

def test_model_service_initialization(model_service):
    """Test ModelService initialization."""
    assert model_service.model is not None
    assert model_service.config is not None
    assert isinstance(model_service.device, torch.device)

def test_model_service_prediction(model_service, sample_request):
    """Test ModelService prediction functionality."""
    result = model_service.predict(
        sample_request["sequence"],
        sample_request["prediction_horizon"]
    )
    
    assert "predictions" in result
    assert "confidence_intervals" in result
    assert len(result["predictions"]) == sample_request["prediction_horizon"]
    assert len(result["confidence_intervals"]["lower_95"]) == sample_request["prediction_horizon"]
    assert len(result["confidence_intervals"]["upper_95"]) == sample_request["prediction_horizon"]

def test_invalid_sequence_length(test_client, sample_request):
    """Test handling of invalid sequence length."""
    # Create request with too short sequence
    invalid_request = sample_request.copy()
    invalid_request["sequence"] = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    
    response = test_client.post("/predict", json=invalid_request)
    assert response.status_code == 500
    assert "error" in response.json()["detail"].lower()

def test_invalid_feature_count(test_client, sample_request):
    """Test handling of invalid feature count."""
    # Create request with wrong number of features
    invalid_request = sample_request.copy()
    invalid_request["sequence"] = [[0.1, 0.2] for _ in range(10)]
    
    response = test_client.post("/predict", json=invalid_request)
    assert response.status_code == 500
    assert "error" in response.json()["detail"].lower()

def test_model_service_error_handling(model_service):
    """Test ModelService error handling."""
    with pytest.raises(RuntimeError):
        # Reset model to None to simulate loading error
        model_service.model = None
        model_service.predict([[0.1, 0.2, 0.3, 0.4, 0.5]], 4)