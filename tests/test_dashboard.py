import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import json
import requests

from src.dashboard.app import (
    load_model_metrics,
    plot_predictions,
    plot_error_distribution,
    make_prediction_request
)

@pytest.fixture
def sample_metrics():
    """Create sample model metrics."""
    return {
        "train_loss": [0.5, 0.4, 0.3, 0.2, 0.1],
        "val_loss": [0.6, 0.5, 0.4, 0.3, 0.2],
        "lstm_weights": [0.6, 0.55, 0.52, 0.51, 0.50],
        "transformer_weights": [0.4, 0.45, 0.48, 0.49, 0.50]
    }

@pytest.fixture
def sample_predictions():
    """Create sample prediction data."""
    return {
        "predictions": [0.1, 0.2, 0.3, 0.4],
        "confidence_intervals": {
            "lower_95": [0.05, 0.15, 0.25, 0.35],
            "upper_95": [0.15, 0.25, 0.35, 0.45]
        }
    }

@pytest.fixture
def sample_history_file(tmp_path, sample_metrics):
    """Create a sample history.json file."""
    history_path = tmp_path / "history.json"
    with open(history_path, "w") as f:
        json.dump(sample_metrics, f)
    return history_path

def test_load_model_metrics(sample_history_file):
    """Test loading model metrics from file."""
    metrics = load_model_metrics(str(sample_history_file))
    
    assert isinstance(metrics, dict)
    assert "train_loss" in metrics
    assert "val_loss" in metrics
    assert "lstm_weights" in metrics
    assert "transformer_weights" in metrics
    
    assert len(metrics["train_loss"]) > 0
    assert len(metrics["val_loss"]) > 0
    assert len(metrics["lstm_weights"]) > 0
    assert len(metrics["transformer_weights"]) > 0

@patch("matplotlib.pyplot")
def test_plot_predictions(mock_plt, sample_predictions):
    """Test prediction plotting functionality."""
    timestamps = pd.date_range(start="2023-01-01", periods=4, freq="15min")
    plot_predictions(timestamps, sample_predictions)
    
    # Verify plot was created
    mock_plt.figure.assert_called_once()
    mock_plt.plot.assert_called()
    mock_plt.fill_between.assert_called()
    mock_plt.xlabel.assert_called_once()
    mock_plt.ylabel.assert_called_once()
    mock_plt.title.assert_called_once()

@patch("matplotlib.pyplot")
def test_plot_error_distribution(mock_plt):
    """Test error distribution plotting functionality."""
    errors = np.random.normal(0, 1, 1000)
    plot_error_distribution(errors)
    
    # Verify histogram was created
    mock_plt.figure.assert_called_once()
    mock_plt.hist.assert_called_once()
    mock_plt.xlabel.assert_called_once()
    mock_plt.ylabel.assert_called_once()
    mock_plt.title.assert_called_once()

@patch("requests.post")
def test_make_prediction_request_success(mock_post):
    """Test successful prediction request."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "predictions": [0.1, 0.2],
        "confidence_intervals": {
            "lower_95": [0.05, 0.15],
            "upper_95": [0.15, 0.25]
        }
    }
    mock_post.return_value = mock_response
    
    # Make request
    sequence = [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(10)]
    result = make_prediction_request(sequence, 2)
    
    # Verify request
    assert result is not None
    assert "predictions" in result
    assert "confidence_intervals" in result
    mock_post.assert_called_once()

@patch("requests.post")
def test_make_prediction_request_failure(mock_post):
    """Test failed prediction request."""
    # Mock failed response
    mock_post.side_effect = requests.exceptions.RequestException("API Error")
    
    # Make request
    sequence = [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(10)]
    with pytest.raises(Exception):
        make_prediction_request(sequence, 2)
    
    # Verify request
    mock_post.assert_called_once()