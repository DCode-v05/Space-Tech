import pytest
import torch
import numpy as np
from src.models.model import LSTMModel, TransformerModel, EnsembleModel

@pytest.fixture
def model_params():
    """Create sample model parameters."""
    return {
        'input_size': 5,
        'sequence_length': 10,
        'batch_size': 32
    }

@pytest.fixture
def sample_batch(model_params):
    """Create sample batch data."""
    batch_size = model_params['batch_size']
    seq_len = model_params['sequence_length']
    input_size = model_params['input_size']
    
    X = torch.randn(batch_size, seq_len, input_size)
    y = torch.randn(batch_size, input_size)
    return X, y

def test_lstm_model(model_params, sample_batch):
    """Test LSTM model functionality."""
    model = LSTMModel(input_size=model_params['input_size'])
    X, _ = sample_batch
    
    # Test forward pass
    output = model(X)
    assert output.shape == (model_params['batch_size'], model_params['input_size'])
    
    # Test model parameters
    assert isinstance(model.lstm, torch.nn.LSTM)
    assert isinstance(model.fc, torch.nn.Linear)
    
    # Test output range (after sigmoid)
    assert torch.all(output >= 0) and torch.all(output <= 1)

def test_transformer_model(model_params, sample_batch):
    """Test Transformer model functionality."""
    model = TransformerModel(input_size=model_params['input_size'])
    X, _ = sample_batch
    
    # Test forward pass
    output = model(X)
    assert output.shape == (model_params['batch_size'], model_params['input_size'])
    
    # Test model components
    assert isinstance(model.transformer_encoder, torch.nn.TransformerEncoder)
    assert isinstance(model.fc, torch.nn.Linear)

def test_ensemble_model(model_params, sample_batch):
    """Test Ensemble model functionality."""
    model = EnsembleModel(input_size=model_params['input_size'])
    X, _ = sample_batch
    
    # Test forward pass
    output = model(X)
    assert output.shape == (model_params['batch_size'], model_params['input_size'])
    
    # Test submodels
    assert isinstance(model.lstm_model, LSTMModel)
    assert isinstance(model.transformer_model, TransformerModel)
    
    # Test ensemble weights
    assert isinstance(model.lstm_weight, torch.nn.Parameter)
    assert isinstance(model.transformer_weight, torch.nn.Parameter)
    
    # Test weight normalization
    weights_sum = torch.sigmoid(model.lstm_weight) + torch.sigmoid(model.transformer_weight)
    assert abs(weights_sum.item() - 1.0) < 1e-6

def test_model_training(model_params, sample_batch):
    """Test model training functionality."""
    model = EnsembleModel(input_size=model_params['input_size'])
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    
    X, y = sample_batch
    initial_loss = None
    
    # Train for a few steps
    for _ in range(5):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        
        if initial_loss is None:
            initial_loss = loss.item()
            
        loss.backward()
        optimizer.step()
    
    # Check if loss decreased
    final_loss = loss.item()
    assert final_loss < initial_loss

def test_model_save_load(model_params, sample_batch, tmp_path):
    """Test model saving and loading functionality."""
    model = EnsembleModel(input_size=model_params['input_size'])
    X, _ = sample_batch
    
    # Get initial predictions
    initial_output = model(X)
    
    # Save model
    save_path = tmp_path / "test_model.pth"
    torch.save(model.state_dict(), save_path)
    
    # Create new model and load weights
    loaded_model = EnsembleModel(input_size=model_params['input_size'])
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model.eval()
    
    # Get predictions from loaded model
    loaded_output = loaded_model(X)
    
    # Check if outputs match
    assert torch.allclose(initial_output, loaded_output)