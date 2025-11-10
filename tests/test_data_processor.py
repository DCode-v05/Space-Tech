import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.data_processor import GNSSDataProcessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='15min'),
        'satellite_id': ['G01'] * 100,
        'clock_error': np.random.normal(0, 0.1, 100),
        'ephemeris_error': np.random.normal(0, 0.2, 100),
        'system_type': ['GPS'] * 100
    })
    return data

@pytest.fixture
def data_processor():
    """Create a GNSSDataProcessor instance."""
    return GNSSDataProcessor()

def test_load_data(data_processor, tmp_path):
    """Test data loading functionality."""
    # Create temporary test data
    test_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='15min'),
        'satellite_id': ['G01'] * 10,
        'clock_error': np.random.normal(0, 0.1, 10),
        'ephemeris_error': np.random.normal(0, 0.2, 10),
        'system_type': ['GPS'] * 10
    })
    
    # Save test data
    test_file = tmp_path / "test_data.csv"
    test_data.to_csv(test_file, index=False)
    
    # Test loading
    loaded_data = data_processor.load_data(str(test_file))
    assert isinstance(loaded_data, pd.DataFrame)
    assert len(loaded_data) == len(test_data)
    assert all(col in loaded_data.columns for col in test_data.columns)

def test_preprocess_data(data_processor, sample_data):
    """Test data preprocessing functionality."""
    processed_data = data_processor.preprocess_data(sample_data)
    
    # Check output type and shape
    assert isinstance(processed_data, pd.DataFrame)
    assert len(processed_data) == len(sample_data)
    
    # Check if timestamps are sorted
    assert processed_data['timestamp'].is_monotonic_increasing
    
    # Check if categorical variables are encoded
    assert 'system_type' not in processed_data.columns
    assert 'satellite_id' not in processed_data.columns
    
    # Check if numerical features are normalized
    for col in ['clock_error', 'ephemeris_error']:
        assert processed_data[col].mean() == pytest.approx(0, abs=0.1)
        assert processed_data[col].std() == pytest.approx(1, abs=0.1)

def test_create_sequences(data_processor, sample_data):
    """Test sequence creation functionality."""
    processed_data = data_processor.preprocess_data(sample_data)
    X, y = data_processor.create_sequences(processed_data)
    
    # Check output types
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    
    # Check shapes
    sequence_length = data_processor.sequence_length
    assert X.shape[1] == sequence_length
    assert X.shape[0] == y.shape[0]
    assert y.shape[1] == processed_data.shape[1]

def test_split_data(data_processor):
    """Test data splitting functionality."""
    # Create sample sequences
    X = np.random.rand(100, 10, 5)
    y = np.random.rand(100, 5)
    
    # Test splitting
    X_train, X_val, y_train, y_val = data_processor.split_data(X, y)
    
    # Check shapes
    assert len(X_train) > len(X_val)
    assert len(X_train) + len(X_val) == len(X)
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)

def test_process_pipeline(data_processor, sample_data):
    """Test the complete processing pipeline."""
    X_train, X_val, y_train, y_val = data_processor.process_pipeline(sample_data)
    
    # Check output types
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_val, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_val, np.ndarray)
    
    # Check shapes
    assert X_train.shape[2] == y_train.shape[1]
    assert X_val.shape[2] == y_val.shape[1]
    assert X_train.shape[1] == data_processor.sequence_length