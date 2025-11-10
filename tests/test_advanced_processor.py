"""
Tests for AdvancedGNSSProcessor class.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add source directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.advanced_processor import AdvancedGNSSProcessor

# Test data generation
def create_test_data(n_samples=100, n_satellites=3, freq='H'):
    """Create test data for GNSS processor."""
    np.random.seed(42)
    base_date = datetime(2023, 1, 1)
    timestamps = [base_date + timedelta(hours=i) for i in range(n_samples)]
    
    data = []
    for sat_id in [f'G{i:02d}' for i in range(1, n_satellites + 1)]:
        sat_data = {
            'utc_time': timestamps,
            'sat_id': [sat_id] * n_samples,
            'clock_error': np.random.normal(0, 1, n_samples).cumsum(),
            'ephemeris_error': np.random.normal(0, 0.5, n_samples).cumsum(),
            'elevation': np.random.uniform(5, 90, n_samples),
            'azimuth': np.random.uniform(0, 360, n_samples)
        }
        data.append(pd.DataFrame(sat_data))
    
    return pd.concat(data).sort_values(['utc_time', 'sat_id']).reset_index(drop=True)

# Fixtures
@pytest.fixture
def sample_data():
    """Sample data fixture."""
    return create_test_data(n_samples=100, n_satellites=3)

@pytest.fixture
def processor_config():
    """Processor configuration fixture."""
    return {
        'scaler_params': {'with_centering': True, 'with_scaling': True},
        'data': {
            'rolling_windows': [6, 12, 24],
            'target_columns': ['clock_error', 'ephemeris_error']
        }
    }

# Tests
def test_processor_initialization(processor_config):
    """Test processor initialization."""
    processor = AdvancedGNSSProcessor(processor_config)
    assert processor is not None
    assert not processor._fitted

def test_fit_processor(sample_data, processor_config):
    """Test fitting the processor."""
    processor = AdvancedGNSSProcessor(processor_config)
    target_cols = processor_config['data']['target_columns']
    
    # Test successful fit
    processor = processor.fit(sample_data, target_columns=target_cols)
    assert processor._fitted
    assert processor.target_columns == target_cols
    assert processor.feature_processor is not None

def test_transform_processor(sample_data, processor_config):
    """Test transforming data with fitted processor."""
    # Setup
    processor = AdvancedGNSSProcessor(processor_config)
    target_cols = processor_config['data']['target_columns']
    processor = processor.fit(sample_data, target_columns=target_cols)
    
    # Test transform
    X, y = processor.transform(sample_data)
    
    # Assertions
    assert X.shape[0] == len(sample_data)
    assert y.shape == (len(sample_data), len(target_cols))
    assert not np.any(np.isnan(X))
    assert not np.any(np.isnan(y))

def test_inverse_transform(processor_config):
    """Test inverse transformation of targets."""
    # Create test data with known range
    data = pd.DataFrame({
        'utc_time': pd.date_range('2023-01-01', periods=10, freq='H'),
        'sat_id': ['G01'] * 10,
        'clock_error': np.linspace(0, 10, 10),
        'ephemeris_error': np.linspace(-5, 5, 10)
    })
    
    # Fit processor
    processor = AdvancedGNSSProcessor(processor_config)
    processor = processor.fit(data, target_columns=['clock_error', 'ephemeris_error'])
    
    # Transform and inverse transform
    _, y_scaled = processor.transform(data)
    y_inv = processor.inverse_transform_targets(y_scaled)
    
    # Check if inverse transform recovers original data (approximately)
    np.testing.assert_allclose(y_inv, data[['clock_error', 'ephemeris_error']], rtol=1e-6)

def test_error_handling(processor_config):
    """Test error conditions."""
    # Test empty DataFrame
    with pytest.raises(ValueError):
        processor = AdvancedGNSSProcessor(processor_config)
        processor.fit(pd.DataFrame(), target_columns=['clock_error'])
    
    # Test missing target columns
    with pytest.raises(ValueError):
        data = pd.DataFrame({'utc_time': [datetime.now()], 'value': [1.0]})
        processor = AdvancedGNSSProcessor(processor_config)
        processor.fit(data, target_columns=['nonexistent'])
    
    # Test transform before fit
    with pytest.raises(RuntimeError):
        processor = AdvancedGNSSProcessor(processor_config)
        processor.transform(pd.DataFrame({'utc_time': [datetime.now()]}))

def test_save_load_processor(tmp_path, sample_data, processor_config):
    """Test saving and loading the processor."""
    # Setup
    save_path = tmp_path / "test_processor.joblib"
    target_cols = processor_config['data']['target_columns']
    
    # Create and fit processor
    processor = AdvancedGNSSProcessor(processor_config)
    processor = processor.fit(sample_data, target_columns=target_cols)
    
    # Save and load
    processor.save(save_path)
    loaded_processor = AdvancedGNSSProcessor.load(save_path)
    
    # Test loaded processor
    assert loaded_processor._fitted
    assert loaded_processor.target_columns == target_cols
    
    # Test transform with loaded processor
    X_orig, y_orig = processor.transform(sample_data)
    X_loaded, y_loaded = loaded_processor.transform(sample_data)
    
    np.testing.assert_array_equal(X_orig, X_loaded)
    np.testing.assert_array_equal(y_orig, y_loaded)

def test_feature_generation(sample_data, processor_config):
    """Test if features are generated correctly."""
    # Fit processor
    processor = AdvancedGNSSProcessor(processor_config)
    target_cols = processor_config['data']['target_columns']
    processor = processor.fit(sample_data, target_columns=target_cols)
    
    # Get transformed data
    X, _ = processor.transform(sample_data)
    
    # Check if temporal features were added
    assert 'hour_sin' in processor.feature_processor.get_feature_names_out()
    assert 'hour_cos' in processor.feature_processor.get_feature_names_out()
    
    # Check if rolling features were added
    for window in processor_config['data']['rolling_windows']:
        assert f'clock_error_rolling_mean_{window}' in processor.feature_processor.get_feature_names_out()
        assert f'ephemeris_error_rolling_std_{window}' in processor.feature_processor.get_feature_names_out()
    
    # Check if satellite type was one-hot encoded
    assert 'satellite_type_GEO' in processor.feature_processor.get_feature_names_out()
    assert 'satellite_type_MEO' in processor.feature_processor.get_feature_names_out()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
