"""
Shared test fixtures and configuration.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture(scope="session")
def sample_data():
    """Create sample GNSS data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_satellites = 3
    
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

@pytest.fixture(scope="session")
def processor_config():
    """Default processor configuration for testing."""
    return {
        'scaler_params': {'with_centering': True, 'with_scaling': True},
        'data': {
            'rolling_windows': [6, 12, 24],
            'target_columns': ['clock_error', 'ephemeris_error']
        }
    }

@pytest.fixture(scope="session")
def fitted_processor(sample_data, processor_config):
    """Return a pre-fitted processor."""
    from src.data.advanced_processor import AdvancedGNSSProcessor
    processor = AdvancedGNSSProcessor(processor_config)
    return processor.fit(sample_data, target_columns=processor_config['data']['target_columns'])
