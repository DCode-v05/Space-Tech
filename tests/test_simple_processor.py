"""
Simple test for AdvancedGNSSProcessor.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add source directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.advanced_processor import AdvancedGNSSProcessor

def create_test_data(n_samples=10, n_satellites=2):
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

def test_processor():
    """Test the AdvancedGNSSProcessor with a simple case."""
    # Redirect stdout to a file for better debugging
    import sys
    original_stdout = sys.stdout
    debug_file = open('test_debug_output.txt', 'w')
    sys.stdout = debug_file
    
    try:
        # Create test data
        data = create_test_data(n_samples=10, n_satellites=2)
    
        # Create processor config
        config = {
            'scaler_params': {'with_centering': True, 'with_scaling': True},
            'data': {
                'rolling_windows': [6, 12],
                'target_columns': ['clock_error', 'ephemeris_error']
            }
        }
        
        # Initialize processor
        processor = AdvancedGNSSProcessor(config)
        
        # Fit processor
        processor = processor.fit(data, target_columns=config['data']['target_columns'])
        
        # Transform data
        X, y = processor.transform(data)
    
        # Print feature names and their values for debugging
        print("\nFeature names:")
        if hasattr(processor.feature_processor, 'get_feature_names_out'):
            feature_names = processor.feature_processor.get_feature_names_out()
            print(f"Number of features: {len(feature_names)}")
            print("First 10 feature names:", feature_names[:10])
        else:
            print("Feature processor does not support get_feature_names_out()")
        
        # Check results
        print(f"\nX shape: {X.shape}, y shape: {y.shape}")
        assert X.shape[0] == len(data), f"Expected X to have {len(data)} rows, got {X.shape[0]}"
        assert y.shape == (len(data), len(config['data']['target_columns'])), \
            f"Expected y to have shape {(len(data), len(config['data']['target_columns']))}, got {y.shape}"
        
        # Check for NaN values in X with more details
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            nan_count = np.sum(nan_mask, axis=0)
            print("\nNaN values per feature:")
            for i, count in enumerate(nan_count):
                if count > 0:
                    feat_name = feature_names[i] if 'feature_names' in locals() else f"feature_{i}"
                    print(f"  {feat_name}: {count} NaN values")
            
            # Print first few rows with NaN values
            print("\nFirst 5 rows with NaN values:")
            nan_rows = np.any(nan_mask, axis=1)
            for i in range(min(5, np.sum(nan_rows))):
                row_idx = np.where(nan_rows)[0][i]
                print(f"\nRow {row_idx} (NaN at columns: {np.where(nan_mask[row_idx])[0].tolist()}):")
                for j in range(X.shape[1]):
                    if nan_mask[row_idx, j]:
                        feat_name = feature_names[j] if 'feature_names' in locals() else f"feature_{j}"
                        print(f"  {feat_name}: {X[row_idx, j]}")
        
        assert not np.any(nan_mask), "Found NaN values in X"
        
        # Check for NaN values in y
        nan_mask_y = np.isnan(y)
        if np.any(nan_mask_y):
            print("\nNaN values in y:")
            nan_count_y = np.sum(nan_mask_y, axis=0)
            for i, count in enumerate(nan_count_y):
                if count > 0:
                    print(f"  Target {i} has {count} NaN values")
        
        assert not np.any(nan_mask_y), "Found NaN values in y"
        
        # Test inverse transform
        y_inv = processor.inverse_transform_targets(y)
        assert y_inv.shape == (len(data), len(config['data']['target_columns']))
        
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise
    finally:
        # Restore stdout and close the file
        sys.stdout = original_stdout
        debug_file.close()
        print("Test completed. Check test_debug_output.txt for detailed output.")

if __name__ == "__main__":
    test_processor()
    print("All tests passed!")
