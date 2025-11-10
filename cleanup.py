"""Cleanup script to remove unnecessary files."""
import os
import shutil
from pathlib import Path

def remove_file(file_path):
    """Remove a file if it exists."""
    try:
        if file_path.exists():
            if file_path.is_file():
                file_path.unlink()
                print(f"Removed file: {file_path}")
            else:
                shutil.rmtree(file_path)
                print(f"Removed directory: {file_path}")
    except Exception as e:
        print(f"Error removing {file_path}: {e}")

def cleanup_project():
    """Remove unnecessary files and directories."""
    base_dir = Path(__file__).parent
    
    # Files to remove
    files_to_remove = [
        'dashboard.py',
        'test_auth.py',
        'test_dashboard.py',
        'test_processor.py',
        'test_output.py',
        'demo_processor.py',
        'debug_data_processor.py',
        'debug_data_processor.log',
        'dashboard_output.log',
        'train_output.log',
        'api_output.log',
        'error_output.txt',
        'demo_output.txt',
        'test_debug_output.txt',
        'output_and_error.log',
        'simple_test.py',
        'train.py',
        'setup.py',
        'pytest.ini'
    ]
    
    # Directories to remove
    dirs_to_remove = [
        'gnss_error_prediction.egg-info',
        '__pycache__',
        'results/analysis',
        'notebooks'
    ]
    
    # Remove files
    for file_name in files_to_remove:
        file_path = base_dir / file_name
        remove_file(file_path)
    
    # Remove directories
    for dir_name in dirs_to_remove:
        dir_path = base_dir / dir_name
        remove_file(dir_path)
    
    # Remove .pyc files
    for pyc_file in base_dir.rglob('*.pyc'):
        remove_file(pyc_file)
    
    # Remove __pycache__ directories
    for pycache_dir in base_dir.rglob('__pycache__'):
        remove_file(pycache_dir)

if __name__ == "__main__":
    print("Cleaning up project...")
    cleanup_project()
    print("Cleanup complete!")
