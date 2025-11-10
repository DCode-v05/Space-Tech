"""
Run tests for the GNSS Error Prediction System.
"""
import sys
import pytest

def main():
    """Run the test suite."""
    # Add the project root to the Python path
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Run the tests
    exit_code = pytest.main([
        "tests/test_advanced_processor.py",
        "-v",
        "--import-mode=importlib"
    ])
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
