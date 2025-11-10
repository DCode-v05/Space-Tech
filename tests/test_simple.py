def test_simple():
    assert 1 + 1 == 2

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
