"""Configure pytest for AI Disruption Agent testing."""

import pytest
import logging

# Register custom markers
def pytest_configure(config):
    """Register custom markers to avoid warnings."""
    config.addinivalue_line(
        "markers", 
        "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )

@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for all tests to work with caplog."""
    # Get the root logger and set it up for testing
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers to ensure clean test state
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Don't add any handlers - let caplog handle capture
    yield logger
    
    # Cleanup after test
    for handler in logger.handlers[:]:
        logger.removeHandler(handler) 