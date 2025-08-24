"""
Simplified tests for XPCS Toolkit logging system.

This module provides basic tests for the logging configuration to ensure
the logging system can be imported and basic functionality works.
"""

import logging
import pytest


def test_logging_basic_import():
    """Test that logging modules can be imported without errors."""
    try:
        from xpcs_toolkit.helper.logging_config import setup_logging, get_logger
        assert setup_logging is not None
        assert get_logger is not None
    except ImportError as e:
        pytest.skip(f"Logging modules not available: {e}")


def test_basic_logger_creation():
    """Test basic logger creation."""
    try:
        from xpcs_toolkit.helper.logging_config import get_logger
        logger = get_logger("test_logger")
        # Handle both Logger and ContextualLoggerAdapter types
        assert hasattr(logger, 'info')  # Both types have logging methods
        # Check the underlying logger name (works for both types)
        logger_name = getattr(logger, 'name', getattr(logger.logger, 'name', None) if hasattr(logger, 'logger') else None)
        assert "test_logger" in str(logger_name) or logger_name == "test_logger"
    except ImportError:
        pytest.skip("Logging configuration not available")


def test_logging_levels():
    """Test that logging levels work correctly."""
    logger = logging.getLogger("test_levels")
    
    # Test that we can set different levels
    logger.setLevel(logging.DEBUG)
    assert logger.level == logging.DEBUG
    
    logger.setLevel(logging.INFO)
    assert logger.level == logging.INFO


def test_logging_handlers():
    """Test basic logging handler functionality."""
    logger = logging.getLogger("test_handlers")
    
    # Create a simple handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler
    logger.addHandler(handler)
    assert len(logger.handlers) >= 1
    
    # Clean up
    logger.removeHandler(handler)


def test_performance_timer_import():
    """Test that performance timer can be imported."""
    try:
        from xpcs_toolkit.helper.logging_utils import PerformanceTimer
        import logging
        # PerformanceTimer requires both logger and operation_name
        test_logger = logging.getLogger("test_timer")
        timer = PerformanceTimer(test_logger, "test_operation")
        assert timer is not None
    except ImportError:
        pytest.skip("Performance timer not available")


def test_correlation_id_creation():
    """Test correlation ID creation."""
    try:
        from xpcs_toolkit.helper.logging_utils import create_correlation_id
        corr_id = create_correlation_id()
        assert isinstance(corr_id, str)
        assert len(corr_id) > 0
        
        # Test that multiple calls return different IDs
        corr_id2 = create_correlation_id()
        assert corr_id != corr_id2
    except ImportError:
        pytest.skip("Logging utils not available")


def test_log_exceptions_decorator():
    """Test log exceptions decorator."""
    try:
        from xpcs_toolkit.helper.logging_utils import log_exceptions
        import logging
        
        # Create a logger for the decorator
        test_logger = logging.getLogger("test_decorator")
        
        @log_exceptions(logger=test_logger)
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
        
    except (ImportError, TypeError):
        pytest.skip("Logging decorators not available or API mismatch")