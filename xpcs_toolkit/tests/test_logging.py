"""
Unit tests for XPCS Toolkit logging system.

This module provides comprehensive tests for the centralized logging configuration,
performance monitoring, rate limiting, and contextual logging features.
"""

import logging
import os
import tempfile
import time
import json
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Import logging modules
try:
    from xpcs_toolkit.helper.logging_config import (
        setup_logging, get_default_config, get_logger, 
        JSONFormatter, ContextFilter, ContextualLoggerAdapter
    )
    from xpcs_toolkit.helper.logging_utils import (
        PerformanceTimer, RateLimitFilter, ProcessInfoFilter,
        add_process_info, create_correlation_id, log_exceptions,
        monitor_performance
    )
    from xpcs_toolkit.helper.logwriter import (
        LoggerWriter, AsyncLoggerWriter, redirect_std_streams
    )
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False

pytestmark = pytest.mark.skipif(not LOGGING_AVAILABLE, reason="Logging modules not available")


class TestLoggingConfiguration:
    """Test centralized logging configuration."""
    
    def test_get_default_config(self):
        """Test default configuration generation."""
        config = get_default_config()
        
        # Verify basic structure
        assert config['version'] == 1
        assert 'formatters' in config
        assert 'handlers' in config
        assert 'loggers' in config
        assert 'root' in config
        
        # Verify formatters
        assert 'simple' in config['formatters']
        assert 'detailed' in config['formatters']
        assert 'json' in config['formatters']
        
        # Verify handlers
        assert 'console' in config['handlers']
        assert 'rotating_file' in config['handlers']
        
        # Verify XPCS logger configuration
        assert 'xpcs_toolkit' in config['loggers']
        xpcs_logger = config['loggers']['xpcs_toolkit']
        assert xpcs_logger['level'] == 'DEBUG'
        assert not xpcs_logger['propagate']
    
    def test_get_default_config_custom_parameters(self):
        """Test default configuration with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            config = get_default_config(
                level="WARNING",
                file_path=str(log_file),
                format_style="json",
                max_size_mb=20,
                backup_count=3,
                include_console=False
            )\
            
            # Verify custom parameters
            assert config['root']['level'] == 'WARNING'
            assert 'console' not in config['handlers']
            
            file_handler = config['handlers']['rotating_file']
            assert file_handler['maxBytes'] == 20 * 1024 * 1024
            assert file_handler['backupCount'] == 3
            assert file_handler['formatter'] == 'json'
    
    def test_setup_logging_default(self):
        """Test setup_logging with defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)  # Change to temp dir for log file
            
            setup_logging()
            
            # Verify logger is configured
            logger = logging.getLogger('xpcs_toolkit.test')
            assert logger.isEnabledFor(logging.DEBUG)
            
            # Test logging works
            logger.info("Test message")
            assert Path("xpcs_toolkit.log").exists()
    
    def test_setup_logging_with_config(self):
        """Test setup_logging with custom configuration."""
        config = {
            "version": 1,
            "formatters": {
                "simple": {"format": "%(levelname)s - %(message)s"}
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "simple",
                    "level": "ERROR"
                }
            },
            "root": {
                "level": "ERROR",
                "handlers": ["console"]
            }
        }
        
        setup_logging(config)
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.ERROR
    
    def test_setup_logging_environment_variables(self):
        """Test setup_logging with environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "env_test.log"
            
            env_vars = {
                'XPCS_LOG_LEVEL': 'WARNING',
                'XPCS_LOG_FILE': str(log_file),
                'XPCS_LOG_FORMAT': 'simple'
            }
            
            with patch.dict(os.environ, env_vars):
                setup_logging()
                
                # Verify configuration applied
                logger = logging.getLogger('xpcs_toolkit.test')
                logger.warning("Test warning")
                
                assert log_file.exists()
    
    def test_get_logger_contextual(self):
        """Test contextual logger creation."""
        setup_logging()
        
        logger = get_logger(__name__, 
                           experiment_id="exp001",
                           user="test_user")
        
        assert isinstance(logger, ContextualLoggerAdapter)
        assert logger.extra['experiment_id'] == "exp001"
        assert logger.extra['user'] == "test_user"
        
        # Test logging with context
        with patch.object(logger.logger, 'log') as mock_log:
            logger.info("Test message", extra={"additional": "info"})
            
            # Verify context is merged
            call_args = mock_log.call_args
            extra = call_args[1]['extra']
            assert 'experiment_id' in extra
            assert 'user' in extra
            assert 'additional' in extra


class TestJSONFormatter:
    """Test JSON formatter functionality."""
    
    def test_json_formatter_basic(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        
        # Create test record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        data = json.loads(formatted)
        
        # Verify JSON structure
        assert data['level'] == 'INFO'
        assert data['logger'] == 'test.logger'
        assert data['message'] == 'Test message'
        assert data['line'] == 42
        assert 'timestamp' in data
        assert 'process_id' in data
    
    def test_json_formatter_with_exception(self):
        """Test JSON formatting with exception information."""
        formatter = JSONFormatter()
        
        try:
            raise ValueError("Test exception")
        except Exception:
            exc_info = True
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="/test/path.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=exc_info
            )
            
            formatted = formatter.format(record)
            data = json.loads(formatted)
            
            assert 'exception' in data
            assert data['exception']['type'] == 'ValueError'
            assert data['exception']['message'] == 'Test exception'
            assert 'traceback' in data['exception']
    
    def test_json_formatter_with_extra_fields(self):
        """Test JSON formatting with extra context fields."""
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.experiment_id = "exp001"
        record.file_count = 5
        record.custom_data = {"nested": "value"}
        
        formatted = formatter.format(record)
        data = json.loads(formatted)
        
        assert 'context' in data
        context = data['context']
        assert context['experiment_id'] == "exp001"
        assert context['file_count'] == 5
        assert context['custom_data'] == {"nested": "value"}


class TestPerformanceTimer:
    """Test performance monitoring functionality."""
    
    def test_performance_timer_context_manager(self):
        """Test PerformanceTimer as context manager."""
        logger = MagicMock()
        
        with PerformanceTimer(logger, "Test operation") as timer:
            time.sleep(0.1)  # Small delay for timing
        
        # Verify logging calls
        assert logger.log.call_count >= 2  # Start and completion
        
        # Check completion log call
        completion_calls = [call for call in logger.log.call_args_list 
                          if 'timing_complete' in str(call)]
        assert len(completion_calls) > 0
    
    def test_performance_timer_decorator(self):
        """Test PerformanceTimer as decorator."""
        logger = MagicMock()
        
        @PerformanceTimer(logger, "Decorated function")
        def test_function():
            time.sleep(0.05)
            return "result"
        
        result = test_function()
        
        assert result == "result"
        assert logger.log.call_count >= 2
    
    def test_performance_timer_with_item_count(self):
        """Test PerformanceTimer with throughput calculation."""
        logger = MagicMock()
        
        with PerformanceTimer(logger, "Batch processing", item_count=100):
            time.sleep(0.1)
        
        # Verify throughput was calculated
        completion_calls = [call for call in logger.log.call_args_list 
                          if 'throughput_items_per_sec' in str(call)]
        assert len(completion_calls) > 0
    
    def test_performance_timer_exception_handling(self):
        """Test PerformanceTimer behavior with exceptions."""
        logger = MagicMock()
        
        with pytest.raises(ValueError):
            with PerformanceTimer(logger, "Failing operation"):
                raise ValueError("Test error")
        
        # Verify exception was logged
        error_calls = [call for call in logger.log.call_args_list 
                      if call[0][0] == logging.ERROR]
        assert len(error_calls) > 0


class TestRateLimitFilter:
    """Test rate limiting functionality."""
    
    def test_rate_limit_filter_basic(self):
        """Test basic rate limiting."""
        rate_filter = RateLimitFilter(rate=2, per_seconds=1)
        
        # Create test records
        record1 = self._create_log_record("Test message 1")
        record2 = self._create_log_record("Test message 2")
        record3 = self._create_log_record("Test message 3")  # Should be filtered
        
        # Test filtering
        assert rate_filter.filter(record1) is True
        assert rate_filter.filter(record2) is True
        assert rate_filter.filter(record3) is False  # Over rate limit
    
    def test_rate_limit_filter_time_window(self):
        """Test rate limit time window reset."""
        rate_filter = RateLimitFilter(rate=1, per_seconds=0.1)
        
        record = self._create_log_record("Test message")
        
        # First message should pass
        assert rate_filter.filter(record) is True
        
        # Second message should be filtered
        assert rate_filter.filter(record) is False
        
        # Wait for time window to reset
        time.sleep(0.15)
        
        # Third message should pass after time window reset
        assert rate_filter.filter(record) is True
    
    def test_rate_limit_filter_pattern_grouping(self):
        """Test message pattern grouping."""
        rate_filter = RateLimitFilter(rate=1, per_seconds=1)
        
        # Different patterns should have separate rate limits
        record1 = self._create_log_record("Processing file 1")
        record2 = self._create_log_record("Processing file 2")  # Similar pattern
        record3 = self._create_log_record("Different message type")
        
        assert rate_filter.filter(record1) is True
        assert rate_filter.filter(record2) is False  # Same pattern, over limit
        assert rate_filter.filter(record3) is True   # Different pattern
    
    def _create_log_record(self, message):
        """Helper to create log records."""
        return logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=1,
            msg=message,
            args=(),
            exc_info=None
        )


class TestProcessInfoFilter:
    """Test process information injection."""
    
    def test_process_info_filter_basic(self):
        """Test basic process info injection."""
        process_filter = ProcessInfoFilter(include_system_info=False)
        
        record = self._create_log_record("Test message")
        
        result = process_filter.filter(record)
        
        assert result is True
        assert hasattr(record, 'process_id')
        assert hasattr(record, 'hostname')
        assert hasattr(record, 'python_version')
        assert hasattr(record, 'thread_name')
    
    def test_process_info_filter_with_system_info(self):
        """Test process info with system information."""
        process_filter = ProcessInfoFilter(include_system_info=True)
        
        record = self._create_log_record("Test message")
        process_filter.filter(record)
        
        # System info fields may not be available in test environment
        # but filter should not fail
        assert hasattr(record, 'process_id')
    
    def _create_log_record(self, message):
        """Helper to create log records."""
        return logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=1,
            msg=message,
            args=(),
            exc_info=None
        )


class TestLoggerWriter:
    """Test logger writer functionality."""
    
    def test_logger_writer_basic(self):
        """Test basic LoggerWriter functionality."""
        logger = MagicMock()
        writer = LoggerWriter(logger.info)
        
        # Test writing
        chars_written = writer.write("Test message\\n")
        
        assert chars_written == len("Test message\\n")
        logger.info.assert_called_once_with("Test message")
    
    def test_logger_writer_line_buffering(self):
        """Test line buffering behavior."""
        logger = MagicMock()
        writer = LoggerWriter(logger.info)
        
        # Write partial line
        writer.write("Partial ")
        logger.info.assert_not_called()
        
        # Complete the line
        writer.write("message\\n")
        logger.info.assert_called_once_with("Partial message")
    
    def test_logger_writer_writelines(self):
        """Test writelines method."""
        logger = MagicMock()
        writer = LoggerWriter(logger.info)
        
        lines = ["Line 1\\n", "Line 2\\n", "Line 3\\n"]
        writer.writelines(lines)
        
        assert logger.info.call_count == 3
    
    def test_logger_writer_flush(self):
        """Test flush method."""
        logger = MagicMock()
        writer = LoggerWriter(logger.info)
        
        # Write incomplete line
        writer.write("Incomplete")
        logger.info.assert_not_called()
        
        # Flush should log the incomplete line
        writer.flush()
        logger.info.assert_called_once_with("Incomplete")
    
    def test_redirect_std_streams(self):
        """Test stdout/stderr redirection."""
        import sys
        
        logger = MagicMock()
        original_stdout = sys.stdout
        
        with redirect_std_streams(stdout_logger=logger.info):
            print("Redirected message")
        
        # Verify stdout was restored
        assert sys.stdout is original_stdout
        
        # Verify message was logged (may need adjustment based on implementation)
        assert logger.info.called


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_correlation_id(self):
        """Test correlation ID generation."""
        correlation_id = create_correlation_id()
        
        assert isinstance(correlation_id, str)
        assert len(correlation_id) == 8  # Default length
        
        # Test custom length
        long_id = create_correlation_id(length=16)
        assert len(long_id) == 16
        
        # Test uniqueness
        id1 = create_correlation_id()
        id2 = create_correlation_id()
        assert id1 != id2
    
    def test_log_exceptions(self):
        """Test exception logging context manager."""
        logger = MagicMock()
        
        with pytest.raises(ValueError):
            with log_exceptions(logger, "Test operation failed"):
                raise ValueError("Test error")
        
        # Verify exception was logged
        logger.log.assert_called()
        call_args = logger.log.call_args
        assert call_args[0][0] == logging.ERROR  # Log level
        assert "Test operation failed" in call_args[0][1]  # Message
        assert call_args[1]['exc_info'] is True  # Exception info included
    
    def test_log_exceptions_no_reraise(self):
        """Test exception logging without re-raising."""
        logger = MagicMock()
        
        # Should not raise exception
        with log_exceptions(logger, "Test operation", reraise=False):
            raise ValueError("Test error")
        
        # Verify exception was still logged
        logger.log.assert_called()
    
    def test_monitor_performance_decorator(self):
        """Test performance monitoring decorator."""
        logger = MagicMock()
        
        @monitor_performance(logger=logger, operation_name="Test operation")
        def test_function():
            time.sleep(0.01)
            return "result"
        
        result = test_function()
        
        assert result == "result"
        assert logger.log.call_count >= 2  # Start and completion logs
    
    def test_add_process_info(self):
        """Test add_process_info convenience function."""
        logger = logging.getLogger("test.logger")
        
        # Clear existing filters
        logger.filters.clear()
        
        add_process_info(logger)
        
        # Verify filter was added
        assert len(logger.filters) == 1
        assert isinstance(logger.filters[0], ProcessInfoFilter)


class TestIntegration:
    """Integration tests for the complete logging system."""
    
    def test_full_logging_setup_and_usage(self):
        """Test complete logging system setup and usage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "integration_test.log"
            
            # Setup logging
            config = get_default_config(
                level="DEBUG",
                file_path=str(log_file),
                format_style="detailed"
            )
            setup_logging(config)
            
            # Create contextual logger
            logger = get_logger(__name__, 
                               test_id="integration_test",
                               component="test_suite")
            
            # Test various logging scenarios
            logger.info("Integration test started")
            
            with PerformanceTimer(logger, "Test operation"):
                time.sleep(0.01)
            
            with log_exceptions(logger, "Expected test exception", reraise=False):
                raise RuntimeError("Test integration error")
            
            logger.warning("Integration test completed")
            
            # Verify log file was created and contains expected content
            assert log_file.exists()
            log_content = log_file.read_text()
            
            assert "Integration test started" in log_content
            assert "Test operation" in log_content
            assert "Expected test exception" in log_content
            assert "Integration test completed" in log_content
    
    def test_concurrent_logging(self):
        """Test logging system under concurrent access."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "concurrent_test.log"
            
            setup_logging(get_default_config(file_path=str(log_file)))
            
            def log_messages(thread_id):
                logger = get_logger(__name__, thread_id=thread_id)
                for i in range(10):
                    logger.info(f"Message {i} from thread {thread_id}")
                    time.sleep(0.001)
            
            # Create multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=log_messages, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify all messages were logged
            log_content = log_file.read_text()
            for thread_id in range(5):
                for msg_id in range(10):
                    assert f"Message {msg_id} from thread {thread_id}" in log_content


if __name__ == "__main__":
    pytest.main([__file__])
