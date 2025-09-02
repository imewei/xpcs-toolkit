"""
Simplified tests for xpcs_toolkit.helper.logwriter module.

This module tests the LoggerWriter class with tests that match the actual implementation.
"""

import pytest
import logging
import sys
import threading
from io import StringIO
from unittest.mock import Mock, patch

from xpcs_toolkit.helper.logwriter import LoggerWriter


class TestLoggerWriterBasics:
    """Test suite for LoggerWriter basic functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_writer')
        self.logger.handlers.clear()
        self.log_capture = StringIO()
        handler = logging.StreamHandler(self.log_capture)
        handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
    
    def test_init_with_logger_and_level(self):
        """Test initialization with logger instance and level."""
        writer = LoggerWriter(self.logger, logging.INFO)
        
        assert hasattr(writer, 'buffer')
        assert hasattr(writer, 'lock')
        assert isinstance(writer.buffer, list)
        assert isinstance(writer.lock, threading.Lock)
    
    def test_init_with_logging_function(self):
        """Test initialization with logging function directly."""
        writer = LoggerWriter(self.logger.info)
        
        assert hasattr(writer, 'level_func')
        assert writer.level_func == self.logger.info
    
    def test_init_with_logger_no_level_raises_error(self):
        """Test that logger without level raises ValueError."""
        with pytest.raises(ValueError, match="Level must be specified"):
            LoggerWriter(self.logger)
    
    def test_write_single_line(self):
        """Test writing a single line."""
        writer = LoggerWriter(self.logger, logging.INFO)
        
        writer.write("Test message\n")
        
        log_content = self.log_capture.getvalue()
        assert "INFO:Test message" in log_content
    
    def test_write_multiple_lines(self):
        """Test writing multiple lines."""
        writer = LoggerWriter(self.logger, logging.DEBUG)
        
        writer.write("Line 1\nLine 2\nLine 3\n")
        
        log_content = self.log_capture.getvalue()
        assert "DEBUG:Line 1" in log_content
        assert "DEBUG:Line 2" in log_content
        assert "DEBUG:Line 3" in log_content
    
    def test_write_partial_lines(self):
        """Test writing partial lines that need buffering."""
        writer = LoggerWriter(self.logger, logging.INFO)
        
        writer.write("Partial ")
        writer.write("line ")
        writer.write("complete\n")
        
        log_content = self.log_capture.getvalue()
        assert "INFO:Partial line complete" in log_content
    
    def test_flush_buffered_content(self):
        """Test flushing buffered content."""
        writer = LoggerWriter(self.logger, logging.INFO)
        
        writer.write("Buffered message")  # No newline
        # Should not be logged yet
        log_content = self.log_capture.getvalue()
        assert "Buffered message" not in log_content
        
        writer.flush()
        log_content = self.log_capture.getvalue()
        assert "INFO:Buffered message" in log_content
    
    def test_writelines(self):
        """Test writelines method."""
        writer = LoggerWriter(self.logger, logging.WARNING)
        
        lines = ["Warning 1\n", "Warning 2\n", "Warning 3\n"]
        writer.writelines(lines)
        
        log_content = self.log_capture.getvalue()
        for i in range(1, 4):
            assert f"WARNING:Warning {i}" in log_content
    
    def test_empty_messages_filtered(self):
        """Test that empty messages are filtered out."""
        writer = LoggerWriter(self.logger, logging.INFO)
        
        writer.write("")
        writer.write("   \n")  # Whitespace only
        writer.write("Actual message\n")
        
        log_content = self.log_capture.getvalue()
        assert "INFO:Actual message" in log_content
        # Should only have one log entry
        assert log_content.count("INFO:") == 1
    
    def test_file_interface_methods(self):
        """Test file-like interface methods."""
        writer = LoggerWriter(self.logger, logging.INFO)
        
        assert writer.readable() is False
        assert writer.writable() is True
        assert writer.isatty() is False
    
    def test_close_flushes_buffer(self):
        """Test that close flushes any remaining buffer."""
        writer = LoggerWriter(self.logger, logging.INFO)
        
        writer.write("Message without newline")
        writer.close()
        
        log_content = self.log_capture.getvalue()
        assert "INFO:Message without newline" in log_content
    
    def test_thread_safety(self):
        """Test basic thread safety of write operations."""
        writer = LoggerWriter(self.logger, logging.INFO)
        
        def write_messages(thread_id):
            for i in range(10):
                writer.write(f"Thread {thread_id} Message {i}\n")
        
        threads = []
        for i in range(3):
            thread = threading.Thread(target=write_messages, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        log_content = self.log_capture.getvalue()
        # Should have 30 messages total
        assert log_content.count("INFO:Thread") == 30


class TestStdoutRedirection:
    """Test suite for stdout/stderr redirection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_redirect')
        self.logger.handlers.clear()
        self.log_capture = StringIO()
        handler = logging.StreamHandler(self.log_capture)
        handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Store original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
    
    def teardown_method(self):
        """Restore original stdout/stderr."""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
    
    def test_stdout_redirection(self):
        """Test redirecting stdout to logger."""
        writer = LoggerWriter(self.logger, logging.INFO)
        
        sys.stdout = writer
        print("Redirected print statement")
        sys.stdout = self.original_stdout
        
        log_content = self.log_capture.getvalue()
        assert "INFO:Redirected print statement" in log_content
    
    def test_stderr_redirection(self):
        """Test redirecting stderr to logger."""
        writer = LoggerWriter(self.logger, logging.ERROR)
        
        sys.stderr = writer
        print("Error message", file=sys.stderr)
        sys.stderr = self.original_stderr
        
        log_content = self.log_capture.getvalue()
        assert "ERROR:Error message" in log_content


class TestLoggerWriterIntegration:
    """Integration tests for LoggerWriter."""
    
    def test_with_real_logger_configuration(self):
        """Test with realistic logger configuration."""
        logger = logging.getLogger('xpcs_toolkit.test_integration')
        logger.setLevel(logging.DEBUG)
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        writer = LoggerWriter(logger, logging.INFO)
        
        writer.write("Integration test message 1\n")
        writer.write("Integration test message 2\n")
        
        log_content = log_capture.getvalue()
        assert "xpcs_toolkit.test_integration" in log_content
        assert "INFO" in log_content
        assert "Integration test message 1" in log_content
        assert "Integration test message 2" in log_content
        
        logger.handlers.clear()
    
    def test_error_handling(self):
        """Test error handling in LoggerWriter."""
        logger = Mock()
        logger.side_effect = Exception("Logging error")
        
        writer = LoggerWriter(logger)
        
        # Should not raise exception even if logging fails
        writer.write("Test message\n")  # Should not raise
        
        # Verify the logger was called
        logger.assert_called_with("Test message")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])