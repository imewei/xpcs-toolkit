import sys
import logging
import threading
import queue
from contextlib import contextmanager
from typing import Optional, Callable, List, Union


class LoggerWriter:
    """
    Enhanced logger writer for redirecting stdout/stderr to logging.
    
    This class provides a file-like interface that can capture output from
    print statements and other stdout/stderr writes, redirecting them to
    the Python logging system with proper level handling and thread safety.
    
    Features:
    - Thread-safe operation using queue-based buffering
    - Support for both logger instances and logging functions
    - Proper handling of line buffering and message assembly
    - Compatible with all standard file-like operations
    
    Examples:
    --------
    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> writer = LoggerWriter(logger, logging.INFO)
    >>> print("Hello world", file=writer)
    >>> # Message appears in logger at INFO level
    
    >>> # Using with logging function directly
    >>> writer = LoggerWriter(logger.debug)
    >>> sys.stdout = writer
    >>> print("Debug message")  # Redirected to logger.debug
    """
    
    def __init__(self, logger_or_func: Union[logging.Logger, Callable], level: Optional[int] = None):
        """
        Initialize LoggerWriter with logger and level.
        
        Parameters
        ----------
        logger_or_func : logging.Logger or callable
            Either a Logger instance (requires level parameter) or a 
            logging function like logger.info, logger.debug, etc.
        level : int, optional
            Logging level (only required if logger_or_func is a Logger instance)
        """
        if isinstance(logger_or_func, logging.Logger):
            if level is None:
                raise ValueError("Level must be specified when passing Logger instance")
            self.logger = logger_or_func
            self.level_func = lambda msg: self.logger.log(level, msg)
        else:
            # Assume it's a logging function like logger.debug
            self.level_func = logger_or_func
            
        self.buffer = []
        self.lock = threading.Lock()
        
    def write(self, message: str) -> int:
        """
        Write message to logger, handling line buffering appropriately.
        
        Parameters
        ----------
        message : str
            Message to write
            
        Returns
        -------
        int
            Number of characters written
        """
        if not message:
            return 0
            
        with self.lock:
            # Split message into lines, preserving empty lines
            lines = message.splitlines(True)  # Keep line endings
            
            for line in lines:
                if line.endswith('\n'):
                    # Complete line - add to buffer and flush
                    self.buffer.append(line.rstrip('\n'))
                    complete_message = ''.join(self.buffer)
                    if complete_message.strip():  # Only log non-empty messages
                        try:
                            self.level_func(complete_message)
                        except Exception:
                            # Continue even if logging fails
                            pass
                    self.buffer.clear()
                else:
                    # Incomplete line - add to buffer
                    self.buffer.append(line)
                    
        return len(message)
    
    def writelines(self, lines: List[str]) -> None:
        """
        Write multiple lines to the logger.
        
        Parameters
        ----------
        lines : list of str
            Lines to write
        """
        for line in lines:
            self.write(line)
    
    def flush(self) -> None:
        """
        Flush any buffered content to the logger.
        
        This method ensures that any incomplete lines in the buffer
        are written to the logger.
        """
        with self.lock:
            if self.buffer:
                complete_message = ''.join(self.buffer)
                if complete_message.strip():  # Only log non-empty messages
                    try:
                        self.level_func(complete_message)
                    except Exception:
                        # Continue even if logging fails
                        pass
                self.buffer.clear()
    
    def isatty(self) -> bool:
        """Return False - logger writers are never TTY devices."""
        return False
    
    def readable(self) -> bool:
        """Return False - logger writers are write-only."""
        return False
    
    def writable(self) -> bool:
        """Return True - logger writers are writable."""
        return True
    
    def close(self) -> None:
        """Close the writer, flushing any remaining content."""
        self.flush()


class AsyncLoggerWriter(LoggerWriter):
    """
    Asynchronous version of LoggerWriter using queue-based processing.
    
    This version uses a separate thread to handle logging operations,
    preventing potential deadlocks or performance issues when logging
    from performance-critical code paths.
    """
    
    def __init__(self, logger_or_func: Union[logging.Logger, Callable], level: Optional[int] = None):
        super().__init__(logger_or_func, level)
        self.message_queue = queue.Queue(maxsize=1000)  # Prevent unbounded growth
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.shutdown_event = threading.Event()
        self.worker_thread.start()
    
    def _worker(self):
        """Worker thread that processes messages from the queue."""
        while not self.shutdown_event.is_set():
            try:
                message = self.message_queue.get(timeout=0.1)
                if message is None:  # Shutdown signal
                    break
                if message.strip():  # Only log non-empty messages
                    self.level_func(message)
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                # Ignore errors in worker thread to prevent crashes
                pass
    
    def write(self, message: str) -> int:
        """Write message to async queue for processing."""
        if not message:
            return 0
            
        with self.lock:
            lines = message.splitlines(True)
            for line in lines:
                if line.endswith('\n'):
                    self.buffer.append(line.rstrip('\n'))
                    complete_message = ''.join(self.buffer)
                    if complete_message.strip():
                        try:
                            self.message_queue.put_nowait(complete_message)
                        except queue.Full:
                            # If queue is full, skip this message to prevent blocking
                            pass
                    self.buffer.clear()
                else:
                    self.buffer.append(line)
                    
        return len(message)
    
    def close(self):
        """Close the async writer and wait for worker thread to finish."""
        self.flush()
        self.shutdown_event.set()
        self.message_queue.put(None)  # Signal worker to shutdown
        self.worker_thread.join(timeout=1.0)


@contextmanager
def redirect_std_streams(stdout_logger=None, stderr_logger=None, async_mode=False):
    """
    Context manager to temporarily redirect stdout/stderr to loggers.
    
    This provides a convenient way to capture all print statements and
    error output within a specific code block and redirect them to the
    logging system.
    
    Parameters
    ----------
    stdout_logger : logging.Logger or callable, optional
        Logger or logging function for stdout redirection
    stderr_logger : logging.Logger or callable, optional  
        Logger or logging function for stderr redirection
    async_mode : bool
        If True, use AsyncLoggerWriter for better performance
        
    Examples
    --------
    >>> logger = logging.getLogger(__name__)
    >>> with redirect_std_streams(stdout_logger=logger.info, stderr_logger=logger.error):
    ...     print("This goes to logger.info")
    ...     print("This goes to stderr", file=sys.stderr)  # Goes to logger.error
    
    >>> # Using with Logger instances
    >>> with redirect_std_streams(stdout_logger=(logger, logging.INFO)):
    ...     print("Logged at INFO level")
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    writer_class = AsyncLoggerWriter if async_mode else LoggerWriter
    
    try:
        if stdout_logger is not None:
            if isinstance(stdout_logger, tuple):
                # Handle (logger, level) tuple format
                logger_instance, level = stdout_logger
                sys.stdout = writer_class(logger_instance, level)
            else:
                # Handle logging function or configured logger
                sys.stdout = writer_class(stdout_logger)
        
        if stderr_logger is not None:
            if isinstance(stderr_logger, tuple):
                # Handle (logger, level) tuple format
                logger_instance, level = stderr_logger
                sys.stderr = writer_class(logger_instance, level)
            else:
                # Handle logging function or configured logger
                sys.stderr = writer_class(stderr_logger)
        
        yield
        
    finally:
        # Restore original streams
        if hasattr(sys.stdout, 'close') and sys.stdout != original_stdout:
            sys.stdout.close()
        if hasattr(sys.stderr, 'close') and sys.stderr != original_stderr:
            sys.stderr.close()
            
        sys.stdout = original_stdout
        sys.stderr = original_stderr


# Backward compatibility - keep old class name as alias
class LogWriter(LoggerWriter):
    """Deprecated alias for LoggerWriter. Use LoggerWriter instead."""
    def __init__(self, level):
        import warnings
        warnings.warn(
            "LogWriter is deprecated. Use LoggerWriter instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Assume level is a logging function for backward compatibility
        super().__init__(level)
