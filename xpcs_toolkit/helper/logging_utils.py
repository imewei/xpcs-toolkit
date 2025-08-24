"""
XPCS Toolkit - Advanced Logging Utilities

This module provides sophisticated logging utilities for performance monitoring,
rate limiting, and contextual information enhancement. These utilities are designed
to work seamlessly with the centralized logging configuration system.

## Key Features

### Performance Monitoring
- `PerformanceTimer`: Context manager and decorator for timing operations
- Automatic CPU and wall clock time measurement
- Memory usage tracking in debug mode
- Throughput calculation for data processing operations

### Rate Limiting
- `RateLimitFilter`: Prevent log spam from high-frequency operations
- Configurable time windows and message counts
- Automatic message aggregation and summaries

### Context Enhancement
- Process information injection (PID, hostname, thread info)
- Automatic correlation ID generation for request tracking
- Custom metadata attachment to log records

### Debug Diagnostics
- Memory profiling with tracemalloc integration
- System resource monitoring with psutil
- Performance bottleneck identification

## Usage Examples

### Basic Performance Timing
```python
from xpcs_toolkit.helper.logging_utils import PerformanceTimer
import logging

logger = logging.getLogger(__name__)

# As context manager
with PerformanceTimer(logger, "Data processing"):
    process_large_dataset(data)

# As decorator
@PerformanceTimer(logger, "File analysis")
def analyze_file(filename):
    return perform_analysis(filename)
```

### Rate Limited Logging
```python
from xpcs_toolkit.helper.logging_utils import RateLimitFilter

# Add rate limiting to prevent log spam
rate_filter = RateLimitFilter(rate=10, per_seconds=60)  # Max 10 messages per minute
logger.addFilter(rate_filter)

# These messages will be rate-limited
for i in range(1000):
    logger.info(f"Processing item {i}")  # Only ~10 will actually be logged
```

### Enhanced Context Information
```python
from xpcs_toolkit.helper.logging_utils import add_process_info

# Automatically add process info to all log messages
add_process_info(logger)
logger.info("Processing started")  # Will include PID, hostname, etc.
```
"""

import logging
import time
import functools
import threading
import collections
import hashlib
import socket
import os
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta


# Optional imports with graceful degradation
try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    tracemalloc = None  # type: ignore
    HAS_TRACEMALLOC = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    HAS_PSUTIL = False


class PerformanceTimer:
    """
    Context manager and decorator for performance monitoring with comprehensive timing.
    
    This utility automatically measures elapsed time, CPU time, and optionally memory
    usage for code blocks or functions. It integrates seamlessly with the logging
    system to provide detailed performance insights.
    
    Features:
    - Wall clock and CPU time measurement
    - Memory usage tracking (when tracemalloc is available)
    - Throughput calculation for data processing operations
    - Nested timing support with hierarchical reporting
    - Exception-safe cleanup and reporting
    
    Examples:
    --------
    # Context manager usage
    with PerformanceTimer(logger, "Data loading", extra_context={"file_size": "10MB"}):
        data = load_large_file("experiment.hdf")
    
    # Decorator usage
    @PerformanceTimer(logger, "Analysis function")
    def analyze_data(dataset):
        return perform_complex_analysis(dataset)
    
    # With throughput calculation
    with PerformanceTimer(logger, "File processing", item_count=len(files)) as timer:
        for file in files:
            process_file(file)
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        operation_name: str,
        level: int = logging.DEBUG,
        item_count: Optional[int] = None,
        extra_context: Optional[Dict[str, Any]] = None,
        enable_memory_tracking: Optional[bool] = None
    ):
        """
        Initialize PerformanceTimer.
        
        Parameters
        ----------
        logger : logging.Logger
            Logger to use for timing reports
        operation_name : str
            Human-readable name for the operation being timed
        level : int
            Logging level for timing reports (default: DEBUG)
        item_count : int, optional
            Number of items being processed (enables throughput calculation)
        extra_context : dict, optional
            Additional context information to include in log messages
        enable_memory_tracking : bool, optional
            Enable memory usage tracking (auto-detected if None)
        """
        self.logger = logger
        self.operation_name = operation_name
        self.level = level
        self.item_count = item_count
        self.extra_context = extra_context or {}
        
        # Auto-detect memory tracking capability
        if enable_memory_tracking is None:
            enable_memory_tracking = HAS_TRACEMALLOC and logger.isEnabledFor(logging.DEBUG)
        self.enable_memory_tracking = enable_memory_tracking
        
        # Timing variables
        self.start_time = None
        self.end_time = None
        self.start_cpu_time = None
        self.end_cpu_time = None
        
        # Memory tracking variables
        self.start_memory = None
        self.peak_memory = None
        self.memory_snapshot_start = None
        
        # Process information (if available)
        self.process_info = {}
        if HAS_PSUTIL and psutil is not None:
            try:
                process = psutil.Process()
                self.process_info = {
                    "cpu_count": psutil.cpu_count(),
                    "available_memory_gb": psutil.virtual_memory().available / (1024**3),
                }
            except Exception:  # Catch all exceptions since psutil might raise various errors
                pass
    
    def __enter__(self):
        """Start timing when entering context."""
        self._start_timing()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log results when exiting context."""
        self._stop_timing()
        self._log_results(exception_occurred=exc_type is not None)
        return False  # Don't suppress exceptions
    
    def __call__(self, func):
        """Decorator interface."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.operation_name = f"{func.__name__}() - {self.operation_name}"
            with self:
                return func(*args, **kwargs)
        return wrapper
    
    def _start_timing(self):
        """Initialize timing measurements."""
        # Start memory tracking if enabled
        if self.enable_memory_tracking and HAS_TRACEMALLOC and tracemalloc is not None:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            self.memory_snapshot_start = tracemalloc.take_snapshot()
        
        # Record start times
        self.start_time = time.perf_counter()
        self.start_cpu_time = time.process_time()
        
        # Log operation start
        self.logger.log(
            self.level,
            "Started %s",
            self.operation_name,
            extra={
                "operation": self.operation_name,
                "timing_start": True,
                **self.extra_context,
                **self.process_info
            }
        )
    
    def _stop_timing(self):
        """Finalize timing measurements."""
        self.end_time = time.perf_counter()
        self.end_cpu_time = time.process_time()
        
        # Capture peak memory if tracking enabled
        if self.enable_memory_tracking and HAS_TRACEMALLOC and tracemalloc is not None:
            if tracemalloc.is_tracing():
                current_memory, peak_memory = tracemalloc.get_traced_memory()
                self.peak_memory = peak_memory / (1024 * 1024)  # Convert to MB
                tracemalloc.stop()
    
    def _log_results(self, exception_occurred=False):
        """Log comprehensive timing results."""
        if self.start_time is None or self.end_time is None:
            return
            
        elapsed_time = self.end_time - self.start_time
        cpu_time = (self.end_cpu_time - self.start_cpu_time) if (self.start_cpu_time is not None and self.end_cpu_time is not None) else None
        
        # Build timing summary
        timing_info = {
            "operation": self.operation_name,
            "timing_complete": True,
            "elapsed_seconds": round(elapsed_time, 4),
            "cpu_seconds": round(cpu_time, 4) if cpu_time else None,
            "exception_occurred": exception_occurred,
        }
        
        # Add memory information if available
        if self.peak_memory is not None:
            timing_info["peak_memory_mb"] = round(self.peak_memory, 2)
        
        # Calculate throughput if item count provided
        if self.item_count and elapsed_time > 0:
            throughput = self.item_count / elapsed_time
            timing_info.update({
                "item_count": self.item_count,
                "throughput_items_per_sec": round(throughput, 2)
            })
        
        # Add any additional context
        timing_info.update(self.extra_context)
        timing_info.update(self.process_info)
        
        # Format human-readable message
        status = "failed" if exception_occurred else "completed"
        message_parts = [f"{self.operation_name} {status} in {elapsed_time:.3f}s"]
        
        if cpu_time is not None:
            cpu_efficiency = (cpu_time / elapsed_time) * 100 if elapsed_time > 0 else 0
            message_parts.append(f"CPU: {cpu_time:.3f}s ({cpu_efficiency:.1f}%)")
        
        if self.peak_memory is not None:
            message_parts.append(f"Peak memory: {self.peak_memory:.1f}MB")
        
        if self.item_count:
            throughput = self.item_count / elapsed_time if elapsed_time > 0 else 0
            message_parts.append(f"Throughput: {throughput:.1f} items/sec")
        
        message = " | ".join(message_parts)
        
        # Choose appropriate log level based on results
        log_level = self.level
        if exception_occurred:
            log_level = logging.ERROR
        elif elapsed_time > 30:  # Long operations get elevated to INFO
            log_level = logging.INFO
            
        self.logger.log(log_level, message, extra=timing_info)


class RateLimitFilter(logging.Filter):
    """
    Logging filter that limits the rate of messages to prevent log spam.
    
    This filter tracks message patterns and suppresses repeated messages
    that exceed the specified rate limit. It provides periodic summaries
    of suppressed messages to maintain visibility into system behavior.
    
    Features:
    - Configurable rate limits per time window
    - Message pattern recognition and grouping
    - Automatic summary reporting of suppressed messages
    - Thread-safe operation for concurrent logging
    
    Examples:
    --------
    # Limit to 10 messages per minute
    rate_filter = RateLimitFilter(rate=10, per_seconds=60)
    logger.addFilter(rate_filter)
    
    # More restrictive limiting for verbose operations
    rate_filter = RateLimitFilter(rate=1, per_seconds=5, 
                                 summary_interval_seconds=300)
    logger.addFilter(rate_filter)
    """
    
    def __init__(
        self,
        rate: int = 10,
        per_seconds: int = 60,
        summary_interval_seconds: int = 300,
        message_pattern_length: int = 50
    ):
        """
        Initialize rate limiting filter.
        
        Parameters
        ----------
        rate : int
            Maximum number of messages to allow per time window
        per_seconds : int
            Time window in seconds for rate limiting
        summary_interval_seconds : int
            How often to emit summaries of suppressed messages
        message_pattern_length : int
            Length of message pattern used for grouping similar messages
        """
        super().__init__()
        self.rate = rate
        self.per_seconds = per_seconds
        self.summary_interval = summary_interval_seconds
        self.pattern_length = message_pattern_length
        
        # Thread-safe tracking structures
        self.lock = threading.Lock()
        self.message_counts = collections.defaultdict(list)  # pattern -> [timestamps]
        self.suppressed_counts = collections.defaultdict(int)  # pattern -> count
        self.last_summary_time = time.time()
    
    def filter(self, record):
        """
        Filter log records based on rate limits.
        
        Parameters
        ----------
        record : logging.LogRecord
            Log record to potentially filter
            
        Returns
        -------
        bool
            True to allow the message, False to suppress it
        """
        current_time = time.time()
        
        # Create a pattern from the message for grouping
        message_pattern = self._extract_pattern(record.getMessage())
        
        with self.lock:
            # Clean up old entries
            cutoff_time = current_time - self.per_seconds
            self.message_counts[message_pattern] = [
                t for t in self.message_counts[message_pattern] if t > cutoff_time
            ]
            
            # Check if we're over the rate limit
            if len(self.message_counts[message_pattern]) >= self.rate:
                self.suppressed_counts[message_pattern] += 1
                
                # Check if it's time for a summary
                if current_time - self.last_summary_time > self.summary_interval:
                    # LogRecord has 'name' attribute for logger name, not 'logger'
                    self._emit_summary(getattr(record, 'name', None))
                    self.last_summary_time = current_time
                
                return False  # Suppress this message
            else:
                # Allow this message through
                self.message_counts[message_pattern].append(current_time)
                return True
    
    def _extract_pattern(self, message: str) -> str:
        """Extract a pattern from a message for grouping similar messages."""
        # Truncate long messages and create a hash for uniqueness
        truncated = message[:self.pattern_length]
        # Replace numbers with placeholders to group similar numeric messages
        import re
        pattern = re.sub(r'\d+', 'N', truncated)
        return pattern
    
    def _emit_summary(self, logger):
        """Emit a summary of suppressed messages."""
        if not self.suppressed_counts or not logger:
            return
            
        total_suppressed = sum(self.suppressed_counts.values())
        summary_parts = [f"Rate limiter suppressed {total_suppressed} messages:"]
        
        # Sort by suppression count for most relevant first
        sorted_patterns = sorted(
            self.suppressed_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for pattern, count in sorted_patterns[:5]:  # Top 5 patterns
            summary_parts.append(f"  '{pattern}': {count} times")
        
        if len(sorted_patterns) > 5:
            remaining = len(sorted_patterns) - 5
            summary_parts.append(f"  ... and {remaining} other patterns")
        
        logger.warning("\\n".join(summary_parts))
        
        # Reset counters
        self.suppressed_counts.clear()


class ProcessInfoFilter(logging.Filter):
    """
    Filter that automatically adds process and system information to log records.
    
    This filter enriches log records with contextual information about the
    current process, system resources, and execution environment. This information
    is valuable for debugging and monitoring in production environments.
    
    Added Information:
    - Process ID and parent process ID
    - Hostname and username
    - Thread information
    - Python version and executable path
    - Current working directory
    - System resource usage (if psutil available)
    
    Examples:
    --------
    # Add to logger to enrich all messages
    process_filter = ProcessInfoFilter()
    logger.addFilter(process_filter)
    
    # Now all log messages will include process context
    logger.info("Application started")  # Includes PID, hostname, etc.
    """
    
    def __init__(self, include_system_info: bool = True):
        """
        Initialize process information filter.
        
        Parameters
        ----------
        include_system_info : bool
            Whether to include system resource information (requires psutil)
        """
        super().__init__()
        self.include_system_info = include_system_info
        
        # Cache static information
        self._cached_info = self._collect_static_info()
        
        # Dynamic info collection interval (seconds)
        self._last_system_check = 0
        self._system_check_interval = 30  # Update system info every 30 seconds
        self._cached_system_info = {}
    
    def _collect_static_info(self) -> Dict[str, Any]:
        """Collect static system information that doesn't change."""
        import sys
        import getpass
        
        info = {
            'process_id': os.getpid(),
            'parent_process_id': os.getppid(),
            'hostname': socket.gethostname(),
            'python_version': sys.version.split()[0],
            'python_executable': sys.executable,
            'working_directory': os.getcwd(),
        }
        
        # Add username if available
        try:
            info['username'] = getpass.getuser()
        except Exception:
            info['username'] = 'unknown'
        
        # Add platform information
        try:
            import platform
            info['platform'] = platform.system()
            info['platform_version'] = platform.version()
        except Exception:
            pass
            
        return info
    
    def _collect_dynamic_info(self) -> Dict[str, Any]:
        """Collect dynamic system information that changes over time."""
        current_time = time.time()
        
        # Only update system info periodically to avoid performance impact
        if (current_time - self._last_system_check < self._system_check_interval and 
            self._cached_system_info):
            return self._cached_system_info
        
        info = {}
        
        if self.include_system_info and HAS_PSUTIL and psutil is not None:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                
                info.update({
                    'cpu_percent': process.cpu_percent(),
                    'memory_rss_mb': memory_info.rss / (1024 * 1024),
                    'memory_vms_mb': memory_info.vms / (1024 * 1024),
                    'num_threads': process.num_threads(),
                    'num_fds': process.num_fds() if hasattr(process, 'num_fds') else None,
                })
                
                # System-wide information
                vm = psutil.virtual_memory()
                info.update({
                    'system_cpu_percent': psutil.cpu_percent(interval=None),
                    'system_memory_percent': vm.percent,
                    'system_memory_available_gb': vm.available / (1024**3),
                })
                
            except Exception:  # Catch all exceptions since psutil might raise various errors
                # Ignore errors in system info collection
                pass
        
        self._cached_system_info = info
        self._last_system_check = current_time
        return info
    
    def filter(self, record):
        """Add process and system information to the log record."""
        # Add static information
        for key, value in self._cached_info.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        
        # Add dynamic information
        dynamic_info = self._collect_dynamic_info()
        for key, value in dynamic_info.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        
        # Add thread information
        if not hasattr(record, 'thread_name'):
            record.thread_name = threading.current_thread().name
        
        return True


def add_process_info(logger: logging.Logger, include_system_info: bool = True) -> None:
    """
    Convenience function to add process information to a logger.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger to enhance with process information
    include_system_info : bool
        Whether to include system resource information
    
    Examples:
    --------
    >>> logger = logging.getLogger(__name__)
    >>> add_process_info(logger)
    >>> logger.info("Message")  # Now includes PID, hostname, etc.
    """
    process_filter = ProcessInfoFilter(include_system_info=include_system_info)
    logger.addFilter(process_filter)


def create_correlation_id(length: int = 8) -> str:
    """
    Generate a short correlation ID for request tracking.
    
    Parameters
    ----------
    length : int
        Length of the correlation ID
        
    Returns
    -------
    str
        Hexadecimal correlation ID
    
    Examples:
    --------
    >>> correlation_id = create_correlation_id()
    >>> logger = get_logger(__name__, correlation_id=correlation_id)
    >>> logger.info("Processing started")  # Includes correlation_id in context
    """
    # Use time and random data for uniqueness
    data = f"{time.time()}{threading.current_thread().ident}{os.getpid()}".encode()
    hash_obj = hashlib.sha256(data)
    return hash_obj.hexdigest()[:length]


@contextmanager
def log_exceptions(logger: logging.Logger, 
                  message: str = "Exception occurred", 
                  level: int = logging.ERROR,
                  reraise: bool = True):
    """
    Context manager for automatic exception logging with context.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger to use for exception reporting
    message : str
        Custom message to include with exception
    level : int
        Logging level for exception reports
    reraise : bool
        Whether to re-raise the exception after logging
    
    Examples:
    --------
    >>> with log_exceptions(logger, "Failed to process file"):
    ...     risky_file_operation()
    >>> # Exception is automatically logged with context and re-raised
    """
    try:
        yield
    except Exception as e:
        logger.log(
            level,
            f"{message}: {type(e).__name__}: {str(e)}",
            exc_info=True,
            extra={
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "exception_context": message
            }
        )
        if reraise:
            raise


# Performance monitoring decorator factory
def monitor_performance(logger: Optional[logging.Logger] = None,
                       operation_name: Optional[str] = None,
                       level: int = logging.DEBUG):
    """
    Decorator factory for automatic performance monitoring.
    
    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use (defaults to function's module logger)
    operation_name : str, optional
        Name for the operation (defaults to function name)
    level : int
        Logging level for performance reports
    
    Returns
    -------
    callable
        Decorator function
    
    Examples:
    --------
    >>> @monitor_performance()
    ... def complex_calculation(data):
    ...     return expensive_operation(data)
    
    >>> @monitor_performance(logger=my_logger, operation_name="Data Analysis")
    ... def analyze_data(dataset):
    ...     return perform_analysis(dataset)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use provided logger or try to get module logger
            actual_logger = logger
            if actual_logger is None:
                module = func.__module__
                actual_logger = logging.getLogger(module)
            
            # Use provided operation name or function name
            actual_name = operation_name or f"{func.__name__}()"
            
            with PerformanceTimer(actual_logger, actual_name, level=level):
                return func(*args, **kwargs)
        return wrapper
    return decorator
