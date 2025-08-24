# XPCS Toolkit - Logging System Documentation

## Overview

The XPCS Toolkit features a comprehensive, centralized logging system designed for both development and production environments. The logging system provides structured output, performance monitoring, contextual information, and flexible configuration suitable for synchrotron beamline operations.

## Key Features

### 🚀 **Centralized Configuration**
- Single point of configuration for all logging
- Environment variable support for deployment flexibility
- External configuration file support (YAML/JSON)
- Backward compatibility with existing code

### 📊 **Performance Monitoring**
- Built-in timing for operations with CPU and wall clock time
- Memory usage tracking in debug mode
- Throughput calculation for data processing
- Automatic performance level escalation for slow operations

### 🔧 **Advanced Features**
- Rate limiting to prevent log spam
- Contextual logging with automatic metadata injection
- Thread-safe operations for concurrent processing
- JSON structured logging for production environments

### 🎯 **Production Ready**
- Rotating file handlers with configurable sizes
- Process and system information injection
- Exception handling with full context
- Correlation ID support for request tracking

## Quick Start

### Basic Setup

```python
from xpcs_toolkit.helper.logging_config import setup_logging, get_logger

# Initialize with defaults
setup_logging()

# Get a contextual logger
logger = get_logger(__name__, experiment_id="exp001", user="researcher")
logger.info("Starting analysis", extra={"q_range": (0.01, 0.1)})
```

### CLI Usage

```bash
# Basic usage with default INFO level
xpcs-toolkit saxs2d /data/experiment/

# Debug mode with detailed logs
xpcs-toolkit --log-level DEBUG saxs2d /data/experiment/

# Custom log file location
xpcs-toolkit --log-file /var/log/xpcs/analysis.log saxs2d /data/

# JSON structured logging for production
xpcs-toolkit --log-format json --log-file analysis.json saxs2d /data/

# External configuration file
xpcs-toolkit --log-config /etc/xpcs/logging.yaml saxs2d /data/
```

## Environment Variables

### Core Configuration
- `XPCS_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `XPCS_LOG_FILE`: Specify log file path (default: ./xpcs_toolkit.log)
- `XPCS_LOG_FORMAT`: Log format style (simple, detailed, json)
- `XPCS_LOG_CONFIG`: Path to external YAML/JSON configuration file

### Advanced Configuration
- `XPCS_LOG_MAX_SIZE`: Maximum size for rotating log files in MB (default: 10)
- `XPCS_LOG_BACKUP_COUNT`: Number of backup files to keep (default: 5)

### Usage Examples
```bash
# Development environment
export XPCS_LOG_LEVEL=DEBUG
export XPCS_LOG_FILE=debug.log

# Production environment  
export XPCS_LOG_LEVEL=INFO
export XPCS_LOG_FORMAT=json
export XPCS_LOG_FILE=/var/log/xpcs/production.json
export XPCS_LOG_MAX_SIZE=50
export XPCS_LOG_BACKUP_COUNT=10

# Run analysis
python -m xpcs_toolkit.cli_headless saxs2d /data/experiment/
```

## Advanced Features

### Performance Monitoring

```python
from xpcs_toolkit.helper.logging_utils import PerformanceTimer
import logging

logger = logging.getLogger(__name__)

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
from xpcs_toolkit.helper.logging_utils import add_process_info, create_correlation_id
from xpcs_toolkit.helper.logging_config import get_logger

# Automatically add process info to all log messages
add_process_info(logger)

# Create correlation ID for request tracking
correlation_id = create_correlation_id()
context_logger = get_logger(__name__, 
                           correlation_id=correlation_id,
                           experiment="exp001")

context_logger.info("Processing started")  # Includes PID, hostname, correlation ID
```

### Exception Handling with Context

```python
from xpcs_toolkit.helper.logging_utils import log_exceptions

with log_exceptions(logger, "Failed to process file"):
    risky_file_operation()
# Exception is automatically logged with context and re-raised
```

## Configuration Files

### YAML Configuration Example

```yaml
# /etc/xpcs/logging.yaml
version: 1
disable_existing_loggers: false

formatters:
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'
  json:
    class: 'xpcs_toolkit.helper.logging_config.JSONFormatter'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: detailed
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: json
    filename: /var/log/xpcs/analysis.json
    maxBytes: 52428800  # 50MB
    backupCount: 10
    encoding: utf-8

loggers:
  xpcs_toolkit:
    level: DEBUG
    handlers: [console, file]
    propagate: false
    
  # Suppress verbose third-party logging
  matplotlib:
    level: WARNING
  h5py:
    level: WARNING
  numpy:
    level: WARNING

root:
  level: INFO
  handlers: [console]
```

### JSON Configuration Example

```json
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "production": {
      "class": "xpcs_toolkit.helper.logging_config.JSONFormatter"
    }
  },
  "handlers": {
    "file": {
      "class": "logging.handlers.RotatingFileHandler",
      "level": "INFO",
      "formatter": "production",
      "filename": "/var/log/xpcs/production.json",
      "maxBytes": 104857600,
      "backupCount": 5,
      "encoding": "utf-8"
    }
  },
  "loggers": {
    "xpcs_toolkit": {
      "level": "INFO",
      "handlers": ["file"],
      "propagate": false
    }
  }
}
```

## Log Message Examples

### Standard Messages
```
2024-08-22 16:30:45 - xpcs_toolkit.cli_headless - INFO - main:89 - XPCS Toolkit CLI started
2024-08-22 16:30:45 - xpcs_toolkit.fileIO.hdf_reader - INFO - put:67 - Writing HDF5 data
2024-08-22 16:30:46 - xpcs_toolkit.fileIO.hdf_reader - DEBUG - put:128 - HDF5 write operation completed in 0.234s
```

### JSON Structured Messages
```json
{
  "timestamp": "2024-08-22T16:30:45.123456",
  "level": "INFO",
  "logger": "xpcs_toolkit.analysis_kernel",
  "message": "Analysis completed successfully",
  "module": "analysis_kernel",
  "function": "plot_g2_function",
  "line": 245,
  "process_id": 12345,
  "hostname": "beamline-workstation",
  "context": {
    "experiment_id": "exp001",
    "q_range": [0.01, 0.1],
    "file_count": 4,
    "elapsed_seconds": 15.67
  }
}
```

### Performance Monitoring
```
2024-08-22 16:31:02 - xpcs_toolkit.analysis_kernel - DEBUG - Data processing completed in 2.345s | CPU: 1.876s (80.0%) | Peak memory: 245.3MB | Throughput: 42.6 items/sec
```

## Integration Patterns

### Beamline Integration

```python
# Beamline control system integration
from xpcs_toolkit.helper.logging_config import setup_logging

# Configure for beamline environment
setup_logging({
    "level": "INFO",
    "file_path": "/beamline/logs/xpcs_analysis.json",
    "format": "json",
    "max_size_mb": 100,
    "backup_count": 20
})

# Create logger with beamline context
logger = get_logger(__name__, 
                   beamline="8-ID-I", 
                   proposal="12345",
                   shift="evening")

logger.info("Beamline analysis started")
```

### Automated Pipeline Integration

```python
# Pipeline monitoring with correlation IDs
import uuid
from xpcs_toolkit.helper.logging_utils import create_correlation_id

pipeline_id = create_correlation_id()
logger = get_logger(__name__, 
                   pipeline_id=pipeline_id,
                   stage="preprocessing")

# Process each file with same correlation ID
for file_path in dataset_files:
    file_logger = get_logger(__name__, 
                           pipeline_id=pipeline_id,
                           file=file_path.name)
    
    with PerformanceTimer(file_logger, "File processing"):
        process_xpcs_file(file_path)
```

## Best Practices

### 🎯 **Development**
- Use DEBUG level for internal state logging
- Add context information to understand complex operations
- Use PerformanceTimer for operations that might be slow
- Test logging configuration in development environment

### 🏭 **Production**
- Use INFO level for normal operations
- Enable JSON formatting for log aggregation systems
- Set appropriate log rotation limits
- Monitor log file sizes and performance impact

### 🔧 **Performance**
- Use contextual loggers to avoid repeated metadata
- Apply rate limiting for high-frequency operations
- Enable memory tracking only in DEBUG mode
- Use async logging for performance-critical paths

### 🚨 **Error Handling**
- Always use exc_info=True for exception logging
- Provide context about what operation failed
- Include relevant parameters and state information
- Use appropriate log levels for error severity

## Troubleshooting

### Common Issues

**1. Log files not created**
- Check directory permissions
- Verify XPCS_LOG_FILE path is writable
- Use absolute paths in configuration

**2. Too much debug output**
- Check XPCS_LOG_LEVEL environment variable
- Verify third-party logger levels (matplotlib, h5py)
- Use rate limiting for verbose operations

**3. Performance impact**
- Disable memory tracking in production
- Use appropriate log levels
- Consider async logging for high-throughput operations

**4. Missing context information**
- Ensure process filters are added to loggers
- Check contextual logger configuration
- Verify extra fields are properly passed

### Debug Commands

```bash
# Test logging configuration
python -c "
from xpcs_toolkit.helper.logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)
logger.info('Logging test successful')
"

# Validate configuration file
python -c "
import yaml
with open('/etc/xpcs/logging.yaml') as f:
    config = yaml.safe_load(f)
    print('Configuration valid')
"

# Check environment variables
env | grep XPCS_LOG
```

## Migration Guide

### From Basic Logging

```python
# Old approach
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# New approach
from xpcs_toolkit.helper.logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__, module_context="important_info")
```

### From Print Statements

```python
# Old approach
print(f"Processing file {filename}")
print(f"Completed in {elapsed:.2f} seconds")

# New approach
logger.info("Processing file", extra={"filename": filename})
with PerformanceTimer(logger, "File processing"):
    # ... processing code ...
```

## API Reference

### Core Functions
- `setup_logging(config=None, env_prefix="XPCS_", config_file=None)`
- `get_logger(name, **context) -> ContextualLoggerAdapter`
- `get_default_config(**kwargs) -> Dict`

### Utility Classes
- `PerformanceTimer`: Performance monitoring context manager/decorator
- `RateLimitFilter`: Rate limiting filter for high-frequency messages
- `ProcessInfoFilter`: Automatic process information injection
- `LoggerWriter`: Stdout/stderr redirection to logging

### Context Managers
- `log_exceptions(logger, message, level, reraise)`
- `redirect_std_streams(stdout_logger, stderr_logger, async_mode)`

For complete API documentation, see the inline documentation in the source code modules.
