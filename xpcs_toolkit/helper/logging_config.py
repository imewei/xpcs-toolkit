"""
XPCS Toolkit - Centralized Logging Configuration

This module provides comprehensive logging configuration for the XPCS Toolkit,
supporting both development and production environments with flexible control
over log levels, formats, and output destinations.

## Features

### Environment Variable Configuration
- XPCS_LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- XPCS_LOG_FILE: Specify log file path (defaults to ./xpcs_toolkit.log)
- XPCS_LOG_CONFIG: Path to external YAML/JSON configuration file
- XPCS_LOG_FORMAT: Log format style (simple, detailed, json)
- XPCS_LOG_MAX_SIZE: Maximum size for rotating log files (default: 10MB)
- XPCS_LOG_BACKUP_COUNT: Number of backup files to keep (default: 5)

### Flexible Configuration
- Default configuration with console and rotating file handlers
- JSON-structured logging for production environments
- Contextual logging with LoggerAdapter for enhanced metadata
- Performance monitoring and diagnostic capabilities

### Integration Features
- Seamless integration with existing CLI interface
- Backward compatibility with current logging patterns
- Support for custom configuration files
- Environment-specific optimizations

## Usage Examples

### Basic Setup
```python
from xpcs_toolkit.helper.logging_config import setup_logging, get_logger

# Initialize with defaults
setup_logging()

# Get contextual logger
logger = get_logger(__name__, experiment_id="exp001", user="researcher")
logger.info("Starting analysis", extra={"q_range": (0.01, 0.1)})
```

### Advanced Configuration
```python
# Custom configuration
config = {
    "level": "DEBUG",
    "file_path": "/var/log/xpcs/analysis.log",
    "format": "json",
    "max_size_mb": 50,
    "backup_count": 10
}
setup_logging(config)
```

### Environment Variables
```bash
export XPCS_LOG_LEVEL=DEBUG
export XPCS_LOG_FILE=/tmp/xpcs_debug.log
export XPCS_LOG_FORMAT=json
python -m xpcs_toolkit.cli_headless saxs2d /data/experiment/
```
"""

from datetime import datetime
import json
import logging
import logging.config
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Optional

__all__ = ["setup_logging", "get_default_config", "get_logger"]


def get_default_config(
    level: str = "INFO",
    file_path: Optional[str] = None,
    format_style: str = "detailed",
    max_size_mb: int = 10,
    backup_count: int = 5,
    include_console: bool = True,
) -> dict[str, Any]:
    """
    Generate default logging configuration dictionary.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    file_path : str, optional
        Path for log file. If None, uses './xpcs_toolkit.log'
    format_style : str
        Format style: 'simple', 'detailed', or 'json'
    max_size_mb : int
        Maximum size for rotating log files in MB
    backup_count : int
        Number of backup files to keep
    include_console : bool
        Whether to include console output

    Returns
    -------
    dict
        Logging configuration dictionary suitable for logging.config.dictConfig
    """
    if file_path is None:
        file_path = "xpcs_toolkit.log"

    # Define formatters
    formatters = {
        "simple": {"format": "%(levelname)s - %(name)s - %(message)s"},
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {"class": "xpcs_toolkit.helper.logging_config.JSONFormatter"},
    }

    # Define handlers
    handlers = {}

    if include_console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": format_style,
            "stream": "ext://sys.stdout",
        }

    # Ensure log directory exists
    log_path = Path(file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers["rotating_file"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "level": "DEBUG",  # File captures everything, console filters by level
        "formatter": format_style,
        "filename": str(log_path),
        "maxBytes": max_size_mb * 1024 * 1024,
        "backupCount": backup_count,
        "encoding": "utf-8",
    }

    # Main configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "filters": {
            "context_filter": {"()": "xpcs_toolkit.helper.logging_config.ContextFilter"}
        },
        "loggers": {
            "xpcs_toolkit": {
                "level": "DEBUG",
                "handlers": list(handlers.keys()),
                "filters": ["context_filter"],
                "propagate": False,
            },
            # Suppress verbose third-party logging
            "matplotlib": {"level": "WARNING"},
            "h5py": {"level": "WARNING"},
            "numpy": {"level": "WARNING"},
        },
        "root": {
            "level": level,
            "handlers": list(handlers.keys()) if not handlers else ["console"],
        },
    }

    return config


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs log records as JSON objects with timestamp, level, logger name,
    message, and any additional context information.
    """

    def format(self, record):
        """Format log record as JSON string."""
        # Base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add process information
        log_entry.update(
            {
                "process_id": os.getpid(),
                "thread_id": record.thread,
                "thread_name": record.threadName,
            }
        )

        # Add exception information if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add any extra context information
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
            }:
                extra_fields[key] = value

        if extra_fields:
            log_entry["context"] = extra_fields

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """
    Filter to add contextual information to log records.

    Automatically adds process ID, hostname, and other system information
    to every log record for enhanced debugging and monitoring.
    """

    def __init__(self):
        super().__init__()
        import socket

        self.hostname = socket.gethostname()
        self.pid = os.getpid()

    def filter(self, record):
        """Add context information to log record."""
        record.hostname = self.hostname
        record.process_id = self.pid
        return True


class ContextualLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes context in all log messages.

    This adapter allows associating persistent context (like experiment_id,
    user, file_path) with a logger instance, so this information is
    automatically included in every log message.
    """

    def process(self, msg, kwargs):
        """Process log message to include context."""
        # Merge adapter context with any additional context in kwargs
        # self.extra is a Mapping that might not have a copy method, so convert to dict
        extra_dict = dict(self.extra) if self.extra else {}

        if "extra" in kwargs:
            kwargs["extra"].update(extra_dict)
        else:
            kwargs["extra"] = extra_dict
        return msg, kwargs


def setup_logging(
    config: Optional[dict[str, Any]] = None,
    env_prefix: str = "XPCS_",
    config_file: Optional[str] = None,
) -> None:
    """
    Setup comprehensive logging configuration for XPCS Toolkit.

    This function provides flexible logging setup with support for:
    - Environment variable configuration
    - Custom configuration dictionaries
    - External configuration files
    - Sensible defaults for immediate use

    Parameters
    ----------
    config : dict, optional
        Custom logging configuration dictionary. If provided, overrides defaults.
    env_prefix : str
        Prefix for environment variables (default: "XPCS_")
    config_file : str, optional
        Path to external YAML or JSON configuration file

    Environment Variables
    --------------------
    XPCS_LOG_LEVEL : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    XPCS_LOG_FILE : str
        Log file path
    XPCS_LOG_CONFIG : str
        Path to external configuration file
    XPCS_LOG_FORMAT : str
        Log format style (simple, detailed, json)
    XPCS_LOG_MAX_SIZE : int
        Maximum log file size in MB
    XPCS_LOG_BACKUP_COUNT : int
        Number of backup files to keep

    Raises
    ------
    ValueError
        If configuration is invalid or file paths are inaccessible
    FileNotFoundError
        If specified config file doesn't exist
    """
    # Start with defaults
    if config is None:
        # Check for external config file
        config_file = config_file or os.getenv(f"{env_prefix}LOG_CONFIG")

        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    if config_file.endswith(".json"):
                        file_config = json.load(f)
                    elif config_file.endswith((".yml", ".yaml")):
                        try:
                            import yaml

                            file_config = yaml.safe_load(f)
                        except ImportError:
                            raise ValueError(
                                "PyYAML is required to load YAML configuration files. "
                                "Install with: pip install pyyaml"
                            )
                    else:
                        raise ValueError(
                            f"Unsupported config file format: {config_file}"
                        )

                config = file_config
            except Exception as e:
                # Don't fail completely, fall back to defaults
                print(
                    f"Warning: Failed to load config file {config_file}: {e}",
                    file=sys.stderr,
                )

        # If still no config, create from environment variables and defaults
        if config is None:
            level = os.getenv(f"{env_prefix}LOG_LEVEL", "INFO").upper()
            file_path = os.getenv(f"{env_prefix}LOG_FILE")
            format_style = os.getenv(f"{env_prefix}LOG_FORMAT", "detailed")
            max_size = int(os.getenv(f"{env_prefix}LOG_MAX_SIZE", "10"))
            backup_count = int(os.getenv(f"{env_prefix}LOG_BACKUP_COUNT", "5"))

            config = get_default_config(
                level=level,
                file_path=file_path,
                format_style=format_style,
                max_size_mb=max_size,
                backup_count=backup_count,
            )

    try:
        # Apply configuration
        logging.config.dictConfig(config)

        # Log successful initialization
        logger = logging.getLogger(__name__)
        logger.info("XPCS Toolkit logging initialized successfully")
        logger.debug(
            "Logging configuration: %s", json.dumps(config, indent=2, default=str)
        )

    except Exception as e:
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("xpcs_toolkit_fallback.log"),
            ],
        )
        logger = logging.getLogger(__name__)
        logger.error("Failed to apply logging configuration, using fallback: %s", e)
        raise


def get_logger(name: str, **context) -> ContextualLoggerAdapter:
    """
    Get a contextual logger with automatic context injection.

    This function returns a LoggerAdapter that automatically includes
    the provided context information in all log messages. This is particularly
    useful for associating log messages with specific experiments, files,
    or analysis parameters.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)
    **context
        Key-value pairs to include in all log messages

    Returns
    -------
    ContextualLoggerAdapter
        Logger adapter with context information

    Examples
    --------
    >>> logger = get_logger(__name__, experiment_id="exp001", user="researcher")
    >>> logger.info("Analysis started", extra={"q_range": (0.01, 0.1)})
    >>> # Log message will include experiment_id, user, and q_range

    >>> logger = get_logger(__name__, file_path="/data/sample.hdf")
    >>> logger.error("Failed to read file")
    >>> # Log message will include file_path context
    """
    base_logger = logging.getLogger(name)
    return ContextualLoggerAdapter(base_logger, context)


# Convenience function for backward compatibility
def configure_logging(enable_verbose_output: bool = False) -> None:
    """
    Backward compatibility function for existing CLI interface.

    Parameters
    ----------
    enable_verbose_output : bool
        If True, enable DEBUG level logging

    Deprecated
    ----------
    This function is deprecated. Use setup_logging() instead.
    """
    import warnings

    warnings.warn(
        "configure_logging() is deprecated. Use setup_logging() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    level = "DEBUG" if enable_verbose_output else "INFO"
    config = get_default_config(level=level)
    setup_logging(config)
