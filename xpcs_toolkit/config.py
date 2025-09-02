"""
Centralized configuration management for XPCS Toolkit.

This module provides a unified configuration system for all XPCS Toolkit
components, including file I/O settings, analysis parameters, logging
configuration, and performance tuning options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any


@dataclass
class XpcsConfig:
    """Main configuration class for XPCS Toolkit."""

    # File I/O settings
    default_file_format: str = "nexus"
    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "xpcs_toolkit"
    )
    temp_dir: Path | None = None
    max_cache_size_mb: int = 1000

    # Analysis settings
    default_correlation_type: str = "multitau"
    max_workers: int = field(default_factory=lambda: os.cpu_count() or 4)
    chunk_size: int = 1000

    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Path | None = None

    # Performance settings
    use_parallel_processing: bool = True
    memory_limit_gb: float = 8.0
    enable_caching: bool = True

    # Plotting settings
    default_plot_backend: str = "matplotlib"
    figure_dpi: int = 100
    save_plots: bool = False
    plot_format: str = "png"

    @classmethod
    def from_env(cls) -> XpcsConfig:
        """
        Create configuration from environment variables.

        Environment variables are prefixed with XPCS_ and follow the pattern:
        XPCS_<SETTING_NAME> where setting names are uppercase.

        Returns
        -------
        XpcsConfig
            Configuration instance populated from environment variables
        """
        kwargs = {}

        # Map environment variables to config fields
        env_mapping = {
            "XPCS_DEFAULT_FILE_FORMAT": "default_file_format",
            "XPCS_CACHE_DIR": ("cache_dir", Path),
            "XPCS_TEMP_DIR": ("temp_dir", lambda x: Path(x) if x else None),
            "XPCS_MAX_CACHE_SIZE_MB": ("max_cache_size_mb", int),
            "XPCS_DEFAULT_CORRELATION_TYPE": "default_correlation_type",
            "XPCS_MAX_WORKERS": ("max_workers", int),
            "XPCS_CHUNK_SIZE": ("chunk_size", int),
            "XPCS_LOG_LEVEL": "log_level",
            "XPCS_LOG_FORMAT": "log_format",
            "XPCS_LOG_FILE": ("log_file", lambda x: Path(x) if x else None),
            "XPCS_USE_PARALLEL_PROCESSING": (
                "use_parallel_processing",
                lambda x: x.lower() == "true",
            ),
            "XPCS_MEMORY_LIMIT_GB": ("memory_limit_gb", float),
            "XPCS_ENABLE_CACHING": ("enable_caching", lambda x: x.lower() == "true"),
        }

        for env_var, mapping in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                if isinstance(mapping, tuple):
                    field_name, converter = mapping
                    kwargs[field_name] = converter(value)
                else:
                    kwargs[mapping] = value

        return cls(**kwargs)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> XpcsConfig:
        """
        Create configuration from a dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary containing configuration values

        Returns
        -------
        XpcsConfig
            Configuration instance
        """
        # Convert string paths to Path objects
        if "cache_dir" in config_dict and isinstance(config_dict["cache_dir"], str):
            config_dict["cache_dir"] = Path(config_dict["cache_dir"])
        if "temp_dir" in config_dict and isinstance(config_dict["temp_dir"], str):
            config_dict["temp_dir"] = Path(config_dict["temp_dir"])
        if "log_file" in config_dict and isinstance(config_dict["log_file"], str):
            config_dict["log_file"] = Path(config_dict["log_file"])

        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        dict
            Configuration as dictionary
        """
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, Path):
                value = str(value)
            result[field_name] = value
        return result

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


# Global configuration instance
from typing import Optional

_global_config: Optional[XpcsConfig] = None


def get_config() -> XpcsConfig:
    """
    Get the global configuration instance.

    Returns
    -------
    XpcsConfig
        Global configuration instance
    """
    global _global_config
    if _global_config is None:
        _global_config = XpcsConfig()
    return _global_config


def set_config(config: XpcsConfig) -> None:
    """
    Set the global configuration instance.

    Parameters
    ----------
    config : XpcsConfig
        Configuration instance to use globally
    """
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _global_config
    _global_config = XpcsConfig()
