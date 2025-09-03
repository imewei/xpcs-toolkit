"""
Tests for the XPCS Toolkit configuration system.

This module tests the centralized configuration management including
XpcsConfig class, environment variable handling, and global configuration.
"""

import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from xpcs_toolkit.config import XpcsConfig, get_config, reset_config, set_config


class TestXpcsConfig:
    """Test suite for XpcsConfig class."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = XpcsConfig()

        assert config.default_file_format == "nexus"
        assert config.cache_dir == Path.home() / ".cache" / "xpcs_toolkit"
        assert config.temp_dir is None
        assert config.max_cache_size_mb == 1000
        assert config.default_correlation_type == "multitau"
        assert config.max_workers == os.cpu_count() or 4
        assert config.chunk_size == 1000
        assert config.log_level == "INFO"
        assert (
            config.log_format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        assert config.log_file is None
        assert config.use_parallel_processing is True
        assert config.memory_limit_gb == 8.0
        assert config.enable_caching is True
        assert config.default_plot_backend == "matplotlib"
        assert config.figure_dpi == 100
        assert config.save_plots is False
        assert config.plot_format == "png"

    def test_custom_configuration(self):
        """Test creating configuration with custom values."""
        custom_cache_dir = Path("/tmp/custom_cache")  # nosec B108
        custom_temp_dir = Path("/tmp/custom_temp")  # nosec B108
        custom_log_file = Path("/tmp/custom.log")  # nosec B108

        config = XpcsConfig(
            default_file_format="hdf5",
            cache_dir=custom_cache_dir,
            temp_dir=custom_temp_dir,
            max_cache_size_mb=2000,
            default_correlation_type="linear",
            max_workers=16,
            chunk_size=2000,
            log_level="DEBUG",
            log_file=custom_log_file,
            use_parallel_processing=False,
            memory_limit_gb=16.0,
            enable_caching=False,
            figure_dpi=300,
            save_plots=True,
            plot_format="pdf",
        )

        assert config.default_file_format == "hdf5"
        assert config.cache_dir == custom_cache_dir
        assert config.temp_dir == custom_temp_dir
        assert config.max_cache_size_mb == 2000
        assert config.default_correlation_type == "linear"
        assert config.max_workers == 16
        assert config.chunk_size == 2000
        assert config.log_level == "DEBUG"
        assert config.log_file == custom_log_file
        assert config.use_parallel_processing is False
        assert config.memory_limit_gb == 16.0
        assert config.enable_caching is False
        assert config.figure_dpi == 300
        assert config.save_plots is True
        assert config.plot_format == "pdf"

    def test_from_env_basic(self):
        """Test creating configuration from environment variables."""
        env_vars = {
            "XPCS_DEFAULT_FILE_FORMAT": "hdf5",
            "XPCS_MAX_CACHE_SIZE_MB": "2000",
            "XPCS_MAX_WORKERS": "8",
            "XPCS_LOG_LEVEL": "DEBUG",
            "XPCS_USE_PARALLEL_PROCESSING": "false",
            "XPCS_ENABLE_CACHING": "true",
            "XPCS_MEMORY_LIMIT_GB": "16.5",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = XpcsConfig.from_env()

            assert config.default_file_format == "hdf5"
            assert config.max_cache_size_mb == 2000
            assert config.max_workers == 8
            assert config.log_level == "DEBUG"
            assert config.use_parallel_processing is False
            assert config.enable_caching is True
            assert config.memory_limit_gb == 16.5

    def test_from_env_with_paths(self):
        """Test environment variables with path values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            log_file = Path(temp_dir) / "app.log"

            env_vars = {
                "XPCS_CACHE_DIR": str(cache_dir),
                "XPCS_TEMP_DIR": str(temp_dir),
                "XPCS_LOG_FILE": str(log_file),
            }

            with patch.dict(os.environ, env_vars, clear=False):
                config = XpcsConfig.from_env()

                assert config.cache_dir == cache_dir
                assert config.temp_dir == Path(temp_dir)
                assert config.log_file == log_file

    def test_from_env_empty_temp_dir(self):
        """Test that empty TEMP_DIR environment variable results in None."""
        env_vars = {"XPCS_TEMP_DIR": "", "XPCS_LOG_FILE": ""}

        with patch.dict(os.environ, env_vars, clear=False):
            config = XpcsConfig.from_env()

            assert config.temp_dir is None
            assert config.log_file is None

    def test_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "default_file_format": "custom",
            "max_workers": 12,
            "log_level": "WARNING",
            "cache_dir": "/tmp/cache",  # nosec B108
            "temp_dir": "/tmp/temp",  # nosec B108
            "log_file": "/tmp/app.log",  # nosec B108
            "use_parallel_processing": False,
            "memory_limit_gb": 32.0,
        }

        config = XpcsConfig.from_dict(config_dict)

        assert config.default_file_format == "custom"
        assert config.max_workers == 12
        assert config.log_level == "WARNING"
        assert config.cache_dir == Path("/tmp/cache")  # nosec B108
        assert config.temp_dir == Path("/tmp/temp")  # nosec B108
        assert config.log_file == Path("/tmp/app.log")  # nosec B108
        assert config.use_parallel_processing is False
        assert config.memory_limit_gb == 32.0

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = XpcsConfig(
            default_file_format="test",
            max_workers=6,
            cache_dir=Path("/test/cache"),
            log_level="ERROR",
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["default_file_format"] == "test"
        assert config_dict["max_workers"] == 6
        assert config_dict["cache_dir"] == "/test/cache"  # Path converted to string
        assert config_dict["log_level"] == "ERROR"

        # Check all expected fields are present
        expected_fields = {
            "default_file_format",
            "cache_dir",
            "temp_dir",
            "max_cache_size_mb",
            "default_correlation_type",
            "max_workers",
            "chunk_size",
            "log_level",
            "log_format",
            "log_file",
            "use_parallel_processing",
            "memory_limit_gb",
            "enable_caching",
            "default_plot_backend",
            "figure_dpi",
            "save_plots",
            "plot_format",
        }
        assert set(config_dict.keys()) == expected_fields

    def test_ensure_directories(self):
        """Test directory creation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            temp_subdir = Path(temp_dir) / "temp"
            log_file = Path(temp_dir) / "logs" / "app.log"

            config = XpcsConfig(
                cache_dir=cache_dir, temp_dir=temp_subdir, log_file=log_file
            )

            # Directories should not exist initially
            assert not cache_dir.exists()
            assert not temp_subdir.exists()
            assert not log_file.parent.exists()

            config.ensure_directories()

            # Directories should be created
            assert cache_dir.exists()
            assert temp_subdir.exists()
            assert log_file.parent.exists()

    def test_ensure_directories_with_none_values(self):
        """Test ensure_directories with None values doesn't crash."""
        config = XpcsConfig(temp_dir=None, log_file=None)

        # Should not raise an exception
        config.ensure_directories()


class TestGlobalConfiguration:
    """Test suite for global configuration management."""

    def setup_method(self):
        """Reset global configuration before each test."""
        reset_config()

    def teardown_method(self):
        """Reset global configuration after each test."""
        reset_config()

    def test_get_default_config(self):
        """Test getting default global configuration."""
        config = get_config()

        assert isinstance(config, XpcsConfig)
        assert config.default_file_format == "nexus"
        assert config.log_level == "INFO"

    def test_set_and_get_config(self):
        """Test setting and getting global configuration."""
        custom_config = XpcsConfig(
            default_file_format="custom", log_level="DEBUG", max_workers=16
        )

        set_config(custom_config)
        retrieved_config = get_config()

        assert retrieved_config is custom_config
        assert retrieved_config.default_file_format == "custom"
        assert retrieved_config.log_level == "DEBUG"
        assert retrieved_config.max_workers == 16

    def test_reset_config(self):
        """Test resetting global configuration to defaults."""
        # Set custom config
        custom_config = XpcsConfig(log_level="DEBUG", max_workers=32)
        set_config(custom_config)

        # Verify it's set
        assert get_config().log_level == "DEBUG"
        assert get_config().max_workers == 32

        # Reset and verify defaults
        reset_config()
        config = get_config()
        assert config.log_level == "INFO"
        assert config.max_workers == os.cpu_count() or 4

    def test_multiple_get_config_calls_return_same_instance(self):
        """Test that multiple get_config calls return the same instance."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_environment_integration(self):
        """Test full environment variable integration."""
        env_vars = {
            "XPCS_DEFAULT_FILE_FORMAT": "integration_test",
            "XPCS_LOG_LEVEL": "WARNING",
            "XPCS_MAX_WORKERS": "4",
            "XPCS_USE_PARALLEL_PROCESSING": "false",
            "XPCS_ENABLE_CACHING": "true",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = XpcsConfig.from_env()

            # Test round-trip conversion
            config_dict = config.to_dict()
            config2 = XpcsConfig.from_dict(config_dict)

            assert config2.default_file_format == "integration_test"
            assert config2.log_level == "WARNING"
            assert config2.max_workers == 4
            assert config2.use_parallel_processing is False
            assert config2.enable_caching is True

    def test_configuration_with_real_paths(self):
        """Test configuration with real filesystem paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            config = XpcsConfig(
                cache_dir=temp_path / "cache",
                temp_dir=temp_path / "temp",
                log_file=temp_path / "logs" / "test.log",
            )

            # Test directory creation
            config.ensure_directories()

            assert config.cache_dir.exists()
            assert config.temp_dir.exists()
            assert config.log_file.parent.exists()

            # Test serialization with real paths
            config_dict = config.to_dict()
            assert isinstance(config_dict["cache_dir"], str)
            assert isinstance(config_dict["temp_dir"], str)
            assert isinstance(config_dict["log_file"], str)

            # Test deserialization
            config2 = XpcsConfig.from_dict(config_dict)
            assert config2.cache_dir == config.cache_dir
            assert config2.temp_dir == config.temp_dir
            assert config2.log_file == config.log_file


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
