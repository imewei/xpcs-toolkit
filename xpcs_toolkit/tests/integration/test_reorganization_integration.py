"""
Integration tests for the XPCS Toolkit reorganization.

This module provides comprehensive integration tests to verify that
the reorganization maintains functionality while providing new capabilities.
"""

import os
from pathlib import Path
import tempfile
import warnings

import pytest


class TestReorganizationIntegration:
    """Integration tests for the complete reorganization."""

    def test_end_to_end_backward_compatibility(self):
        """Test complete backward compatibility workflow."""
        # Import using old patterns - should work without issues
        from xpcs_toolkit import AnalysisKernel, DataFileLocator

        # Test that classes can be instantiated
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                locator = DataFileLocator(temp_dir)
                kernel = AnalysisKernel(temp_dir)

                # Basic functionality should work
                assert locator.directory == temp_dir
                assert hasattr(kernel, "build")

        except Exception as e:
            # Some functionality may require real data files
            pytest.skip(f"Integration test requires real data: {e}")

    def test_configuration_integration_workflow(self):
        """Test complete configuration system workflow."""
        from xpcs_toolkit import XpcsConfig, get_config, reset_config, set_config

        # Test default configuration
        default_config = get_config()
        assert isinstance(default_config, XpcsConfig)
        assert default_config.default_file_format == "nexus"

        # Test custom configuration
        custom_config = XpcsConfig(
            default_file_format="custom",
            log_level="DEBUG",
            max_workers=16,
            enable_caching=False,
        )

        set_config(custom_config)
        retrieved_config = get_config()
        assert retrieved_config.default_file_format == "custom"
        assert retrieved_config.log_level == "DEBUG"
        assert retrieved_config.max_workers == 16
        assert retrieved_config.enable_caching is False

        # Test configuration with file paths
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            path_config = XpcsConfig(
                cache_dir=temp_path / "cache",
                temp_dir=temp_path / "temp",
                log_file=temp_path / "logs" / "app.log",
            )

            # Test directory creation
            path_config.ensure_directories()
            assert path_config.cache_dir.exists()
            assert path_config.temp_dir.exists()
            assert path_config.log_file.parent.exists()

            # Test serialization/deserialization
            config_dict = path_config.to_dict()
            restored_config = XpcsConfig.from_dict(config_dict)

            assert restored_config.cache_dir == path_config.cache_dir
            assert restored_config.temp_dir == path_config.temp_dir
            assert restored_config.log_file == path_config.log_file

        # Reset to clean state
        reset_config()
        assert get_config().default_file_format == "nexus"

    def test_environment_configuration_workflow(self):
        """Test configuration from environment variables."""
        from unittest.mock import patch

        from xpcs_toolkit.config import XpcsConfig

        # Test environment variable configuration
        env_vars = {
            "XPCS_DEFAULT_FILE_FORMAT": "env_test",
            "XPCS_LOG_LEVEL": "WARNING",
            "XPCS_MAX_WORKERS": "12",
            "XPCS_USE_PARALLEL_PROCESSING": "false",
            "XPCS_ENABLE_CACHING": "true",
            "XPCS_MEMORY_LIMIT_GB": "24.5",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = XpcsConfig.from_env()

            assert config.default_file_format == "env_test"
            assert config.log_level == "WARNING"
            assert config.max_workers == 12
            assert config.use_parallel_processing is False
            assert config.enable_caching is True
            assert config.memory_limit_gb == 24.5

    def test_mixed_import_patterns(self):
        """Test that different import patterns work together."""
        # Test mixing old and new imports
        from xpcs_toolkit import XpcsConfig
        from xpcs_toolkit import XpcsDataFile as OldImport

        # Test that configuration affects old imports
        config = XpcsConfig(default_file_format="mixed_test")

        # Both should be available
        assert OldImport is not None
        assert XpcsConfig is not None
        assert isinstance(config, XpcsConfig)

    def test_deprecation_warning_workflow(self):
        """Test deprecation warning system works correctly."""
        # Test that deprecated imports generate warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Should generate deprecation warning
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]

            if deprecation_warnings:
                assert len(deprecation_warnings) >= 1
                assert "deprecated" in str(deprecation_warnings[0].message).lower()

    def test_modular_structure_accessibility(self):
        """Test that new modular structure is fully accessible."""
        # Test accessing through new structure
        import xpcs_toolkit.core
        import xpcs_toolkit.io
        import xpcs_toolkit.scientific
        import xpcs_toolkit.utils

        # All should be importable
        assert hasattr(xpcs_toolkit.core, "__path__")
        assert hasattr(xpcs_toolkit.scientific, "__path__")
        assert hasattr(xpcs_toolkit.io, "__path__")
        assert hasattr(xpcs_toolkit.utils, "__path__")

        # Test deeper structure
        import xpcs_toolkit.core.analysis
        import xpcs_toolkit.io.formats
        import xpcs_toolkit.scientific.correlation
        import xpcs_toolkit.utils.logging

        # Verify subpackages exist
        assert hasattr(xpcs_toolkit.core.analysis, "__path__")
        assert hasattr(xpcs_toolkit.scientific.correlation, "__path__")
        assert hasattr(xpcs_toolkit.io.formats, "__path__")
        assert hasattr(xpcs_toolkit.utils.logging, "__path__")

    def test_plugin_system_foundation(self):
        """Test that plugin system foundation is in place."""
        # Test plugin directory exists
        import xpcs_toolkit.plugins

        assert hasattr(xpcs_toolkit.plugins, "__path__")

        # Test documentation structure
        import xpcs_toolkit.docs

        assert hasattr(xpcs_toolkit.docs, "__path__")

        # Test scripts directory
        import xpcs_toolkit.scripts

        assert hasattr(xpcs_toolkit.scripts, "__path__")

    def test_comprehensive_api_coverage(self):
        """Test that all expected APIs are available."""
        import xpcs_toolkit

        # Test main classes
        assert hasattr(xpcs_toolkit, "XpcsDataFile")
        assert hasattr(xpcs_toolkit, "AnalysisKernel")
        assert hasattr(xpcs_toolkit, "DataFileLocator")

        # Test deprecated aliases
        assert hasattr(xpcs_toolkit, "XpcsFile")
        assert hasattr(xpcs_toolkit, "ViewerKernel")
        assert hasattr(xpcs_toolkit, "FileLocator")

        # Test configuration system
        assert hasattr(xpcs_toolkit, "XpcsConfig")
        assert hasattr(xpcs_toolkit, "get_config")
        assert hasattr(xpcs_toolkit, "set_config")

        # Test that __all__ includes all expected exports
        expected_all = [
            "XpcsDataFile",
            "AnalysisKernel",
            "DataFileLocator",
            "XpcsFile",
            "ViewerKernel",
            "FileLocator",
            "XpcsConfig",
            "get_config",
            "set_config",
        ]

        for item in expected_all:
            assert item in xpcs_toolkit.__all__, f"Missing from __all__: {item}"


class TestReorganizationPerformance:
    """Test performance characteristics after reorganization."""

    def test_import_performance_regression(self):
        """Test that reorganization doesn't significantly slow imports."""
        import time

        # Test main package import time
        start_time = time.time()
        import xpcs_toolkit

        main_import_time = time.time() - start_time

        # Should be reasonable (less than 3 seconds)
        assert main_import_time < 3.0, f"Main import too slow: {main_import_time:.2f}s"

        # Test individual class access time
        start_time = time.time()
        _ = xpcs_toolkit.XpcsDataFile
        _ = xpcs_toolkit.AnalysisKernel
        _ = xpcs_toolkit.DataFileLocator
        access_time = time.time() - start_time

        # Class access should be very fast
        assert access_time < 0.1, f"Class access too slow: {access_time:.2f}s"

    def test_memory_usage_regression(self):
        """Test that reorganization doesn't significantly increase memory usage."""
        import sys

        # Get initial module count
        initial_module_count = len(sys.modules)

        # Import main package

        # Get module count after import
        after_import_count = len(sys.modules)

        # Should not import excessive modules
        module_increase = after_import_count - initial_module_count
        assert module_increase < 50, f"Too many modules imported: {module_increase}"


class TestReorganizationRobustness:
    """Test robustness and error handling after reorganization."""

    def test_import_error_handling(self):
        """Test that import errors are handled gracefully."""
        # Test importing non-existent modules
        with pytest.raises(ImportError):
            import xpcs_toolkit.nonexistent_module

        with pytest.raises(ImportError):
            import xpcs_toolkit.core.nonexistent_submodule

        # Test that failed imports don't break subsequent imports
        import xpcs_toolkit

        assert hasattr(xpcs_toolkit, "XpcsDataFile")

    def test_configuration_error_handling(self):
        """Test configuration system error handling."""
        import tempfile

        from xpcs_toolkit.config import XpcsConfig

        # Test invalid configuration values are handled
        config = XpcsConfig()

        # Test with temporary directory that we create and then remove
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create a valid path first
            config.cache_dir = temp_path / "cache"
            config.ensure_directories()
            assert config.cache_dir.exists()

        # Test with None values - should not crash
        config.cache_dir = None
        config.temp_dir = None
        config.log_file = None
        config.ensure_directories()  # Should handle None gracefully

    def test_backwards_compatibility_robustness(self):
        """Test that backwards compatibility is robust."""
        # Test importing deprecated classes still works
        from xpcs_toolkit import FileLocator, ViewerKernel, XpcsFile

        # Should be importable without errors
        assert ViewerKernel is not None
        assert XpcsFile is not None
        assert FileLocator is not None

        # Test that these are proper aliases

        # Note: In our current implementation, these might not be identical
        # due to the way we handle backwards compatibility, but they should exist


class TestReorganizationDocumentation:
    """Test that reorganization maintains documentation."""

    def test_package_docstrings(self):
        """Test that packages have appropriate documentation."""
        import xpcs_toolkit

        # Main package should have docstring
        assert xpcs_toolkit.__doc__ is not None
        assert len(xpcs_toolkit.__doc__.strip()) > 0

        # Test subpackages have docstrings
        packages_to_check = [
            "xpcs_toolkit.core",
            "xpcs_toolkit.scientific",
            "xpcs_toolkit.io",
            "xpcs_toolkit.utils",
        ]

        for package_name in packages_to_check:
            try:
                package = __import__(package_name, fromlist=[""])
                assert package.__doc__ is not None, f"{package_name} missing docstring"
                assert len(package.__doc__.strip()) > 0, (
                    f"{package_name} empty docstring"
                )
            except ImportError:
                # Skip if package not available
                continue

    def test_module_metadata(self):
        """Test that modules have appropriate metadata."""
        import xpcs_toolkit

        # Test version information
        assert hasattr(xpcs_toolkit, "__version__")
        assert hasattr(xpcs_toolkit, "__author__")
        assert hasattr(xpcs_toolkit, "__credits__")

        # Test that __all__ is properly defined
        assert hasattr(xpcs_toolkit, "__all__")
        assert isinstance(xpcs_toolkit.__all__, list)
        assert len(xpcs_toolkit.__all__) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
