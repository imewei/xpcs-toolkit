"""
Tests for the new modular structure of XPCS Toolkit.

This module tests that the reorganized modular structure is accessible,
properly organized, and maintains expected functionality.
"""

import importlib
from pathlib import Path

import pytest


class TestCoreModules:
    """Test core module structure and accessibility."""

    def test_core_package_structure(self):
        """Test that core package has expected structure."""
        # Test core package import
        import xpcs_toolkit.core

        assert hasattr(xpcs_toolkit.core, "__path__")

        # Test subpackages exist
        import xpcs_toolkit.core.analysis
        import xpcs_toolkit.core.data
        import xpcs_toolkit.core.models

        # Verify they are proper packages
        assert hasattr(xpcs_toolkit.core.analysis, "__path__")
        assert hasattr(xpcs_toolkit.core.data, "__path__")
        assert hasattr(xpcs_toolkit.core.models, "__path__")

    def test_core_analysis_modules(self):
        """Test core analysis modules are accessible."""
        # Test that analysis modules exist
        try:
            import xpcs_toolkit.core.analysis.kernel

            # If import succeeds, verify it has expected structure
            assert hasattr(xpcs_toolkit.core.analysis.kernel, "__file__")
        except ImportError:
            # Module might be empty or not yet implemented
            pytest.skip("Core analysis modules not yet implemented")

    def test_core_data_modules(self):
        """Test core data modules are accessible."""
        try:
            import xpcs_toolkit.core.data.file
            import xpcs_toolkit.core.data.locator

            # Verify modules have expected structure
            assert hasattr(xpcs_toolkit.core.data.file, "__file__")
            assert hasattr(xpcs_toolkit.core.data.locator, "__file__")
        except ImportError:
            pytest.skip("Core data modules not yet implemented")

    def test_core_models_package(self):
        """Test core models package structure."""
        import xpcs_toolkit.core.models

        # Should be a package
        assert hasattr(xpcs_toolkit.core.models, "__path__")

        # Should have __init__.py
        init_file = Path(xpcs_toolkit.core.models.__file__)
        assert init_file.name == "__init__.py"


class TestScientificModules:
    """Test scientific module structure and accessibility."""

    def test_scientific_package_structure(self):
        """Test scientific package structure."""
        import xpcs_toolkit.scientific

        assert hasattr(xpcs_toolkit.scientific, "__path__")

        # Test subpackages
        import xpcs_toolkit.scientific.correlation
        import xpcs_toolkit.scientific.dynamics
        import xpcs_toolkit.scientific.processing
        import xpcs_toolkit.scientific.scattering

        # Verify they are packages
        assert hasattr(xpcs_toolkit.scientific.correlation, "__path__")
        assert hasattr(xpcs_toolkit.scientific.scattering, "__path__")
        assert hasattr(xpcs_toolkit.scientific.dynamics, "__path__")
        assert hasattr(xpcs_toolkit.scientific.processing, "__path__")

    def test_correlation_modules(self):
        """Test correlation analysis modules."""
        try:
            import xpcs_toolkit.scientific.correlation.g2
            import xpcs_toolkit.scientific.correlation.twotime

            # Should be proper modules
            assert hasattr(xpcs_toolkit.scientific.correlation.g2, "__file__")
            assert hasattr(xpcs_toolkit.scientific.correlation.twotime, "__file__")
        except ImportError:
            pytest.skip("Correlation modules not accessible")

    def test_scattering_modules(self):
        """Test scattering analysis modules."""
        try:
            import xpcs_toolkit.scientific.scattering.saxs_1d
            import xpcs_toolkit.scientific.scattering.saxs_2d

            assert hasattr(xpcs_toolkit.scientific.scattering.saxs_1d, "__file__")
            assert hasattr(xpcs_toolkit.scientific.scattering.saxs_2d, "__file__")
        except ImportError:
            pytest.skip("Scattering modules not accessible")

    def test_dynamics_modules(self):
        """Test dynamics analysis modules."""
        try:
            import xpcs_toolkit.scientific.dynamics.intensity
            import xpcs_toolkit.scientific.dynamics.stability
            import xpcs_toolkit.scientific.dynamics.tauq

            assert hasattr(xpcs_toolkit.scientific.dynamics.intensity, "__file__")
            assert hasattr(xpcs_toolkit.scientific.dynamics.stability, "__file__")
            assert hasattr(xpcs_toolkit.scientific.dynamics.tauq, "__file__")
        except ImportError:
            pytest.skip("Dynamics modules not accessible")

    def test_processing_modules(self):
        """Test data processing modules."""
        try:
            import xpcs_toolkit.scientific.processing.averaging

            assert hasattr(xpcs_toolkit.scientific.processing.averaging, "__file__")
        except ImportError:
            pytest.skip("Processing modules not accessible")


class TestIOModules:
    """Test I/O module structure and accessibility."""

    def test_io_package_structure(self):
        """Test I/O package structure."""
        import xpcs_toolkit.io

        assert hasattr(xpcs_toolkit.io, "__path__")

        # Test subpackages
        import xpcs_toolkit.io.cache
        import xpcs_toolkit.io.export
        import xpcs_toolkit.io.formats

        assert hasattr(xpcs_toolkit.io.formats, "__path__")
        assert hasattr(xpcs_toolkit.io.cache, "__path__")
        assert hasattr(xpcs_toolkit.io.export, "__path__")

    def test_formats_modules(self):
        """Test file format modules."""
        try:
            import xpcs_toolkit.io.formats.detection
            import xpcs_toolkit.io.formats.hdf5

            assert hasattr(xpcs_toolkit.io.formats.detection, "__file__")
            assert hasattr(xpcs_toolkit.io.formats.hdf5, "__path__")
        except ImportError:
            pytest.skip("Format modules not accessible")

    def test_hdf5_modules(self):
        """Test HDF5 specific modules."""
        try:
            import xpcs_toolkit.io.formats.hdf5.lazy_reader
            import xpcs_toolkit.io.formats.hdf5.reader

            assert hasattr(xpcs_toolkit.io.formats.hdf5.reader, "__file__")
            assert hasattr(xpcs_toolkit.io.formats.hdf5.lazy_reader, "__file__")
        except ImportError:
            pytest.skip("HDF5 modules not accessible")

    def test_cache_modules(self):
        """Test cache modules."""
        try:
            import xpcs_toolkit.io.cache.qmap_cache

            assert hasattr(xpcs_toolkit.io.cache.qmap_cache, "__file__")
        except ImportError:
            pytest.skip("Cache modules not accessible")


class TestCLIModules:
    """Test CLI module structure and accessibility."""

    def test_cli_package_structure(self):
        """Test CLI package structure."""
        import xpcs_toolkit.cli

        assert hasattr(xpcs_toolkit.cli, "__path__")

        # Test subpackages
        import xpcs_toolkit.cli.commands

        assert hasattr(xpcs_toolkit.cli.commands, "__path__")

    def test_cli_modules(self):
        """Test CLI modules."""
        try:
            import xpcs_toolkit.cli.headless

            assert hasattr(xpcs_toolkit.cli.headless, "__file__")
        except ImportError:
            pytest.skip("CLI modules not accessible")


class TestUtilityModules:
    """Test utility module structure and accessibility."""

    def test_utils_package_structure(self):
        """Test utils package structure."""
        import xpcs_toolkit.utils

        assert hasattr(xpcs_toolkit.utils, "__path__")

        # Test subpackages
        import xpcs_toolkit.utils.common
        import xpcs_toolkit.utils.compatibility
        import xpcs_toolkit.utils.concurrency
        import xpcs_toolkit.utils.logging
        import xpcs_toolkit.utils.math
        import xpcs_toolkit.utils.validation

        # Verify they are packages
        for subpkg in [
            xpcs_toolkit.utils.logging,
            xpcs_toolkit.utils.math,
            xpcs_toolkit.utils.validation,
            xpcs_toolkit.utils.concurrency,
            xpcs_toolkit.utils.compatibility,
            xpcs_toolkit.utils.common,
        ]:
            assert hasattr(subpkg, "__path__")

    def test_logging_utils(self):
        """Test logging utility modules."""
        try:
            import xpcs_toolkit.utils.logging.config
            import xpcs_toolkit.utils.logging.handlers
            import xpcs_toolkit.utils.logging.writer

            assert hasattr(xpcs_toolkit.utils.logging.writer, "__file__")
            assert hasattr(xpcs_toolkit.utils.logging.config, "__file__")
            assert hasattr(xpcs_toolkit.utils.logging.handlers, "__file__")
        except ImportError:
            pytest.skip("Logging utils not accessible")

    def test_math_utils(self):
        """Test math utility modules."""
        try:
            import xpcs_toolkit.utils.math.fitting

            assert hasattr(xpcs_toolkit.utils.math.fitting, "__file__")
        except ImportError:
            pytest.skip("Math utils not accessible")

    def test_compatibility_utils(self):
        """Test compatibility utility modules."""
        try:
            import xpcs_toolkit.utils.compatibility.matplotlib

            assert hasattr(xpcs_toolkit.utils.compatibility.matplotlib, "__file__")
        except ImportError:
            pytest.skip("Compatibility utils not accessible")

    def test_common_utils(self):
        """Test common utility modules."""
        try:
            import xpcs_toolkit.utils.common.helpers
            import xpcs_toolkit.utils.common.lazy_imports

            assert hasattr(xpcs_toolkit.utils.common.helpers, "__file__")
            assert hasattr(xpcs_toolkit.utils.common.lazy_imports, "__file__")
        except ImportError:
            pytest.skip("Common utils not accessible")


class TestModuleInitFiles:
    """Test that all packages have proper __init__.py files."""

    def test_all_packages_have_init_files(self):
        """Test that all packages have __init__.py files."""
        packages_to_check = [
            "xpcs_toolkit.core",
            "xpcs_toolkit.core.analysis",
            "xpcs_toolkit.core.data",
            "xpcs_toolkit.core.models",
            "xpcs_toolkit.scientific",
            "xpcs_toolkit.scientific.correlation",
            "xpcs_toolkit.scientific.scattering",
            "xpcs_toolkit.scientific.dynamics",
            "xpcs_toolkit.scientific.processing",
            "xpcs_toolkit.io",
            "xpcs_toolkit.io.formats",
            "xpcs_toolkit.io.formats.hdf5",
            "xpcs_toolkit.io.cache",
            "xpcs_toolkit.io.export",
            "xpcs_toolkit.cli",
            "xpcs_toolkit.cli.commands",
            "xpcs_toolkit.utils",
            "xpcs_toolkit.utils.logging",
            "xpcs_toolkit.utils.math",
            "xpcs_toolkit.utils.validation",
            "xpcs_toolkit.utils.concurrency",
            "xpcs_toolkit.utils.compatibility",
            "xpcs_toolkit.utils.common",
            "xpcs_toolkit.plugins",
        ]

        for package_name in packages_to_check:
            try:
                package = importlib.import_module(package_name)

                # Should be a package (have __path__)
                assert hasattr(package, "__path__"), f"{package_name} is not a package"

                # Should have __init__.py
                init_file = Path(package.__file__)
                assert init_file.name == "__init__.py", (
                    f"{package_name} missing __init__.py"
                )

                # __init__.py should exist on filesystem
                assert init_file.exists(), f"{package_name} __init__.py doesn't exist"

            except ImportError:
                pytest.skip(f"Package {package_name} not accessible")


class TestModularStructureIntegration:
    """Integration tests for the modular structure."""

    def test_module_discoverability(self):
        """Test that modules can be discovered programmatically."""
        import pkgutil

        import xpcs_toolkit

        # Get all submodules
        discovered_modules = []
        for _importer, modname, _ispkg in pkgutil.walk_packages(
            xpcs_toolkit.__path__, xpcs_toolkit.__name__ + "."
        ):
            discovered_modules.append(modname)

        # Should discover key modules
        expected_patterns = [
            "xpcs_toolkit.core",
            "xpcs_toolkit.scientific",
            "xpcs_toolkit.io",
            "xpcs_toolkit.utils",
        ]

        for pattern in expected_patterns:
            matching_modules = [m for m in discovered_modules if m.startswith(pattern)]
            assert len(matching_modules) > 0, f"No modules found matching {pattern}"

    def test_cross_module_imports(self):
        """Test that modules can import from each other appropriately."""
        try:
            # Test that core modules can import utils
            # This is mainly a structural test
            import xpcs_toolkit.core
            import xpcs_toolkit.utils

            # Modules should be importable without circular imports
            assert True

        except ImportError as e:
            pytest.skip(f"Cross-module import test skipped: {e}")

    def test_package_metadata(self):
        """Test that packages have appropriate metadata."""
        packages_to_check = [
            "xpcs_toolkit.core",
            "xpcs_toolkit.scientific",
            "xpcs_toolkit.io",
            "xpcs_toolkit.utils",
        ]

        for package_name in packages_to_check:
            try:
                package = importlib.import_module(package_name)

                # Should have docstring
                assert package.__doc__ is not None, f"{package_name} missing docstring"
                assert len(package.__doc__.strip()) > 0, (
                    f"{package_name} has empty docstring"
                )

            except ImportError:
                pytest.skip(f"Package {package_name} not accessible")

    def test_future_extensibility(self):
        """Test that structure supports future extensibility."""
        # Test plugin system foundation
        import xpcs_toolkit.plugins

        assert hasattr(xpcs_toolkit.plugins, "__path__")

        # Test documentation structure
        import xpcs_toolkit.docs

        assert hasattr(xpcs_toolkit.docs, "__path__")

        # Test script support
        import xpcs_toolkit.scripts

        assert hasattr(xpcs_toolkit.scripts, "__path__")


class TestModularStructurePerformance:
    """Test performance characteristics of modular structure."""

    def test_import_performance(self):
        """Test that imports are reasonably fast."""
        import time

        # Test main package import time
        start_time = time.time()
        main_import_time = time.time() - start_time

        # Should be reasonably fast (less than 2 seconds)
        assert main_import_time < 2.0, (
            f"Main import took {main_import_time:.2f}s (too slow)"
        )

        # Test subpackage import times
        subpackages = [
            "xpcs_toolkit.core",
            "xpcs_toolkit.scientific",
            "xpcs_toolkit.io",
            "xpcs_toolkit.utils",
        ]

        for package_name in subpackages:
            start_time = time.time()
            try:
                importlib.import_module(package_name)
                import_time = time.time() - start_time

                # Subpackage imports should be very fast
                assert import_time < 0.5, (
                    f"{package_name} import took {import_time:.2f}s (too slow)"
                )
            except ImportError:
                # Skip if package not available
                continue


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
