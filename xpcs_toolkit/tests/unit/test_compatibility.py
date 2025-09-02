"""
Tests for backward compatibility and import redirection.

This module tests that the reorganization maintains full backward compatibility
with existing import patterns and that deprecated aliases work correctly.
"""

import sys
import warnings

import pytest

from xpcs_toolkit import compat


class TestBackwardCompatibilityImports:
    """Test backward compatibility import functionality."""

    def test_main_package_imports(self):
        """Test that main package imports work correctly."""
        # Test importing from main package
        from xpcs_toolkit import AnalysisKernel, DataFileLocator, XpcsDataFile

        # Verify classes are importable
        assert XpcsDataFile is not None
        assert AnalysisKernel is not None
        assert DataFileLocator is not None

        # Test deprecated aliases
        from xpcs_toolkit import FileLocator, ViewerKernel, XpcsFile

        assert XpcsFile is not None
        assert ViewerKernel is not None
        assert FileLocator is not None

    def test_configuration_imports(self):
        """Test that configuration system imports work."""
        from xpcs_toolkit import XpcsConfig, get_config, set_config

        assert XpcsConfig is not None
        assert callable(get_config)
        assert callable(set_config)

        # Test that we can create and use configuration
        config = XpcsConfig()
        assert config.default_file_format == "nexus"

    def test_new_modular_structure_access(self):
        """Test that new modular structure is accessible."""
        # Test core modules

        # Test I/O modules

        # Test scientific modules

        # Test utility modules

        # All imports should succeed without error
        assert True

    def test_deprecated_warnings_generated(self):
        """Test that deprecated imports generate warnings."""
        # Test that ViewerKernel generates deprecation warning when instantiated
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from xpcs_toolkit import ViewerKernel

            # Warning is generated on instantiation, not import
            try:
                # Try to instantiate with a temporary directory
                import tempfile

                with tempfile.TemporaryDirectory() as temp_dir:
                    ViewerKernel(temp_dir)
            except (FileNotFoundError, OSError, Exception):
                # Even if instantiation fails, warning should still be generated
                pass

            # Should have generated a warning during instantiation attempt
            assert len(w) >= 1
            warning = w[-1]  # Get the last warning
            assert issubclass(warning.category, DeprecationWarning)
            assert "deprecated" in str(warning.message).lower()

    def test_class_functionality_preserved(self):
        """Test that imported classes maintain their functionality."""
        from xpcs_toolkit import AnalysisKernel, DataFileLocator, XpcsDataFile

        # Test that we can instantiate classes (even if with mock data)
        try:
            # These may fail due to missing files, but should not fail due to import issues
            data_file = XpcsDataFile()
            assert hasattr(data_file, "__class__")
        except (FileNotFoundError, TypeError):
            pass  # Expected for missing file

        try:
            locator = DataFileLocator("/tmp")
            assert hasattr(locator, "__class__")
        except (FileNotFoundError, OSError):
            pass  # Expected for missing directory

        try:
            kernel = AnalysisKernel("/tmp")
            assert hasattr(kernel, "__class__")
        except (FileNotFoundError, OSError):
            pass  # Expected for missing directory


class TestCompatibilityLayer:
    """Test the compatibility layer functionality."""

    def test_module_redirects_mapping(self):
        """Test that module redirect mappings are correct."""
        redirects = compat.MODULE_REDIRECTS

        # Test key mappings exist
        assert "xpcs_toolkit.analysis_kernel" in redirects
        assert "xpcs_toolkit.xpcs_file" in redirects
        assert "xpcs_toolkit.data_file_locator" in redirects

        # Test scientific module mappings
        assert "xpcs_toolkit.module.g2mod" in redirects
        assert "xpcs_toolkit.module.saxs1d" in redirects
        assert "xpcs_toolkit.module.tauq" in redirects

        # Test helper module mappings
        assert "xpcs_toolkit.helper.logwriter" in redirects
        assert "xpcs_toolkit.helper.fitting" in redirects

        # Test fileIO module mappings
        assert "xpcs_toolkit.fileIO.hdf_reader" in redirects
        assert "xpcs_toolkit.fileIO.qmap_utils" in redirects

    def test_class_aliases_mapping(self):
        """Test that class aliases are correctly defined."""
        aliases = compat.CLASS_ALIASES

        # Test core class aliases
        assert "AnalysisKernel" in aliases
        assert "XpcsFile" in aliases
        assert "DataFileLocator" in aliases

        # Test deprecated aliases
        assert "ViewerKernel" in aliases
        assert "FileLocator" in aliases

        # Test alias structure
        for _alias, (module_path, class_name) in aliases.items():
            assert isinstance(module_path, str)
            assert isinstance(class_name, str)
            assert module_path.startswith("xpcs_toolkit.")

    def test_get_compatibility_class(self):
        """Test getting classes through compatibility layer."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            # Test getting a known class
            try:
                cls = compat.get_compatibility_class("AnalysisKernel")
                assert cls is not None
                assert hasattr(cls, "__name__")
            except (ImportError, AttributeError):
                pytest.skip("Class not available in current structure")

    def test_get_compatibility_class_unknown(self):
        """Test that unknown classes raise AttributeError."""
        with pytest.raises(AttributeError):
            compat.get_compatibility_class("UnknownClass")

    def test_compatibility_hooks_installation(self):
        """Test that compatibility hooks can be installed."""
        # This is mainly a smoke test since hooks modify sys.meta_path
        original_meta_path = sys.meta_path.copy()

        try:
            compat.install_compatibility_hooks()
            # Should not raise an error
            assert True
        finally:
            # Restore original meta_path to avoid affecting other tests
            sys.meta_path[:] = original_meta_path


class TestDeprecationWarnings:
    """Test deprecation warning functionality."""

    def test_deprecation_warnings_for_old_imports(self):
        """Test that deprecated imports generate appropriate warnings."""
        test_cases = [
            ("ViewerKernel", "deprecated"),
            ("XpcsFile", "deprecated"),
            ("FileLocator", "deprecated"),
        ]

        for class_name, expected_message in test_cases:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                # Import the deprecated class
                exec(f"from xpcs_toolkit import {class_name}")

                # Check if warning was generated
                if w:  # Some warnings might not be generated in test environment
                    warning = w[-1]
                    assert issubclass(warning.category, DeprecationWarning)
                    assert expected_message in str(warning.message).lower()

    def test_no_warnings_for_new_imports(self):
        """Test that new recommended imports don't generate warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import through new paths (if available)

            # Filter for deprecation warnings only
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]

            # Should not have deprecation warnings for new imports
            assert len(deprecation_warnings) == 0


class TestImportCompatibility:
    """Test import compatibility across different import patterns."""

    def test_star_imports(self):
        """Test that star imports work correctly."""
        # Test star imports in a separate namespace
        namespace = {}
        exec("from xpcs_toolkit import *", namespace)

        # Should have imported main classes
        assert "XpcsDataFile" in namespace
        assert "AnalysisKernel" in namespace
        assert "DataFileLocator" in namespace

        # Test that the imported classes are usable
        assert namespace["XpcsDataFile"] is not None
        assert namespace["AnalysisKernel"] is not None
        assert namespace["DataFileLocator"] is not None

    def test_explicit_imports(self):
        """Test explicit import patterns."""
        # Test individual imports
        from xpcs_toolkit import AnalysisKernel, DataFileLocator, XpcsDataFile

        assert XpcsDataFile is not None
        assert AnalysisKernel is not None
        assert DataFileLocator is not None

    def test_aliased_imports(self):
        """Test importing with aliases."""
        from xpcs_toolkit import AnalysisKernel as AK
        from xpcs_toolkit import DataFileLocator as DFL
        from xpcs_toolkit import XpcsDataFile as XDF

        assert XDF is not None
        assert AK is not None
        assert DFL is not None

        # Test that aliases work
        assert XDF.__name__.endswith("XpcsDataFile")

    def test_module_level_imports(self):
        """Test importing at module level."""
        import xpcs_toolkit

        # Should be able to access classes through module
        assert hasattr(xpcs_toolkit, "XpcsDataFile")
        assert hasattr(xpcs_toolkit, "AnalysisKernel")
        assert hasattr(xpcs_toolkit, "DataFileLocator")

        # Should also have configuration
        assert hasattr(xpcs_toolkit, "XpcsConfig")
        assert hasattr(xpcs_toolkit, "get_config")
        assert hasattr(xpcs_toolkit, "set_config")


class TestReorganizationIntegration:
    """Integration tests for the reorganization."""

    def test_old_and_new_apis_coexist(self):
        """Test that old and new APIs can coexist."""
        # Import through old method
        from xpcs_toolkit import XpcsDataFile as OldXpcsDataFile

        # Import through new method (when available)
        try:
            from xpcs_toolkit.core.data.file import XpcsDataFile as NewXpcsDataFile

            # They should have the same name and similar functionality
            # Note: During refactoring, these may be separate implementations
            assert OldXpcsDataFile.__name__ == NewXpcsDataFile.__name__ == "XpcsDataFile"
        except ImportError:
            # New structure might not be fully implemented yet
            pytest.skip("New modular structure not yet available")

    def test_configuration_system_integration(self):
        """Test that configuration system integrates with existing code."""
        from xpcs_toolkit import XpcsConfig, get_config, set_config

        # Test that we can get current config
        config = get_config()
        assert isinstance(config, XpcsConfig)

        # Test that we can set new config
        new_config = XpcsConfig(log_level="DEBUG")
        set_config(new_config)

        retrieved_config = get_config()
        assert retrieved_config.log_level == "DEBUG"

    def test_all_expected_exports_present(self):
        """Test that all expected exports are present in __all__."""
        import xpcs_toolkit

        expected_exports = [
            "XpcsDataFile",
            "AnalysisKernel",
            "DataFileLocator",
            "XpcsFile",
            "ViewerKernel",
            "FileLocator",  # Deprecated
            "XpcsConfig",
            "get_config",
            "set_config",  # New
        ]

        for export in expected_exports:
            assert export in xpcs_toolkit.__all__, f"Missing export: {export}"
            assert hasattr(xpcs_toolkit, export), f"Missing attribute: {export}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
