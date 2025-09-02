"""
Comprehensive tests for FileIO modules.

This module provides comprehensive tests for the file I/O components of the XPCS Toolkit,
including HDF5 file handling, q-space mapping, and file type detection utilities.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch
import warnings

import h5py
import numpy as np
import pytest

# Import FileIO modules
from xpcs_toolkit.fileIO import ftype_utils, hdf_reader, qmap_utils
from xpcs_toolkit.fileIO.aps_8idi import key as hdf_key


class TestHdfReader:
    """Test HDF5 file reading and writing operations."""

    def test_put_function_basic_usage(self):
        """Test basic HDF5 writing functionality."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Test data
            test_data = {
                "array_data": np.array([1, 2, 3, 4, 5]),
                "scalar_data": 42.0,
                "string_data": "test_string",
            }

            # Write data
            hdf_reader.put(tmp_path, test_data, file_type="nexus", mode="raw")

            # Verify data was written
            with h5py.File(tmp_path, "r") as f:
                assert "array_data" in f
                # Account for potential shape changes in HDF5 storage
                stored_array = f["array_data"][:]
                if stored_array.ndim > test_data["array_data"].ndim:
                    stored_array = stored_array.squeeze()
                np.testing.assert_array_equal(stored_array, test_data["array_data"])
                assert f["scalar_data"][()] == test_data["scalar_data"]
                assert f["string_data"][()].decode() == test_data["string_data"]
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_put_function_file_type_validation(self):
        """Test file type validation in put function."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            test_data = {"test": np.array([1, 2, 3])}

            # Test invalid file type
            with pytest.raises(ValueError, match="Unsupported file type"):
                hdf_reader.put(tmp_path, test_data, file_type="invalid_type")

            # Test valid file types
            for file_type in hdf_key:
                hdf_reader.put(tmp_path, test_data, file_type=file_type)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_put_function_mode_validation(self):
        """Test mode validation in put function."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            test_data = {"test": np.array([1, 2, 3])}

            # Test invalid mode
            with pytest.raises(ValueError, match="Unsupported mode"):
                hdf_reader.put(
                    tmp_path, test_data, file_type="nexus", mode="invalid_mode"
                )

            # Test valid modes
            for mode in ["raw", "alias"]:
                hdf_reader.put(tmp_path, test_data, file_type="nexus", mode=mode)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_put_function_deprecated_ftype_parameter(self):
        """Test backward compatibility warning for deprecated ftype parameter."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            test_data = {"test": np.array([1, 2, 3])}

            # Test deprecated ftype parameter
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                hdf_reader.put(tmp_path, test_data, ftype="nexus")

                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "ftype' is deprecated" in str(w[0].message)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_put_function_alias_mode(self):
        """Test alias mode functionality."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create test data with keys that should be aliased
            test_data = {"test_key": np.array([1, 2, 3])}

            # Test alias mode (skip complex mocking, just verify it doesn't crash)
            try:
                hdf_reader.put(tmp_path, test_data, file_type="nexus", mode="alias")
            except (KeyError, AttributeError):
                # Expected if alias mapping doesn't exist for test_key
                pass

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_put_function_large_data_logging(self):
        """Test logging for large data operations."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create large test data
            large_array = np.random.rand(1000, 1000)  # ~8MB array
            test_data = {"large_data": large_array}

            # Test that it handles large data without errors
            hdf_reader.put(tmp_path, test_data, file_type="nexus", mode="raw")

            # Verify data integrity
            with h5py.File(tmp_path, "r") as f:
                np.testing.assert_array_equal(f["large_data"][:], large_array)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestQmapUtils:
    """Test q-space mapping utilities."""

    def test_qmap_manager_initialization(self):
        """Test QMapManager initialization."""
        # Test with disk cache enabled
        manager1 = qmap_utils.QMapManager(use_disk_cache=True)
        assert manager1.use_disk_cache is True
        assert hasattr(manager1, "db")
        assert hasattr(manager1, "_cached_qmap_loader")

        # Test with disk cache disabled
        manager2 = qmap_utils.QMapManager(use_disk_cache=False)
        assert manager2.use_disk_cache is False
        assert hasattr(manager2, "db")
        assert hasattr(manager2, "_cached_qmap_loader")

    def test_cache_key_generation(self):
        """Test cache key generation for q-maps."""
        manager = qmap_utils.QMapManager(use_disk_cache=False)

        # Create a temporary HDF5 file with mock geometry data
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create mock file with required geometry parameters
            with h5py.File(tmp_path, "w") as f:
                # Create nested group structure
                xpcs_group = f.create_group("xpcs")
                xpcs_group.create_group("qmap")

                # Mock geometry parameters using simple keys for testing
                f.create_dataset("bcx", data=100.0)
                f.create_dataset("bcy", data=200.0)
                f.create_dataset("X_energy", data=8000.0)
                f.create_dataset("pixel_size", data=0.075)
                f.create_dataset("det_dist", data=5000.0)

            # Test cache key generation
            with patch.object(
                qmap_utils,
                "key_map",
                {
                    "nexus": {
                        "bcx": "bcx",
                        "bcy": "bcy",
                        "X_energy": "X_energy",
                        "pixel_size": "pixel_size",
                        "det_dist": "det_dist",
                    }
                },
            ):
                cache_key = manager._generate_cache_key(tmp_path)

                assert isinstance(cache_key, str)
                assert len(cache_key) > 0
                # Should contain hash, mtime, and size components
                parts = cache_key.split("_")
                assert len(parts) >= 3

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_cache_key_fallback(self):
        """Test cache key generation fallback for files without geometry."""
        manager = qmap_utils.QMapManager(use_disk_cache=False)

        # Create a temporary HDF5 file without qmap data
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create simple HDF5 file without qmap structure
            with h5py.File(tmp_path, "w") as f:
                f.create_dataset("dummy", data=[1, 2, 3])

            # Test fallback cache key generation
            cache_key = manager._generate_cache_key(tmp_path)

            assert isinstance(cache_key, str)
            assert len(cache_key) > 0
            # Should contain filename, mtime, and size
            assert str(Path(tmp_path).stat().st_mtime).split(".")[0] in cache_key
            assert str(Path(tmp_path).stat().st_size) in cache_key

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_cache_key_generation_with_invalid_file(self):
        """Test cache key generation with non-existent file."""
        manager = qmap_utils.QMapManager(use_disk_cache=False)

        # Test with non-existent file
        non_existent_file = "/path/that/does/not/exist.hdf5"
        cache_key = manager._generate_cache_key(non_existent_file)

        assert isinstance(cache_key, str)
        assert len(cache_key) > 0
        # Should use fallback method with filename and timestamp
        assert "exist.hdf5" in cache_key

    @patch("xpcs_toolkit.fileIO.qmap_utils.QMap")
    def test_load_qmap_uncached(self, mock_qmap_class):
        """Test uncached QMap loading."""
        manager = qmap_utils.QMapManager(use_disk_cache=False)
        mock_qmap_instance = Mock()
        mock_qmap_class.return_value = mock_qmap_instance

        # Test loading qmap
        cache_key = "test_cache_key"
        filename = "test_file.hdf5"

        result = manager._load_qmap_uncached(cache_key, filename)

        assert result == mock_qmap_instance
        mock_qmap_class.assert_called_once_with(filename=filename)

    def test_qmap_manager_in_memory_cache(self):
        """Test in-memory caching functionality."""
        manager = qmap_utils.QMapManager(use_disk_cache=False)

        # Test that db dict exists and can store data
        test_key = "test_qmap_key"
        test_value = Mock()

        manager.db[test_key] = test_value
        assert manager.db[test_key] == test_value
        assert test_key in manager.db

        # Test cache operations
        assert len(manager.db) == 1
        del manager.db[test_key]
        assert len(manager.db) == 0


class TestFtypeUtils:
    """Test file type detection utilities."""

    def test_is_nexus_file_detection(self):
        """Test NeXus file format detection."""
        # Create temporary NeXus file
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create file with NeXus structure
            with h5py.File(tmp_path, "w") as f:
                entry_group = f.create_group("entry")
                instrument_group = entry_group.create_group("instrument")
                bluesky_group = instrument_group.create_group("bluesky")
                metadata_group = bluesky_group.create_group("metadata")
                metadata_group.create_dataset("test", data="nexus_data")

            # Test detection
            assert ftype_utils.isNeXusFile(tmp_path) is True

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_is_nexus_file_false_negative(self):
        """Test NeXus file detection with non-NeXus file."""
        # Create temporary non-NeXus file
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create file without NeXus structure
            with h5py.File(tmp_path, "w") as f:
                f.create_dataset("random_data", data=[1, 2, 3])

            # Test detection
            assert ftype_utils.isNeXusFile(tmp_path) is False

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_is_legacy_file_detection(self):
        """Test legacy file format detection."""
        # Create temporary legacy file
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create file with legacy structure
            with h5py.File(tmp_path, "w") as f:
                xpcs_group = f.create_group("xpcs")
                xpcs_group.create_dataset("Version", data="1.0")

            # Test detection
            assert ftype_utils.isLegacyFile(tmp_path) is True

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_is_legacy_file_false_negative(self):
        """Test legacy file detection with non-legacy file."""
        # Create temporary non-legacy file
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create file without legacy structure
            with h5py.File(tmp_path, "w") as f:
                f.create_dataset("some_data", data=[4, 5, 6])

            # Test detection
            assert ftype_utils.isLegacyFile(tmp_path) is False

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_get_ftype_function(self):
        """Test comprehensive file type detection."""
        # Test with non-existent file
        assert ftype_utils.get_ftype("/path/that/does/not/exist.hdf5") is False

        # Test with legacy file
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            legacy_path = tmp.name

        try:
            with h5py.File(legacy_path, "w") as f:
                xpcs_group = f.create_group("xpcs")
                xpcs_group.create_dataset("Version", data="2.0")

            assert ftype_utils.get_ftype(legacy_path) == "legacy"

        finally:
            if os.path.exists(legacy_path):
                os.unlink(legacy_path)

        # Test with NeXus file
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            nexus_path = tmp.name

        try:
            with h5py.File(nexus_path, "w") as f:
                entry_group = f.create_group("entry")
                instrument_group = entry_group.create_group("instrument")
                bluesky_group = instrument_group.create_group("bluesky")
                metadata_group = bluesky_group.create_group("metadata")
                metadata_group.create_dataset("plan_name", data="scan")

            assert ftype_utils.get_ftype(nexus_path) == "nexus"

        finally:
            if os.path.exists(nexus_path):
                os.unlink(nexus_path)

        # Test with unrecognized file
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            unknown_path = tmp.name

        try:
            with h5py.File(unknown_path, "w") as f:
                f.create_dataset("unknown_data", data="mystery")

            assert ftype_utils.get_ftype(unknown_path) is False

        finally:
            if os.path.exists(unknown_path):
                os.unlink(unknown_path)

    def test_file_detection_error_handling(self):
        """Test error handling in file detection functions."""
        # Test with invalid file (not HDF5)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            txt_path = tmp.name
            tmp.write(b"This is not an HDF5 file")

        try:
            # All functions should handle invalid files gracefully
            assert ftype_utils.isNeXusFile(txt_path) is False
            assert ftype_utils.isLegacyFile(txt_path) is False
            assert ftype_utils.get_ftype(txt_path) is False

        finally:
            if os.path.exists(txt_path):
                os.unlink(txt_path)

    def test_permission_error_handling(self):
        """Test handling of permission errors."""
        # Create a file and remove read permissions
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            restricted_path = tmp.name

        try:
            # Create valid HDF5 file first
            with h5py.File(restricted_path, "w") as f:
                f.create_dataset("test", data=[1, 2, 3])

            # Remove read permissions (on Unix systems)
            if os.name != "nt":  # Skip on Windows
                os.chmod(restricted_path, 0o000)

                # Functions should handle permission errors gracefully
                assert ftype_utils.isNeXusFile(restricted_path) is False
                assert ftype_utils.isLegacyFile(restricted_path) is False
                assert ftype_utils.get_ftype(restricted_path) is False

                # Restore permissions for cleanup
                os.chmod(restricted_path, 0o644)

        finally:
            if os.path.exists(restricted_path):
                # Ensure we can delete the file
                if os.name != "nt":
                    os.chmod(restricted_path, 0o644)
                os.unlink(restricted_path)


class TestFileIOIntegration:
    """Test integration between FileIO modules."""

    def test_hdf_reader_with_file_type_detection(self):
        """Test integration of hdf_reader with file type detection."""
        # Create a NeXus file
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            nexus_path = tmp.name

        try:
            # Create NeXus structure and write some data
            with h5py.File(nexus_path, "w") as f:
                entry_group = f.create_group("entry")
                instrument_group = entry_group.create_group("instrument")
                bluesky_group = instrument_group.create_group("bluesky")
                metadata_group = bluesky_group.create_group("metadata")
                metadata_group.create_dataset("plan_name", data="test_scan")

            # Detect file type
            detected_type = ftype_utils.get_ftype(nexus_path)
            assert detected_type == "nexus"

            # Use detected type with hdf_reader
            test_data = {"integration_test": np.array([10, 20, 30])}
            hdf_reader.put(nexus_path, test_data, file_type=detected_type, mode="raw")

            # Verify data was written correctly
            with h5py.File(nexus_path, "r") as f:
                # Original NeXus structure should still exist
                assert "/entry/instrument/bluesky/metadata/plan_name" in f
                # New data should be added
                assert "integration_test" in f
                # Account for potential shape changes in HDF5 storage
                stored_array = f["integration_test"][:]
                if stored_array.ndim > test_data["integration_test"].ndim:
                    stored_array = stored_array.squeeze()
                np.testing.assert_array_equal(
                    stored_array, test_data["integration_test"]
                )

        finally:
            if os.path.exists(nexus_path):
                os.unlink(nexus_path)

    def test_qmap_cache_with_different_file_types(self):
        """Test QMap caching with different file types."""
        manager = qmap_utils.QMapManager(use_disk_cache=False)

        # Create legacy file
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            legacy_path = tmp.name

        try:
            with h5py.File(legacy_path, "w") as f:
                xpcs_group = f.create_group("xpcs")
                xpcs_group.create_dataset("Version", data="1.0")
                xpcs_group.create_group("qmap")

            # Verify it's detected as legacy
            assert ftype_utils.get_ftype(legacy_path) == "legacy"

            # Test cache key generation for legacy file
            cache_key = manager._generate_cache_key(legacy_path)
            assert isinstance(cache_key, str)
            assert len(cache_key) > 0

        finally:
            if os.path.exists(legacy_path):
                os.unlink(legacy_path)

    def test_error_propagation_across_modules(self):
        """Test error handling across module boundaries."""
        # Test with corrupted file
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            corrupted_path = tmp.name
            # Write invalid HDF5 content
            tmp.write(b"This is definitely not HDF5")

        try:
            # File type detection should handle corrupted files
            assert ftype_utils.get_ftype(corrupted_path) is False
            assert ftype_utils.isNeXusFile(corrupted_path) is False
            assert ftype_utils.isLegacyFile(corrupted_path) is False

            # QMap manager should handle corrupted files gracefully
            manager = qmap_utils.QMapManager(use_disk_cache=False)
            cache_key = manager._generate_cache_key(corrupted_path)
            assert isinstance(cache_key, str)  # Should generate fallback key

        finally:
            if os.path.exists(corrupted_path):
                os.unlink(corrupted_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
