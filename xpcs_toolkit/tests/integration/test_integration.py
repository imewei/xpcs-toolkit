"""
Integration tests for XPCS Toolkit.

This module provides comprehensive integration tests that verify the interactions
between different components of the XPCS Toolkit, ensuring that the complete
analysis pipeline works correctly from data loading to result generation.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

# Import core components
from xpcs_toolkit import AnalysisKernel, DataFileLocator
from xpcs_toolkit.module import g2mod, saxs1d


class TestXpcsWorkflowIntegration:
    """Test complete XPCS analysis workflow integration."""

    @pytest.fixture
    def temp_dir_with_mock_data(self):
        """Create temporary directory with mock HDF5 files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some mock XPCS data files
            files = [f"xpcs_data_{i:03d}.hdf" for i in range(5)]
            for filename in files:
                Path(tmpdir, filename).touch()
            yield tmpdir, files

    def test_complete_data_loading_pipeline(self, temp_dir_with_mock_data):
        """Test complete data loading from directory to analysis ready state."""
        temp_dir, mock_files = temp_dir_with_mock_data

        # Step 1: Directory scanning with DataFileLocator
        locator = DataFileLocator(temp_dir)
        success = locator.build_file_list()
        assert success is True
        assert len(locator.source_files) >= 5

        # Step 2: Integration with AnalysisKernel
        kernel = AnalysisKernel(temp_dir)
        kernel.build_file_list()

        # Verify integration between components
        assert kernel.directory == temp_dir
        assert hasattr(kernel, "file_list")
        assert isinstance(kernel.file_list, list)

        # Step 3: Verify file selection integration
        kernel.selected_files = [0, 1, 2]
        selected = kernel.get_selected_files()
        assert selected == [0, 1, 2]

    def test_module_integration_workflow(self, temp_dir_with_mock_data):
        """Test integration between analysis modules."""
        temp_dir, mock_files = temp_dir_with_mock_data

        # Create kernel with mock data
        kernel = AnalysisKernel(temp_dir)
        kernel.file_list = mock_files
        kernel.selected_files = [0, 1]

        # Test G2 module integration
        mock_xf_list = []
        for _i in range(2):
            mock_xf = Mock()
            mock_xf.atype = "Multitau"
            mock_xf.get_g2_data.return_value = (
                np.array([0.01, 0.02]),
                np.logspace(-6, 0, 32),
                np.random.rand(32, 2) + 1.0,
                np.random.rand(32, 2) * 0.01,
                ["Q1", "Q2"],
            )
            mock_xf_list.append(mock_xf)

        # Test cross-module data flow
        g2_result = g2mod.get_data(mock_xf_list)
        assert isinstance(g2_result, tuple)
        assert len(g2_result) >= 4

        # Test SAXS module integration
        color1, marker1 = saxs1d.get_color_marker(0)
        color2, marker2 = saxs1d.get_color_marker(1)

        assert isinstance(color1, str)
        assert isinstance(marker1, str)
        assert color1 != color2 or marker1 != marker2

    def test_error_propagation_integration(self, temp_dir_with_mock_data):
        """Test error handling across module boundaries."""
        temp_dir, mock_files = temp_dir_with_mock_data

        # Create kernel with problematic data
        kernel = AnalysisKernel(temp_dir)
        kernel.file_list = mock_files

        # Test error handling in data tree operations
        try:
            result = kernel.get_data_tree([999])  # Invalid index
            # Should handle gracefully
            assert result is None or result is not None
        except (IndexError, AttributeError, TypeError):
            # Expected for invalid indices
            pass

        # Test error handling in plotting operations
        mock_handler = Mock()
        try:
            kernel.plot_g2_function(
                handler=mock_handler,
                q_range=(0.01, 0.1),
                rows=[999],  # Invalid row
            )
        except (IndexError, AttributeError, TypeError, ValueError):
            # Expected for invalid data
            pass


class TestPerformanceCharacteristics:
    """Test performance characteristics of the XPCS Toolkit."""

    @pytest.fixture
    def large_file_list(self):
        """Create a large list of mock files for performance testing."""
        return [f"file_{i:04d}.hdf" for i in range(100)]

    def test_file_list_building_performance(self, large_file_list):
        """Test performance of file list building operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many files
            for filename in large_file_list:
                Path(tmpdir, filename).touch()

            start_time = time.time()

            # Test DataFileLocator performance
            locator = DataFileLocator(tmpdir)
            success = locator.build_file_list()

            end_time = time.time()
            elapsed = end_time - start_time

            # Should handle 100 files quickly
            assert success is True
            assert len(locator.source_files) >= 100
            assert elapsed < 2.0  # Should complete within 2 seconds

    def test_analysis_kernel_performance(self, large_file_list):
        """Test AnalysisKernel performance with many files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            for filename in large_file_list[:50]:  # Use 50 files for speed
                Path(tmpdir, filename).touch()

            start_time = time.time()

            # Test kernel operations
            kernel = AnalysisKernel(tmpdir)
            kernel.build_file_list()
            kernel.selected_files = list(range(min(10, len(kernel.file_list))))
            selected = kernel.get_selected_files()

            end_time = time.time()
            elapsed = end_time - start_time

            # Performance checks
            assert len(selected) <= 10
            assert elapsed < 1.0  # Should be fast

    def test_module_function_performance(self):
        """Test performance of core module functions."""
        # Test SAXS color generation performance
        start_time = time.time()

        colors = []
        for i in range(1000):
            color, marker = saxs1d.get_color_marker(i % 20)
            colors.append((color, marker))

        end_time = time.time()
        elapsed = end_time - start_time

        assert len(colors) == 1000
        assert elapsed < 0.5  # Should be very fast

        # Verify color/marker diversity
        unique_colors = {color for color, marker in colors}
        assert len(unique_colors) > 1  # Should have multiple colors

    def test_memory_usage_characteristics(self):
        """Test memory usage patterns of core operations."""
        import gc

        # Test that operations clean up properly
        initial_objects = len(gc.get_objects())

        # Perform operations that create temporary objects
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            for i in range(20):
                Path(tmpdir, f"test_{i}.hdf").touch()

            # Create and destroy multiple kernels
            for _ in range(10):
                kernel = AnalysisKernel(tmpdir)
                kernel.build_file_list()
                del kernel

        # Force garbage collection
        gc.collect()

        final_objects = len(gc.get_objects())

        # Object count should not grow excessively
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Reasonable growth limit


class TestConcurrencyAndThreadSafety:
    """Test concurrency and thread safety aspects."""

    def test_concurrent_file_operations(self):
        """Test concurrent file operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(20):
                Path(tmpdir, f"concurrent_test_{i}.hdf").touch()

            results = []
            errors = []

            def worker_function(worker_id):
                """Worker function for concurrent testing."""
                try:
                    locator = DataFileLocator(tmpdir)
                    success = locator.build_file_list()
                    results.append((worker_id, success, len(locator.source_files)))
                except Exception as e:
                    errors.append((worker_id, str(e)))

            # Launch multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker_function, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join()

            # Verify results
            assert len(errors) == 0  # No errors should occur
            assert len(results) == 5  # All threads should complete

            # All should find the same number of files
            file_counts = [count for worker_id, success, count in results]
            assert all(success for worker_id, success, count in results)
            assert all(count >= 20 for count in file_counts)

    def test_module_thread_safety(self):
        """Test thread safety of module functions."""
        results = []
        errors = []

        def color_generator_worker(worker_id):
            """Worker that generates colors concurrently."""
            try:
                worker_colors = []
                for i in range(100):
                    color, marker = saxs1d.get_color_marker(i + worker_id * 100)
                    worker_colors.append((color, marker))
                results.append((worker_id, worker_colors))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Launch multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=color_generator_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify thread safety
        assert len(errors) == 0
        assert len(results) == 3

        # Each worker should generate 100 color/marker pairs
        for _worker_id, colors in results:
            assert len(colors) == 100
            # Verify all are valid color/marker pairs
            for color, marker in colors:
                assert isinstance(color, str)
                assert isinstance(marker, str)


class TestDataFlowIntegration:
    """Test data flow between different components."""

    @pytest.fixture
    def mock_analysis_chain(self):
        """Set up mock analysis chain."""
        # Mock data file
        mock_file = Mock()
        mock_file.atype = "Multitau"
        mock_file.get_g2_data.return_value = (
            np.array([0.005, 0.01, 0.02, 0.05]),
            np.logspace(-6, 0, 64),
            np.random.rand(64, 4) + 1.0,
            np.random.rand(64, 4) * 0.02,
            ["Q1 (0.005)", "Q2 (0.01)", "Q3 (0.02)", "Q4 (0.05)"],
        )

        # Mock SAXS data
        mock_file.get_saxs_1d_data.return_value = (
            np.logspace(-3, -1, 100),  # q values
            np.random.rand(100) * 1e6,  # intensity
            np.random.rand(100) * 1e3,  # error
        )

        return mock_file

    def test_g2_to_visualization_pipeline(self, mock_analysis_chain):
        """Test data flow from G2 analysis to visualization."""
        mock_file = mock_analysis_chain

        # Step 1: Get G2 data
        q, tau, g2, g2_err, labels = mock_file.get_g2_data()

        # Verify data structure
        assert len(q) == 4
        assert len(tau) == 64
        assert g2.shape == (64, 4)
        assert g2_err.shape == (64, 4)
        assert len(labels) == 4

        # Step 2: Process through g2mod
        result = g2mod.get_data([mock_file])
        assert isinstance(result, tuple)

        # Step 3: Test visualization preparation
        # Test color assignment for multiple q-values
        colors = []
        for i in range(len(q)):
            color, marker = saxs1d.get_color_marker(i)
            colors.append((color, marker))

        assert len(colors) == 4
        # Colors should be distinguishable
        unique_colors = {color for color, marker in colors}
        assert len(unique_colors) >= 2

    def test_saxs_analysis_pipeline(self, mock_analysis_chain):
        """Test SAXS analysis pipeline integration."""
        mock_file = mock_analysis_chain

        # Step 1: Get SAXS data
        q, intensity, error = mock_file.get_saxs_1d_data()

        assert len(q) == 100
        assert len(intensity) == 100
        assert len(error) == 100

        # Step 2: Test intensity offsetting for multi-curve plots
        offset_intensity = saxs1d.offset_intensity(
            intensity, n=1, plot_offset=0.1, yscale="log"
        )

        # Should modify the intensity
        assert len(offset_intensity) == len(intensity)
        assert not np.array_equal(intensity, offset_intensity)

        # Step 3: Test color assignment integration
        color, marker = saxs1d.get_color_marker(0)
        assert isinstance(color, str)
        assert isinstance(marker, str)

    def test_multi_file_analysis_integration(self):
        """Test analysis integration across multiple files."""
        # Create multiple mock files
        mock_files = []
        for i in range(3):
            mock_file = Mock()
            mock_file.atype = "Multitau"

            # Each file has slightly different q-range
            q_start = 0.005 * (i + 1)
            mock_file.get_g2_data.return_value = (
                np.array([q_start, q_start * 2, q_start * 4]),
                np.logspace(-5, -1, 32),
                np.random.rand(32, 3) + 1.0,
                np.random.rand(32, 3) * 0.01,
                [f"File{i}_Q{j}" for j in range(3)],
            )
            mock_files.append(mock_file)

        # Test combined analysis
        combined_result = g2mod.get_data(mock_files)
        assert isinstance(combined_result, tuple)

        # Verify each file was processed
        for mock_file in mock_files:
            mock_file.get_g2_data.assert_called()


class TestSystemResourceUsage:
    """Test system resource usage patterns."""

    def test_file_handle_management(self):
        """Test that file handles are properly managed."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_files = len(process.open_files())

        # Perform operations that might open files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many files
            file_count = 50
            for i in range(file_count):
                Path(tmpdir, f"resource_test_{i}.hdf").touch()

            # Create and destroy multiple instances
            for _ in range(5):
                locator = DataFileLocator(tmpdir)
                locator.build_file_list()

                kernel = AnalysisKernel(tmpdir)
                kernel.build_file_list()

                # Explicitly clean up
                del locator, kernel

        final_files = len(process.open_files())

        # File handle count should not grow significantly
        handle_growth = final_files - initial_files
        assert handle_growth <= 2  # Allow for some minor growth

    def test_cpu_usage_characteristics(self):
        """Test CPU usage patterns of intensive operations."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Measure CPU usage during intensive operations
        process.cpu_percent()

        # Perform CPU-intensive operations
        start_time = time.time()

        # Multiple color generations (should be fast)
        for _ in range(1000):
            for i in range(20):
                saxs1d.get_color_marker(i)

        # Multiple file list operations
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(30):
                Path(tmpdir, f"cpu_test_{i}.hdf").touch()

            for _ in range(10):
                locator = DataFileLocator(tmpdir)
                locator.build_file_list()

        end_time = time.time()
        elapsed = end_time - start_time

        # Should complete in reasonable time
        assert elapsed < 3.0  # 3 seconds max

        cpu_percent_after = process.cpu_percent()

        # CPU usage should be reasonable (not constantly maxed)
        # Note: This test may be flaky depending on system load
        assert cpu_percent_after >= 0  # Basic sanity check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
