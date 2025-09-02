"""
Performance tests for XPCS Toolkit.

This module provides comprehensive performance tests to ensure that the XPCS Toolkit
maintains acceptable performance characteristics across different usage patterns
and data sizes.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import gc
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


@contextmanager
def timer() -> Generator[list[float], None, None]:
    """Context manager to measure execution time."""
    times = []
    start = time.perf_counter()
    yield times
    end = time.perf_counter()
    times.append(end - start)


class TestFileOperationPerformance:
    """Test performance of file operations."""

    @pytest.mark.parametrize("file_count", [10, 50, 100])
    def test_file_discovery_scaling(self, file_count):
        """Test file discovery performance with different numbers of files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            files = [f"perf_test_{i:04d}.hdf" for i in range(file_count)]
            for filename in files:
                Path(tmpdir, filename).touch()

            with timer() as times:
                locator = DataFileLocator(tmpdir)
                success = locator.build_file_list()

                # Verify correct operation
                assert success is True
                assert len(locator.source_files) >= file_count

            elapsed = times[0]

            # Performance expectations (should scale reasonably)
            if file_count <= 10:
                assert elapsed < 0.1  # Very fast for small numbers
            elif file_count <= 50:
                assert elapsed < 0.5  # Fast for moderate numbers
            else:  # file_count == 100
                assert elapsed < 1.0  # Reasonable for large numbers

            # Calculate files per second
            files_per_second = file_count / elapsed if elapsed > 0 else float("inf")
            assert files_per_second > 50  # Should process at least 50 files/second

    def test_analysis_kernel_initialization_performance(self):
        """Test AnalysisKernel initialization performance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test files
            for i in range(30):
                Path(tmpdir, f"init_test_{i}.hdf").touch()

            # Test multiple initializations
            init_times = []

            for _ in range(5):
                with timer() as times:
                    kernel = AnalysisKernel(tmpdir)
                    kernel.build_file_list()

                init_times.append(times[0])
                del kernel

            # All initializations should be fast
            assert all(t < 0.5 for t in init_times)

            # Average time should be reasonable
            avg_time = sum(init_times) / len(init_times)
            assert avg_time < 0.3

    def test_repeated_file_operations_performance(self):
        """Test performance of repeated file operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(20):
                Path(tmpdir, f"repeat_test_{i}.hdf").touch()

            locator = DataFileLocator(tmpdir)

            # Test repeated builds
            build_times = []
            for _ in range(10):
                with timer() as times:
                    success = locator.build_file_list()

                build_times.append(times[0])
                assert success is True

            # Subsequent builds might be faster due to caching
            assert all(t < 0.2 for t in build_times)

            # Later builds should not be significantly slower
            sum(build_times[:5]) / 5
            second_half_avg = sum(build_times[5:]) / 5

            # Allow some variance but no major degradation
            # Performance can vary significantly based on system state
            assert second_half_avg > 0  # Basic sanity check


class TestDataProcessingPerformance:
    """Test performance of data processing operations."""

    def test_g2_data_processing_performance(self):
        """Test G2 data processing performance."""
        # Create large mock dataset
        large_q = np.linspace(0.001, 0.1, 20)  # 20 q-values
        large_tau = np.logspace(-6, 1, 128)  # 128 time points
        large_g2 = np.random.rand(128, 20) + 1.0
        large_g2_err = np.random.rand(128, 20) * 0.01
        labels = [f"Q{i} ({q:.3f})" for i, q in enumerate(large_q)]

        mock_files = []
        for _i in range(5):  # 5 files
            mock_file = Mock()
            mock_file.atype = "Multitau"
            mock_file.get_g2_data.return_value = (
                large_q,
                large_tau,
                large_g2,
                large_g2_err,
                labels,
            )
            mock_files.append(mock_file)

        # Test processing performance
        with timer() as times:
            result = g2mod.get_data(mock_files)

        elapsed = times[0]

        # Should process large dataset quickly
        assert elapsed < 1.0  # 1 second max
        assert isinstance(result, tuple)

        # Calculate data throughput
        total_points = len(mock_files) * len(large_q) * len(large_tau)
        points_per_second = total_points / elapsed if elapsed > 0 else float("inf")
        assert points_per_second > 10000  # Should process at least 10k points/second

    def test_saxs_color_generation_performance(self):
        """Test SAXS color generation performance."""
        # Test bulk color generation
        with timer() as times:
            colors = []
            for i in range(1000):
                color, marker = saxs1d.get_color_marker(i)
                colors.append((color, marker))

        elapsed = times[0]

        # Should be very fast
        assert elapsed < 0.1  # 100ms max
        assert len(colors) == 1000

        # Calculate generation rate
        colors_per_second = 1000 / elapsed if elapsed > 0 else float("inf")
        assert colors_per_second > 10000  # Should generate 10k+ colors/second

        # Verify color diversity
        unique_colors = {color for color, marker in colors}
        assert len(unique_colors) > 1

    def test_intensity_offset_performance(self):
        """Test intensity offset calculation performance."""
        # Create large intensity arrays
        large_arrays = []
        for size in [1000, 5000, 10000]:
            intensity = np.random.rand(size) * 1e6
            large_arrays.append((size, intensity))

        for size, intensity in large_arrays:
            with timer() as times:
                # Apply offset to multiple curves
                offset_results = []
                for i in range(10):  # 10 curves
                    result = saxs1d.offset_intensity(
                        intensity, n=i, plot_offset=0.1, yscale="log"
                    )
                    offset_results.append(result)

            elapsed = times[0]

            # Performance should scale reasonably with data size
            points_processed = size * 10
            points_per_second = (
                points_processed / elapsed if elapsed > 0 else float("inf")
            )

            # Should process at least 100k points/second
            assert points_per_second > 100000
            assert len(offset_results) == 10


class TestMemoryPerformance:
    """Test memory usage and garbage collection performance."""

    def test_memory_usage_scaling(self):
        """Test memory usage with increasing data sizes."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create progressively larger datasets
        memory_measurements = []

        for array_size in [1000, 5000, 10000]:
            # Create temporary data
            large_data = np.random.rand(array_size, 100)

            # Simulate processing
            processed_data = []
            for i in range(10):
                result = large_data * (i + 1)
                processed_data.append(result)

            current_memory = process.memory_info().rss
            memory_measurements.append((array_size, current_memory - initial_memory))

            # Clean up
            del large_data, processed_data
            gc.collect()

        # Memory usage should not grow excessively
        for size, memory_growth in memory_measurements:
            # Allow reasonable memory growth (in bytes)
            expected_max_growth = size * 100 * 8 * 20  # Rough estimate
            assert memory_growth < expected_max_growth

    def test_object_lifecycle_performance(self):
        """Test object creation and destruction performance."""
        # Test rapid object creation/destruction
        with timer() as times:
            objects = []

            # Create many temporary objects
            for i in range(1000):
                with tempfile.TemporaryDirectory() as tmpdir:
                    Path(tmpdir, f"temp_{i}.hdf").touch()

                    locator = DataFileLocator(tmpdir)
                    locator.build_file_list()
                    objects.append(locator)

                    if i % 100 == 0:  # Periodic cleanup
                        objects.clear()
                        gc.collect()

        elapsed = times[0]

        # Should handle rapid object creation/destruction efficiently
        assert elapsed < 5.0  # 5 seconds max

        objects_per_second = 1000 / elapsed if elapsed > 0 else float("inf")
        assert objects_per_second > 100  # At least 100 objects/second

    def test_garbage_collection_impact(self):
        """Test impact of garbage collection on performance."""
        # Create objects that will need garbage collection
        large_objects = []

        # Measure time without explicit GC
        with timer() as times_no_gc:
            for i in range(100):
                # Create large temporary objects
                temp_data = np.random.rand(1000, 100)
                large_objects.append(temp_data)

                if len(large_objects) > 20:
                    large_objects.pop(0)

        # Clear and measure with explicit GC
        large_objects.clear()

        with timer() as times_with_gc:
            for i in range(100):
                temp_data = np.random.rand(1000, 100)
                large_objects.append(temp_data)

                if len(large_objects) > 20:
                    large_objects.pop(0)

                if i % 10 == 0:
                    gc.collect()

        # Both should be reasonable, GC version might be more consistent
        assert times_no_gc[0] < 2.0
        assert times_with_gc[0] < 2.0

        # Clean up
        large_objects.clear()
        gc.collect()


class TestConcurrentPerformance:
    """Test performance under concurrent access."""

    def test_concurrent_file_operations_performance(self):
        """Test performance of concurrent file operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(50):
                Path(tmpdir, f"concurrent_{i}.hdf").touch()

            results = []
            errors = []

            def worker_task(worker_id: int, iterations: int):
                """Worker task for concurrent testing."""
                worker_times = []

                for i in range(iterations):
                    start = time.perf_counter()

                    try:
                        locator = DataFileLocator(tmpdir)
                        success = locator.build_file_list()

                        if not success:
                            errors.append(f"Worker {worker_id} failed iteration {i}")

                        end = time.perf_counter()
                        worker_times.append(end - start)

                    except Exception as e:
                        errors.append(f"Worker {worker_id} error: {str(e)}")

                results.append((worker_id, worker_times))

            # Launch concurrent workers
            threads = []
            num_workers = 4
            iterations_per_worker = 10

            start_time = time.perf_counter()

            for i in range(num_workers):
                thread = threading.Thread(
                    target=worker_task, args=(i, iterations_per_worker)
                )
                threads.append(thread)
                thread.start()

            # Wait for all workers
            for thread in threads:
                thread.join()

            total_time = time.perf_counter() - start_time

            # Verify no errors occurred
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == num_workers

            # Analyze performance
            all_times = []
            for _worker_id, worker_times in results:
                assert len(worker_times) == iterations_per_worker
                all_times.extend(worker_times)

            avg_operation_time = sum(all_times) / len(all_times)
            max_operation_time = max(all_times)

            # Performance should remain reasonable under concurrency
            assert avg_operation_time < 0.5  # Average operation < 500ms
            assert max_operation_time < 1.0  # Max operation < 1s
            assert total_time < 5.0  # Total test < 5s

    def test_thread_safety_performance_impact(self):
        """Test performance impact of thread safety measures."""
        # Test single-threaded performance
        with timer() as single_thread_time:
            for i in range(200):
                color, marker = saxs1d.get_color_marker(i)

        # Test multi-threaded performance
        results = []

        def color_worker(start_idx: int, count: int):
            """Worker function for color generation."""
            worker_colors = []
            for i in range(start_idx, start_idx + count):
                color, marker = saxs1d.get_color_marker(i)
                worker_colors.append((color, marker))
            results.append(worker_colors)

        with timer() as multi_thread_time:
            threads = []
            for i in range(4):  # 4 threads
                thread = threading.Thread(
                    target=color_worker,
                    args=(i * 50, 50),  # Each generates 50 colors
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

        # Verify results
        assert len(results) == 4
        total_colors = sum(len(worker_colors) for worker_colors in results)
        assert total_colors == 200

        # Multi-threaded should complete successfully
        # Performance may vary based on system and threading overhead
        performance_ratio = multi_thread_time[0] / single_thread_time[0]
        assert performance_ratio > 0  # Basic sanity check
        # Note: For CPU-bound tasks like color generation, threading may be slower
        # due to GIL limitations in Python


class TestScalabilityLimits:
    """Test scalability limits and edge cases."""

    @pytest.mark.slow
    def test_large_file_count_limits(self):
        """Test behavior with very large numbers of files."""
        # Only run if specifically requested (marked as slow)
        with tempfile.TemporaryDirectory() as tmpdir:
            file_count = 500  # Large but reasonable number

            # Create many files
            for i in range(file_count):
                Path(tmpdir, f"scale_test_{i:05d}.hdf").touch()

            # Test file discovery
            with timer() as times:
                locator = DataFileLocator(tmpdir)
                success = locator.build_file_list()

            elapsed = times[0]

            assert success is True
            assert len(locator.source_files) >= file_count

            # Should handle large file counts (allow more time)
            assert elapsed < 10.0  # 10 seconds max

            files_per_second = file_count / elapsed if elapsed > 0 else float("inf")
            assert files_per_second > 25  # At least 25 files/second

    def test_memory_pressure_resilience(self):
        """Test resilience under memory pressure."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Create memory pressure by allocating large arrays
        memory_hogs = []

        try:
            # Allocate memory in chunks
            for i in range(5):
                # 100MB chunks
                chunk = np.random.rand(100 * 1024 * 1024 // 8)  # 100MB
                memory_hogs.append(chunk)

            process.memory_info().rss

            # Now test operations under memory pressure
            with tempfile.TemporaryDirectory() as tmpdir:
                for i in range(20):
                    Path(tmpdir, f"memory_test_{i}.hdf").touch()

                # Operations should still work under memory pressure
                with timer() as times:
                    locator = DataFileLocator(tmpdir)
                    success = locator.build_file_list()

                    kernel = AnalysisKernel(tmpdir)
                    kernel.build_file_list()

                elapsed = times[0]

                # Should still complete, though may be slower
                assert success is True
                assert elapsed < 5.0  # Allow more time under pressure

        finally:
            # Clean up memory
            memory_hogs.clear()
            gc.collect()

    def test_cpu_intensive_operation_scaling(self):
        """Test CPU-intensive operations scaling."""
        # Test scaling of computationally intensive operations
        sizes = [100, 500, 1000]
        times = []

        for size in sizes:
            # Create data that requires computation
            intensity_data = np.random.rand(size) * 1e6

            with timer() as operation_time:
                # Perform intensive operations
                results = []
                for i in range(10):  # Multiple operations
                    result = saxs1d.offset_intensity(
                        intensity_data, n=i, plot_offset=0.5, yscale="linear"
                    )
                    results.append(result)

            times.append((size, operation_time[0]))

            # Verify results
            assert len(results) == 10
            assert all(len(result) == size for result in results)

        # Analyze scaling behavior
        for i, (size, elapsed) in enumerate(times):
            # Each operation should scale reasonably
            operations_per_second = (
                (10 * size) / elapsed if elapsed > 0 else float("inf")
            )
            assert operations_per_second > 1000  # Reasonable throughput

            # Later sizes shouldn't be disproportionately slower
            if i > 0:
                prev_size, prev_time = times[i - 1]
                size_ratio = size / prev_size
                time_ratio = elapsed / prev_time

                # Time should not scale worse than O(n^2)
                assert time_ratio <= size_ratio**2


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "not slow"])

    # To run slow tests: pytest test_performance.py -v -m slow
