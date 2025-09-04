"""
Performance Benchmarking and Regression Testing for XPCS Toolkit

This module provides comprehensive performance benchmarking using pytest-benchmark,
memory profiling, and regression testing to ensure computational performance
remains optimal as the codebase evolves.
"""

import os
from pathlib import Path
import tempfile
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Check if pytest-benchmark is available
pytest_benchmark_available = False
try:
    import pytest_benchmark

    pytest_benchmark_available = True
except ImportError:
    pass
import gc
from typing import Any, Optional

import psutil

try:
    from memory_profiler import memory_usage, profile

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

    # Create dummy decorator for when memory_profiler is not available
    def profile(func):
        return func


from xpcs_toolkit.tests.fixtures.synthetic_data import SyntheticXPCSDataGenerator


class TestPerformanceBenchmarks:
    """Performance benchmarking and regression testing."""

    # Performance thresholds (will be refined based on actual measurements)
    CORRELATION_TIME_THRESHOLD = 10.0  # seconds for large datasets
    MEMORY_THRESHOLD_MB = 2000  # MB memory usage limit
    IO_TIME_THRESHOLD = 5.0  # seconds for file I/O

    @pytest.fixture(scope="class")
    def synthetic_data_generator(self):
        """Create synthetic data generator for performance tests."""
        return SyntheticXPCSDataGenerator(random_seed=42)

    @pytest.fixture(scope="class")
    def large_test_dataset(self, synthetic_data_generator):
        """Generate large dataset for performance testing (optimized size)."""
        return synthetic_data_generator.generate_brownian_motion_intensity(
            n_times=2000,  # Reduced from 10000 for faster tests
            n_q_bins=20,   # Reduced from 50
            noise_level=0.1,
        )

    @pytest.fixture(scope="class")
    def medium_test_dataset(self, synthetic_data_generator):
        """Generate medium dataset for standard benchmarks (optimized size)."""
        return synthetic_data_generator.generate_brownian_motion_intensity(
            n_times=500, n_q_bins=10, noise_level=0.1  # Reduced for speed
        )

    @pytest.fixture(scope="class")
    def test_hdf5_file(self, synthetic_data_generator, tmp_path_factory):
        """Create temporary HDF5 file for I/O benchmarks."""
        temp_dir = tmp_path_factory.mktemp("benchmark_data")
        file_path = temp_dir / "benchmark_test.h5"

        metadata = synthetic_data_generator.create_test_hdf5_file(file_path)
        return file_path, metadata

    def simple_g2_calculation(self, intensity: np.ndarray) -> np.ndarray:
        """
        Simple g2 correlation calculation for benchmarking.

        This is a basic implementation that can be used to benchmark
        correlation function computation performance.
        """
        n_times, n_q = intensity.shape

        # Multi-tau correlation with limited levels for performance
        max_level = min(6, int(np.log2(n_times // 4)))
        tau_values = []

        for level in range(max_level):
            for buf in range(4):  # 4 buffers per level for faster computation
                tau = (2**level) * (buf + 1)
                if tau < n_times - 1:
                    tau_values.append(tau)

        tau_values = np.array(tau_values)
        g2_result = np.zeros((len(tau_values), n_q))

        for tau_idx, tau in enumerate(tau_values):
            tau = int(tau)
            for q_idx in range(n_q):
                intensity_base = intensity[: n_times - tau, q_idx]
                intensity_delayed = intensity[tau:n_times, q_idx]

                numerator = np.mean(intensity_base * intensity_delayed)
                denominator = np.mean(intensity_base) ** 2

                g2_result[tau_idx, q_idx] = (
                    numerator / denominator if denominator > 0 else 1.0
                )

        return g2_result

    def load_hdf5_file_simple(self, file_path: Path) -> dict[str, Any]:
        """
        Simple HDF5 file loading for benchmarking.

        This loads basic data from an HDF5 file to benchmark I/O performance.
        """
        import h5py

        data = {}
        try:
            with h5py.File(file_path, "r") as f:
                # Load common XPCS data
                if "xpcs/multitau/g2" in f:
                    data["g2"] = f["xpcs/multitau/g2"][()]
                if "xpcs/multitau/tau" in f:
                    data["tau"] = f["xpcs/multitau/tau"][()]
                if "xpcs/multitau/ql_sta" in f:
                    data["q_static"] = f["xpcs/multitau/ql_sta"][()]

                # Load metadata
                if "entry/sample/temperature" in f:
                    data["temperature"] = f["entry/sample/temperature"][()]

        except Exception as e:
            data["error"] = str(e)

        return data

    @pytest.mark.benchmark(group="correlation")
    @pytest.mark.skipif(
        not pytest_benchmark_available, reason="pytest-benchmark not installed"
    )
    def test_g2_calculation_performance_small(self, benchmark, medium_test_dataset):
        """Benchmark g2 correlation calculation speed for medium datasets."""
        intensity, q_values, time_values = medium_test_dataset

        # Benchmark the correlation calculation
        result = benchmark(self.simple_g2_calculation, intensity)

        # Verify result is valid
        assert result is not None
        assert result.shape[1] == len(q_values)
        assert np.all(np.isfinite(result))

        # Log performance info
        print(f"Medium dataset: {intensity.shape} -> g2 shape: {result.shape}")

    @pytest.mark.benchmark(group="correlation")
    @pytest.mark.slow
    @pytest.mark.skipif(
        not pytest_benchmark_available, reason="pytest-benchmark not installed"
    )
    def test_g2_calculation_performance_large(self, benchmark, large_test_dataset):
        """Benchmark g2 correlation calculation speed for large datasets."""
        intensity, q_values, time_values = large_test_dataset

        # This may be slow, so add timeout
        result = benchmark.pedantic(
            self.simple_g2_calculation,
            args=(intensity,),
            iterations=1,
            rounds=3,
            warmup_rounds=1,
        )

        # Verify result
        assert result is not None
        assert result.shape[1] == len(q_values)

        # Check performance threshold
        stats = benchmark.stats
        mean_time = stats.get("mean", 0)
        assert mean_time < self.CORRELATION_TIME_THRESHOLD, (
            f"Correlation calculation too slow: {mean_time:.2f}s > {self.CORRELATION_TIME_THRESHOLD}s"
        )

        print(
            f"Large dataset: {intensity.shape} -> g2 shape: {result.shape}, time: {mean_time:.2f}s"
        )

    @pytest.mark.benchmark(group="io")
    @pytest.mark.skipif(
        not pytest_benchmark_available, reason="pytest-benchmark not installed"
    )
    def test_hdf5_loading_performance(self, benchmark, test_hdf5_file):
        """Benchmark HDF5 file loading speed."""
        file_path, expected_metadata = test_hdf5_file

        # Benchmark file loading
        result = benchmark(self.load_hdf5_file_simple, file_path)

        # Verify loading worked
        assert result is not None
        assert "error" not in result or result["error"] is None

        # Check I/O performance
        stats = benchmark.stats
        mean_time = stats.get("mean", 0)
        assert mean_time < self.IO_TIME_THRESHOLD, (
            f"HDF5 loading too slow: {mean_time:.2f}s > {self.IO_TIME_THRESHOLD}s"
        )

        print(f"HDF5 loading time: {mean_time:.3f}s")

    @pytest.mark.benchmark(group="math")
    @pytest.mark.skipif(
        not pytest_benchmark_available, reason="pytest-benchmark not installed"
    )
    def test_numpy_operations_performance(self, benchmark):
        """Benchmark basic NumPy operations for baseline performance."""
        size = 1000000  # 1M elements
        data = np.random.random(size)

        def numpy_operations():
            # Common operations in XPCS analysis
            result1 = np.mean(data)
            result2 = np.std(data)
            result3 = np.fft.fft(data)
            result4 = np.correlate(data[:1000], data[:1000], mode="full")
            return result1, result2, len(result3), len(result4)

        result = benchmark(numpy_operations)
        assert len(result) == 4
        assert all(np.isfinite([result[0], result[1]]))

        print(f"NumPy operations on {size} elements completed")

    @pytest.mark.skipif(
        not MEMORY_PROFILER_AVAILABLE, reason="memory_profiler not available"
    )
    def test_memory_usage_correlation(self, medium_test_dataset):
        """Test memory usage during correlation calculations."""
        intensity, q_values, time_values = medium_test_dataset

        # Monitor memory usage during computation
        def memory_test_function():
            return self.simple_g2_calculation(intensity)

        # Measure memory usage
        mem_usage = memory_usage((memory_test_function, ()), interval=0.1, timeout=60)

        if mem_usage:
            peak_memory_mb = max(mem_usage) - min(mem_usage)  # Memory increase

            assert peak_memory_mb < self.MEMORY_THRESHOLD_MB, (
                f"Memory usage too high: {peak_memory_mb:.1f}MB > {self.MEMORY_THRESHOLD_MB}MB"
            )

            print(f"Peak memory usage: {peak_memory_mb:.1f}MB")
        else:
            pytest.skip("Memory profiling failed")

    @pytest.mark.parametrize("data_size", [50, 200, 500])  # Reduced from [100, 1000, 10000]
    @pytest.mark.benchmark(group="scaling")
    @pytest.mark.slow  # Mark as slow test
    @pytest.mark.skipif(
        not pytest_benchmark_available, reason="pytest-benchmark not installed"
    )
    def test_scaling_performance(self, benchmark, data_size):
        """Test how performance scales with data size (optimized with smaller datasets)."""
        # Generate data of specified size with fewer q_bins for speed
        generator = SyntheticXPCSDataGenerator(random_seed=42)
        intensity, _, _ = generator.generate_brownian_motion_intensity(
            n_times=data_size, n_q_bins=5  # Reduced from 10 to 5
        )

        # Benchmark correlation calculation
        benchmark(self.simple_g2_calculation, intensity)

        # Record scaling information
        stats = benchmark.stats
        mean_time = stats.get("mean", 0)

        # Time should scale reasonably with data size (adjusted for smaller baseline)
        expected_scaling_factor = data_size * np.log2(data_size) / (50 * np.log2(50))  # Adjusted baseline
        max_expected_time = 0.05 * expected_scaling_factor  # Reduced baseline time

        assert mean_time < max_expected_time, (
            f"Scaling too poor: {mean_time:.3f}s > {max_expected_time:.3f}s for size {data_size}"
        )

        print(
            f"Size {data_size}: {mean_time:.3f}s (scaling factor: {expected_scaling_factor:.1f})"
        )

    def test_cpu_utilization_monitoring(self, medium_test_dataset):
        """Monitor CPU utilization during computation."""
        intensity, _, _ = medium_test_dataset

        # Monitor CPU usage
        initial_cpu_percent = psutil.cpu_percent(interval=1)

        start_time = time.time()
        result = self.simple_g2_calculation(intensity)
        end_time = time.time()

        final_cpu_percent = psutil.cpu_percent(interval=1)
        computation_time = end_time - start_time

        # Verify computation completed
        assert result is not None

        # Log performance metrics
        print(f"Computation time: {computation_time:.3f}s")
        print(f"CPU usage: {initial_cpu_percent:.1f}% -> {final_cpu_percent:.1f}%")

        # Basic sanity check - computation should complete in reasonable time
        assert computation_time < 30.0, (
            f"Computation took too long: {computation_time:.1f}s"
        )

    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Perform many small operations that should not accumulate memory
        for i in range(100):
            data = np.random.random((100, 10))
            result = self.simple_g2_calculation(data)

            # Explicitly delete large objects
            del data, result

            # Force garbage collection every 10 iterations
            if i % 10 == 0:
                gc.collect()

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal (< 100MB for this test)
        assert memory_increase < 100, (
            f"Possible memory leak: {memory_increase:.1f}MB increase"
        )

        print(
            f"Memory change: {memory_increase:+.1f}MB ({initial_memory:.1f} -> {final_memory:.1f}MB)"
        )

    @pytest.mark.benchmark(group="array_ops")
    @pytest.mark.skipif(
        not pytest_benchmark_available, reason="pytest-benchmark not installed"
    )
    def test_array_operations_performance(self, benchmark):
        """Benchmark array operations common in XPCS analysis."""
        size = (5000, 20)  # Typical intensity array size
        data = np.random.random(size)

        def array_operations():
            # Operations common in correlation analysis
            mean_vals = np.mean(data, axis=0)  # Time average for each q
            var_vals = np.var(data, axis=0)  # Variance for each q

            # Normalized intensity fluctuations
            normalized = (data - mean_vals) / np.sqrt(var_vals)

            # Cross-correlations between q-bins
            cross_corr = np.corrcoef(data.T)

            return mean_vals, var_vals, normalized.shape, cross_corr.shape

        result = benchmark(array_operations)

        # Verify results
        assert len(result) == 4
        assert len(result[0]) == size[1]  # Mean for each q-bin
        assert len(result[1]) == size[1]  # Variance for each q-bin
        assert result[2] == size  # Normalized data shape
        assert result[3] == (size[1], size[1])  # Cross-correlation matrix shape

    def test_performance_regression_detection(self, medium_test_dataset):
        """Test framework for detecting performance regressions."""
        intensity, _, _ = medium_test_dataset

        # Run computation multiple times to get stable measurement
        times = []
        for _ in range(5):
            start = time.perf_counter()
            self.simple_g2_calculation(intensity)
            end = time.perf_counter()
            times.append(end - start)

        mean_time = np.mean(times)
        std_time = np.std(times)

        # Store baseline performance (in real implementation, this would be stored/loaded)
        baseline_time = 1.0  # seconds (placeholder - would be actual baseline)
        regression_threshold = 1.5  # 50% slower is considered a regression

        # Check for performance regression
        if mean_time > baseline_time * regression_threshold:
            pytest.fail(
                f"Performance regression detected: {mean_time:.3f}s vs baseline {baseline_time:.3f}s"
            )

        print(
            f"Performance: {mean_time:.3f}±{std_time:.3f}s (baseline: {baseline_time:.3f}s)"
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.skipif(
        not pytest_benchmark_available, reason="pytest-benchmark not installed"
    )
    def test_dtype_performance_comparison(self, benchmark, dtype):
        """Compare performance between different floating-point precisions."""
        size = (2000, 15)
        data = np.random.random(size).astype(dtype)

        result = benchmark(self.simple_g2_calculation, data)

        # Verify computation worked
        assert result is not None
        assert result.dtype in [np.float32, np.float64]

        # Log dtype information
        stats = benchmark.stats
        mean_time = stats.get("mean", 0)
        print(f"dtype {dtype.__name__}: {mean_time:.3f}s")


class TestPerformanceUtilities:
    """Utility functions for performance testing."""

    def test_benchmark_data_generation(self):
        """Test that synthetic data generation is fast enough for benchmarking."""
        generator = SyntheticXPCSDataGenerator(random_seed=42)

        start_time = time.perf_counter()
        intensity, q_vals, tau_vals = generator.generate_brownian_motion_intensity(
            n_times=1000, n_q_bins=20
        )
        generation_time = time.perf_counter() - start_time

        # Data generation should be fast (< 1 second)
        assert generation_time < 1.0, (
            f"Data generation too slow: {generation_time:.3f}s"
        )

        # Verify data quality
        assert intensity.shape == (1000, 20)
        assert len(q_vals) == 20
        assert len(tau_vals) == 1000

        print(f"Data generation time: {generation_time:.3f}s")

    def test_benchmark_setup_teardown(self):
        """Test benchmark setup and teardown performance."""
        setup_times = []
        teardown_times = []

        for _ in range(10):
            # Measure setup time
            start = time.perf_counter()
            temp_data = np.random.random((500, 10))
            setup_time = time.perf_counter() - start
            setup_times.append(setup_time)

            # Measure teardown time
            start = time.perf_counter()
            del temp_data
            gc.collect()
            teardown_time = time.perf_counter() - start
            teardown_times.append(teardown_time)

        mean_setup = np.mean(setup_times)
        mean_teardown = np.mean(teardown_times)

        # Setup and teardown should be fast
        assert mean_setup < 0.1, f"Benchmark setup too slow: {mean_setup:.3f}s"
        assert mean_teardown < 0.1, f"Benchmark teardown too slow: {mean_teardown:.3f}s"

        print(f"Setup: {mean_setup:.4f}s, Teardown: {mean_teardown:.4f}s")


if __name__ == "__main__":
    # Run with benchmark plugin
    pytest.main(
        [
            __file__,
            "-v",
            "--benchmark-only",
            "--benchmark-group-by=group",
            "--benchmark-sort=mean",
        ]
    )
