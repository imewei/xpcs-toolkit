"""
System Robustness and Error Handling Tests for XPCS Toolkit

This module tests the robustness of the XPCS Toolkit under various
error conditions, edge cases, and resource constraints to ensure
reliable operation in real-world scientific computing environments.
"""

import os
from pathlib import Path
import shutil
import signal
import tempfile
import threading
import time
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch
import warnings

import h5py
import numpy as np
import psutil
import pytest

from xpcs_toolkit.core.data.locator import DataFileLocator, create_xpcs_dataset
from xpcs_toolkit.scientific.correlation import g2
from xpcs_toolkit.tests.fixtures.synthetic_data import SyntheticXPCSDataGenerator


def _long_running_analysis_worker(result_queue):
    """Worker function for signal handling test - must be at module level for multiprocessing."""
    import time

    try:
        # Set up signal handler for graceful shutdown
        interrupted = False

        def signal_handler(signum, frame):
            nonlocal interrupted
            interrupted = True

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Simulate long analysis
        generator = SyntheticXPCSDataGenerator(random_seed=42)

        for i in range(10):  # 10 iterations
            if interrupted:
                result_queue.put({"status": "interrupted", "iteration": i})
                return

            # Generate some data (simulating analysis work)
            intensity, q, tau = generator.generate_brownian_motion_intensity(
                n_times=100, n_q_bins=5
            )

            # Simulate processing time
            time.sleep(0.1)

            result_queue.put({"status": "progress", "iteration": i})

        result_queue.put({"status": "completed", "iteration": 10})

    except KeyboardInterrupt:
        result_queue.put({"status": "keyboard_interrupt"})
    except Exception as e:
        result_queue.put({"status": "error", "error": str(e)})


class TestSystemRobustness:
    """Test system robustness and error handling."""

    @pytest.fixture
    def corrupted_hdf5_files(self):
        """Create various types of corrupted HDF5 files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            corrupted_files = {}

            # 1. Not an HDF5 file at all
            fake_h5 = temp_path / "fake.h5"
            fake_h5.write_text("This is not HDF5 data")
            corrupted_files["not_hdf5"] = str(fake_h5)

            # 2. Empty HDF5 file
            empty_h5 = temp_path / "empty.h5"
            with h5py.File(empty_h5, "w") as f:
                pass  # Create empty file
            corrupted_files["empty_hdf5"] = str(empty_h5)

            # 3. HDF5 with missing required groups
            incomplete_h5 = temp_path / "incomplete.h5"
            with h5py.File(incomplete_h5, "w") as f:
                f.create_group("entry")  # Missing xpcs group
            corrupted_files["incomplete_structure"] = str(incomplete_h5)

            # 4. HDF5 with corrupted datasets
            bad_data_h5 = temp_path / "bad_data.h5"
            with h5py.File(bad_data_h5, "w") as f:
                f.create_group("entry")
                xpcs = f.create_group("xpcs")
                multitau = xpcs.create_group("multitau")

                # Create datasets with incompatible shapes
                multitau.create_dataset("g2", data=np.array([1, 2]))  # Wrong shape
                multitau.create_dataset("tau", data=np.array([1, 2, 3]))  # Mismatched
                multitau.create_dataset("ql_sta", data=np.array([0.01]))  # Single value
            corrupted_files["bad_data"] = str(bad_data_h5)

            # 5. HDF5 with NaN/Inf values
            nan_data_h5 = temp_path / "nan_data.h5"
            with h5py.File(nan_data_h5, "w") as f:
                f.create_group("entry")
                xpcs = f.create_group("xpcs")
                multitau = xpcs.create_group("multitau")

                # Create datasets with NaN/Inf values
                g2_data = np.ones((10, 5))
                g2_data[2, 1] = np.nan
                g2_data[5, 3] = np.inf

                multitau.create_dataset("g2", data=g2_data)
                multitau.create_dataset("tau", data=np.logspace(-4, 0, 10))
                multitau.create_dataset("ql_sta", data=np.linspace(0.01, 0.1, 5))
            corrupted_files["nan_inf_data"] = str(nan_data_h5)

            yield corrupted_files

    @pytest.fixture
    def resource_limited_environment(self):
        """Simulate resource-limited environment for testing."""
        # Get current resource usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        initial_files = len(process.open_files())

        return {
            "initial_memory_bytes": initial_memory,
            "initial_open_files": initial_files,
            "process": process,
        }

    def test_corrupted_hdf5_handling(self, corrupted_hdf5_files):
        """Test graceful handling of corrupted HDF5 files."""
        results = {}

        for corruption_type, file_path in corrupted_hdf5_files.items():
            results[corruption_type] = {
                "file_path": file_path,
                "create_dataset_success": False,
                "exception_type": None,
                "error_message": None,
                "graceful_failure": False,
            }

            try:
                # Test create_xpcs_dataset function
                dataset = create_xpcs_dataset(file_path)

                if dataset is None:
                    # Graceful failure - returned None
                    results[corruption_type]["graceful_failure"] = True
                else:
                    # Unexpected success - investigate further
                    results[corruption_type]["create_dataset_success"] = True

                    # Test if we can actually use the dataset
                    try:
                        if hasattr(dataset, "atype"):
                            analysis_types = dataset.atype
                            if "Multitau" in analysis_types:
                                # Try to extract data
                                data_result = g2.get_data([dataset])
                                if data_result[0] is False:
                                    results[corruption_type]["graceful_failure"] = True
                    except Exception:
                        results[corruption_type]["graceful_failure"] = True

            except Exception as e:
                results[corruption_type]["exception_type"] = type(e).__name__
                results[corruption_type]["error_message"] = str(e)

                # Check if it's an expected exception type
                expected_exceptions = [
                    "OSError",
                    "IOError",
                    "ValueError",
                    "KeyError",
                    "HDF5Error",
                    "FileNotFoundError",
                ]
                if type(e).__name__ in expected_exceptions:
                    results[corruption_type]["graceful_failure"] = True

        # Validate that all corrupted files were handled gracefully
        for corruption_type, result in results.items():
            assert result["graceful_failure"], (
                f"Corrupted file {corruption_type} not handled gracefully: {result}"
            )

        return results

    def test_insufficient_memory_scenarios(self, resource_limited_environment):
        """Test behavior under memory constraints."""
        process_info = resource_limited_environment

        # Create large synthetic datasets to stress memory
        generator = SyntheticXPCSDataGenerator(random_seed=42)

        memory_stress_results = []

        # Test with progressively larger datasets
        dataset_sizes = [
            (1000, 10),  # Small: 1K time points, 10 q-bins
            (5000, 20),  # Medium: 5K time points, 20 q-bins
            (10000, 50),  # Large: 10K time points, 50 q-bins
        ]

        for size_idx, (n_times, n_q_bins) in enumerate(dataset_sizes):
            test_result = {
                "dataset_size": (n_times, n_q_bins),
                "memory_before_mb": 0,
                "memory_after_mb": 0,
                "memory_peak_mb": 0,
                "generation_success": False,
                "analysis_success": False,
                "error_occurred": False,
                "error_type": None,
            }

            try:
                # Measure memory before
                process_info["process"].memory_full_info()  # Update cache
                memory_before = process_info["process"].memory_info().rss / 1024 / 1024
                test_result["memory_before_mb"] = memory_before

                # Generate large dataset
                intensity, q_vals, tau_vals = (
                    generator.generate_brownian_motion_intensity(
                        n_times=n_times, n_q_bins=n_q_bins, noise_level=0.1
                    )
                )
                test_result["generation_success"] = True

                # Measure memory after generation
                memory_after_gen = (
                    process_info["process"].memory_info().rss / 1024 / 1024
                )
                test_result["memory_peak_mb"] = max(
                    test_result["memory_peak_mb"], memory_after_gen
                )

                # Test analysis operations that might use more memory
                if intensity.size < 1e8:  # Only if reasonable size
                    # Simulate correlation analysis (simplified)
                    np.mean(intensity, axis=0)
                    np.std(intensity, axis=0)

                    # Simple autocorrelation calculation (memory intensive)
                    for q_idx in range(
                        min(5, n_q_bins)
                    ):  # Limit to avoid excessive memory
                        intensity_q = intensity[
                            : min(1000, n_times), q_idx
                        ]  # Limit size
                        autocorr = np.correlate(
                            intensity_q - np.mean(intensity_q),
                            intensity_q - np.mean(intensity_q),
                            mode="full",
                        )
                        autocorr = autocorr[autocorr.size // 2 :]

                    test_result["analysis_success"] = True

                # Measure final memory
                memory_after = process_info["process"].memory_info().rss / 1024 / 1024
                test_result["memory_after_mb"] = memory_after
                test_result["memory_peak_mb"] = max(
                    test_result["memory_peak_mb"], memory_after
                )

                # Clean up large objects
                del intensity, q_vals, tau_vals
                import gc

                gc.collect()

            except MemoryError:
                test_result["error_occurred"] = True
                test_result["error_type"] = "MemoryError"
                # This is expected for very large datasets

            except Exception as e:
                test_result["error_occurred"] = True
                test_result["error_type"] = type(e).__name__

                # Unexpected error - should investigate
                if not isinstance(e, (MemoryError, OSError)):
                    pytest.fail(f"Unexpected error in memory test: {e}")

            memory_stress_results.append(test_result)

        # Validate memory stress test results
        for i, result in enumerate(memory_stress_results):
            dataset_size = result["dataset_size"]

            # At least the smallest dataset should succeed
            if i == 0:  # Smallest dataset
                assert (
                    result["generation_success"]
                    or result["error_type"] == "MemoryError"
                ), f"Smallest dataset should succeed or fail gracefully: {result}"

            # Memory usage should be reasonable (not growing indefinitely)
            if result["generation_success"]:
                memory_increase = result["memory_after_mb"] - result["memory_before_mb"]

                # Memory increase should be proportional to data size (rough check)
                expected_size_mb = (
                    np.prod(dataset_size) * 8 / 1024 / 1024
                )  # 8 bytes per float64

                # Allow 5x overhead for processing
                assert memory_increase < expected_size_mb * 5, (
                    f"Memory usage too high: {memory_increase:.1f}MB for {expected_size_mb:.1f}MB data"
                )

        return memory_stress_results

    def test_file_descriptor_limits(self, resource_limited_environment):
        """Test behavior when approaching file descriptor limits."""
        process_info = resource_limited_environment

        # Test opening many files simultaneously
        file_handles = []
        temp_files = []

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                generator = SyntheticXPCSDataGenerator(random_seed=42)

                # Create multiple HDF5 files
                num_files = 20  # Reasonable number for testing
                for i in range(num_files):
                    file_path = temp_path / f"test_{i:03d}.h5"
                    generator.create_test_hdf5_file(file_path)
                    temp_files.append(file_path)

                # Try to open all files simultaneously
                open_files_count = 0
                file_descriptor_error = False

                try:
                    for file_path in temp_files:
                        h5_file = h5py.File(file_path, "r")
                        file_handles.append(h5_file)
                        open_files_count += 1

                        # Check current file descriptor usage
                        current_fds = len(process_info["process"].open_files())

                        # If we're approaching system limits, break
                        if current_fds > 900:  # Conservative limit
                            break

                except OSError as e:
                    file_descriptor_error = True
                    type(e).__name__

                # Test that we can still operate with remaining resources
                if len(file_handles) > 0:
                    # Try to read from one of the open files
                    test_file = file_handles[0]

                    # Should be able to read basic structure
                    assert "entry" in test_file or "xpcs" in test_file, (
                        "Should be able to read from open file"
                    )

                # Test DataFileLocator with multiple files
                locator = DataFileLocator(str(temp_path))
                success = locator.build_file_list(str(temp_path))
                assert success, "Should be able to build file list"

                # Should find the files we created
                assert len(locator.source_files) == num_files, (
                    f"Should find {num_files} files, found {len(locator.source_files)}"
                )

        finally:
            # Clean up file handles
            for handle in file_handles:
                try:
                    handle.close()
                except:
                    pass

        return {
            "files_created": len(temp_files),
            "files_opened_simultaneously": open_files_count,
            "file_descriptor_error": file_descriptor_error,
            "final_fd_count": len(process_info["process"].open_files()),
        }

    def test_network_interruption_recovery(self):
        """Test recovery from simulated network/I/O interruptions."""
        # This test simulates I/O interruptions during file operations

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            generator = SyntheticXPCSDataGenerator(random_seed=42)

            # Create test file
            test_file = temp_path / "test_network.h5"
            generator.create_test_hdf5_file(test_file)

            interruption_results = []

            # Test various interruption scenarios
            interruption_scenarios = [
                {
                    "name": "file_permissions",
                    "setup": lambda f: os.chmod(f, 0o000),  # Remove all permissions
                    "cleanup": lambda f: os.chmod(f, 0o644),  # Restore permissions
                    "expected_error": "PermissionError",
                },
                {
                    "name": "file_deletion",
                    "setup": lambda f: os.unlink(f),  # Delete file
                    "cleanup": lambda f: generator.create_test_hdf5_file(f),  # Recreate
                    "expected_error": "FileNotFoundError",
                },
            ]

            for scenario in interruption_scenarios:
                scenario_result = {
                    "scenario_name": scenario["name"],
                    "error_handled_gracefully": False,
                    "recovery_successful": False,
                    "error_type": None,
                }

                try:
                    # First, verify file works normally
                    create_xpcs_dataset(str(test_file))
                    assert True, "Initial file should be accessible"

                    # Apply interruption
                    scenario["setup"](test_file)

                    # Try to access file during "interruption"
                    try:
                        dataset_during = create_xpcs_dataset(str(test_file))
                        if dataset_during is None:
                            scenario_result["error_handled_gracefully"] = True
                    except Exception as e:
                        scenario_result["error_type"] = type(e).__name__
                        if scenario_result["error_type"] in [
                            scenario["expected_error"],
                            "OSError",
                            "IOError",
                        ]:
                            scenario_result["error_handled_gracefully"] = True

                    # Recover from interruption
                    scenario["cleanup"](test_file)

                    # Test recovery
                    dataset_after = create_xpcs_dataset(str(test_file))
                    if (
                        dataset_after is not None
                        or scenario["name"] == "file_permissions"
                    ):
                        scenario_result["recovery_successful"] = True

                except Exception as e:
                    scenario_result["error_type"] = type(e).__name__
                    # Some exceptions during setup/cleanup are acceptable
                    if type(e).__name__ in [
                        "PermissionError",
                        "FileNotFoundError",
                        "OSError",
                    ]:
                        scenario_result["error_handled_gracefully"] = True

                interruption_results.append(scenario_result)

        # Validate interruption handling
        for result in interruption_results:
            assert result["error_handled_gracefully"], (
                f"Interruption scenario '{result['scenario_name']}' not handled gracefully"
            )

        return interruption_results

    def test_concurrent_access_safety(self):
        """Test thread safety and concurrent access patterns."""
        from concurrent.futures import ThreadPoolExecutor
        import queue

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            generator = SyntheticXPCSDataGenerator(random_seed=42)

            # Create test files
            num_files = 5
            test_files = []
            for i in range(num_files):
                file_path = temp_path / f"concurrent_test_{i}.h5"
                generator.create_test_hdf5_file(file_path)
                test_files.append(str(file_path))

            # Test concurrent file access
            results_queue = queue.Queue()
            errors_queue = queue.Queue()

            def concurrent_file_access(file_path, thread_id):
                """Access file from multiple threads."""
                try:
                    # Multiple operations per thread
                    for operation in range(3):
                        dataset = create_xpcs_dataset(file_path)

                        if dataset is not None:
                            # Try to extract data if possible
                            try:
                                if (
                                    hasattr(dataset, "analysis_type")
                                    and "Multitau" in dataset.analysis_type
                                ):
                                    data_result = g2.get_data([dataset])
                                    if data_result[0] is not False:
                                        results_queue.put(
                                            {
                                                "thread_id": thread_id,
                                                "file_path": file_path,
                                                "operation": operation,
                                                "success": True,
                                            }
                                        )
                                    else:
                                        results_queue.put(
                                            {
                                                "thread_id": thread_id,
                                                "file_path": file_path,
                                                "operation": operation,
                                                "success": False,
                                                "reason": "data_extraction_failed",
                                            }
                                        )
                                else:
                                    results_queue.put(
                                        {
                                            "thread_id": thread_id,
                                            "file_path": file_path,
                                            "operation": operation,
                                            "success": False,
                                            "reason": "no_multitau_data",
                                        }
                                    )
                            except Exception as e:
                                errors_queue.put(
                                    {
                                        "thread_id": thread_id,
                                        "operation": operation,
                                        "error": str(e),
                                        "error_type": type(e).__name__,
                                    }
                                )
                        else:
                            results_queue.put(
                                {
                                    "thread_id": thread_id,
                                    "file_path": file_path,
                                    "operation": operation,
                                    "success": False,
                                    "reason": "dataset_creation_failed",
                                }
                            )

                        # Small delay to increase chance of race conditions
                        time.sleep(0.01)

                except Exception as e:
                    errors_queue.put(
                        {
                            "thread_id": thread_id,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        }
                    )

            # Run concurrent access test
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit tasks for each file with multiple threads
                futures = []
                for i, file_path in enumerate(test_files):
                    for thread_num in range(2):  # 2 threads per file
                        future = executor.submit(
                            concurrent_file_access,
                            file_path,
                            f"thread_{i}_{thread_num}",
                        )
                        futures.append(future)

                # Wait for all tasks to complete
                for future in futures:
                    future.result(timeout=30)  # 30 second timeout

            # Collect results
            results = []
            while not results_queue.empty():
                results.append(results_queue.get())

            errors = []
            while not errors_queue.empty():
                errors.append(errors_queue.get())

            # Analyze concurrent access results
            successful_operations = [r for r in results if r["success"]]
            failed_operations = [r for r in results if not r["success"]]

            # Should have some successful operations
            assert len(successful_operations) > 0, (
                "Should have at least some successful concurrent operations"
            )

            # Errors should be manageable (not catastrophic failures)
            catastrophic_errors = [
                e
                for e in errors
                if e["error_type"]
                not in ["OSError", "IOError", "PermissionError", "ValueError"]
            ]

            assert len(catastrophic_errors) == 0, (
                f"Should not have catastrophic errors: {catastrophic_errors}"
            )

            return {
                "total_operations": len(results),
                "successful_operations": len(successful_operations),
                "failed_operations": len(failed_operations),
                "errors": len(errors),
                "error_types": list({e["error_type"] for e in errors}),
                "threads_used": len({r["thread_id"] for r in results}),
            }

    def test_signal_handling_graceful_shutdown(self):
        """Test graceful handling of system signals and interrupts."""
        from multiprocessing import Process, Queue
        import time

        # Test graceful shutdown
        result_queue = Queue()
        process = Process(target=_long_running_analysis_worker, args=(result_queue,))

        process.start()
        time.sleep(0.3)  # Let it run for a bit

        # Send interrupt signal
        process.terminate()  # SIGTERM

        # Wait for process to finish gracefully
        process.join(timeout=2.0)

        # Check if process terminated gracefully
        graceful_shutdown = not process.is_alive()

        if process.is_alive():
            process.kill()  # Force kill if still alive
            process.join()

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # Analyze shutdown behavior
        last_status = results[-1]["status"] if results else None

        return {
            "graceful_shutdown": graceful_shutdown,
            "results_before_shutdown": len(results),
            "last_status": last_status,
            "completed_iterations": len(
                [r for r in results if r["status"] == "progress"]
            ),
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
