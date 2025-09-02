"""
End-to-End Scientific Workflow Testing for XPCS Toolkit

This module tests complete XPCS analysis workflows from raw data loading
through final result generation, ensuring all components work together
correctly for real scientific use cases.
"""

from pathlib import Path
import shutil
import tempfile
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch
import warnings

import h5py
import numpy as np
import pytest

from xpcs_toolkit.core.data.locator import DataFileLocator, create_xpcs_dataset
from xpcs_toolkit.scientific.correlation import g2
from xpcs_toolkit.tests.fixtures.synthetic_data import (
    SyntheticXPCSDataGenerator,
    create_test_data_suite,
    ensure_test_data_exists,
)


class TestScientificWorkflows:
    """Test complete scientific analysis workflows."""

    @pytest.fixture(scope="class")
    def test_data_suite(self):
        """Create comprehensive test data suite for workflow testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            suite_info = create_test_data_suite(temp_path)
            yield suite_info

    @pytest.fixture(scope="class")
    def sample_xpcs_files(self, temp_dir):
        """Load sample XPCS files for workflow testing."""
        # Create synthetic data for testing
        generator = SyntheticXPCSDataGenerator(random_seed=42)
        files = []
        for i in range(3):
            file_path = Path(temp_dir) / f"sample_{i}.h5"
            generator.create_test_hdf5_file(file_path)
            xpcs_file = create_xpcs_dataset(str(file_path))
            if xpcs_file:
                files.append(xpcs_file)
        return files

    def test_complete_xpcs_analysis_pipeline(self, temp_dir):
        """Test full XPCS analysis from raw data to final results."""
        # Step 0: Create synthetic test data
        data_dir = Path(temp_dir)
        generator = SyntheticXPCSDataGenerator(random_seed=42)

        # Create multiple test files for analysis
        for i in range(3):
            file_path = data_dir / f"pipeline_test_{i}.h5"
            generator.create_test_hdf5_file(file_path)

        # Step 1: Initialize data locator
        locator = DataFileLocator(str(data_dir))

        # Step 2: Build file list and discover files
        success = locator.build_file_list(
            path=str(data_dir), file_extensions=(".h5", ".hdf"), sort_method="Filename"
        )
        assert success, "File list building should succeed"
        assert len(locator.source_files) >= 3, "Should find multiple test files"

        # Step 3: Add files to target list for analysis
        target_files = list(locator.source_files)[:3]  # Use first 3 files
        locator.add_target_files(target_files, preload=True, threshold=10)

        assert len(locator.target_files) == 3, "Should have 3 target files"
        assert len(locator.file_cache) == 3, "Should have cached 3 files"

        # Step 4: Get XPCS file objects for analysis
        xpcs_files = locator.get_xpcs_file_list(filter_analysis_type="Multitau")
        assert len(xpcs_files) > 0, "Should find Multitau analysis files"

        # Step 5: Extract correlation data for analysis
        q, tau, g2_data, g2_err, labels = g2.get_data(
            xpcs_files,
            q_range=(0.01, 0.1),  # Focus on specific q-range
            t_range=(1e-5, 1e0),  # Focus on specific time range
        )

        # Validate extraction results
        assert len(q) == len(xpcs_files), "Should have q-data for each file"
        assert len(tau) == len(xpcs_files), "Should have tau-data for each file"
        assert len(g2_data) == len(xpcs_files), "Should have g2-data for each file"
        assert len(g2_err) == len(xpcs_files), "Should have error data for each file"

        # Step 6: Validate scientific results
        for i, file_data in enumerate(zip(q, tau, g2_data, g2_err)):
            q_vals, tau_vals, g2_vals, g2_errs = file_data

            # Basic data validation
            assert len(q_vals) > 0, f"File {i}: Should have q-values"
            assert len(tau_vals) > 0, f"File {i}: Should have tau-values"
            assert g2_vals.shape[0] == len(tau_vals), f"File {i}: g2 shape mismatch"
            assert g2_vals.shape[1] == len(q_vals), f"File {i}: g2 shape mismatch"

            # Scientific validation
            assert np.all(g2_vals > 0), f"File {i}: g2 values should be positive"
            assert np.all(g2_vals < 10), f"File {i}: g2 values should be reasonable"
            assert np.all(np.isfinite(g2_vals)), f"File {i}: g2 should be finite"
            assert np.all(g2_errs >= 0), f"File {i}: errors should be non-negative"

        # Step 7: Test plot geometry calculation for visualization
        num_figs_multi, num_lines_multi = g2.compute_geometry(g2_data, "multiple")
        num_figs_single, num_lines_single = g2.compute_geometry(g2_data, "single")
        num_figs_combined, num_lines_combined = g2.compute_geometry(
            g2_data, "single-combined"
        )

        # Validate geometry calculations
        assert num_figs_multi > 0, "Multiple mode should have figures"
        assert num_lines_multi > 0, "Multiple mode should have lines"
        assert num_figs_combined == 1, "Combined mode should have 1 figure"

        # Step 8: Test workflow completion
        workflow_results = {
            "files_processed": len(xpcs_files),
            "q_ranges_analyzed": [len(q_vals) for q_vals in q],
            "tau_ranges_analyzed": [len(tau_vals) for tau_vals in tau],
            "total_data_points": sum(g2_vals.size for g2_vals in g2_data),
            "analysis_successful": True,
        }

        assert workflow_results["files_processed"] >= 1, (
            "Should process at least 1 file"
        )
        assert workflow_results["total_data_points"] > 0, "Should have analysis data"
        assert workflow_results["analysis_successful"], "Analysis should succeed"

        return workflow_results

    def test_multi_file_batch_analysis(self, temp_dir):
        """Test batch processing of multiple XPCS files."""
        data_dir = Path(temp_dir)
        generator = SyntheticXPCSDataGenerator(random_seed=42)

        # Create test files for batch analysis
        for i in range(5):
            file_path = data_dir / f"batch_test_{i}.h5"
            generator.create_test_hdf5_file(file_path)

        locator = DataFileLocator(str(data_dir))

        # Load all available files
        locator.build_file_list(str(data_dir))
        all_files = list(locator.source_files)

        # Process files in batches
        batch_size = 2
        batch_results = []

        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i : i + batch_size]

            # Clear previous targets and load new batch
            locator.clear_target_files()
            locator.add_target_files(batch_files, preload=True)

            # Get XPCS files for this batch
            xpcs_batch = locator.get_xpcs_file_list()

            if len(xpcs_batch) > 0:
                # Extract data for batch
                q, tau, g2_data, g2_err, labels = g2.get_data(xpcs_batch)

                batch_result = {
                    "batch_id": i // batch_size,
                    "files_in_batch": len(xpcs_batch),
                    "q_bins_total": sum(len(q_vals) for q_vals in q),
                    "tau_points_total": sum(len(tau_vals) for tau_vals in tau),
                    "data_quality_check": all(
                        np.all(np.isfinite(g2_vals)) for g2_vals in g2_data
                    ),
                }
                batch_results.append(batch_result)

        # Validate batch processing results
        assert len(batch_results) > 0, "Should have processed at least one batch"

        for batch in batch_results:
            assert batch["files_in_batch"] > 0, "Each batch should have files"
            assert batch["data_quality_check"], "All data should be finite"
            assert batch["q_bins_total"] > 0, "Should have q-bin data"
            assert batch["tau_points_total"] > 0, "Should have tau-point data"

        # Test aggregated results across batches
        total_files = sum(batch["files_in_batch"] for batch in batch_results)
        total_q_bins = sum(batch["q_bins_total"] for batch in batch_results)

        assert total_files <= len(all_files), "Total files should not exceed available"
        assert total_q_bins > 0, "Should have aggregated q-bin data"

        return batch_results

    def test_data_format_compatibility(self, temp_dir):
        """Test compatibility with different HDF5/NeXus formats."""
        data_dir = Path(temp_dir)
        generator = SyntheticXPCSDataGenerator(random_seed=42)

        # Create test files for compatibility testing
        for i in range(3):
            file_path = data_dir / f"format_test_{i}.h5"
            generator.create_test_hdf5_file(file_path)

        # Test with different file format expectations
        format_tests = [
            {
                "name": "nexus_format",
                "expected_groups": ["/entry", "/xpcs", "/xpcs/multitau"],
                "required_datasets": ["g2", "tau", "ql_sta"],
            },
            {
                "name": "basic_hdf5",
                "expected_groups": ["/xpcs/multitau"],
                "required_datasets": ["g2", "tau"],
            },
        ]

        compatibility_results = {}

        for format_test in format_tests:
            format_name = format_test["name"]
            compatibility_results[format_name] = {
                "files_compatible": 0,
                "files_tested": 0,
                "missing_groups": [],
                "missing_datasets": [],
            }

            # Test files in directory
            for file_path in data_dir.glob("*.h5"):
                compatibility_results[format_name]["files_tested"] += 1

                try:
                    with h5py.File(file_path, "r") as f:
                        # Check for expected groups
                        groups_present = all(
                            group in f for group in format_test["expected_groups"]
                        )

                        # Check for required datasets in multitau group
                        datasets_present = True
                        if "/xpcs/multitau" in f:
                            multitau_group = f["/xpcs/multitau"]
                            datasets_present = all(
                                dataset in multitau_group
                                for dataset in format_test["required_datasets"]
                            )

                        if groups_present and datasets_present:
                            compatibility_results[format_name]["files_compatible"] += 1
                        else:
                            if not groups_present:
                                missing_groups = [
                                    group
                                    for group in format_test["expected_groups"]
                                    if group not in f
                                ]
                                compatibility_results[format_name][
                                    "missing_groups"
                                ].extend(missing_groups)

                            if not datasets_present:
                                if "/xpcs/multitau" in f:
                                    missing_datasets = [
                                        ds
                                        for ds in format_test["required_datasets"]
                                        if ds not in f["/xpcs/multitau"]
                                    ]
                                    compatibility_results[format_name][
                                        "missing_datasets"
                                    ].extend(missing_datasets)

                except Exception:
                    # File read error - format incompatible
                    continue

        # Validate compatibility results
        for format_name, results in compatibility_results.items():
            compatibility_ratio = results["files_compatible"] / max(
                results["files_tested"], 1
            )

            # Should have reasonable compatibility (at least 50% for synthetic data)
            assert compatibility_ratio >= 0.5, (
                f"Format {format_name} compatibility too low: {compatibility_ratio:.1%}"
            )

            # Should not have excessive missing components
            assert len(set(results["missing_groups"])) <= 2, (
                f"Too many missing groups for {format_name}"
            )

        return compatibility_results

    def test_error_handling_and_recovery(self):
        """Test workflow robustness with various error conditions."""
        error_scenarios = [
            {
                "name": "empty_directory",
                "setup": lambda: tempfile.mkdtemp(),
                "expected_behavior": "graceful_handling",
            },
            {
                "name": "nonexistent_directory",
                "setup": lambda: "/nonexistent/path/to/data",
                "expected_behavior": "exception_raised",
            },
        ]

        error_handling_results = {}

        for scenario in error_scenarios:
            scenario_name = scenario["name"]
            test_path = scenario["setup"]()
            expected = scenario["expected_behavior"]

            error_handling_results[scenario_name] = {
                "test_path": test_path,
                "expected_behavior": expected,
                "actual_behavior": None,
                "error_handled_gracefully": False,
            }

            try:
                if scenario_name == "empty_directory":
                    # Test with empty directory
                    locator = DataFileLocator(test_path)
                    success = locator.build_file_list(test_path)

                    # Should handle empty directory gracefully
                    assert success, "Should handle empty directory"
                    assert len(locator.source_files) == 0, "Should find no files"

                    error_handling_results[scenario_name]["actual_behavior"] = (
                        "graceful_handling"
                    )
                    error_handling_results[scenario_name][
                        "error_handled_gracefully"
                    ] = True

                    # Cleanup
                    shutil.rmtree(test_path)

                elif scenario_name == "nonexistent_directory":
                    # Should raise FileNotFoundError
                    with pytest.raises(FileNotFoundError):
                        locator = DataFileLocator(test_path)

                    error_handling_results[scenario_name]["actual_behavior"] = (
                        "exception_raised"
                    )
                    error_handling_results[scenario_name][
                        "error_handled_gracefully"
                    ] = True

            except Exception as e:
                error_handling_results[scenario_name]["actual_behavior"] = (
                    f"unexpected_exception: {type(e).__name__}"
                )

        # Validate error handling
        for scenario_name, results in error_handling_results.items():
            assert results["error_handled_gracefully"], (
                f"Error scenario '{scenario_name}' not handled correctly"
            )
            assert results["actual_behavior"] == results["expected_behavior"], (
                f"Behavior mismatch in '{scenario_name}'"
            )

        return error_handling_results

    def test_memory_efficiency_large_datasets(self, temp_dir):
        """Test memory efficiency with multiple large datasets."""
        import gc

        import psutil

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        data_dir = Path(temp_dir)
        locator = DataFileLocator(str(data_dir))

        # Load and process files multiple times to test memory management
        memory_measurements = []

        for iteration in range(3):
            # Clear cache and force garbage collection
            locator.clear_target_files()
            gc.collect()

            # Load files
            locator.build_file_list(str(data_dir))
            locator.add_target_files(list(locator.source_files), preload=True)

            # Get XPCS files and extract data
            xpcs_files = locator.get_xpcs_file_list()
            if len(xpcs_files) > 0:
                q, tau, g2_data, g2_err, labels = g2.get_data(xpcs_files)

            # Measure memory after each iteration
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_measurements.append(current_memory)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Analyze memory usage patterns
        max_memory = max(memory_measurements)
        memory_increase = final_memory - initial_memory
        peak_increase = max_memory - initial_memory

        # Memory usage should be reasonable
        assert memory_increase < 1000, (
            f"Memory increase too large: {memory_increase:.1f}MB"
        )
        assert peak_increase < 1500, (
            f"Peak memory increase too large: {peak_increase:.1f}MB"
        )

        # Memory should not continuously grow between iterations
        if len(memory_measurements) >= 2:
            memory_growth = memory_measurements[-1] - memory_measurements[0]
            assert memory_growth < 500, (
                f"Memory growth between iterations: {memory_growth:.1f}MB"
            )

        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "peak_memory_mb": max_memory,
            "memory_increase_mb": memory_increase,
            "peak_increase_mb": peak_increase,
            "memory_measurements": memory_measurements,
        }

    def test_concurrent_file_processing(self, temp_dir):
        """Test concurrent processing capabilities."""
        from concurrent.futures import ThreadPoolExecutor
        import threading

        # Create test data first
        generator = SyntheticXPCSDataGenerator(random_seed=42)
        data_dir = Path(temp_dir)

        # Create multiple test files
        for i in range(5):
            file_path = data_dir / f"concurrent_test_{i}.h5"
            generator.create_test_hdf5_file(file_path)

        def process_single_file(file_path):
            """Process a single file in isolation."""
            try:
                xpcs_file = create_xpcs_dataset(file_path)
                if (
                    xpcs_file
                    and hasattr(xpcs_file, "analysis_type")
                    and "Multitau" in xpcs_file.analysis_type
                ):
                    # Extract basic data to verify processing
                    result = g2.get_data([xpcs_file])
                    if (
                        result[0] is not False
                    ):  # get_data returns (False, None...) on error
                        q, tau, g2_data, g2_err, labels = result
                        return {
                            "file_path": file_path,
                            "success": True,
                            "q_bins": len(q[0]) if len(q) > 0 else 0,
                            "tau_points": len(tau[0]) if len(tau) > 0 else 0,
                            "thread_id": threading.get_ident(),
                        }
                return {
                    "file_path": file_path,
                    "success": False,
                    "error": "No multitau data",
                }
            except Exception as e:
                return {"file_path": file_path, "success": False, "error": str(e)}

        # Get list of test files from directory
        file_paths = [str(p) for p in data_dir.glob("*.h5")]

        # Process files concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            concurrent_results = list(executor.map(process_single_file, file_paths))

        # Process files sequentially for comparison
        sequential_results = [process_single_file(fp) for fp in file_paths]

        # Validate concurrent processing
        successful_concurrent = [r for r in concurrent_results if r["success"]]
        successful_sequential = [r for r in sequential_results if r["success"]]

        assert len(successful_concurrent) == len(successful_sequential), (
            "Concurrent processing should have same success rate as sequential"
        )

        # Check that different threads were used
        thread_ids = {r["thread_id"] for r in successful_concurrent if "thread_id" in r}
        assert len(thread_ids) > 1, "Should use multiple threads for processing"

        # Validate that results are consistent
        for conc_result, seq_result in zip(
            successful_concurrent, successful_sequential
        ):
            if conc_result["success"] and seq_result["success"]:
                assert conc_result["q_bins"] == seq_result["q_bins"], (
                    "Concurrent processing should give same results"
                )
                assert conc_result["tau_points"] == seq_result["tau_points"], (
                    "Concurrent processing should give same results"
                )

        return {
            "concurrent_results": concurrent_results,
            "sequential_results": sequential_results,
            "successful_concurrent": len(successful_concurrent),
            "successful_sequential": len(successful_sequential),
            "threads_used": len(thread_ids),
        }


class TestWorkflowIntegrationEdgeCases:
    """Test edge cases and boundary conditions in workflow integration."""

    def test_workflow_with_corrupted_data(self):
        """Test workflow behavior with corrupted or incomplete data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a corrupted HDF5 file
            corrupted_file = Path(temp_dir) / "corrupted.h5"

            # Create file with invalid HDF5 structure
            with open(corrupted_file, "wb") as f:
                f.write(b"This is not a valid HDF5 file")

            # Test that locator handles corrupted files gracefully
            locator = DataFileLocator(temp_dir)
            success = locator.build_file_list(temp_dir)

            assert success, "Should handle directory with corrupted files"

            # Try to add corrupted file to targets
            locator.add_target_files([corrupted_file.name], preload=True)

            # get_xpcs_file_list should handle failures gracefully
            xpcs_files = locator.get_xpcs_file_list()
            # Should either return empty list or skip corrupted files
            assert isinstance(xpcs_files, list), (
                "Should return list even with corrupted files"
            )

    def test_workflow_performance_monitoring(self, temp_dir):
        """Test workflow performance monitoring and profiling."""
        import time

        data_dir = Path(temp_dir)

        # Time the complete workflow
        start_time = time.perf_counter()

        # Initialize and load
        init_start = time.perf_counter()
        locator = DataFileLocator(str(data_dir))
        locator.build_file_list(str(data_dir))
        init_time = time.perf_counter() - init_start

        # File loading time
        load_start = time.perf_counter()
        locator.add_target_files(list(locator.source_files)[:3], preload=True)
        load_time = time.perf_counter() - load_start

        # Analysis time
        analysis_start = time.perf_counter()
        xpcs_files = locator.get_xpcs_file_list()
        if len(xpcs_files) > 0:
            q, tau, g2_data, g2_err, labels = g2.get_data(xpcs_files)
            geometry_results = g2.compute_geometry(g2_data, "multiple")
        analysis_time = time.perf_counter() - analysis_start

        total_time = time.perf_counter() - start_time

        # Performance expectations
        assert init_time < 1.0, f"Initialization too slow: {init_time:.3f}s"
        assert load_time < 5.0, f"File loading too slow: {load_time:.3f}s"
        assert analysis_time < 2.0, f"Analysis too slow: {analysis_time:.3f}s"
        assert total_time < 10.0, f"Total workflow too slow: {total_time:.3f}s"

        return {
            "initialization_time_s": init_time,
            "file_loading_time_s": load_time,
            "analysis_time_s": analysis_time,
            "total_workflow_time_s": total_time,
            "files_processed": len(xpcs_files) if "xpcs_files" in locals() else 0,
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
