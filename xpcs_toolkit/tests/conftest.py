"""
Pytest configuration for XPCS Toolkit test suite.

This module configures pytest settings, fixtures, and markers for the organized test structure.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from unittest.mock import Mock

import numpy as np
import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running tests")
    config.addinivalue_line("markers", "fileio: marks tests as file I/O related")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


# Shared fixtures for all test modules
@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for tests (session-scoped for efficiency)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def mock_xpcs_data():
    """Create mock XPCS data for testing (session-scoped, deterministic)."""
    # Use fixed seed for reproducible, fast data generation
    np.random.seed(42)
    q_values = np.array([0.005, 0.01, 0.02, 0.05])
    tau_values = np.logspace(-6, 0, 32)  # Reduced from 64 to 32 for speed
    g2_values = np.random.rand(32, 4) + 1.0
    g2_errors = np.random.rand(32, 4) * 0.01
    labels = [f"Q{i + 1} ({q:.3f})" for i, q in enumerate(q_values)]
    np.random.seed()  # Reset seed

    return {
        "q_values": q_values,
        "tau_values": tau_values,
        "g2_values": g2_values,
        "g2_errors": g2_errors,
        "labels": labels,
    }


@pytest.fixture(scope="session")
def mock_saxs_data():
    """Create mock SAXS data for testing (session-scoped, smaller dataset)."""
    # Use fixed seed and smaller dataset for speed
    np.random.seed(42)
    q_values = np.logspace(-3, -1, 50)  # Reduced from 100 to 50 for speed
    intensity = np.random.rand(50) * 1e6
    errors = np.random.rand(50) * 1e3
    np.random.seed()  # Reset seed

    return {"q_values": q_values, "intensity": intensity, "errors": errors}


@pytest.fixture(scope="session")
def mock_hdf5_file(temp_dir):
    """Create a mock HDF5 file for testing (session-scoped)."""
    import h5py

    file_path = temp_dir / "test_data.h5"

    # Only create if it doesn't exist (session persistence)
    if not file_path.exists():
        with h5py.File(file_path, "w") as f:
            # Create basic structure with smaller datasets for speed
            f.create_dataset("test_array", data=np.array([1, 2, 3, 4, 5]))
            f.create_dataset("test_scalar", data=42.0)
            f.create_dataset("test_string", data="test_data")

            # Create group structure with smaller data
            group = f.create_group("test_group")
            np.random.seed(42)
            group.create_dataset("nested_data", data=np.random.rand(5, 5))  # Reduced from 10x10
            np.random.seed()

    return file_path


@pytest.fixture(scope="session")
def mock_analysis_kernel():
    """Create a mock AnalysisKernel for testing (session-scoped)."""
    kernel = Mock()
    kernel.directory = "/mock/directory"
    kernel.file_list = ["file1.h5", "file2.h5", "file3.h5"]
    kernel.selected_files = [0, 1]
    kernel.get_selected_files.return_value = [0, 1]
    return kernel


@pytest.fixture(scope="session")
def mock_data_file():
    """Create a mock XpcsDataFile for testing (session-scoped with deterministic data)."""
    data_file = Mock()
    data_file.filename = "mock_data.h5"
    data_file.analysis_type = ("Multitau",)
    data_file.atype = "Multitau"

    # Pre-generate deterministic mock data to avoid repeated random generation
    np.random.seed(42)
    g2_data = np.random.rand(32, 2) + 1.0
    g2_errors = np.random.rand(32, 2) * 0.01
    saxs_intensity = np.random.rand(25) * 1e5  # Reduced from 50 to 25
    saxs_errors = np.random.rand(25) * 1e2
    np.random.seed()

    # Mock g2 data
    data_file.get_g2_data.return_value = (
        np.array([0.01, 0.02]),
        np.logspace(-6, 0, 32),
        g2_data,
        g2_errors,
        ["Q1", "Q2"],
    )

    # Mock SAXS data
    data_file.get_saxs_1d_data.return_value = (
        np.logspace(-3, -1, 25),
        saxs_intensity,
        saxs_errors,
    )

    return data_file


@pytest.fixture(scope="session")
def synthetic_data_generator():
    """Create a synthetic data generator for testing (session-scoped)."""
    try:
        from xpcs_toolkit.tests.fixtures.synthetic_data import SyntheticXPCSDataGenerator
        return SyntheticXPCSDataGenerator(random_seed=42)
    except ImportError:
        return None


@pytest.fixture(scope="session")
def shared_temp_files(temp_dir):
    """Create shared temporary files for testing (session-scoped)."""
    files = {}
    
    # Create test HDF5 files with minimal data
    import h5py
    
    test_file = temp_dir / "shared_test.h5"
    with h5py.File(test_file, "w") as f:
        f.create_dataset("test_data", data=np.arange(100))
        f.create_dataset("metadata", data="test")
        
    files["hdf5_test"] = test_file
    
    # Create test text file
    text_file = temp_dir / "test.txt"
    text_file.write_text("test data\nline 2\n")
    files["text_test"] = text_file
    
    return files


# Performance test configuration
def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location and optimize for parallel execution."""
    for item in items:
        # Add markers based on test path
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        if "fileio" in item.nodeid:
            item.add_marker(pytest.mark.fileio)
        if "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)

        # Add slow marker for performance tests
        if "performance" in item.nodeid or "slow" in item.name:
            item.add_marker(pytest.mark.slow)
            
        # Mark tests that must run serially (not in parallel)
        if any(keyword in item.nodeid for keyword in ["cli", "subprocess", "system"]):
            item.add_marker(pytest.mark.serial)
            
        # Mark benchmark tests
        if "benchmark" in item.nodeid or "benchmark" in item.name:
            item.add_marker(pytest.mark.benchmark)


# Test discovery helpers
def pytest_sessionstart(session):
    """Print test organization info at start of session."""
    print("\n" + "=" * 50)
    print("XPCS Toolkit Test Suite")
    print("=" * 50)
    print("Test Structure:")
    print("  • unit/        - Unit tests for individual components")
    print("  • integration/ - Integration tests for component interactions")
    print("  • performance/ - Performance and scalability tests")
    print("  • fileio/      - File I/O and format handling tests")
    print("=" * 50)
