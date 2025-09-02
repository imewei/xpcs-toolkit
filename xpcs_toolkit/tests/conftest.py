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
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_xpcs_data():
    """Create mock XPCS data for testing."""
    q_values = np.array([0.005, 0.01, 0.02, 0.05])
    tau_values = np.logspace(-6, 0, 64)
    g2_values = np.random.rand(64, 4) + 1.0
    g2_errors = np.random.rand(64, 4) * 0.01
    labels = [f"Q{i + 1} ({q:.3f})" for i, q in enumerate(q_values)]

    return {
        "q_values": q_values,
        "tau_values": tau_values,
        "g2_values": g2_values,
        "g2_errors": g2_errors,
        "labels": labels,
    }


@pytest.fixture
def mock_saxs_data():
    """Create mock SAXS data for testing."""
    q_values = np.logspace(-3, -1, 100)
    intensity = np.random.rand(100) * 1e6
    errors = np.random.rand(100) * 1e3

    return {"q_values": q_values, "intensity": intensity, "errors": errors}


@pytest.fixture
def mock_hdf5_file(temp_dir):
    """Create a mock HDF5 file for testing."""
    import h5py

    file_path = temp_dir / "test_data.h5"

    with h5py.File(file_path, "w") as f:
        # Create basic structure
        f.create_dataset("test_array", data=np.array([1, 2, 3, 4, 5]))
        f.create_dataset("test_scalar", data=42.0)
        f.create_dataset("test_string", data="test_data")

        # Create group structure
        group = f.create_group("test_group")
        group.create_dataset("nested_data", data=np.random.rand(10, 10))

    return file_path


@pytest.fixture
def mock_analysis_kernel():
    """Create a mock AnalysisKernel for testing."""
    kernel = Mock()
    kernel.directory = "/mock/directory"
    kernel.file_list = ["file1.h5", "file2.h5", "file3.h5"]
    kernel.selected_files = [0, 1]
    kernel.get_selected_files.return_value = [0, 1]
    return kernel


@pytest.fixture
def mock_data_file():
    """Create a mock XpcsDataFile for testing."""
    data_file = Mock()
    data_file.filename = "mock_data.h5"
    data_file.analysis_type = ("Multitau",)
    data_file.atype = "Multitau"

    # Mock g2 data
    data_file.get_g2_data.return_value = (
        np.array([0.01, 0.02]),
        np.logspace(-6, 0, 32),
        np.random.rand(32, 2) + 1.0,
        np.random.rand(32, 2) * 0.01,
        ["Q1", "Q2"],
    )

    # Mock SAXS data
    data_file.get_saxs_1d_data.return_value = (
        np.logspace(-3, -1, 50),
        np.random.rand(50) * 1e5,
        np.random.rand(50) * 1e2,
    )

    return data_file


# Performance test configuration
def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location."""
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
