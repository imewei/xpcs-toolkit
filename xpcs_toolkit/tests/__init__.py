"""
Test suite for XPCS Toolkit.

This comprehensive test suite is organized into the following categories:

## Test Organization

### Unit Tests (`unit/`)
Individual component tests that verify the behavior of single classes, functions, or modules:
- `test_xpcs_file.py` - XpcsDataFile class tests
- `test_analysis_kernel.py` - AnalysisKernel class tests
- `test_data_file_locator.py` - DataFileLocator class tests
- `test_modules.py` - Scientific analysis modules tests
- `test_logging.py` - Logging system tests
- `test_cli_headless.py` - Command-line interface tests

### Integration Tests (`integration/`)
Multi-component tests that verify interactions between different parts of the system:
- `test_integration.py` - Complete workflow integration tests
- Cross-module compatibility tests
- End-to-end analysis pipeline tests

### Performance Tests (`performance/`)
Performance, scalability, and resource usage tests:
- `test_performance.py` - Comprehensive performance benchmarks
- Memory usage monitoring tests
- Concurrency and thread safety tests
- Large dataset handling tests

### FileIO Tests (`fileio/`)
File format handling and I/O operation tests:
- `test_comprehensive.py` - Complete FileIO module tests
- `test_basic.py` - Basic file operations tests
- HDF5 format validation tests
- File type detection tests

## Running Tests

```bash
# Run all tests
pytest xpcs_toolkit/tests/

# Run specific test categories
pytest xpcs_toolkit/tests/unit/           # Unit tests only
pytest xpcs_toolkit/tests/integration/   # Integration tests only
pytest xpcs_toolkit/tests/performance/   # Performance tests only
pytest xpcs_toolkit/tests/fileio/        # FileIO tests only

# Run with markers
pytest -m "unit"                         # All unit tests
pytest -m "integration"                  # All integration tests
pytest -m "performance"                  # All performance tests
pytest -m "not slow"                     # Exclude slow tests
pytest -m "fileio"                       # All file I/O tests

# Coverage reporting
pytest --cov=xpcs_toolkit --cov-report=html xpcs_toolkit/tests/
```

## Test Configuration

Test configuration is managed through:
- `conftest.py` - Shared fixtures and pytest configuration
- `pyproject.toml` - Test tool configuration
- Custom markers for test categorization

## Fixtures

Common test fixtures are available for all test modules:
- `temp_dir` - Temporary directory for file operations
- `mock_xpcs_data` - Mock XPCS correlation data
- `mock_saxs_data` - Mock SAXS scattering data
- `mock_hdf5_file` - Mock HDF5 file for testing
- `mock_analysis_kernel` - Mock AnalysisKernel instance
- `mock_data_file` - Mock XpcsDataFile instance
"""
