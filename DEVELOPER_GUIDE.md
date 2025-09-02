# XPCS Toolkit Developer Guide

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)

**Complete guide for developing, testing, and contributing to the XPCS Toolkit.**

## Table of Contents

- [Development Setup](#development-setup)
- [Development Workflows](#development-workflows)
- [Quality Standards](#quality-standards)
- [Build & Release](#build--release)
- [Contributing Guidelines](#contributing-guidelines)
- [CI/CD Pipeline](#cicd-pipeline)

---

## Development Setup

### Quick Start

```bash
# Clone repository
git clone https://github.com/imewei/xpcs-toolkit.git
cd xpcs-toolkit

# Set up development environment
make venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
make dev

# Verify setup
make status
make test/cli
```

### Prerequisites

- **Python 3.9+** (recommended: Python 3.11+)
- **Git** for version control
- **Make** for build automation
- **Virtual environment** for isolated development

### Project Structure

```
xpcs_toolkit/
├── 🏗️  Makefile              # Build automation
├── 📋  pyproject.toml        # Modern packaging configuration
├── 🧪  tests/               # Comprehensive test suite
├── 📚  docs/                # Documentation files
├── 🔧  xpcs_toolkit/        # Source code
│   ├── core/               # Core business logic
│   ├── scientific/         # Scientific analysis modules  
│   ├── io/                 # Input/output operations
│   ├── cli/                # Command-line interface
│   └── utils/              # Utilities and helpers
└── 🚀  scripts/            # Development scripts
```

---

## Development Workflows

### Daily Development Commands

#### 📊 Project Status
```bash
# Check overall project status
make status

# Check environment setup
make check-venv

# Show project statistics
make stats
```

#### 🧪 Testing Commands
```bash
# Quick test run during development
make quick-test

# Run comprehensive test suite
make test

# Specific test categories
make test/unit              # Unit tests only
make test/integration       # Integration tests
make test/cli              # CLI functionality tests
make test/performance      # Performance benchmarks
make test/all              # Everything

# Generate coverage report
make coverage
```

#### ✨ Code Quality
```bash
# Format code automatically
make format

# Check formatting without changes
make format/check

# Run all linting checks
make lint

# Specific linting tools
make lint/ruff             # Code style and errors
make lint/mypy             # Type checking
make security/bandit       # Security analysis
```

#### 🏗️ Build & Install
```bash
# Install in development mode
make install

# Install with development dependencies
make install/dev

# Build distribution packages
make build

# Check build requirements
make build/check
```

#### 🧹 Maintenance
```bash
# Clean build artifacts
make clean

# Deep clean everything
make clean-all

# Clean specific targets
make clean-build           # Build artifacts only
make clean-pyc            # Python bytecode files
make clean-test           # Test artifacts
```

### Advanced Commands

#### 📈 Performance Analysis
```bash
# Profile package import performance
make profile

# Run performance benchmarks
make test/performance

# Monitor memory usage during tests
make test/memory
```

#### 📚 Documentation
```bash
# Generate documentation
make docs

# Serve documentation locally
make docs/serve

# Clean documentation artifacts
make clean-docs
```

#### 🔄 Dependency Management
```bash
# Check dependency compatibility
make deps/check

# Show outdated packages
make deps/outdated

# Update dependencies
make deps/update

# Display dependency tree
make deps/tree
```

### Development Workflow Examples

#### New Feature Development
```bash
# 1. Start with clean environment
make status
make clean

# 2. Create feature branch
git checkout -b feature/new-analysis-method

# 3. Develop with rapid feedback
make quick-test          # Fast tests during development

# 4. Pre-commit checks
make format             # Auto-format code
make lint               # Check code quality
make test              # Full test suite
make coverage          # Coverage analysis

# 5. Commit changes
git add .
git commit -m "Add new analysis method with tests"

# 6. Final validation
make ci/check          # Complete CI pipeline locally
```

#### Bug Fix Workflow
```bash
# 1. Reproduce the issue
make test/specific-module

# 2. Write failing test
# Add test that reproduces the bug

# 3. Fix the bug
# Implement fix

# 4. Verify fix
make test              # All tests should pass
make coverage          # Ensure coverage maintained

# 5. Commit fix
git commit -m "Fix: description of bug fix with test"
```

#### Release Preparation
```bash
# 1. Full quality assurance
make ci/check          # Complete CI pipeline

# 2. Build distributions
make build             # Create wheel and source dist

# 3. Test release (optional)
make release/test      # Upload to test PyPI

# 4. Production release
make release/prod      # Upload to production PyPI
```

---

## Quality Standards

### Code Quality Metrics

| Metric | Target | Tool | Command |
|--------|--------|------|---------|
| **Test Coverage** | ≥ 85% | pytest-cov | `make coverage` |
| **Code Style** | PEP 8 | Ruff | `make lint/ruff` |
| **Type Coverage** | ≥ 80% | MyPy | `make lint/mypy` |
| **Security** | Zero critical | Bandit | `make security/bandit` |
| **Import Time** | < 10s | Custom | `make profile` |

### Automated Quality Gates

#### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**Pre-commit checks include:**
- Code formatting (Ruff)
- Type checking (MyPy)  
- Security scanning (Bandit)
- Import performance validation
- Scientific accuracy tests

#### Required Standards

**✅ Passing Requirements:**
- All tests passing (446+ tests)
- Coverage ≥ 85%
- Zero critical security issues
- Type checking with minimal errors
- Import performance < 10s
- Documentation coverage ≥ 80%

**❌ Blocking Conditions:**
- Any test failures
- Coverage below 85%
- Critical security vulnerabilities
- Severe performance regressions
- Import failures

### Scientific Computing Standards

#### Numerical Accuracy
```python
# Use appropriate precision for scientific calculations
import numpy as np

# Prefer float64 for critical calculations
result = np.array(data, dtype=np.float64)

# Use proper error handling for numerical edge cases
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = numerator / denominator

# Validate results against analytical solutions
np.testing.assert_allclose(computed, analytical, rtol=1e-10)
```

#### Performance Optimization
```python
# Prefer vectorized operations
result = np.sum(array * weights)  # Good

# Avoid Python loops for array operations
# result = sum(a * w for a, w in zip(array, weights))  # Avoid

# Use appropriate data types
intensity = np.array(data, dtype=np.float32)  # 32-bit sufficient
precision_calc = np.array(data, dtype=np.float64)  # 64-bit for precision
```

#### Error Handling
```python
# Use specific exception types
try:
    data = load_xpcs_file(filename)
except (OSError, IOError) as e:
    logger.error("File access error: %s", e)
    raise
except ValueError as e:
    logger.error("Data format error: %s", e)
    return None
except Exception as e:
    logger.error("Unexpected error: %s", e)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Full traceback:", exc_info=True)
    raise
```

---

## Build & Release

### Build System

The project uses a modern Python packaging setup:
- **pyproject.toml** - Primary configuration  
- **Makefile** - Development automation
- **setup.py** - Backward compatibility

#### Build Commands
```bash
# Check build requirements
make build/check

# Build both wheel and source distribution
make build

# Build specific formats
make build/wheel          # Wheel package only
make build/sdist          # Source distribution only

# Verify build
python -m pip install dist/xpcs_toolkit-*.whl
```

### Release Process

#### Version Management
```bash
# Version is managed by setuptools_scm from git tags
git tag v1.2.3
git push origin v1.2.3
```

#### Release Workflow
```bash
# 1. Complete quality checks
make ci/check

# 2. Update documentation
make docs

# 3. Build distributions  
make build

# 4. Test release (optional)
make release/test

# 5. Production release (maintainers only)
make release/prod
```

### Package Configuration

Key configuration files:

#### pyproject.toml
```toml
[build-system]
requires = ["setuptools>=64", "setuptools_scm>=8", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "xpcs-toolkit"
dynamic = ["version"]
description = "Advanced X-ray Photon Correlation Spectroscopy Analysis"
authors = [{name = "Wei Chen", email = "weichen@anl.gov"}]
license = {text = "MIT"}
requires-python = ">=3.9"

# Dependencies and optional groups defined here
```

#### Makefile Variables
```makefile
PROJECT_NAME := xpcs-toolkit
PACKAGE_NAME := xpcs_toolkit
PYTHON := python
PIP := pip
PYTEST := pytest
```

---

## Contributing Guidelines

### Code Style

#### Python Style
- **Follow PEP 8** - Use `make format` for automatic formatting
- **Use type hints** - All public functions should have type annotations
- **Document functions** - Include docstrings with parameters and examples
- **Write tests** - New features require comprehensive test coverage

#### Example Function
```python
def analyze_correlation(
    g2_data: np.ndarray,
    tau_values: np.ndarray,
    q_range: Optional[Tuple[float, float]] = None
) -> Dict[str, Any]:
    """
    Analyze correlation function data with optional q-filtering.
    
    Args:
        g2_data: Correlation function array [n_q, n_tau]
        tau_values: Time delay values in seconds
        q_range: Optional (q_min, q_max) filter in Å⁻¹
        
    Returns:
        Dict with analysis results including fit parameters
        
    Example:
        >>> g2 = np.random.random((10, 100)) + 1
        >>> tau = np.logspace(-6, 0, 100)
        >>> results = analyze_correlation(g2, tau, q_range=(0.01, 0.1))
        >>> print(f"Found {len(results['fit_params'])} successful fits")
    """
    # Implementation here
    return results
```

### Testing Requirements

#### Test Categories
1. **Unit Tests** - Test individual functions and classes
2. **Integration Tests** - Test module interactions
3. **Performance Tests** - Benchmark critical functions
4. **CLI Tests** - Validate command-line interface

#### Test Structure
```python
import pytest
import numpy as np
from xpcs_toolkit import XpcsDataFile

class TestXpcsDataFile:
    """Test suite for XpcsDataFile class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample XPCS data for testing."""
        # Create synthetic data
        return synthetic_data
    
    def test_data_loading(self, sample_data):
        """Test basic data loading functionality."""
        data = XpcsDataFile(sample_data.filename)
        
        assert data.g2.shape[0] > 0
        assert data.tau.shape[0] > 0
        assert data.analysis_type in ['Multitau', 'Twotime']
    
    @pytest.mark.parametrize("q_range", [
        (0.01, 0.1),
        (0.005, 0.05),
        None
    ])
    def test_q_range_filtering(self, sample_data, q_range):
        """Test q-range filtering functionality."""
        data = XpcsDataFile(sample_data.filename)
        result = data.get_correlation_data(q_range=q_range)
        
        if q_range is not None:
            q_min, q_max = q_range
            assert np.all(result['q_values'] >= q_min)
            assert np.all(result['q_values'] <= q_max)
```

### Pull Request Process

#### Before Submitting
```bash
# 1. Ensure all quality checks pass
make ci/check

# 2. Update documentation if needed
make docs

# 3. Add appropriate tests
make test

# 4. Check coverage
make coverage
```

#### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature  
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Coverage maintained/improved

## Quality Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or migration guide provided)
```

---

## CI/CD Pipeline

### GitHub Actions Workflows

The project uses comprehensive CI/CD with multiple workflow jobs:

#### 1. Quality Gates (`quality-checks.yml`)
**Triggers:** Every push and PR
**Purpose:** Fast feedback on code quality

```yaml
jobs:
  quality-gates:
    runs-on: ubuntu-latest
    steps:
      - name: Code Quality
        run: |
          make lint
          make format/check
          make security/bandit
      - name: Type Checking  
        run: make lint/mypy
      - name: Quick Tests
        run: make quick-test
```

#### 2. Comprehensive Testing (`comprehensive-testing.yml`)
**Triggers:** PR and main branch pushes
**Purpose:** Full validation across Python versions

```yaml
strategy:
  matrix:
    python-version: ["3.9", "3.10", "3.11", "3.12"]
    os: [ubuntu-latest, windows-latest, macos-latest]

jobs:
  test:
    steps:
      - name: Full Test Suite
        run: make test/all
      - name: Coverage Analysis
        run: make coverage
      - name: Performance Benchmarks
        run: make test/performance
```

#### 3. Nightly Comprehensive
**Triggers:** Daily schedule
**Purpose:** Extended testing and monitoring

### Local CI Simulation

Run the complete CI pipeline locally:

```bash
# Complete CI check (same as GitHub Actions)
make ci/check

# This runs:
# - make check-venv     # Environment validation
# - make lint           # Code quality
# - make test           # Full test suite  
# - make coverage       # Coverage analysis
# - make security/bandit # Security scan
```

### Performance Monitoring

Track key performance metrics:

```bash
# Import performance
make profile

# Memory usage
make test/memory

# Execution benchmarks
make test/performance
```

Expected benchmarks:
- **Package import:** < 10 seconds
- **CLI startup:** < 5 seconds  
- **Basic analysis:** 2-10 seconds (data dependent)
- **Memory usage:** 50-200MB baseline

### Quality Monitoring Dashboard

Key metrics tracked over time:
- **Test Coverage Trends** - Target: maintain ≥85%
- **Performance Baselines** - Flag >20% regressions
- **Security Posture** - Zero critical vulnerabilities
- **Code Quality Scores** - Ruff/MyPy compliance
- **Documentation Coverage** - Target: ≥80%

---

## Advanced Topics

### Custom Analysis Modules

#### Creating New Analysis Functions
```python
# xpcs_toolkit/scientific/custom/new_analysis.py
import numpy as np
from typing import Dict, Any, Optional

def custom_analysis(
    data: np.ndarray,
    parameters: Dict[str, Any],
    options: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Template for custom analysis function.
    
    Args:
        data: Input data array
        parameters: Analysis parameters
        options: Optional configuration
        
    Returns:
        Dict with analysis results
    """
    # Implement custom analysis
    results = {
        'success': True,
        'values': computed_values,
        'errors': error_estimates,
        'metadata': analysis_metadata
    }
    
    return results
```

#### Integration with CLI
```python
# Add to CLI in xpcs_toolkit/cli/commands/
@click.command()
@click.argument('data_path')
@click.option('--parameter', default=1.0, help='Analysis parameter')
def custom_command(data_path: str, parameter: float):
    """Custom analysis command."""
    from xpcs_toolkit.scientific.custom import new_analysis
    
    # Load data
    data = load_data(data_path)
    
    # Run analysis
    results = new_analysis.custom_analysis(
        data.arrays, 
        {'param': parameter}
    )
    
    # Output results
    click.echo(f"Analysis complete: {results['success']}")
```

### Plugin System

The modular architecture supports plugin development:

```python
# Example plugin structure
xpcs_toolkit_plugins/
├── __init__.py
├── my_plugin/
│   ├── __init__.py
│   ├── analysis.py      # Custom analysis methods
│   ├── cli.py          # CLI extensions
│   └── utils.py        # Helper functions
└── setup.py            # Plugin packaging
```

### Performance Optimization

#### Profiling Code
```python
# Use built-in profiling
import cProfile
import pstats

def profile_analysis():
    pr = cProfile.Profile()
    pr.enable()
    
    # Your analysis code here
    result = run_analysis()
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative').print_stats(10)
    
    return result
```

#### Memory Optimization
```python
# Monitor memory usage
import tracemalloc
import psutil

def monitor_memory():
    tracemalloc.start()
    process = psutil.Process()
    
    # Your code here
    
    # Memory statistics
    current, peak = tracemalloc.get_traced_memory()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    print(f"Current memory: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
    print(f"Process memory: {memory_mb:.1f} MB")
```

---

## Support & Resources

### Documentation
- **User Guide**: [USER_GUIDE.md](USER_GUIDE.md) - End-user documentation
- **API Reference**: Generated from docstrings via Sphinx
- **Scientific Background**: [SCIENTIFIC_BACKGROUND.md](SCIENTIFIC_BACKGROUND.md)
- **File Formats**: [FILE_FORMAT_GUIDE.md](FILE_FORMAT_GUIDE.md)

### Community
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community support
- **Email**: weichen@anl.gov for direct contact

### Development Environment
```bash
# Get help with any make command
make help

# Show current project status
make status

# Run diagnostics
make check-venv
make deps/check
```

### Troubleshooting

#### Common Development Issues
```bash
# Virtual environment issues
make check-venv
make venv  # Recreate if needed

# Dependency conflicts
make deps/check
make deps/update

# Test failures
make test/unit          # Isolate unit tests
make test/integration   # Check integration tests
make coverage          # Coverage analysis

# Build issues
make clean
make build/check
make build
```

This developer guide provides everything needed to contribute effectively to the XPCS Toolkit. For questions not covered here, please open a GitHub issue or discussion.