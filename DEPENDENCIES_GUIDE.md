# XPCS Toolkit Dependencies Guide

## Overview

This guide explains the comprehensive dependency management system for XPCS Toolkit, including installation options, requirements files, and dependency categories.

## 📋 Installation Options

### Quick Start (Core Installation)
```bash
pip install xpcs-toolkit
```

### Development Installation
```bash
pip install xpcs-toolkit[dev]
# OR
pip install -r requirements-dev.txt
```

### Full Feature Installation
```bash
pip install xpcs-toolkit[all]
# OR  
pip install -r requirements-full.txt
```

## 📁 Requirements Files

The project includes several requirements files for different use cases:

### Core Requirements
- **`requirements.txt`** - Minimum dependencies for basic functionality
- **`requirements-minimal.txt`** - Absolute minimum for constrained environments

### Development Requirements  
- **`requirements-dev.txt`** - Full development environment
- **`requirements-docs.txt`** - Documentation building only

### Extended Requirements
- **`requirements-full.txt`** - All optional features and dependencies

## 🏗️ Configuration Files

### pyproject.toml
Primary configuration with modern Python packaging standards:
- Core dependencies with version constraints
- Optional dependency groups (dev, docs, gui, performance, extended)
- Tool configurations (pytest, mypy, ruff, coverage)
- Build system requirements
- Package metadata and entry points

### setup.py
Backward compatibility for older build systems:
- Mirrors pyproject.toml configuration
- Supports legacy pip versions
- Includes installation message
- Dynamic version from setuptools_scm

## 📦 Dependency Categories

### Core Dependencies
Essential packages required for basic XPCS Toolkit functionality:

```python
numpy>=1.20.0              # Array processing
scipy>=1.7.0               # Scientific computing
h5py>=3.0.0                # HDF5 file format
hdf5plugin>=3.0.0          # HDF5 compression plugins
pandas>=1.3.0              # Data analysis
matplotlib>=3.5.0          # Plotting
scikit-learn>=1.0.0        # Machine learning
joblib>=1.0.0              # Parallel processing
tqdm>=4.60.0               # Progress bars
```

### Development Dependencies (`[dev]`)
Tools for developing, testing, and maintaining the codebase:

#### Testing Framework
```python
pytest>=6.0.0             # Test runner
pytest-cov>=3.0.0         # Coverage integration
pytest-xdist>=2.5.0       # Parallel testing
pytest-mock>=3.6.0        # Mocking support
coverage[toml]>=6.0.0      # Code coverage
```

#### Code Quality
```python
ruff>=0.1.0               # Fast linter/formatter
mypy>=1.0.0               # Type checking  
pre-commit>=2.15.0        # Git hooks
bandit>=1.7.0             # Security analysis
safety>=2.0.0             # Vulnerability scanning
```

#### Build and Release
```python
build>=0.8.0              # Modern build tool
twine>=4.0.0              # PyPI uploads
wheel>=0.37.0             # Wheel packaging
```

### GUI Dependencies (`[gui]`)
Optional graphical user interface components:

```python
pyqtgraph>=0.12.0         # Fast plotting
PyQt5>=5.15.0             # GUI framework
ipywidgets>=7.6.0         # Interactive widgets
```

### Performance Dependencies (`[performance]`)
Optional performance optimization packages:

```python
numba>=0.56.0             # JIT compilation
psutil>=5.8.0             # System monitoring
cython>=0.29.0            # C extensions
```

### Extended Dependencies (`[extended]`)
Advanced scientific computing capabilities:

```python
xarray>=0.20.0            # Labeled arrays
zarr>=2.10.0              # Chunked arrays
dask>=2021.0.0            # Parallel computing
```

### Documentation Dependencies (`[docs]`)
Tools for building and maintaining documentation:

```python
sphinx>=4.0.0             # Documentation generator
sphinx-rtd-theme>=1.0.0   # Read the Docs theme
myst-parser>=0.18.0       # Markdown support
nbsphinx>=0.8.0           # Jupyter integration
```

## 🎯 Installation Scenarios

### Minimal Installation (Containers/Embedded)
For Docker containers or resource-constrained environments:
```bash
pip install -r requirements-minimal.txt
```

Features available:
- Basic XPCS analysis
- Core file I/O (HDF5)
- Essential plotting
- Command-line interface

Features NOT available:
- Advanced file formats
- Machine learning analysis
- GUI components
- Performance optimizations

### Standard User Installation
For typical scientific computing workflows:
```bash
pip install xpcs-toolkit
```

Includes all core dependencies for full XPCS analysis capabilities.

### Development Installation
For contributors and advanced users:
```bash
# Clone repository
git clone https://github.com/imewei/xpcs-toolkit.git
cd xpcs-toolkit

# Install with development dependencies
pip install -e .[dev]
# OR
make install/dev
```

### Full Feature Installation
For maximum functionality:
```bash
pip install xpcs-toolkit[all]
```

Includes ALL optional dependencies for complete feature set.

## 🔧 Version Constraints

### Philosophy
- **Minimum versions**: Ensure compatibility with tested features
- **Maximum versions**: Avoid for maximum flexibility (except breaking changes)
- **Conservative approach**: Prefer stability over bleeding-edge features

### Version Constraint Examples
```python
numpy>=1.20.0              # Minimum version for required features
scipy>=1.7.0,<2.0.0        # Major version cap for stability
h5py>=3.0.0                # Flexible upper bound
```

## 🚀 Performance Considerations

### Import Time Optimization
XPCS Toolkit uses lazy imports for heavy dependencies:

```python
# Heavy imports are deferred until needed
np = lazy_import('numpy')
plt = lazy_import('matplotlib.pyplot')
```

Benefits:
- Faster CLI startup (~1.5s vs 3-5s)
- Reduced memory footprint for basic operations
- Graceful degradation if optional dependencies missing

### Memory Usage
Different installation profiles have different memory footprints:

- **Minimal**: ~50MB base + data
- **Standard**: ~150MB base + data  
- **Full**: ~500MB+ base + data

## 🔍 Dependency Resolution

### Common Issues and Solutions

#### Import Errors
```bash
# Problem: "No module named 'package_name'"
pip install xpcs-toolkit[dev]  # Install missing optional dependencies
```

#### Version Conflicts
```bash
# Problem: Dependency version conflicts
pip install --upgrade xpcs-toolkit  # Update to latest compatible versions
make deps/check                      # Check for conflicts
```

#### Build Failures
```bash
# Problem: Missing build dependencies
pip install build wheel setuptools   # Install build tools
make build/check                     # Verify build requirements
```

## 🌍 Environment-Specific Considerations

### Python Version Support
- **Minimum**: Python 3.9
- **Recommended**: Python 3.11+
- **Tested**: Python 3.9, 3.10, 3.11, 3.12, 3.13

### Operating System Support
- **Linux**: Full support (recommended for HPC)
- **macOS**: Full support
- **Windows**: Core functionality (some optional features may be limited)

### HPC/Cluster Environments
```bash
# Load required modules first
module load python/3.11 hdf5 openmpi

# Install with minimal dependencies
pip install xpcs-toolkit --user

# Or use requirements file
pip install -r requirements-minimal.txt --user
```

### Cloud/Container Environments
```dockerfile
# Minimal Docker installation
FROM python:3.11-slim
RUN pip install -r requirements-minimal.txt
```

## 📊 Dependency Monitoring

### Security Scanning
```bash
make security/bandit    # Security vulnerability scanning
pip-audit              # Check for known vulnerabilities
```

### Dependency Updates
```bash
make deps/outdated     # Show outdated packages
make deps/update       # Update dependencies
make deps/tree         # Show dependency tree
```

### Compatibility Testing
```bash
make test/all          # Test with current dependencies
tox                    # Test multiple Python versions (if configured)
```

## 🛠️ Development Workflow

### Setting Up Development Environment
```bash
# 1. Clone and setup
git clone https://github.com/imewei/xpcs-toolkit.git
cd xpcs-toolkit

# 2. Create virtual environment
make venv
source .venv/bin/activate

# 3. Install development dependencies
make install/dev

# 4. Verify installation
make test/cli
make status
```

### Managing Dependencies

#### Adding New Dependencies
1. Add to appropriate section in `pyproject.toml`
2. Update corresponding `requirements-*.txt` files
3. Update this documentation
4. Test installation in clean environment

#### Updating Existing Dependencies
1. Update version constraints in `pyproject.toml`
2. Update requirements files
3. Test compatibility with existing code
4. Update documentation if needed

## 🎯 Best Practices

### For Users
1. **Use virtual environments** to avoid conflicts
2. **Install minimal dependencies** first, add features as needed
3. **Update regularly** for security and bug fixes
4. **Check compatibility** before major updates

### For Developers
1. **Pin exact versions** in requirements files for reproducibility
2. **Use version ranges** in pyproject.toml for flexibility
3. **Test with multiple dependency versions** when possible
4. **Document breaking changes** in dependency updates

### For System Administrators
1. **Use requirements files** for consistent deployments
2. **Monitor security advisories** for dependencies
3. **Cache packages** in internal repositories for offline installs
4. **Test deployments** in staging environments

## 🚨 Troubleshooting

### Common Problems

#### "Package not found" errors
```bash
pip install --upgrade pip    # Update pip
pip cache purge             # Clear package cache
pip install --no-cache-dir xpcs-toolkit
```

#### Version conflict errors
```bash
pip install --force-reinstall xpcs-toolkit
# OR create fresh environment
```

#### Build errors on older systems
```bash
pip install --only-binary=all xpcs-toolkit  # Use pre-built wheels
```

#### Memory issues during installation
```bash
pip install --no-cache-dir xpcs-toolkit     # Reduce memory usage
pip install -r requirements-minimal.txt     # Install minimal version first
```

### Getting Help
1. Check [GitHub Issues](https://github.com/imewei/xpcs-toolkit/issues)
2. Review installation logs for specific error messages  
3. Use `make status` to check environment
4. Try installation in clean virtual environment

## 📚 Additional Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [pip Documentation](https://pip.pypa.io/)
- [Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [Dependency Management Best Practices](https://packaging.python.org/guides/analyzing-pypi-package-downloads/)

## 🔄 Migration from Legacy Versions

If upgrading from an older version of XPCS Toolkit:

1. **Uninstall old version**: `pip uninstall pyxpcsviewer`
2. **Install new version**: `pip install xpcs-toolkit`
3. **Update import statements**: Use new package name in code
4. **Check deprecation warnings**: Update to new API where indicated

The new dependency system is designed for better compatibility, security, and performance while maintaining backward compatibility where possible.