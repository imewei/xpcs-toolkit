# XPCS Toolkit

[![PyPI version](https://badge.fury.io/py/xpcs-toolkit.svg)](https://badge.fury.io/py/xpcs-toolkit)
[![Python Version](https://img.shields.io/pypi/pyversions/xpcs-toolkit)](https://pypi.org/project/xpcs-toolkit/)
[![License](https://img.shields.io/pypi/l/xpcs-toolkit)](https://github.com/imewei/xpcs-toolkit/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/imewei/xpcs-toolkit/ci.yml?branch=main)](https://github.com/imewei/xpcs-toolkit/actions)
[![Documentation Status](https://readthedocs.org/projects/xpcs-toolkit/badge/?version=latest)](https://xpcs-toolkit.readthedocs.io/en/latest/)
[![Coverage Status](https://img.shields.io/codecov/c/github/imewei/xpcs-toolkit)](https://codecov.io/gh/imewei/xpcs-toolkit)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Downloads](https://img.shields.io/pypi/dm/xpcs-toolkit)](https://pypi.org/project/xpcs-toolkit/)
[![GitHub stars](https://img.shields.io/github/stars/imewei/xpcs-toolkit)](https://github.com/imewei/xpcs-toolkit/stargazers)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

A comprehensive command-line tool for X-ray Photon Correlation Spectroscopy (XPCS) analysis, designed for synchrotron beamline operations and research environments.

## ✨ Features

### 🔬 **Scientific Analysis**
- **Multi-tau correlation analysis** with g2(q,t) correlation function processing
- **Two-time correlation analysis** for time-resolved studies
- **Small-Angle X-ray Scattering (SAXS)** visualization and analysis
- **Beam stability monitoring** and quality assessment
- **Advanced data fitting** with single and double exponential models

### 🚀 **Performance & Usability** 
- **Headless operation** optimized for remote and automated analysis
- **Lazy loading** for fast startup (~1.5s import time)
- **Memory efficient** processing of large datasets
- **Parallel processing** support for batch analysis
- **Comprehensive logging** with performance monitoring

### 🛠️ **Technical Excellence**
- **Modern Python packaging** with full dependency management
- **Backward compatibility** for legacy systems
- **Growing test suite** with expanding code coverage
- **Professional documentation** with examples and guides
- **Enterprise-grade** build and deployment system

## 📦 Installation

### Quick Start
```bash
pip install xpcs-toolkit
```

### Development Installation
```bash
git clone https://github.com/imewei/xpcs-toolkit.git
cd xpcs-toolkit
make dev  # or: pip install -e .[dev]
```

### Installation Options
```bash
# Core functionality
pip install xpcs-toolkit

# With specific optional dependencies
pip install xpcs-toolkit[dev]         # Development tools
pip install xpcs-toolkit[docs]        # Documentation tools  
pip install xpcs-toolkit[performance] # Performance optimizations

# With all optional dependencies
pip install xpcs-toolkit[all]

# Minimal (containers/embedded)
pip install -r requirements-minimal.txt
```

## 🚀 Quick Start

### Command Line Interface
```bash
# Display help
xpcs-toolkit --help

# List files in directory
xpcs-toolkit list /path/to/data/

# Generate 2D SAXS patterns
xpcs-toolkit saxs2d /path/to/data/ --outfile pattern.png

# Analyze g2 correlation functions  
xpcs-toolkit g2 /path/to/data/ --qmin 0.01 --qmax 0.1

# Create 1D radial profiles
xpcs-toolkit saxs1d /path/to/data/ --log-x --log-y

# Monitor beam stability
xpcs-toolkit stability /path/to/data/
```

### Python API
```python
import xpcs_toolkit

# Load XPCS data file
data = xpcs_toolkit.XpcsDataFile('experiment.hdf5')

# Initialize analysis kernel
kernel = xpcs_toolkit.AnalysisKernel('/path/to/data/')
kernel.build_file_list()

# Access correlation data
g2_data = data.g2
intensity = data.saxs_2d
```

## 🏗️ Development

### Environment Setup
```bash
# Create virtual environment
make venv
source .venv/bin/activate

# Install development dependencies
make install/dev

# Run tests
make test

# Code quality checks
make lint
make format

# Build documentation  
make docs
```

### Development Commands
```bash
make help           # Show all available commands
make status         # Project status and environment info  
make clean          # Remove build artifacts
make build          # Build distribution packages
make coverage       # Generate test coverage report
```

## 📊 Supported File Formats

### Primary Support
- **APS 8-ID-I NeXus format** - Custom NeXus-based format for XPCS data
- **Legacy HDF5 format** - Backward compatibility with older XPCS files
- **Automatic format detection** - Distinguishes between NeXus and legacy formats

### Data Export
- **High-resolution images** (PNG, PDF, SVG)
- **Publication-ready plots** with LaTeX labels
- **Structured data outputs** (HDF5, NumPy arrays)
- **Analysis reports** (JSON, CSV)

## 🎯 Use Cases

### Synchrotron Beamlines
- **Real-time analysis** during experiments
- **Quality control** and data validation
- **Automated processing** pipelines
- **Remote monitoring** capabilities

### Research Applications
- **Soft matter dynamics** studies
- **Materials characterization** 
- **Time-resolved measurements**
- **Comparative analysis** workflows

### Production Environments
- **Batch processing** of experimental datasets
- **Reproducible analysis** with version control
- **Performance monitoring** and optimization
- **Integration** with existing analysis pipelines

## 📚 Documentation

### 📖 **[Full Documentation](https://xpcs-toolkit.readthedocs.io/)**

#### Quick Links
- **[Installation Guide](https://xpcs-toolkit.readthedocs.io/en/latest/installation.html)** - Detailed setup instructions
- **[Quick Start Tutorial](https://xpcs-toolkit.readthedocs.io/en/latest/quickstart.html)** - Get started in 5 minutes  
- **[API Reference](https://xpcs-toolkit.readthedocs.io/en/latest/api/)** - Complete API documentation
- **[User Guides](https://xpcs-toolkit.readthedocs.io/en/latest/guides/)** - In-depth tutorials and examples
- **[FAQ](https://xpcs-toolkit.readthedocs.io/en/latest/faq.html)** - Common questions and troubleshooting

#### Development Resources
- **[Contributing Guide](https://xpcs-toolkit.readthedocs.io/en/latest/contributing.html)** - How to contribute
- **[Changelog](https://xpcs-toolkit.readthedocs.io/en/latest/changelog.html)** - Release history
- **[Developer Documentation](https://xpcs-toolkit.readthedocs.io/en/latest/development/)** - Development setup

## 🤝 Contributing

We welcome contributions! Please see our development workflow:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Install** development dependencies (`make install/dev`)
4. **Make** your changes with tests
5. **Run** quality checks (`make ci/check`)
6. **Submit** a pull request

### Code Quality Standards
- **Comprehensive testing** for new features
- **Type hints** for all public APIs  
- **Comprehensive documentation** with examples
- **Performance benchmarks** for critical paths

## 🏆 Performance Benchmarks

| Operation | Time | Throughput | Notes |
|-----------|------|------------|-------|
| Package import | ~1.5s | - | First import with lazy loading |
| File discovery | ~0.001s | >10,000 files/s | Directory scanning |
| CLI help | ~1.7s | - | Includes import overhead |
| Basic analysis | ~2-4s | Varies | Depends on data size |

*Benchmarks on Apple Silicon M1/M2, Python 3.13*

**Memory Usage**: Typical analysis operations use 50-200MB depending on dataset size and analysis type.

## 📄 Citation

If you use XPCS Toolkit in your research, please cite:

```bibtex
@software{xpcs_toolkit,
  title={XPCS Toolkit: Advanced X-ray Photon Correlation Spectroscopy Analysis},
  author={Chen, Wei},
  institution={Argonne National Laboratory},
  year={2024},
  url={https://github.com/imewei/xpcs-toolkit}
}
```

## 🏢 Institutional Support

Developed at **Argonne National Laboratory** with support from the U.S. Department of Energy, Office of Science, Office of Basic Energy Sciences.

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/imewei/xpcs-toolkit/issues)
- **Email**: weichen@anl.gov
- **Repository**: [GitHub Repository](https://github.com/imewei/xpcs-toolkit)
- **Documentation**: See included documentation files and tutorials

## 📖 Local Documentation

### User Guides
- **[USER_GUIDE.md](USER_GUIDE.md)** - Complete user guide with CLI and API examples
- **[SCIENTIFIC_BACKGROUND.md](SCIENTIFIC_BACKGROUND.md)** - Theory and scientific methods  
- **[FILE_FORMAT_GUIDE.md](FILE_FORMAT_GUIDE.md)** - File formats and data structures

### Developer Resources  
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - Development setup, testing, and contribution guide
- **[DEPENDENCIES_GUIDE.md](DEPENDENCIES_GUIDE.md)** - Dependency management and installation options
- **[QUALITY_GATES.md](QUALITY_GATES.md)** - Code quality standards and CI/CD processes

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Advanced Photon Source** at Argonne National Laboratory
- **XPCS community** for feedback and requirements
- **Scientific Python ecosystem** for foundational tools
- **Contributors** and early adopters

---

**⚡ Ready to analyze your XPCS data?** Get started with `pip install xpcs-toolkit` and explore the power of modern X-ray photon correlation spectroscopy analysis!