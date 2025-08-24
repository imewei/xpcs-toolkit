#!/usr/bin/env python3
"""
XPCS Toolkit Setup Script

This setup.py file provides backward compatibility for older pip versions
and build systems that don't fully support pyproject.toml.

For modern installations, the project configuration is primarily defined in
pyproject.toml. This file acts as a bridge for compatibility.
"""

import os
import sys
from pathlib import Path

# Ensure we can import setuptools
try:
    from setuptools import setup, find_packages
except ImportError:
    print("Error: setuptools is required to install this package.")
    print("Please install it using: pip install setuptools")
    sys.exit(1)

# Get the long description from README
def get_long_description():
    """Read the README.md file for the long description."""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    else:
        return (
            "XPCS Toolkit: A comprehensive command-line tool for "
            "X-ray Photon Correlation Spectroscopy analysis"
        )

# Get version dynamically
def get_version():
    """Get version from setuptools_scm or fallback."""
    try:
        from setuptools_scm import get_version  # type: ignore[import-untyped]
        return get_version(root='..', relative_to=__file__)
    except (ImportError, LookupError):
        # Fallback version if setuptools_scm is not available or no git
        return "0.1.0.dev0"

# Core dependencies - should match pyproject.toml
CORE_REQUIREMENTS = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "h5py>=3.0.0", 
    "hdf5plugin>=3.0.0",
    "pandas>=1.3.0",
    "matplotlib>=3.5.0",
    "scikit-learn>=1.0.0",
    "joblib>=1.0.0",
    "tqdm>=4.60.0",
]

# Development dependencies
DEV_REQUIREMENTS = [
    "pytest>=6.0.0",
    "pytest-cov>=3.0.0", 
    "pytest-xdist>=2.5.0",
    "pytest-mock>=3.6.0",
    "coverage[toml]>=6.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "pre-commit>=2.15.0",
    "bandit>=1.7.0",
    "safety>=2.0.0",
    "line_profiler>=3.5.0",
    "memory_profiler>=0.60.0",
    "build>=0.8.0",
    "twine>=4.0.0",
    "wheel>=0.37.0",
]

# Documentation dependencies  
DOCS_REQUIREMENTS = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "sphinx-autodoc-typehints>=1.12.0",
    "myst-parser>=0.18.0",
    "sphinx-copybutton>=0.5.0",
    "nbsphinx>=0.8.0",
]

# GUI dependencies
GUI_REQUIREMENTS = [
    "pyqtgraph>=0.12.0",
    "PyQt5>=5.15.0",
    "ipywidgets>=7.6.0",
]

# Performance dependencies
PERFORMANCE_REQUIREMENTS = [
    "numba>=0.56.0",
    "psutil>=5.8.0", 
    "cython>=0.29.0",
]

# Extended scientific computing
EXTENDED_REQUIREMENTS = [
    "xarray>=0.20.0",
    "netcdf4>=1.5.0",
    "zarr>=2.10.0",
    "dask>=2021.0.0",
]

# Define extra requirements
EXTRAS_REQUIRE = {
    "dev": DEV_REQUIREMENTS,
    "docs": DOCS_REQUIREMENTS,
    "gui": GUI_REQUIREMENTS,
    "performance": PERFORMANCE_REQUIREMENTS,
    "extended": EXTENDED_REQUIREMENTS,
    "all": (
        DEV_REQUIREMENTS + 
        DOCS_REQUIREMENTS + 
        GUI_REQUIREMENTS + 
        PERFORMANCE_REQUIREMENTS + 
        EXTENDED_REQUIREMENTS
    ),
}

# Package configuration
setup(
    # Basic package information
    name="xpcs-toolkit",
    use_scm_version={
        "version_scheme": "post-release",
        "local_scheme": "node-and-timestamp",
        "write_to": "xpcs_toolkit/_version.py",
    },
    
    # Package metadata
    description="XPCS Toolkit: A comprehensive command-line tool for X-ray Photon Correlation Spectroscopy analysis",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    # Author and contact information
    author="Wei Chen",
    author_email="weichen@anl.gov",
    maintainer="Wei Chen",
    maintainer_email="weichen@anl.gov",
    
    # URLs and links
    url="https://github.com/imewei/xpcs-toolkit",
    project_urls={
        "Documentation": "https://xpcs-toolkit.readthedocs.io/",
        "Repository": "https://github.com/imewei/xpcs-toolkit.git",
        "Bug Reports": "https://github.com/imewei/xpcs-toolkit/issues",
        "Changelog": "https://github.com/imewei/xpcs-toolkit/blob/main/CHANGELOG.md",
        "Funding": "https://www.anl.gov/",
    },
    
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "*.tests", "*.tests.*"]),
    package_dir={"": "."},
    
    # Include additional files
    package_data={
        "xpcs_toolkit": [
            "configure/*.json",
            "py.typed",
        ],
        "xpcs_toolkit.configure": ["*.json"],
    },
    include_package_data=True,
    
    # Dependencies
    install_requires=CORE_REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Console scripts
    entry_points={
        "console_scripts": [
            "xpcs-toolkit=xpcs_toolkit.cli_headless:main",
            "xpcs=xpcs_toolkit.cli_headless:main",
        ],
    },
    
    # Package classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11", 
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Natural Language :: English",
        "Environment :: Console",
        "Environment :: X11 Applications",
    ],
    
    # Keywords for PyPI
    keywords=[
        "XPCS",
        "X-ray",
        "photon correlation spectroscopy",
        "synchrotron", 
        "visualization",
        "scientific computing",
        "materials science",
        "soft matter",
    ],
    
    # Licensing
    license="MIT",
    
    # Build requirements
    setup_requires=[
        "setuptools>=61.0",
        "setuptools_scm[toml]>=6.2",
        "wheel",
    ],
    
    # Test requirements (for python setup.py test)
    tests_require=[
        "pytest>=6.0.0",
        "pytest-cov>=3.0.0",
    ],
    
    # Zip safety
    zip_safe=False,
    
    # Additional metadata
    platforms=["any"],
)

# Post-installation message
def print_installation_message():
    """Print a helpful message after installation."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                   XPCS Toolkit Installed                  ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║  
║  Thank you for installing XPCS Toolkit!                  ║
║                                                           ║
║  Quick start:                                             ║
║    xpcs-toolkit --help                                    ║
║    xpcs --version                                         ║
║                                                           ║
║  Documentation: https://xpcs-toolkit.readthedocs.io/     ║
║  Issues: https://github.com/imewei/xpcs-toolkit/issues   ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    # Only print message if not in build/test mode
    if len(sys.argv) > 1 and sys.argv[1] not in ['egg_info', 'build', 'develop', 'test']:
        print_installation_message()