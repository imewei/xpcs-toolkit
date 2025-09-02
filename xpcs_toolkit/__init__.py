"""
XPCS Toolkit - Modern Python package for XPCS analysis.

This comprehensive toolkit provides advanced visualization and analysis capabilities
for X-ray Photon Correlation Spectroscopy (XPCS) datasets, with specialized support
for the customized NeXus file format developed at Argonne National Laboratory's
Advanced Photon Source beamline 8-ID-I.

The toolkit has been reorganized into a modern, maintainable structure while
maintaining full backward compatibility with existing code.

## Usage

```python
# Existing imports continue to work
from xpcs_toolkit import XpcsDataFile, AnalysisKernel, DataFileLocator

# Load and analyze data
data_file = XpcsDataFile('experiment.hdf')
kernel = AnalysisKernel('/path/to/data')
```

## New Modular Structure

The codebase has been reorganized into logical modules:

- `core/`: Core analysis engines and data handling
- `scientific/`: Specialized scientific analysis modules
- `io/`: Input/output operations and file format handling
- `cli/`: Command-line interface
- `utils/`: Utilities and helper functions
- `config.py`: Centralized configuration management

For new development, prefer importing from specific modules, but all
existing imports remain functional.

"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .analysis_kernel import AnalysisKernel, ViewerKernel

# Configuration system (new)
from .config import XpcsConfig, get_config, reset_config, set_config
from .data_file_locator import DataFileLocator, FileLocator

# Keep existing imports working (from original locations)
from .xpcs_file import XpcsDataFile, XpcsFile

# Public API exports
__all__ = [
    # Core classes
    "XpcsDataFile",
    "AnalysisKernel",
    "DataFileLocator",
    # Deprecated aliases (for compatibility)
    "XpcsFile",
    "ViewerKernel",
    "FileLocator",
    # Configuration (new)
    "XpcsConfig",
    "get_config",
    "set_config",
    "reset_config",
]

# Version handling
try:
    __version__ = version("xpcs-toolkit")
except PackageNotFoundError:
    try:
        __version__ = version("pyxpcsviewer")  # Backward compatibility
    except PackageNotFoundError:
        __version__ = "0.1.0"  # Fallback

__author__ = "Wei Chen"
__credits__ = "Argonne National Laboratory"
