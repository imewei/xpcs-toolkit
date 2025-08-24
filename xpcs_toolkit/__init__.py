"""
XPCS Toolkit - Python-based Interactive Visualization Tool for XPCS Datasets

This comprehensive toolkit provides advanced visualization and analysis capabilities
for X-ray Photon Correlation Spectroscopy (XPCS) datasets, with specialized support
for the customized NeXus file format developed at Argonne National Laboratory's
Advanced Photon Source beamline 8-ID-I.

## Key Features

### Multi-tau Correlation Analysis
- Complete g2 correlation function analysis and visualization
- Time-delay correlation analysis with customizable q-ranges
- Advanced fitting capabilities for relaxation dynamics
- Statistical error analysis and uncertainty quantification

### Two-time Correlation Analysis  
- Two-dimensional correlation function visualization
- Time-resolved dynamics analysis for non-equilibrium systems
- Interactive exploration of speckle pattern evolution
- Support for aging and non-stationary processes

### Interactive Data Visualization
- 2D SAXS pattern visualization with ROI selection
- 1D radial averaging with customizable integration sectors
- Real-time intensity time-series monitoring
- Beam stability analysis and quality assessment

### File Format Support
- Native support for APS 8-ID-I NeXus format
- Legacy HDF5 format compatibility
- Automated file type detection and handling
- Efficient large dataset management

### Analysis Modules
- Comprehensive correlation analysis (g2mod)
- Small-angle X-ray scattering analysis (saxs1d, saxs2d)
- Two-time correlation analysis (twotime)
- Tau vs q fitting and analysis (tauq)
- Beam stability monitoring (stability)
- Intensity time-series analysis (intt)

### Command-line Interface
- Headless operation for batch processing
- Automated plot generation and export
- Integration with analysis pipelines
- Configurable output formats and parameters

## Usage Examples

```python
# Interactive analysis
from xpcs_toolkit import XpcsDataFile, AnalysisKernel

# Load XPCS data file
xpcs_data = XpcsDataFile('experiment_data.hdf')
print(f"Analysis types: {xpcs_data.analysis_type}")

# Create analysis kernel for batch processing
kernel = AnalysisKernel('/path/to/data/directory')
kernel.build()

# Extract g2 correlation data
q_vals, t_vals, g2, g2_err, labels = xpcs_data.get_g2_data(q_range=(0.01, 0.1))
```

## Supported Analysis Types

- **Multi-tau**: Traditional multi-tau correlation analysis
- **Two-time**: Two-dimensional correlation analysis
- **Mixed**: Files containing both analysis types

## Installation and Setup

The toolkit supports both interactive Python-based analysis and headless
command-line operation, making it suitable for both exploratory data
analysis and automated processing workflows.

"""

from importlib.metadata import version, PackageNotFoundError
# New classes (recommended)
from xpcs_toolkit.xpcs_file import XpcsDataFile
from xpcs_toolkit.analysis_kernel import AnalysisKernel  
from xpcs_toolkit.data_file_locator import DataFileLocator
# Backward compatibility aliases (deprecated)
from xpcs_toolkit.xpcs_file import XpcsFile
from xpcs_toolkit.analysis_kernel import ViewerKernel
from xpcs_toolkit.data_file_locator import FileLocator

# Explicit exports
__all__ = [
    # New classes (recommended)
    'XpcsDataFile',
    'AnalysisKernel', 
    'DataFileLocator',
    # Backward compatibility aliases
    'XpcsFile',
    'ViewerKernel',
    'FileLocator',
]

# Version handling - try both old and new package names for compatibility
try:
    __version__ = version("xpcs-toolkit")
except PackageNotFoundError:
    try:
        __version__ = version("pyxpcsviewer")  # Backward compatibility
    except PackageNotFoundError:
        __version__ = "0.1.0"  # Fallback if package is not installed

__author__ = 'Wei Chen'
__credits__ = 'Argonne National Laboratory'
