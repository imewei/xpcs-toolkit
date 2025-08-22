from importlib.metadata import version, PackageNotFoundError
# New classes (recommended)
from xpcs_toolkit.xpcs_file import XpcsDataFile
from xpcs_toolkit.analysis_kernel import AnalysisKernel  
from xpcs_toolkit.data_file_locator import DataFileLocator
# Backward compatibility aliases (deprecated)
from xpcs_toolkit.xpcs_file import XpcsFile
from xpcs_toolkit.analysis_kernel import ViewerKernel
from xpcs_toolkit.data_file_locator import FileLocator

# Version handling - try both old and new package names for compatibility
try:
    __version__ = version("xpcs-toolkit")
except PackageNotFoundError:
    try:
        __version__ = version("pyxpcsviewer")  # Backward compatibility
    except PackageNotFoundError:
        __version__ = "0.1.0"  # Fallback if package is not installed

__author__ = 'Miaoqi Chu'
__credits__ = 'Argonne National Laboratory'
