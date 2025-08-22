from importlib.metadata import version, PackageNotFoundError
from xpcs_toolkit.xpcs_file import XpcsFile
from xpcs_toolkit.viewer_kernel import ViewerKernel
from xpcs_toolkit.file_locator import FileLocator

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
