"""
Backward compatibility module for XPCS Toolkit.

This module provides import redirections and compatibility wrappers
to ensure existing code continues to work with the reorganized structure.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any
import warnings


def _create_module_redirect(old_path: str, new_path: str) -> Any:
    """
    Create a module redirect for backward compatibility.

    Parameters
    ----------
    old_path : str
        Old module path (e.g., 'xpcs_toolkit.analysis_kernel')
    new_path : str
        New module path (e.g., 'xpcs_toolkit.core.analysis.kernel')
    """

    def _import_redirect() -> Any:
        warnings.warn(
            f"Import from '{old_path}' is deprecated. Please use '{new_path}' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        # Import the new module
        parts = new_path.split(".")
        module = __import__(new_path)
        for part in parts[1:]:
            module = getattr(module, part)
        return module

    return _import_redirect


# Module redirection mappings
MODULE_REDIRECTS = {
    "xpcs_toolkit.analysis_kernel": "xpcs_toolkit.core.analysis.kernel",
    "xpcs_toolkit.xpcs_file": "xpcs_toolkit.core.data.file",
    "xpcs_toolkit.data_file_locator": "xpcs_toolkit.core.data.locator",
    "xpcs_toolkit.cli_headless": "xpcs_toolkit.cli.headless",
    # Scientific modules
    "xpcs_toolkit.module.g2mod": "xpcs_toolkit.scientific.correlation.g2",
    "xpcs_toolkit.module.twotime": "xpcs_toolkit.scientific.correlation.twotime",
    "xpcs_toolkit.module.saxs1d": "xpcs_toolkit.scientific.scattering.saxs_1d",
    "xpcs_toolkit.module.saxs2d": "xpcs_toolkit.scientific.scattering.saxs_2d",
    "xpcs_toolkit.module.tauq": "xpcs_toolkit.scientific.dynamics.tauq",
    "xpcs_toolkit.module.intt": "xpcs_toolkit.scientific.dynamics.intensity",
    "xpcs_toolkit.module.stability": "xpcs_toolkit.scientific.dynamics.stability",
    "xpcs_toolkit.module.average_toolbox": "xpcs_toolkit.scientific.processing.averaging",
    # Helper modules
    "xpcs_toolkit.helper.logwriter": "xpcs_toolkit.utils.logging.writer",
    "xpcs_toolkit.helper.logging_config": "xpcs_toolkit.utils.logging.config",
    "xpcs_toolkit.helper.logging_utils": "xpcs_toolkit.utils.logging.handlers",
    "xpcs_toolkit.helper.fitting": "xpcs_toolkit.utils.math.fitting",
    "xpcs_toolkit.helper.utils": "xpcs_toolkit.utils.common.helpers",
    # FileIO modules
    "xpcs_toolkit.fileIO.hdf_reader": "xpcs_toolkit.io.formats.hdf5.reader",
    "xpcs_toolkit.fileIO.lazy_hdf_reader": "xpcs_toolkit.io.formats.hdf5.lazy_reader",
    "xpcs_toolkit.fileIO.ftype_utils": "xpcs_toolkit.io.formats.detection",
    "xpcs_toolkit.fileIO.qmap_utils": "xpcs_toolkit.io.cache.qmap_cache",
    # Other modules
    "xpcs_toolkit.mpl_compat": "xpcs_toolkit.utils.compatibility.matplotlib",
    "xpcs_toolkit._lazy_imports": "xpcs_toolkit.utils.common.lazy_imports",
}


def install_compatibility_hooks() -> None:
    """
    Install import hooks for backward compatibility.

    This function should be called early in the application startup
    to ensure all imports are properly redirected.
    """
    import importlib

    class CompatibilityFinder:
        """Custom import finder for backward compatibility."""

        def find_spec(
            self,
            fullname: str,
            path: Any | None = None,
            target: Any | None = None,
        ) -> Any | None:
            if fullname in MODULE_REDIRECTS:
                # Redirect to new module
                new_name = MODULE_REDIRECTS[fullname]
                warnings.warn(
                    f"Import of '{fullname}' is deprecated. "
                    f"Please use '{new_name}' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return importlib.util.find_spec(new_name)
            return None

    # Install the finder
    sys.meta_path.insert(0, CompatibilityFinder())


# Class name redirections for commonly used classes
CLASS_ALIASES = {
    "AnalysisKernel": ("xpcs_toolkit.core.analysis.kernel", "AnalysisKernel"),
    "ViewerKernel": (
        "xpcs_toolkit.core.analysis.kernel",
        "ViewerKernel",
    ),  # Deprecated alias
    "XpcsFile": ("xpcs_toolkit.core.data.file", "XpcsFile"),
    "DataFileLocator": ("xpcs_toolkit.core.data.locator", "DataFileLocator"),
    "FileLocator": (
        "xpcs_toolkit.core.data.locator",
        "FileLocator",
    ),  # Deprecated alias
}


def get_compatibility_class(name: str) -> Any:
    """
    Get a class by its old name for backward compatibility.

    Parameters
    ----------
    name : str
        Old class name

    Returns
    -------
    class
        The requested class
    """
    if name in CLASS_ALIASES:
        module_path, class_name = CLASS_ALIASES[name]
        warnings.warn(
            f"Access to '{name}' through compatibility layer is deprecated. "
            f"Please import from '{module_path}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    raise AttributeError(f"No compatibility mapping for class '{name}'")


# Maintain backward compatibility for old import patterns
def __getattr__(name: str) -> Any:
    """Module-level __getattr__ for backward compatibility."""
    if name in CLASS_ALIASES:
        return get_compatibility_class(name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
