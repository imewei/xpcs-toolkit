"""
Core business logic for XPCS Toolkit.

This package contains the fundamental analysis engines, data handling,
and domain models that form the heart of the XPCS analysis system.
"""

from __future__ import annotations

from pathlib import Path

# Ensure subpackages are accessible by explicitly importing them
# This addresses CI environments where package structure isn't fully resolved
import sys

# Add the core directory to Python path to ensure subpackages are discoverable
_core_dir = Path(__file__).parent
if str(_core_dir) not in sys.path:
    sys.path.insert(0, str(_core_dir))

# Import subpackages with fallback handling
try:
    from . import analysis
except ImportError as e:
    import warnings

    warnings.warn(f"Could not import core.analysis: {e}", ImportWarning, stacklevel=2)

try:
    from . import data
except ImportError as e:
    import warnings

    warnings.warn(f"Could not import core.data: {e}", ImportWarning, stacklevel=2)

try:
    from . import models
except ImportError as e:
    import warnings

    warnings.warn(f"Could not import core.models: {e}", ImportWarning, stacklevel=2)

__all__: list[str] = []
