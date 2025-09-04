"""
Core business logic for XPCS Toolkit.

This package contains the fundamental analysis engines, data handling,
and domain models that form the heart of the XPCS analysis system.

To use core components, import directly from subpackages:
- from xpcs_toolkit.core.data.locator import DataFileLocator
- from xpcs_toolkit.core.analysis.kernel import AnalysisKernel
- from xpcs_toolkit.core.models import *
"""

from __future__ import annotations

# Note: This __init__.py is intentionally minimal to avoid circular imports.
# The subpackages (analysis, data, models) should be imported directly
# rather than through this parent package to ensure reliable imports
# across all environments including CI/CD systems.

__all__: list[str] = []
