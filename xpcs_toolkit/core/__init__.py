"""
Core business logic for XPCS Toolkit.

This package contains the fundamental analysis engines, data handling,
and domain models that form the heart of the XPCS analysis system.
"""

from __future__ import annotations

def __getattr__(name: str):
    """Lazy loading for core submodules to avoid circular imports."""
    if name == "data":
        from . import data
        return data
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__: list[str] = [
    "data",
]
