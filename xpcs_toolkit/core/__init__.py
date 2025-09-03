"""
Core business logic for XPCS Toolkit.

This package contains the fundamental analysis engines, data handling,
and domain models that form the heart of the XPCS analysis system.
"""

from __future__ import annotations

# Flag to prevent recursive imports during module loading
_loading_data = False

def __getattr__(name: str):
    """Lazy loading for core submodules to avoid circular imports."""
    global _loading_data
    
    if name == "data":
        if _loading_data:
            raise AttributeError(f"Circular import detected for '{name}'")
        
        _loading_data = True
        try:
            from . import data
            return data
        finally:
            _loading_data = False
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__: list[str] = [
    "data",
]
