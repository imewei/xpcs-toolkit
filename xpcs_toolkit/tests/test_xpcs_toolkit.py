#!/usr/bin/env python

"""Tests for `xpcs-toolkit` package (new name)."""

import pytest
import subprocess
import sys
import tempfile
import os
from pathlib import Path

# Import using the new package name
from xpcs_toolkit.viewer_kernel import ViewerKernel
from xpcs_toolkit.helper.listmodel import ListDataModel


def test_new_package_imports():
    """Test that the new package name imports work correctly."""
    import xpcs_toolkit
    assert xpcs_toolkit is not None
    assert hasattr(xpcs_toolkit, '__version__')
    assert hasattr(xpcs_toolkit, '__author__')


def test_cli_version_new_name():
    """Test that the new CLI commands show correct branding."""
    # Test xpcs-toolkit command
    result = subprocess.run(
        ["xpcs-toolkit", "--version"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        assert "xpcs-toolkit" in result.stdout.lower()
    
    # Test xpcs command  
    result = subprocess.run(
        ["xpcs", "--version"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        assert "xpcs-toolkit" in result.stdout.lower()


def test_cli_help_new_branding():
    """Test that the CLI shows new branding."""
    result = subprocess.run(
        [sys.executable, "-m", "xpcs_toolkit.cli_headless", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "XPCS Toolkit" in result.stdout
    assert "usage" in result.stdout.lower()


def test_backward_compatibility_warning():
    """Test that backward compatibility imports issue deprecation warnings."""
    import warnings
    import subprocess
    import sys
    
    # Test in a subprocess to ensure clean import
    result = subprocess.run(
        [sys.executable, "-c", "import warnings; warnings.simplefilter('always'); import pyxpcsviewer"],
        capture_output=True,
        text=True
    )
    
    # Check that a deprecation warning was issued
    assert "DeprecationWarning" in result.stderr
    assert "deprecated" in result.stderr.lower()
    assert "xpcs_toolkit" in result.stderr


def test_new_package_structure():
    """Test that the new package structure exists."""
    import xpcs_toolkit
    package_dir = Path(xpcs_toolkit.__file__).parent
    
    # Check that key files exist in new structure
    assert (package_dir / "cli_headless.py").exists()
    assert (package_dir / "viewer_kernel.py").exists()
    assert (package_dir / "xpcs_file.py").exists()
    assert (package_dir / "mpl_compat.py").exists()
    
    # Check that key modules exist
    assert (package_dir / "module").exists()
    assert (package_dir / "fileIO").exists()
    assert (package_dir / "helper").exists()


def test_api_equivalence():
    """Test that both old and new APIs provide the same functionality."""
    # Import using both names
    import pyxpcsviewer
    import xpcs_toolkit
    
    # Check that key classes are available from both
    assert hasattr(pyxpcsviewer, 'XpcsFile')
    assert hasattr(xpcs_toolkit, 'XpcsFile')
    assert hasattr(pyxpcsviewer, 'ViewerKernel')
    assert hasattr(xpcs_toolkit, 'ViewerKernel')
    
    # Check that they're the same classes
    assert pyxpcsviewer.XpcsFile is xpcs_toolkit.XpcsFile
    assert pyxpcsviewer.ViewerKernel is xpcs_toolkit.ViewerKernel


def test_module_imports_new_name():
    """Test that analysis modules work with new package name."""
    from xpcs_toolkit.module import g2mod, saxs1d
    from xpcs_toolkit.mpl_compat import PlotWidget, DataTreeWidget
    
    assert g2mod is not None
    assert saxs1d is not None
    assert PlotWidget is not None
    assert DataTreeWidget is not None


def test_cli_commands_available():
    """Test that all CLI commands are available."""
    result = subprocess.run(
        [sys.executable, "-m", "xpcs_toolkit.cli_headless", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    # Check that all expected commands are listed
    expected_commands = ["list", "saxs2d", "g2", "saxs1d", "stability"]
    for cmd in expected_commands:
        assert cmd in result.stdout


def test_version_consistency():
    """Test that version is consistent across old and new package names."""
    import pyxpcsviewer
    import xpcs_toolkit
    
    # Both should have the same version
    assert pyxpcsviewer.__version__ == xpcs_toolkit.__version__
