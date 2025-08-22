#!/usr/bin/env python

"""Tests for `xpcs-toolkit` package (new name)."""

import pytest
import subprocess
import sys
import tempfile
import os
from pathlib import Path

# Import using the new package name
from xpcs_toolkit.analysis_kernel import AnalysisKernel
from xpcs_toolkit.helper.listmodel import ListDataModel

# Also import old class names to test backward compatibility
from xpcs_toolkit.viewer_kernel import ViewerKernel  # This should issue deprecation warning


def test_new_package_imports():
    """Test that the new package name imports work correctly."""
    import xpcs_toolkit
    assert xpcs_toolkit is not None
    assert hasattr(xpcs_toolkit, '__version__')
    assert hasattr(xpcs_toolkit, '__author__')


def test_cli_version_new_name():
    """Test that the new CLI commands show correct branding (if installed as console scripts)."""
    import shutil
    
    # Test xpcs-toolkit command if available
    if shutil.which("xpcs-toolkit"):
        result = subprocess.run(
            ["xpcs-toolkit", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            assert "xpcs-toolkit" in result.stdout.lower()
    
    # Test xpcs command if available  
    if shutil.which("xpcs"):
        result = subprocess.run(
            ["xpcs", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            assert "xpcs-toolkit" in result.stdout.lower()
    
    # At minimum, test the direct module invocation
    result = subprocess.run(
        [sys.executable, "-m", "xpcs_toolkit.cli_headless", "--version"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
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
    
    # Test in a subprocess to ensure clean import and instantiation
    test_code = """
import warnings
warnings.simplefilter('always')
from xpcs_toolkit import FileLocator, ViewerKernel
# Instantiate to trigger deprecation warnings
try:
    fl = FileLocator('.')
    vk = ViewerKernel('.')
except:
    pass  # Don't care about errors, just want warnings
"""
    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        text=True
    )
    
    # Check that a deprecation warning was issued (warnings go to stderr)
    assert "DeprecationWarning" in result.stderr
    assert "deprecated" in result.stderr.lower()


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
    import xpcs_toolkit
    
    # Check that key classes are available (both old and new names)
    assert hasattr(xpcs_toolkit, 'XpcsFile')  # Old name for backward compatibility
    assert hasattr(xpcs_toolkit, 'XpcsDataFile')  # New name
    assert hasattr(xpcs_toolkit, 'ViewerKernel')  # Old name for backward compatibility
    assert hasattr(xpcs_toolkit, 'AnalysisKernel')  # New name
    
    # Check that old and new classes are available
    assert hasattr(xpcs_toolkit, 'DataFileLocator')  # New name
    assert hasattr(xpcs_toolkit, 'FileLocator')  # Old name for backward compatibility
    
    # Test that the classes are properly related
    assert issubclass(xpcs_toolkit.XpcsFile, xpcs_toolkit.XpcsDataFile)
    assert issubclass(xpcs_toolkit.FileLocator, xpcs_toolkit.DataFileLocator)


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


def test_new_class_functionality():
    """Test that the new classes have expected functionality."""
    import xpcs_toolkit
    
    # Test that the new classes exist and are callable
    XpcsDataFile = xpcs_toolkit.XpcsDataFile
    AnalysisKernel = xpcs_toolkit.AnalysisKernel
    DataFileLocator = xpcs_toolkit.DataFileLocator
    
    assert callable(XpcsDataFile)
    assert callable(AnalysisKernel)
    assert callable(DataFileLocator)
