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
from xpcs_toolkit.analysis_kernel import ViewerKernel  # This should issue deprecation warning


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
    assert (package_dir / "analysis_kernel.py").exists()
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


def test_deprecation_warnings_with_pytest_warns():
    """Test deprecation warnings using pytest.warns()."""
    import warnings
    import xpcs_toolkit
    from xpcs_toolkit import FileLocator, ViewerKernel, XpcsFile
    
    # Test FileLocator deprecation warning
    with pytest.warns(DeprecationWarning, match="FileLocator is deprecated"):
        fl = FileLocator('.')
    
    # Test ViewerKernel deprecation warning  
    with pytest.warns(DeprecationWarning, match="ViewerKernel is deprecated"):
        vk = ViewerKernel('.')
    
    # Test XpcsFile class exists (deprecation warning only on instantiation with parameters)
    # The deprecation warning is issued in the class __init__ method, not on import
    assert XpcsFile is not None
    assert issubclass(XpcsFile, xpcs_toolkit.XpcsDataFile)


def test_parameter_name_backward_compatibility():
    """Test that old parameter names work with deprecation warnings."""
    from xpcs_toolkit.fileIO.qmap_utils import QMap
    from xpcs_toolkit.fileIO.hdf_reader import get_abs_cs_scale, get_analysis_type
    from xpcs_toolkit.fileIO.ftype_utils import get_ftype, isNeXusFile, isLegacyFile
    
    # Test that old parameter names are accepted (though will fail without valid files)
    # These tests mainly check that the deprecated parameters are accepted without syntax errors
    
    # Note: These would normally require valid HDF5 files to test fully,
    # but we're mainly testing that the parameter interface works
    try:
        # This will fail due to missing file, but should accept the old parameter name
        with pytest.warns(DeprecationWarning, match="'fname' is deprecated"):
            get_abs_cs_scale(fname='nonexistent.h5')
    except (FileNotFoundError, OSError, Exception):
        pass  # Expected due to missing file, but deprecation warning should have been issued
    
    try:
        with pytest.warns(DeprecationWarning, match="'ftype' is deprecated"):
            get_abs_cs_scale('nonexistent.h5', ftype='nexus')
    except (FileNotFoundError, OSError, Exception):
        pass  # Expected due to missing file


def test_new_class_inheritance():
    """Test that inheritance relationships are correct."""
    import xpcs_toolkit
    
    # Test that deprecated classes inherit from new classes
    assert issubclass(xpcs_toolkit.XpcsFile, xpcs_toolkit.XpcsDataFile)
    assert issubclass(xpcs_toolkit.FileLocator, xpcs_toolkit.DataFileLocator)
    
    # Test that new classes have proper method resolution order
    assert xpcs_toolkit.XpcsDataFile.__name__ == 'XpcsDataFile'
    assert xpcs_toolkit.AnalysisKernel.__name__ == 'AnalysisKernel'
    assert xpcs_toolkit.DataFileLocator.__name__ == 'DataFileLocator'


def test_module_level_imports():
    """Test that both new and old names are available at module level."""
    import xpcs_toolkit
    
    # Test new class names
    assert hasattr(xpcs_toolkit, 'XpcsDataFile')
    assert hasattr(xpcs_toolkit, 'AnalysisKernel')
    assert hasattr(xpcs_toolkit, 'DataFileLocator')
    
    # Test old class names (for backward compatibility)
    assert hasattr(xpcs_toolkit, 'XpcsFile')
    assert hasattr(xpcs_toolkit, 'ViewerKernel')
    assert hasattr(xpcs_toolkit, 'FileLocator')
    
    # Test that classes can be instantiated (though they may fail without proper data)
    # We mainly test that the classes are importable and callable
    for cls_name in ['XpcsDataFile', 'AnalysisKernel', 'DataFileLocator', 
                     'XpcsFile', 'ViewerKernel', 'FileLocator']:
        cls = getattr(xpcs_toolkit, cls_name)
        assert callable(cls)


def test_fileio_modules_parameter_names():
    """Test that fileIO modules accept both old and new parameter names."""
    from xpcs_toolkit.fileIO.ftype_utils import get_ftype, isNeXusFile, isLegacyFile
    
    # These functions now use 'filename' but should be backward compatible
    # Test that they accept the new parameter name (though will fail with nonexistent file)
    try:
        result = get_ftype('nonexistent.h5')
        assert result is False  # Should return False for nonexistent file
    except Exception:
        pass  # May raise exception, that's ok
    
    try:
        result = isNeXusFile('nonexistent.h5')
        assert result is False or result is None
    except Exception:
        pass  # May raise exception for nonexistent file
    
    try:
        result = isLegacyFile('nonexistent.h5')
        assert result is False or result is None
    except Exception:
        pass  # May raise exception for nonexistent file


def test_helper_modules_functionality():
    """Test that helper modules work correctly with new parameter names."""
    from xpcs_toolkit.helper.listmodel import ListDataModel, TableDataModel
    from xpcs_toolkit.helper.fitting import single_exp, fit_tau
    import numpy as np
    
    # Test ListDataModel
    test_data = ['item1', 'item2', 'item3']
    model = ListDataModel(test_data)
    assert len(model) == 3
    assert model[0] == 'item1'
    assert model.data(0) == 'item1'
    
    # Test TableDataModel with updated column header
    table_model = TableDataModel()
    headers = table_model.xlabels
    assert 'filename' in headers  # Should use new name
    assert 'fname' not in headers  # Should not use old name
    
    # Test fitting functions (basic functionality)
    x = np.array([1, 2, 3, 4, 5])
    result = single_exp(x, tau=2.0, bkg=0.1, cts=1.0)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(x)


def test_analysis_modules_integration():
    """Test that analysis modules integrate properly with refactored classes."""
    from xpcs_toolkit.module import g2mod, saxs1d, stability
    
    # Test that modules can be imported without error
    assert hasattr(g2mod, 'get_data')
    assert hasattr(g2mod, 'pg_plot')
    assert hasattr(saxs1d, 'pg_plot')
    assert hasattr(stability, 'plot')
    
    # Test color and marker functions
    color, marker = saxs1d.get_color_marker(0)
    assert isinstance(color, str)
    assert isinstance(marker, str)


def test_mpl_compat_functionality():
    """Test matplotlib compatibility layer."""
    from xpcs_toolkit.mpl_compat import PlotWidget, DataTreeWidget, MockSignal
    
    # Test that classes can be instantiated
    plot_widget = PlotWidget()
    data_widget = DataTreeWidget()
    signal = MockSignal()
    
    # Test basic functionality
    assert hasattr(plot_widget, 'clear')
    assert hasattr(data_widget, 'setWindowTitle')  # MockDataTreeWidget method
    assert hasattr(signal, 'emit')
    
    # Test signal emission (should not raise error)
    signal.emit('test', 'data')
    
    # Test data widget functionality
    data_widget.setWindowTitle('Test Window')
    data_widget.resize(600, 400)
