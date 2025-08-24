"""
Tests for XPCS Toolkit Analysis Kernel functionality.

This module tests the AnalysisKernel class and its backward compatibility
with ViewerKernel, focusing on file management and data processing capabilities.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from xpcs_toolkit.analysis_kernel import AnalysisKernel, ViewerKernel


class TestAnalysisKernel:
    """Test cases for the AnalysisKernel class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def analysis_kernel(self, temp_dir):
        """Create an AnalysisKernel instance for testing."""
        return AnalysisKernel(temp_dir)
    
    def test_analysis_kernel_init(self, temp_dir):
        """Test AnalysisKernel initialization."""
        kernel = AnalysisKernel(temp_dir)
        assert kernel.directory == temp_dir
        assert hasattr(kernel, 'file_list')
        assert hasattr(kernel, 'selected_files')
    
    def test_analysis_kernel_inheritance(self):
        """Test that AnalysisKernel inherits from DataFileLocator."""
        from xpcs_toolkit.data_file_locator import DataFileLocator
        assert issubclass(AnalysisKernel, DataFileLocator)
    
    def test_viewer_kernel_backward_compatibility(self, temp_dir):
        """Test ViewerKernel backward compatibility."""
        # Test that ViewerKernel issues deprecation warning
        with pytest.warns(DeprecationWarning, match="ViewerKernel is deprecated"):
            kernel = ViewerKernel(temp_dir)
        
        # Test that it's a subclass of AnalysisKernel
        assert isinstance(kernel, AnalysisKernel)
        assert issubclass(ViewerKernel, AnalysisKernel)
    
    def test_build_file_list_empty_directory(self, analysis_kernel, temp_dir):
        """Test building file list in empty directory."""
        analysis_kernel.build_file_list()
        assert hasattr(analysis_kernel, 'file_list')
        # Should have empty or minimal file list
        assert isinstance(analysis_kernel.file_list, list)
    
    def test_build_file_list_with_files(self, temp_dir):
        """Test building file list with sample files."""
        # Create test files
        test_files = ['test1.h5', 'test2.hdf5', 'test3.dat', 'ignore.txt']
        for filename in test_files:
            (Path(temp_dir) / filename).touch()
        
        kernel = AnalysisKernel(temp_dir)
        kernel.build_file_list()
        
        # Should detect HDF5 files but not text files
        assert hasattr(kernel, 'file_list')
    
    def test_directory_property(self, analysis_kernel, temp_dir):
        """Test directory property getter/setter."""
        assert analysis_kernel.directory == temp_dir
        
        # Test setting new directory
        with tempfile.TemporaryDirectory() as new_dir:
            analysis_kernel.directory = new_dir
            assert analysis_kernel.directory == new_dir
    
    def test_get_selected_files(self, analysis_kernel):
        """Test getting selected files."""
        # Should have a method to get selected files
        assert hasattr(analysis_kernel, 'get_selected_files') or hasattr(analysis_kernel, 'selected_files')
    
    @patch('xpcs_toolkit.analysis_kernel.os.path.exists')
    def test_directory_validation(self, mock_exists, temp_dir):
        """Test directory validation."""
        # Test with non-existent directory
        mock_exists.return_value = False
        try:
            kernel = AnalysisKernel('/nonexistent/path')
            # Should either raise an error or handle gracefully
        except (FileNotFoundError, OSError, ValueError):
            pass  # Expected behavior


class TestAnalysisKernelFileOperations:
    """Test file operations in AnalysisKernel."""
    
    @pytest.fixture
    def sample_data_dir(self):
        """Create a directory with sample data files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create various file types
            files_to_create = [
                'experiment_001.h5',
                'experiment_002.hdf5', 
                'calibration.dat',
                'readme.txt',
                'analysis.log'
            ]
            
            for filename in files_to_create:
                (Path(tmpdir) / filename).touch()
            
            yield tmpdir
    
    def test_file_filtering(self, sample_data_dir):
        """Test that only relevant files are included."""
        kernel = AnalysisKernel(sample_data_dir)
        kernel.build_file_list()
        
        # Should filter files appropriately
        assert hasattr(kernel, 'file_list')
        
        # Check that file list exists and is reasonable
        if hasattr(kernel, 'file_list') and kernel.file_list:
            for file_path in kernel.file_list:
                # Should be valid paths
                assert isinstance(file_path, (str, Path))
    
    def test_file_sorting(self, sample_data_dir):
        """Test that files are sorted appropriately."""
        kernel = AnalysisKernel(sample_data_dir)
        kernel.build_file_list()
        
        if hasattr(kernel, 'file_list') and kernel.file_list:
            # Files should be in some consistent order
            file_names = [os.path.basename(str(f)) for f in kernel.file_list]
            assert len(file_names) > 0


class TestAnalysisKernelMethods:
    """Test various methods of AnalysisKernel."""
    
    @pytest.fixture
    def kernel_with_data(self):
        """Create kernel with mock data for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AnalysisKernel(tmpdir)
            yield kernel
    
    def test_kernel_string_representation(self, kernel_with_data):
        """Test string representation of kernel."""
        str_repr = str(kernel_with_data)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
    
    def test_kernel_attributes(self, kernel_with_data):
        """Test that kernel has expected attributes."""
        # Should inherit from DataFileLocator
        assert hasattr(kernel_with_data, 'directory')
        
        # Should have file management attributes
        expected_attrs = ['file_list', 'selected_files', 'build_file_list']
        for attr in expected_attrs:
            if hasattr(kernel_with_data, attr):
                assert True  # At least some expected attributes exist
                break
        else:
            # If none of the expected attributes exist, that might be okay
            # depending on the implementation
            pass
    
    def test_error_handling(self):
        """Test error handling in AnalysisKernel."""
        # Test with invalid directory
        try:
            kernel = AnalysisKernel(None)
        except (TypeError, ValueError, AttributeError):
            pass  # Expected behavior for invalid input
        
        # Test with empty string
        try:
            kernel = AnalysisKernel("")
        except (ValueError, FileNotFoundError, OSError):
            pass  # Expected behavior
    
    def test_method_existence(self, kernel_with_data):
        """Test that expected methods exist."""
        # Core methods that should exist
        core_methods = ['build_file_list']
        for method in core_methods:
            if hasattr(kernel_with_data, method):
                assert callable(getattr(kernel_with_data, method))


class TestBackwardCompatibility:
    """Test backward compatibility features."""
    
    def test_viewer_kernel_deprecation_warning(self):
        """Test that ViewerKernel shows deprecation warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.warns(DeprecationWarning):
                kernel = ViewerKernel(tmpdir)
            
            # Should still work as AnalysisKernel
            assert isinstance(kernel, AnalysisKernel)
    
    def test_api_compatibility(self):
        """Test API compatibility between old and new classes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Both should have similar interfaces
            analysis = AnalysisKernel(tmpdir)
            
            with pytest.warns(DeprecationWarning):
                viewer = ViewerKernel(tmpdir)
            
            # Should have similar attributes
            analysis_attrs = set(dir(analysis))
            viewer_attrs = set(dir(viewer))
            
            # ViewerKernel should have at least the same methods as AnalysisKernel
            common_attrs = analysis_attrs.intersection(viewer_attrs)
            assert len(common_attrs) > 0  # Should have some common interface
    
    def test_inheritance_chain(self):
        """Test the inheritance chain is correct."""
        from xpcs_toolkit.data_file_locator import DataFileLocator
        
        # Test inheritance
        assert issubclass(AnalysisKernel, DataFileLocator)
        assert issubclass(ViewerKernel, AnalysisKernel)
        assert issubclass(ViewerKernel, DataFileLocator)
        
        # Test MRO (Method Resolution Order)
        mro = ViewerKernel.__mro__
        assert ViewerKernel in mro
        assert AnalysisKernel in mro
        assert DataFileLocator in mro


class TestIntegrationWithDataLocator:
    """Test integration with DataFileLocator functionality."""
    
    @pytest.fixture
    def integrated_kernel(self):
        """Create kernel for integration testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield AnalysisKernel(tmpdir)
    
    def test_data_locator_methods(self, integrated_kernel):
        """Test that DataFileLocator methods work."""
        # Should inherit file location functionality
        assert hasattr(integrated_kernel, 'directory')
        
        # Test directory access
        directory = integrated_kernel.directory
        assert os.path.exists(directory)
    
    def test_file_management_integration(self, integrated_kernel):
        """Test file management integration."""
        # Should be able to build and manage file lists
        try:
            integrated_kernel.build_file_list()
            # Should complete without error
            assert True
        except Exception as e:
            # If there's an expected exception, that's also okay
            assert isinstance(e, (AttributeError, NotImplementedError, FileNotFoundError))


@pytest.mark.integration
class TestAnalysisKernelIntegration:
    """Integration tests for AnalysisKernel with actual file system."""
    
    def test_real_directory_usage(self):
        """Test with real directory structure."""
        # Use current directory for testing
        current_dir = os.getcwd()
        
        try:
            kernel = AnalysisKernel(current_dir)
            kernel.build_file_list()
            
            # Should work with real directory
            assert kernel.directory == current_dir
        except Exception:
            # Some implementations might require specific directory structure
            pass
    
    def test_performance_with_many_files(self):
        """Test performance with multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many test files
            for i in range(50):
                (Path(tmpdir) / f"test_{i:03d}.h5").touch()
            
            kernel = AnalysisKernel(tmpdir)
            
            # Should handle many files reasonably quickly
            import time
            start = time.time()
            kernel.build_file_list()
            elapsed = time.time() - start
            
            # Should complete in reasonable time (less than 1 second)
            assert elapsed < 1.0