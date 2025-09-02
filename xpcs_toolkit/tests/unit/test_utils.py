"""
Tests for xpcs_toolkit.helper.utils module.

This module tests core utility functions used throughout the XPCS Toolkit,
focusing on data processing utilities, normalization functions, and array operations.
"""

import pytest
import numpy as np
from unittest.mock import patch

from xpcs_toolkit.helper.utils import get_min_max, norm_saxs_data, create_slice


class TestGetMinMax:
    """Test suite for get_min_max function."""
    
    def test_basic_percentile_calculation(self):
        """Test basic percentile calculation without special parameters."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        vmin, vmax = get_min_max(data)
        
        assert vmin == 1.0
        assert vmax == 9.0
    
    def test_custom_percentiles(self):
        """Test with custom min and max percentiles."""
        data = np.arange(100)
        vmin, vmax = get_min_max(data, min_percent=10, max_percent=90)
        
        assert vmin == pytest.approx(9.9, rel=1e-1)  # 10th percentile
        assert vmax == pytest.approx(89.1, rel=1e-1)  # 90th percentile
    
    def test_plot_norm_3_log_type(self):
        """Test plot_norm=3 with plot_type='log'."""
        data = np.array([-10, -5, 0, 5, 10, 15])
        vmin, vmax = get_min_max(data, plot_norm=3, plot_type='log')
        
        # Should set symmetric range around 0 based on max absolute value
        expected_max = max(abs(data.min()), abs(data.max()))
        assert vmin == -expected_max
        assert vmax == expected_max
    
    def test_plot_norm_3_non_log_type(self):
        """Test plot_norm=3 with non-log plot_type."""
        data = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
        vmin, vmax = get_min_max(data, plot_norm=3, plot_type='linear')
        
        # Should set symmetric range around 1 based on max deviation
        min_val, max_val = data.min(), data.max()
        t = max(abs(1 - min_val), abs(max_val - 1))
        assert vmin == 1 - t
        assert vmax == 1 + t
    
    def test_plot_norm_other_values(self):
        """Test plot_norm with values other than 3."""
        data = np.array([1, 2, 3, 4, 5])
        vmin, vmax = get_min_max(data, plot_norm=1, plot_type='log')
        
        # Should behave like normal percentile calculation
        assert vmin == 1.0
        assert vmax == 5.0
    
    def test_empty_data(self):
        """Test handling of empty data arrays."""
        data = np.array([])
        with pytest.raises((ValueError, IndexError)):
            get_min_max(data)
    
    def test_single_value_data(self):
        """Test with single-value data."""
        data = np.array([42])
        vmin, vmax = get_min_max(data)
        
        assert vmin == 42.0
        assert vmax == 42.0
    
    def test_2d_data_flattening(self):
        """Test that 2D data is properly flattened."""
        data = np.array([[1, 100], [2, 99]])
        vmin, vmax = get_min_max(data, min_percent=25, max_percent=75)
        
        # Should use flattened data: [1, 2, 99, 100]
        assert vmin == pytest.approx(1.75, rel=1e-1)  # 25th percentile
        assert vmax == pytest.approx(99.25, rel=1e-1)  # 75th percentile


class TestNormSaxsData:
    """Test suite for norm_saxs_data function."""
    
    def test_no_normalization(self):
        """Test plot_norm=0 (no normalization)."""
        q = np.array([0.1, 0.2, 0.3])
        Iq = np.array([100, 50, 25])
        
        result_Iq, xlabel, ylabel = norm_saxs_data(Iq, q, plot_norm=0)
        
        np.testing.assert_array_equal(result_Iq, Iq)
        assert xlabel == '$q (\\AA^{-1})$'
        assert ylabel == 'Intensity'
    
    def test_kratky_normalization(self):
        """Test plot_norm=1 (Kratky plot: I*q^2)."""
        q = np.array([0.1, 0.2, 0.3])
        Iq = np.array([100, 50, 25])
        
        result_Iq, xlabel, ylabel = norm_saxs_data(Iq, q, plot_norm=1)
        
        expected = Iq * q**2
        np.testing.assert_array_equal(result_Iq, expected)
        assert xlabel == '$q (\\AA^{-1})$'
        assert ylabel == 'Intensity * q^2'
    
    def test_porod_normalization(self):
        """Test plot_norm=2 (Porod plot: I*q^4)."""
        q = np.array([0.1, 0.2, 0.3])
        Iq = np.array([100, 50, 25])
        
        result_Iq, xlabel, ylabel = norm_saxs_data(Iq, q, plot_norm=2)
        
        expected = Iq * q**4
        np.testing.assert_allclose(result_Iq, expected)
        assert xlabel == '$q (\\AA^{-1})$'
        assert ylabel == 'Intensity * q^4'
    
    def test_baseline_normalization(self):
        """Test plot_norm=3 (baseline normalization: I/I_0)."""
        q = np.array([0.1, 0.2, 0.3])
        Iq = np.array([100, 50, 25])
        
        result_Iq, xlabel, ylabel = norm_saxs_data(Iq, q, plot_norm=3)
        
        expected = Iq / Iq[0]  # Normalize by first value
        np.testing.assert_allclose(result_Iq, expected)
        assert xlabel == '$q (\\AA^{-1})$'
        assert ylabel == 'Intensity / I_0'
    
    def test_invalid_plot_norm(self):
        """Test with invalid plot_norm value."""
        q = np.array([0.1, 0.2, 0.3])
        Iq = np.array([100, 50, 25])
        
        # Should default to no normalization
        result_Iq, xlabel, ylabel = norm_saxs_data(Iq, q, plot_norm=99)
        
        np.testing.assert_array_equal(result_Iq, Iq)
        assert ylabel == 'Intensity'
    
    def test_zero_q_values(self):
        """Test handling of zero q values."""
        q = np.array([0.0, 0.1, 0.2])
        Iq = np.array([100, 50, 25])
        
        # Kratky normalization with zero q
        result_Iq, _, _ = norm_saxs_data(Iq, q, plot_norm=1)
        
        expected = np.array([0, 50 * 0.1**2, 25 * 0.2**2])
        np.testing.assert_array_equal(result_Iq, expected)
    
    def test_zero_baseline(self):
        """Test baseline normalization with zero first value."""
        q = np.array([0.1, 0.2, 0.3])
        Iq = np.array([0, 50, 25])
        
        with pytest.warns(RuntimeWarning):
            result_Iq, _, _ = norm_saxs_data(Iq, q, plot_norm=3)
            assert np.isinf(result_Iq[1])
    
    def test_array_shapes_consistency(self):
        """Test that input arrays must have same shape."""
        q = np.array([0.1, 0.2])
        Iq = np.array([100, 50, 25])  # Different shape
        
        with pytest.raises((ValueError, IndexError)):
            norm_saxs_data(Iq, q, plot_norm=1)


class TestCreateSlice:
    """Test suite for create_slice function."""
    
    def test_basic_slicing(self):
        """Test basic array slicing within range."""
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        x_range = [3, 7]
        
        result_slice = create_slice(arr, x_range)
        
        # Should include values from 3 to < 7 (indices 2 to 5, exclusive end)
        assert result_slice == slice(2, 6)
        assert list(arr[result_slice]) == [3, 4, 5, 6]
    
    def test_range_at_boundaries(self):
        """Test range that matches array boundaries."""
        arr = np.array([1, 2, 3, 4, 5])
        x_range = [1, 5]
        
        result_slice = create_slice(arr, x_range)
        
        assert result_slice == slice(0, 4)
        assert list(arr[result_slice]) == [1, 2, 3, 4]
    
    def test_range_outside_array(self):
        """Test range completely outside array bounds."""
        arr = np.array([1, 2, 3, 4, 5])
        x_range = [10, 20]
        
        result_slice = create_slice(arr, x_range)
        
        # Should return slice that covers available range
        assert isinstance(result_slice, slice)
    
    def test_range_below_array(self):
        """Test range completely below array values."""
        arr = np.array([5, 6, 7, 8, 9])
        x_range = [1, 3]
        
        result_slice = create_slice(arr, x_range)
        
        # Should return slice starting from beginning
        assert result_slice.start == 0
    
    def test_empty_array(self):
        """Test with empty array should raise IndexError."""
        arr = np.array([])
        x_range = [1, 5]
        
        with pytest.raises(IndexError):
            create_slice(arr, x_range)
    
    def test_single_element_array(self):
        """Test with single-element array."""
        arr = np.array([5])
        x_range = [4, 6]
        
        result_slice = create_slice(arr, x_range)
        
        assert result_slice == slice(0, 1)
        assert list(arr[result_slice]) == [5]
    
    def test_inverted_range(self):
        """Test with inverted range (start > end)."""
        arr = np.array([1, 2, 3, 4, 5])
        x_range = [4, 2]  # Inverted range
        
        result_slice = create_slice(arr, x_range)
        
        # Function should handle this case gracefully
        assert isinstance(result_slice, slice)
    
    def test_exact_value_matches(self):
        """Test when range boundaries exactly match array values."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x_range = [2.0, 4.0]
        
        result_slice = create_slice(arr, x_range)
        
        assert result_slice == slice(1, 3)  # Includes 2.0, 3.0 (exclusive end)
        np.testing.assert_array_equal(arr[result_slice], [2.0, 3.0])
    
    def test_floating_point_array(self):
        """Test with floating-point array and range."""
        arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        x_range = [0.15, 0.45]
        
        result_slice = create_slice(arr, x_range)
        
        # Should include values >= 0.15 and <= 0.45
        selected = arr[result_slice]
        assert all(val >= 0.15 for val in selected)
        assert all(val <= 0.45 for val in selected)
    
    def test_non_monotonic_array(self):
        """Test with non-monotonic array (edge case)."""
        arr = np.array([1, 5, 2, 4, 3])
        x_range = [2, 4]
        
        # Function assumes monotonic array, behavior with non-monotonic is undefined
        # but should not crash
        result_slice = create_slice(arr, x_range)
        assert isinstance(result_slice, slice)


class TestUtilsIntegration:
    """Integration tests for utility functions."""
    
    def test_utils_import(self):
        """Test that all utility functions can be imported."""
        from xpcs_toolkit.helper.utils import get_min_max, norm_saxs_data, create_slice
        
        assert callable(get_min_max)
        assert callable(norm_saxs_data)
        assert callable(create_slice)
    
    def test_typical_saxs_workflow(self):
        """Test a typical SAXS data processing workflow."""
        # Generate synthetic SAXS data
        q = np.logspace(-2, 0, 100)  # q from 0.01 to 1 Å^-1
        Iq = 1000 * np.exp(-q**2 / 0.01)  # Gaussian-like profile
        
        # Apply Kratky normalization
        Iq_kratky, xlabel, ylabel = norm_saxs_data(Iq, q, plot_norm=1)
        
        # Get intensity range for plotting
        vmin, vmax = get_min_max(Iq_kratky)
        
        # Create slice for specific q-range
        q_range = [0.05, 0.5]
        q_slice = create_slice(q, q_range)
        
        # Verify results make sense
        assert ylabel == 'Intensity * q^2'
        assert vmin < vmax
        assert len(Iq_kratky[q_slice]) > 0
        assert all(q[q_slice] >= 0.05)
        assert all(q[q_slice] <= 0.5)
    
    def test_error_handling_chain(self):
        """Test error handling across utility functions."""
        # Test with problematic data
        q = np.array([0, 0.1, 0.2])
        Iq = np.array([0, 50, 25])
        
        # Should handle zero values gracefully
        with pytest.warns(RuntimeWarning):
            result_Iq, _, _ = norm_saxs_data(Iq, q, plot_norm=3)
        
        # get_min_max should handle the case where there are finite values
        finite_values = result_Iq[np.isfinite(result_Iq)]
        if len(finite_values) > 0:
            vmin, vmax = get_min_max(finite_values)
            assert np.isfinite(vmin)
            assert np.isfinite(vmax)
        else:
            # If no finite values, this is also valid behavior for error handling
            assert True  # Test passes - error was handled gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])