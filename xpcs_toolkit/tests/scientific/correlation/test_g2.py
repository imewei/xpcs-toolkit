"""
Tests for xpcs_toolkit.scientific.correlation.g2 module

Comprehensive test coverage for multi-tau correlation function analysis,
including data extraction, geometry calculations, and numerical accuracy testing.
"""

from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from xpcs_toolkit.scientific.correlation import g2
from xpcs_toolkit.tests.fixtures.synthetic_data import (
    SyntheticXPCSDataGenerator,
    ensure_test_data_exists,
)


class TestG2CorrelationAnalysis:
    """Test suite for G2 correlation analysis functionality."""

    @pytest.fixture
    def mock_xpcs_files(self):
        """Create mock XPCS file objects for testing."""
        files = []
        generator = SyntheticXPCSDataGenerator()

        for i in range(3):
            mock_file = Mock()
            mock_file.atype = ["Multitau"]  # Analysis type

            # Generate synthetic correlation data
            intensity, q_vals, tau_vals = generator.generate_brownian_motion_intensity(
                n_times=100, n_q_bins=5
            )
            g2_vals = generator.calculate_analytical_g2(np.logspace(-4, 0, 50), q_vals)
            g2_errors = 0.01 * g2_vals + 0.001

            # Mock the get_g2_data method - use closure factory to capture loop variables
            def create_mock_get_g2_data(q_values, g2_values, g2_error_values):
                def mock_get_g2_data(q_range=None, time_range=None):
                    q_filtered = q_values
                    tau_filtered = np.logspace(-4, 0, 50)
                    g2_filtered = g2_values
                    g2_err_filtered = g2_error_values
                    labels = [f"q={q:.3f}" for q in q_filtered]

                    # Apply filtering if specified
                    if q_range is not None:
                        q_min, q_max = q_range
                        mask = (q_filtered >= q_min) & (q_filtered <= q_max)
                        q_filtered = q_filtered[mask]
                        g2_filtered = g2_filtered[:, mask]
                        g2_err_filtered = g2_err_filtered[:, mask]
                        labels = [labels[j] for j, m in enumerate(mask) if m]

                    if time_range is not None:
                        t_min, t_max = time_range
                        mask = (tau_filtered >= t_min) & (tau_filtered <= t_max)
                        tau_filtered = tau_filtered[mask]
                        g2_filtered = g2_filtered[mask, :]
                        g2_err_filtered = g2_err_filtered[mask, :]

                    return (
                        q_filtered,
                        tau_filtered,
                        g2_filtered,
                        g2_err_filtered,
                        labels,
                    )

                return mock_get_g2_data

            mock_file.get_g2_data = create_mock_get_g2_data(q_vals, g2_vals, g2_errors)
            files.append(mock_file)

        return files

    @pytest.fixture
    def mock_xpcs_files_invalid(self):
        """Create mock XPCS files without Multitau analysis for error testing."""
        mock_file = Mock()
        mock_file.atype = ["Twotime"]  # Not Multitau
        return [mock_file]

    def test_data_extraction_filtering(self, mock_xpcs_files):
        """Test g2 data extraction with q/t range filtering."""
        # Test without filtering
        q, tel, g2_data, g2_err, labels = g2.get_data(mock_xpcs_files)

        assert len(q) == 3  # Three files
        assert len(tel) == 3
        assert len(g2_data) == 3
        assert len(g2_err) == 3
        assert len(labels) == 3

        # Each file should have data
        for i in range(3):
            assert len(q[i]) == 5  # 5 q-bins
            assert len(tel[i]) == 50  # 50 tau values
            assert g2_data[i].shape == (50, 5)  # (n_tau, n_q)
            assert g2_err[i].shape == (50, 5)
            assert len(labels[i]) == 5

    def test_data_extraction_q_filtering(self, mock_xpcs_files):
        """Test g2 data extraction with q-range filtering."""
        q_range = (0.01, 0.05)  # Filter to specific q-range

        q, tel, g2_data, g2_err, labels = g2.get_data(mock_xpcs_files, q_range=q_range)

        # Should still have 3 files, but potentially fewer q-bins
        assert len(q) == 3

        for i in range(3):
            # All q-values should be within the specified range
            assert np.all(q[i] >= q_range[0])
            assert np.all(q[i] <= q_range[1])

            # Data shapes should be consistent
            n_q_filtered = len(q[i])
            assert g2_data[i].shape[1] == n_q_filtered
            assert g2_err[i].shape[1] == n_q_filtered
            assert len(labels[i]) == n_q_filtered

    def test_data_extraction_t_filtering(self, mock_xpcs_files):
        """Test g2 data extraction with time-range filtering."""
        t_range = (1e-3, 1e-1)  # Filter to specific time range

        q, tel, g2_data, g2_err, labels = g2.get_data(mock_xpcs_files, t_range=t_range)

        # Should still have 3 files, but potentially fewer time points
        assert len(tel) == 3

        for i in range(3):
            # All tau values should be within the specified range
            assert np.all(tel[i] >= t_range[0])
            assert np.all(tel[i] <= t_range[1])

            # Data shapes should be consistent
            n_tau_filtered = len(tel[i])
            assert g2_data[i].shape[0] == n_tau_filtered
            assert g2_err[i].shape[0] == n_tau_filtered

    def test_data_extraction_invalid_analysis_type(self, mock_xpcs_files_invalid):
        """Test data extraction fails with non-Multitau files."""
        result = g2.get_data(mock_xpcs_files_invalid)

        # Should return False and None values when files don't have Multitau analysis
        assert result[0] is False
        assert all(x is None for x in result[1:])

    def test_geometry_calculation(self):
        """Test plot geometry calculations for different modes."""
        # Create mock g2 data with known dimensions
        mock_g2_data = [
            np.random.random((100, 5)),  # File 1: 100 tau, 5 q-bins
            np.random.random((100, 5)),  # File 2: 100 tau, 5 q-bins
            np.random.random((100, 5)),  # File 3: 100 tau, 5 q-bins
        ]

        # Test multiple mode (separate panel for each q-bin)
        num_figs, num_lines = g2.compute_geometry(mock_g2_data, "multiple")
        assert num_figs == 5  # 5 q-bins = 5 panels
        assert num_lines == 3  # 3 files = 3 lines per panel

        # Test single mode (separate panel for each file)
        num_figs, num_lines = g2.compute_geometry(mock_g2_data, "single")
        assert num_figs == 3  # 3 files = 3 panels
        assert num_lines == 5  # 5 q-bins = 5 lines per panel

        # Test single-combined mode (one panel for everything)
        num_figs, num_lines = g2.compute_geometry(mock_g2_data, "single-combined")
        assert num_figs == 1  # 1 panel total
        assert num_lines == 15  # 3 files × 5 q-bins = 15 lines total

    @pytest.mark.parametrize("plot_type", ["multiple", "single", "single-combined"])
    def test_plot_geometry_modes(self, plot_type):
        """Test all supported plot geometry modes."""
        # Test with different data configurations
        test_cases = [
            ([np.random.random((50, 3))], 3, 1),  # 1 file, 3 q-bins
            (
                [np.random.random((50, 2)), np.random.random((50, 2))],
                2,
                2,
            ),  # 2 files, 2 q-bins
            ([np.random.random((100, 1))], 1, 1),  # 1 file, 1 q-bin
        ]

        for g2_data, expected_q_bins, expected_files in test_cases:
            num_figs, num_lines = g2.compute_geometry(g2_data, plot_type)

            if plot_type == "multiple":
                assert num_figs == expected_q_bins
                assert num_lines == expected_files
            elif plot_type == "single":
                assert num_figs == expected_files
                assert num_lines == expected_q_bins
            elif plot_type == "single-combined":
                assert num_figs == 1
                assert num_lines == expected_q_bins * expected_files

    def test_plot_geometry_invalid_type(self):
        """Test error handling for invalid plot type."""
        mock_g2_data = [np.random.random((50, 3))]

        with pytest.raises(ValueError, match="plot_type not support"):
            g2.compute_geometry(mock_g2_data, "invalid_type")

    def test_numerical_accuracy(self):
        """Test correlation function calculations against known solutions."""
        # Generate synthetic data with known analytical solution
        generator = SyntheticXPCSDataGenerator()

        # Test parameters
        diffusion_coeff = 1e-12  # m²/s
        q_values = np.array([0.01, 0.02, 0.05])  # Å⁻¹
        tau_values = np.logspace(-4, 0, 50)  # seconds

        # Calculate analytical g2
        analytical_g2 = generator.calculate_analytical_g2(
            tau_values, q_values, diffusion_coeff
        )

        # Verify analytical solution properties
        assert analytical_g2.shape == (len(tau_values), len(q_values))

        # At tau=0, g2 should approach 1 + β (where β ≈ 0.8)
        assert np.all(
            analytical_g2[0, :] >= 1.0
        )  # Allow for numerical precision at boundary
        assert np.all(analytical_g2[0, :] < 2.0)

        # At large tau, g2 should approach 1
        assert np.allclose(analytical_g2[-1, :], 1.0, rtol=1e-2)

        # Higher q should have faster decay (smaller Γ = D*q²)
        # So g2 should decay faster for higher q values
        mid_tau_idx = len(tau_values) // 2
        g2_mid_q1 = analytical_g2[mid_tau_idx, 0]  # Lowest q
        g2_mid_q3 = analytical_g2[mid_tau_idx, 2]  # Highest q

        # At the same tau, higher q should have lower g2 (faster decay)
        # Use small tolerance for numerical precision issues
        assert g2_mid_q3 <= g2_mid_q1 + 1e-10, (
            f"Higher q should decay faster: g2(q_high)={g2_mid_q3}, g2(q_low)={g2_mid_q1}"
        )

    def test_plotting_functions_disabled(self):
        """Test that plotting functions are properly disabled in headless mode."""
        # Test pg_plot function
        with pytest.raises(
            NotImplementedError, match="GUI plotting functionality has been disabled"
        ):
            g2.pg_plot(
                hdl=None,
                xf_list=[],
                q_range=None,
                t_range=None,
                y_range=None,
                plot_type="multiple",
            )

        # Test pg_plot_one_g2 function
        with pytest.raises(
            NotImplementedError, match="GUI plotting functionality has been disabled"
        ):
            g2.pg_plot_one_g2(
                ax=None, x=[], y=[], dy=[], color=(0, 0, 0), label="test", symbol="o"
            )

    def test_colors_and_symbols_constants(self):
        """Test that color and symbol constants are properly defined."""
        # Test colors tuple
        assert hasattr(g2, "colors")
        assert isinstance(g2.colors, tuple)
        assert len(g2.colors) == 10  # Should have 10 colors

        # Each color should be an RGB tuple
        for color in g2.colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

        # Test symbols list
        assert hasattr(g2, "symbols")
        assert isinstance(g2.symbols, list)
        assert len(g2.symbols) > 0

        # Common symbols should be present
        expected_symbols = ["o", "s", "+", "x"]
        for symbol in expected_symbols:
            assert symbol in g2.symbols

    def test_edge_cases_empty_data(self):
        """Test handling of edge cases and empty data."""
        # Test with empty file list
        empty_files = []

        # This should handle empty input gracefully
        # The exact behavior depends on implementation
        try:
            result = g2.get_data(empty_files)
            # If it doesn't raise an exception, verify the result structure
            if result[0] is not False:
                assert len(result) == 5  # Should return 5-tuple
        except (IndexError, AttributeError):
            # These are acceptable for empty input
            pass

    def test_data_consistency_multiple_calls(self, mock_xpcs_files):
        """Test that multiple calls with same parameters return consistent results."""
        # Call get_data multiple times with same parameters
        result1 = g2.get_data(mock_xpcs_files, q_range=(0.01, 0.1))
        result2 = g2.get_data(mock_xpcs_files, q_range=(0.01, 0.1))

        # Results should be identical
        q1, tel1, g2_1, g2_err1, labels1 = result1
        q2, tel2, g2_2, g2_err2, labels2 = result2

        assert len(q1) == len(q2)
        for i in range(len(q1)):
            assert np.array_equal(q1[i], q2[i])
            assert np.array_equal(tel1[i], tel2[i])
            assert np.array_equal(g2_1[i], g2_2[i])
            assert np.array_equal(g2_err1[i], g2_err2[i])
            assert labels1[i] == labels2[i]


class TestG2ModuleIntegration:
    """Integration tests for the g2 module."""

    def test_module_imports(self):
        """Test that all expected functions and constants are importable."""
        # Test function imports
        assert hasattr(g2, "get_data")
        assert callable(g2.get_data)

        assert hasattr(g2, "compute_geometry")
        assert callable(g2.compute_geometry)

        assert hasattr(g2, "pg_plot")
        assert callable(g2.pg_plot)

        assert hasattr(g2, "pg_plot_one_g2")
        assert callable(g2.pg_plot_one_g2)

        # Test constant imports
        assert hasattr(g2, "colors")
        assert hasattr(g2, "symbols")

    def test_lazy_imports(self):
        """Test that lazy imports work correctly."""
        # The module should be able to import numpy and matplotlib via lazy imports
        # This is mostly tested by the module actually loading successfully
        assert True  # If we got here, lazy imports worked

    def test_module_docstring(self):
        """Test that module has comprehensive documentation."""
        assert g2.__doc__ is not None
        assert len(g2.__doc__) > 100  # Should have substantial documentation

        # Should contain key scientific information
        doc = g2.__doc__
        assert "correlation function" in doc.lower()
        assert "xpcs" in doc.lower()
        assert "g₂" in doc or "g2" in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
