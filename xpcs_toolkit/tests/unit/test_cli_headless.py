"""
Tests for xpcs_toolkit.cli_headless module.

This module tests the headless command-line interface for XPCS data analysis,
including argument parsing, command execution, and output generation.
"""

import pytest
import argparse
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import numpy as np

from xpcs_toolkit import cli_headless
from xpcs_toolkit.cli_headless import (
    create_parser, main, plot_saxs_2d, plot_g2_function, 
    plot_saxs1d, plot_stability, list_files
)


class TestArgumentParser:
    """Test suite for CLI argument parsing."""
    
    def test_create_parser_basic(self):
        """Test basic parser creation."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        assert 'XPCS Toolkit' in parser.description
    
    def test_parser_version_argument(self):
        """Test version argument."""
        parser = create_parser()
        
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(['--version'])
        assert excinfo.value.code == 0
    
    def test_parser_help_argument(self):
        """Test help argument."""
        parser = create_parser()
        
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(['--help'])
        assert excinfo.value.code == 0
    
    def test_parser_verbose_argument(self):
        """Test verbose argument."""
        parser = create_parser()
        args = parser.parse_args(['--verbose', 'list', '/tmp'])
        
        assert args.verbose is True
    
    def test_parser_logging_arguments(self):
        """Test logging-related arguments."""
        parser = create_parser()
        args = parser.parse_args([
            '--log-level', 'DEBUG',
            '--log-file', 'test.log',
            '--log-format', 'json',
            'list', '/tmp'
        ])
        
        assert args.log_level == 'DEBUG'
        assert args.log_file == 'test.log'
        assert args.log_format == 'json'
    
    def test_parser_list_command(self):
        """Test list command parsing."""
        parser = create_parser()
        args = parser.parse_args(['list', '/path/to/data'])
        
        assert args.command == 'list'
        assert args.path == '/path/to/data'
    
    def test_parser_saxs2d_command(self):
        """Test saxs2d command parsing."""
        parser = create_parser()
        args = parser.parse_args([
            'saxs2d', '/path/to/data',
            '--outfile', 'output.png',
            '--log-scale',
            '--max-files', '5',
            '--dpi', '300'
        ])
        
        assert args.command == 'saxs2d'
        assert args.path == '/path/to/data'
        assert args.outfile == 'output.png'
        assert args.log_scale is True
        assert args.max_files == 5
        assert args.dpi == 300
    
    def test_parser_g2_command(self):
        """Test g2 command parsing."""
        parser = create_parser()
        args = parser.parse_args([
            'g2', '/path/to/data',
            '--outfile', 'g2.png',
            '--qmin', '0.01',
            '--qmax', '0.5',
            '--max-files', '10'
        ])
        
        assert args.command == 'g2'
        assert args.qmin == 0.01
        assert args.qmax == 0.5
        assert args.max_files == 10
    
    def test_parser_saxs1d_command(self):
        """Test saxs1d command parsing."""
        parser = create_parser()
        args = parser.parse_args([
            'saxs1d', '/path/to/data',
            '--log-x', '--log-y',
            '--qmin', '0.001',
            '--qmax', '1.0'
        ])
        
        assert args.command == 'saxs1d'
        assert args.log_x is True
        assert args.log_y is True
        assert args.qmin == 0.001
        assert args.qmax == 1.0
    
    def test_parser_stability_command(self):
        """Test stability command parsing."""
        parser = create_parser()
        args = parser.parse_args([
            'stability', '/path/to/data',
            '--outfile', 'stability.pdf',
            '--dpi', '150'
        ])
        
        assert args.command == 'stability'
        assert args.outfile == 'stability.pdf'
        assert args.dpi == 150
    
    def test_parser_no_command(self):
        """Test parser with no command."""
        parser = create_parser()
        args = parser.parse_args([])
        
        assert args.command is None


class TestMainFunction:
    """Test suite for main function."""
    
    def test_main_no_command_prints_help(self):
        """Test main function with no command prints help."""
        with patch('sys.argv', ['cli_headless']):
            with patch.object(argparse.ArgumentParser, 'print_help') as mock_help:
                result = main()
                mock_help.assert_called_once()
                assert result == 1
    
    def test_main_nonexistent_path(self):
        """Test main function with nonexistent path."""
        with patch('sys.argv', ['cli_headless', 'list', '/nonexistent/path']):
            with patch('xpcs_toolkit.cli_headless.get_logger') as mock_logger:
                result = main()
                assert result == 1
                mock_logger.return_value.error.assert_called()
    
    def test_main_path_not_directory(self):
        """Test main function with path that's not a directory."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            with patch('sys.argv', ['cli_headless', 'list', tmp_file.name]):
                with patch('xpcs_toolkit.cli_headless.get_logger') as mock_logger:
                    result = main()
                    assert result == 1
                    mock_logger.return_value.error.assert_called()
    
    @patch('xpcs_toolkit.cli_headless.list_files')
    def test_main_successful_command_execution(self, mock_list_files):
        """Test successful command execution."""
        mock_list_files.return_value = 0
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('sys.argv', ['cli_headless', 'list', tmp_dir]):
                with patch('xpcs_toolkit.cli_headless.get_logger'):
                    result = main()
                    assert result == 0
                    mock_list_files.assert_called_once()
    
    def test_main_exception_handling(self):
        """Test main function exception handling."""
        with patch('sys.argv', ['cli_headless', 'unknown_command', '/tmp']):
            with patch('xpcs_toolkit.cli_headless.get_logger') as mock_logger:
                with pytest.raises(SystemExit) as excinfo:
                    main()
                assert excinfo.value.code == 2  # argparse returns 2 for invalid arguments
    
    @patch('xpcs_toolkit.cli_headless.setup_logging')
    def test_main_logging_configuration(self, mock_setup_logging):
        """Test logging configuration in main function."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('sys.argv', ['cli_headless', '--verbose', 'list', tmp_dir]):
                with patch('xpcs_toolkit.cli_headless.list_files', return_value=0):
                    main()
                    mock_setup_logging.assert_called()


class TestListFilesCommand:
    """Test suite for list_files command."""
    
    def test_list_files_empty_directory(self):
        """Test list_files with empty directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = argparse.Namespace(path=tmp_dir)
            
            with patch('xpcs_toolkit.cli_headless.DataFileLocator') as mock_locator:
                mock_instance = mock_locator.return_value
                mock_instance.source_files.input_list = []
                
                with patch('builtins.print') as mock_print:
                    result = list_files(args)
                    assert result == 1
                    mock_print.assert_called_with("No HDF files found in the specified path")
    
    def test_list_files_with_files(self):
        """Test list_files with files present."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = argparse.Namespace(path=tmp_dir)
            
            mock_files = ['file1.hdf', 'file2.hdf', 'file3.hdf']
            
            with patch('xpcs_toolkit.cli_headless.DataFileLocator') as mock_locator:
                mock_instance = mock_locator.return_value
                mock_instance.source_files.input_list = mock_files
                
                with patch('builtins.print') as mock_print:
                    result = list_files(args)
                    assert result == 0
                    
                    # Check that files were printed
                    expected_calls = [
                        call(f"Found {len(mock_files)} HDF files:"),
                        call(f"  1. {mock_files[0]}"),
                        call(f"  2. {mock_files[1]}"),
                        call(f"  3. {mock_files[2]}")
                    ]
                    mock_print.assert_has_calls(expected_calls)
    
    def test_list_files_locator_build_called(self):
        """Test that DataFileLocator.build() is called."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = argparse.Namespace(path=tmp_dir)
            
            with patch('xpcs_toolkit.cli_headless.DataFileLocator') as mock_locator:
                mock_instance = mock_locator.return_value
                mock_instance.source_files.input_list = ['file1.hdf']
                
                list_files(args)
                mock_instance.build.assert_called_once()


class TestPlotSaxs2dCommand:
    """Test suite for plot_saxs_2d command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.args = argparse.Namespace(
            path=self.temp_dir,
            outfile='test_output.png',
            log_scale=False,
            max_files=None,
            dpi=150
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('xpcs_toolkit.cli_headless.AnalysisKernel')
    @patch('xpcs_toolkit.cli_headless.DataFileLocator')
    @patch('xpcs_toolkit.cli_headless.plt')
    def test_plot_saxs_2d_no_files(self, mock_plt, mock_locator, mock_kernel):
        """Test plot_saxs_2d with no files found."""
        mock_locator_instance = mock_locator.return_value
        mock_locator_instance.source_files.input_list = []
        
        result = plot_saxs_2d(self.args)
        assert result == 1
    
    @patch('xpcs_toolkit.cli_headless.AnalysisKernel')
    @patch('xpcs_toolkit.cli_headless.DataFileLocator')
    @patch('xpcs_toolkit.cli_headless.plt')
    def test_plot_saxs_2d_no_valid_xpcs_files(self, mock_plt, mock_locator, mock_kernel):
        """Test plot_saxs_2d with no valid XPCS files."""
        # Configure matplotlib mocks
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        mock_locator_instance = mock_locator.return_value
        mock_locator_instance.source_files.input_list = ['file1.hdf']
        
        mock_kernel_instance = mock_kernel.return_value
        mock_kernel_instance.get_xf_list.return_value = []
        
        result = plot_saxs_2d(self.args)
        assert result == 1
    
    @patch('xpcs_toolkit.cli_headless.AnalysisKernel')
    @patch('xpcs_toolkit.cli_headless.DataFileLocator')
    @patch('xpcs_toolkit.cli_headless.plt')
    def test_plot_saxs_2d_no_saxs_data(self, mock_plt, mock_locator, mock_kernel):
        """Test plot_saxs_2d with no SAXS 2D data."""
        # Configure matplotlib mocks
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Set up mocks
        mock_locator_instance = mock_locator.return_value
        mock_locator_instance.source_files.input_list = ['file1.hdf']
        
        mock_xpcs_file = Mock()
        mock_xpcs_file.saxs_2d = None
        mock_xpcs_file.saxs_2d_log = None
        
        mock_kernel_instance = mock_kernel.return_value
        mock_kernel_instance.get_xf_list.return_value = [mock_xpcs_file]
        
        with patch('builtins.print') as mock_print:
            result = plot_saxs_2d(self.args)
            mock_print.assert_called_with("Error: SAXS 2D data is not available")
            # Function returns None, not an exit code
            assert result is None
    
    @patch('xpcs_toolkit.cli_headless.AnalysisKernel')
    @patch('xpcs_toolkit.cli_headless.DataFileLocator')
    @patch('xpcs_toolkit.cli_headless.plt')
    def test_plot_saxs_2d_successful_linear(self, mock_plt, mock_locator, mock_kernel):
        """Test successful plot_saxs_2d with linear scale."""
        # Set up mocks
        mock_locator_instance = mock_locator.return_value
        mock_locator_instance.source_files.input_list = ['file1.hdf']
        
        mock_saxs_data = np.random.rand(100, 100)
        mock_xpcs_file = Mock()
        mock_xpcs_file.saxs_2d = mock_saxs_data
        mock_xpcs_file.label = 'Test File'
        
        mock_kernel_instance = mock_kernel.return_value
        mock_kernel_instance.get_xf_list.return_value = [mock_xpcs_file]
        
        # Mock matplotlib
        mock_fig, mock_ax = Mock(), Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_ax.imshow.return_value = Mock()
        
        result = plot_saxs_2d(self.args)
        
        # Verify matplotlib calls
        mock_plt.subplots.assert_called_once_with(figsize=(10, 8))
        mock_ax.imshow.assert_called_once_with(mock_saxs_data, origin='lower', aspect='auto')
        mock_plt.colorbar.assert_called_once()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()
        
        assert result == 0
    
    @patch('xpcs_toolkit.cli_headless.AnalysisKernel')
    @patch('xpcs_toolkit.cli_headless.DataFileLocator')
    @patch('xpcs_toolkit.cli_headless.plt')
    def test_plot_saxs_2d_successful_log_scale(self, mock_plt, mock_locator, mock_kernel):
        """Test successful plot_saxs_2d with log scale."""
        self.args.log_scale = True
        
        # Set up mocks
        mock_locator_instance = mock_locator.return_value
        mock_locator_instance.source_files.input_list = ['file1.hdf']
        
        mock_saxs_data = np.random.rand(100, 100)
        mock_saxs_data_log = np.log(mock_saxs_data + 1)
        mock_xpcs_file = Mock()
        mock_xpcs_file.saxs_2d = mock_saxs_data
        mock_xpcs_file.saxs_2d_log = mock_saxs_data_log
        mock_xpcs_file.label = 'Test File'
        
        mock_kernel_instance = mock_kernel.return_value
        mock_kernel_instance.get_xf_list.return_value = [mock_xpcs_file]
        
        # Mock matplotlib
        mock_fig, mock_ax = Mock(), Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = plot_saxs_2d(self.args)
        
        # Should use log scale data
        mock_ax.imshow.assert_called_once_with(mock_saxs_data_log, origin='lower', aspect='auto')
        assert result == 0
    
    @patch('xpcs_toolkit.cli_headless.AnalysisKernel')
    @patch('xpcs_toolkit.cli_headless.DataFileLocator')
    @patch('xpcs_toolkit.cli_headless.plt')
    def test_plot_saxs_2d_max_files_limit(self, mock_plt, mock_locator, mock_kernel):
        """Test plot_saxs_2d with max_files limit."""
        # Configure matplotlib mocks
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        self.args.max_files = 2
        
        mock_locator_instance = mock_locator.return_value
        mock_locator_instance.source_files.input_list = ['file1.hdf', 'file2.hdf', 'file3.hdf']
        
        plot_saxs_2d(self.args)
        
        # Should only add first 2 files to target
        mock_locator_instance.add_target.assert_called_once_with(['file1.hdf', 'file2.hdf'])


class TestPlotG2FunctionCommand:
    """Test suite for plot_g2_function command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.args = argparse.Namespace(
            path=self.temp_dir,
            outfile='g2_test.png',
            qmin=None,
            qmax=None,
            max_files=None,
            dpi=150
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('xpcs_toolkit.cli_headless.AnalysisKernel')
    @patch('xpcs_toolkit.cli_headless.DataFileLocator')
    @patch('xpcs_toolkit.cli_headless.plt')
    def test_plot_g2_function_no_multitau_files(self, mock_plt, mock_locator, mock_kernel):
        """Test plot_g2_function with no multitau files."""
        mock_locator_instance = mock_locator.return_value
        mock_locator_instance.source_files.input_list = ['file1.hdf']
        
        mock_kernel_instance = mock_kernel.return_value
        mock_kernel_instance.get_xf_list.return_value = []  # No multitau files
        
        result = plot_g2_function(self.args)
        assert result == 1
    
    @patch('xpcs_toolkit.cli_headless.AnalysisKernel')
    @patch('xpcs_toolkit.cli_headless.DataFileLocator')
    @patch('xpcs_toolkit.cli_headless.plt')
    def test_plot_g2_function_successful(self, mock_plt, mock_locator, mock_kernel):
        """Test successful plot_g2_function."""
        # Configure matplotlib mocks
        mock_fig = Mock()
        mock_axes_list = [Mock() for _ in range(4)]
        # Create a mock object that has ravel() method
        mock_axes = Mock()
        mock_axes.ravel.return_value = mock_axes_list
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        # Set up mocks
        mock_locator_instance = mock_locator.return_value
        mock_locator_instance.source_files.input_list = ['file1.hdf']
        
        # Mock G2 data
        mock_q_values = np.array([0.01, 0.02])
        mock_time_elapsed = np.logspace(-5, -1, 50)
        mock_g2 = np.random.rand(50, 2) + 1.0
        mock_g2_error = np.random.rand(50, 2) * 0.1
        mock_q_bin_labels = ['Q1', 'Q2']
        
        mock_xpcs_file = Mock()
        mock_xpcs_file.label = 'Test File'
        mock_xpcs_file.get_g2_data.return_value = (
            mock_q_values, mock_time_elapsed, mock_g2, mock_g2_error, mock_q_bin_labels
        )
        
        mock_kernel_instance = mock_kernel.return_value
        mock_kernel_instance.get_xf_list.return_value = [mock_xpcs_file]
        
        result = plot_g2_function(self.args)
        
        # Verify get_xf_list was called with Multitau filter
        mock_kernel_instance.get_xf_list.assert_called_once_with(filter_atype="Multitau")
        
        # Verify get_g2_data was called
        mock_xpcs_file.get_g2_data.assert_called_once()
        
        # Verify matplotlib calls
        mock_plt.subplots.assert_called_once_with(2, 2, figsize=(12, 10))
        mock_plt.savefig.assert_called_once()
        
        assert result == 0
    
    @patch('xpcs_toolkit.cli_headless.AnalysisKernel')
    @patch('xpcs_toolkit.cli_headless.DataFileLocator')
    @patch('xpcs_toolkit.cli_headless.plt')
    def test_plot_g2_function_with_q_range(self, mock_plt, mock_locator, mock_kernel):
        """Test plot_g2_function with q-range specification."""
        # Configure matplotlib mocks
        mock_fig = Mock()
        mock_axes_list = [Mock() for _ in range(4)]
        # Create a mock object that has ravel() method
        mock_axes = Mock()
        mock_axes.ravel.return_value = mock_axes_list
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        self.args.qmin = 0.01
        self.args.qmax = 0.05
        
        mock_locator_instance = mock_locator.return_value
        mock_locator_instance.source_files.input_list = ['file1.hdf']
        
        mock_xpcs_file = Mock()
        mock_xpcs_file.label = 'Test File'
        mock_xpcs_file.get_g2_data.return_value = (
            np.array([0.02]), np.logspace(-5, -1, 50), 
            np.random.rand(50, 1) + 1.0, np.random.rand(50, 1) * 0.1,
            ['Q1']
        )
        
        mock_kernel_instance = mock_kernel.return_value
        mock_kernel_instance.get_xf_list.return_value = [mock_xpcs_file]
        
        plot_g2_function(self.args)
        
        # Verify q_range was passed to get_g2_data
        mock_xpcs_file.get_g2_data.assert_called_once_with(q_range=(0.01, 0.05))


class TestPlotSaxs1dCommand:
    """Test suite for plot_saxs1d command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.args = argparse.Namespace(
            path=self.temp_dir,
            outfile='saxs1d_test.png',
            qmin=None,
            qmax=None,
            log_x=False,
            log_y=False,
            max_files=None,
            dpi=150
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('xpcs_toolkit.cli_headless.AnalysisKernel')
    @patch('xpcs_toolkit.cli_headless.DataFileLocator')
    @patch('xpcs_toolkit.cli_headless.plt')
    def test_plot_saxs1d_successful(self, mock_plt, mock_locator, mock_kernel):
        """Test successful plot_saxs1d."""
        # Configure matplotlib mocks
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Mock colormap
        mock_cmap = Mock()
        mock_cmap.return_value = ['blue', 'red', 'green']  # Return list of colors
        mock_plt.cm.get_cmap.return_value = mock_cmap
        
        # Set up mocks
        mock_locator_instance = mock_locator.return_value
        mock_locator_instance.source_files.input_list = ['file1.hdf']
        
        # Mock SAXS 1D data
        mock_q = np.logspace(-2, 0, 100)
        mock_Iq = np.random.rand(1, 100)  # Shape: (n_phi, n_q)
        
        mock_xpcs_file = Mock()
        mock_xpcs_file.label = 'Test File'
        mock_xpcs_file.get_saxs1d_data.return_value = (
            mock_q, mock_Iq, "q (Å⁻¹)", "Intensity"
        )
        
        mock_kernel_instance = mock_kernel.return_value
        mock_kernel_instance.get_xf_list.return_value = [mock_xpcs_file]
        
        result = plot_saxs1d(self.args)
        
        # Verify get_saxs1d_data was called
        mock_xpcs_file.get_saxs1d_data.assert_called_once()
        
        # Verify matplotlib calls
        mock_plt.subplots.assert_called_once_with(figsize=(10, 6))
        mock_ax.plot.assert_called()
        mock_plt.savefig.assert_called_once()
        
        assert result == 0
    
    @patch('xpcs_toolkit.cli_headless.AnalysisKernel')
    @patch('xpcs_toolkit.cli_headless.DataFileLocator')
    @patch('xpcs_toolkit.cli_headless.plt')
    def test_plot_saxs1d_with_log_scales(self, mock_plt, mock_locator, mock_kernel):
        """Test plot_saxs1d with log scales."""
        # Configure matplotlib mocks
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Mock colormap
        mock_cmap = Mock()
        mock_cmap.return_value = ['blue', 'red', 'green']  # Return list of colors
        mock_plt.cm.get_cmap.return_value = mock_cmap
        
        self.args.log_x = True
        self.args.log_y = True
        
        mock_locator_instance = mock_locator.return_value
        mock_locator_instance.source_files.input_list = ['file1.hdf']
        
        mock_xpcs_file = Mock()
        mock_xpcs_file.label = 'Test File'
        mock_xpcs_file.get_saxs1d_data.return_value = (
            np.logspace(-2, 0, 100), np.random.rand(1, 100), "q (Å⁻¹)", "Intensity"
        )
        
        mock_kernel_instance = mock_kernel.return_value
        mock_kernel_instance.get_xf_list.return_value = [mock_xpcs_file]
        
        plot_saxs1d(self.args)
        
        # Verify log scales were set
        mock_ax.set_xscale.assert_called_with('log')
        mock_ax.set_yscale.assert_called_with('log')


class TestPlotStabilityCommand:
    """Test suite for plot_stability command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.args = argparse.Namespace(
            path=self.temp_dir,
            outfile='stability_test.png',
            dpi=150
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('xpcs_toolkit.cli_headless.AnalysisKernel')
    @patch('xpcs_toolkit.cli_headless.DataFileLocator')
    @patch('xpcs_toolkit.cli_headless.plt')
    def test_plot_stability_successful(self, mock_plt, mock_locator, mock_kernel):
        """Test successful plot_stability."""
        # Set up mocks
        mock_locator_instance = mock_locator.return_value
        mock_locator_instance.source_files.input_list = ['file1.hdf']
        
        # Mock intensity vs time data
        mock_t_data = np.linspace(0, 100, 1000)
        mock_I_data = np.random.rand(1000) + 1000
        mock_Int_t = (mock_t_data, mock_I_data)
        
        mock_xpcs_file = Mock()
        mock_xpcs_file.label = 'Test File'
        mock_xpcs_file.Int_t = mock_Int_t
        
        mock_kernel_instance = mock_kernel.return_value
        mock_kernel_instance.get_xf_list.return_value = [mock_xpcs_file]
        
        # Mock matplotlib
        mock_fig, mock_ax = Mock(), Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = plot_stability(self.args)
        
        # Verify matplotlib calls
        mock_plt.subplots.assert_called_once_with(figsize=(10, 6))
        mock_ax.plot.assert_called_once_with(
            mock_t_data, mock_I_data, 'b-', linewidth=1, alpha=0.8
        )
        mock_plt.savefig.assert_called_once()
        
        assert result == 0
    
    @patch('xpcs_toolkit.cli_headless.AnalysisKernel')
    @patch('xpcs_toolkit.cli_headless.DataFileLocator')
    @patch('xpcs_toolkit.cli_headless.plt')
    def test_plot_stability_no_intensity_data(self, mock_plt, mock_locator, mock_kernel):
        """Test plot_stability with no intensity vs time data."""
        mock_locator_instance = mock_locator.return_value
        mock_locator_instance.source_files.input_list = ['file1.hdf']
        
        mock_xpcs_file = Mock()
        mock_xpcs_file.label = 'Test File'
        mock_xpcs_file.Int_t = None
        
        mock_kernel_instance = mock_kernel.return_value
        mock_kernel_instance.get_xf_list.return_value = [mock_xpcs_file]
        
        mock_fig, mock_ax = Mock(), Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        with patch('builtins.print') as mock_print:
            result = plot_stability(self.args)
            mock_print.assert_called_with("Error: Intensity vs time data is not available")
            # Function returns None, not an exit code
            assert result is None


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def test_all_commands_exist_in_command_map(self):
        """Test that all commands exist in the command map."""
        from xpcs_toolkit.cli_headless import main
        
        # Extract the command_map from main function
        # This is a bit of a hack, but ensures consistency
        expected_commands = ['list', 'saxs2d', 'g2', 'saxs1d', 'stability']
        
        parser = create_parser()
        
        # Test that all subparsers are created
        for command in expected_commands:
            # This should not raise an error if subparser exists
            args = parser.parse_args([command, '/tmp'])
            assert args.command == command
    
    def test_cli_function_imports(self):
        """Test that all CLI functions can be imported."""
        from xpcs_toolkit.cli_headless import (
            plot_saxs_2d, plot_g2_function, plot_saxs1d, 
            plot_stability, list_files
        )
        
        assert callable(plot_saxs_2d)
        assert callable(plot_g2_function)
        assert callable(plot_saxs1d)
        assert callable(plot_stability)
        assert callable(list_files)
    
    @patch('xpcs_toolkit.cli_headless.get_logger')
    @patch('xpcs_toolkit.cli_headless.setup_logging')
    def test_logging_integration(self, mock_setup, mock_get_logger):
        """Test logging integration throughout CLI."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('sys.argv', ['cli_headless', 'list', tmp_dir]):
                with patch('xpcs_toolkit.cli_headless.list_files', return_value=0):
                    main()
                    
                    # Verify logging was set up
                    mock_setup.assert_called_once()
                    # Verify logger was used
                    mock_get_logger.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])