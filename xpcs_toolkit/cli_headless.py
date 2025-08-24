"""
XPCS Toolkit - Headless Command-Line Interface for XPCS Data Analysis

This module provides a comprehensive command-line interface for headless operation
of X-ray Photon Correlation Spectroscopy (XPCS) data analysis and visualization.
Designed for batch processing, automated workflows, and integration with analysis
pipelines at synchrotron facilities.

## Supported File Formats

### APS 8-ID-I NeXus Format
- Custom NeXus format developed at Argonne National Laboratory
- Multi-tau correlation analysis results
- Two-time correlation analysis results
- Complete experimental metadata and parameters
- Optimized for large-scale XPCS experiments

### Legacy HDF5 Format
- Backward compatibility with older data formats
- Automatic format detection and conversion
- Seamless migration path for existing datasets

## Analysis Capabilities

### 2D SAXS Visualization (saxs2d)
- Small-angle X-ray scattering pattern visualization
- Logarithmic and linear intensity scaling
- Detector geometry correction
- Q-space mapping and calibration
- Beam center determination
- Mask application and bad pixel handling

### G2 Correlation Analysis (g2)
- Multi-tau correlation function analysis
- Time-delay dependent dynamics
- Q-range selection and filtering
- Statistical error propagation
- Relaxation time extraction
- Non-ergodic behavior characterization

### 1D Radial Analysis (saxs1d)
- Radial averaging of 2D patterns
- Angular integration with customizable sectors
- Background subtraction capabilities
- Multi-phi analysis for anisotropic samples
- Logarithmic scaling options
- Export to standard data formats

### Beam Stability Monitoring (stability)
- Intensity time-series analysis
- Beam position stability assessment
- Detector stability monitoring
- Sample drift detection
- Long-term measurement quality control
- Statistical stability metrics

### File Management (list)
- Automatic file discovery and indexing
- Format validation and verification
- Metadata extraction and summary
- Batch operation preparation
- Dataset organization tools

## Command-Line Features

- **Batch Processing**: Process multiple files automatically
- **Parameter Control**: Full control over analysis parameters
- **Output Formats**: PNG, PDF, SVG figure generation
- **Logging**: Detailed processing logs for debugging
- **Error Handling**: Robust error reporting and recovery
- **Integration**: Easy integration with shell scripts and workflows

## Typical Workflow

1. **File Discovery**: List and validate XPCS data files
2. **Quality Assessment**: Check beam stability and data quality
3. **Pattern Analysis**: Visualize 2D scattering patterns
4. **Correlation Analysis**: Extract and fit g2 functions
5. **Radial Averaging**: Generate 1D scattering profiles
6. **Export Results**: Save analysis results and figures

## Usage in Synchrotron Environments

Optimized for use at synchrotron beamlines where:
- Large volumes of data require automated processing
- Consistent analysis protocols are essential
- Integration with beamline control systems is needed
- Remote analysis capabilities are required
- Reproducible analysis workflows are critical

"""
import argparse
import sys
import logging
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np

from xpcs_toolkit import __version__
from xpcs_toolkit.data_file_locator import DataFileLocator
from xpcs_toolkit.analysis_kernel import AnalysisKernel
from xpcs_toolkit.helper.logging_config import setup_logging, get_logger

logger = logging.getLogger(__name__)


def configure_logging(enable_verbose_output=False):
    """Configure logging with appropriate level and format.

    Args:
        enable_verbose_output: If True, enable DEBUG level logging

    Deprecated:
        This function is deprecated. Use setup_logging() from
        xpcs_toolkit.helper.logging_config instead.
    """
    warnings.warn(
        "configure_logging() is deprecated. Use setup_logging() from "
        "xpcs_toolkit.helper.logging_config instead.",
        DeprecationWarning,
        stacklevel=2
    )

    level = "DEBUG" if enable_verbose_output else "INFO"
    setup_logging({"level": level})


def plot_saxs_2d(arguments):
    """Plot 2D SAXS scattering patterns from XPCS data.

    Args:
        arguments: Command line arguments containing path, output settings, etc.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    logger.info(f"Processing 2D scattering analysis for path: {arguments.path}")

    # Initialize file locator and analysis kernel
    file_locator = DataFileLocator(arguments.path)
    file_locator.build()

    if not file_locator.source_files.input_list:
        logger.error("No HDF files found in the specified path")
        return 1

    # Add files to target list
    maximum_files = arguments.max_files
    files_to_process = (file_locator.source_files.input_list[:maximum_files]
                       if maximum_files else file_locator.source_files.input_list)
    file_locator.add_target(files_to_process)

    analysis_kernel = AnalysisKernel(arguments.path)

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get first file for plotting
    xpcs_file_list = analysis_kernel.get_xf_list(rows=[0])
    if not xpcs_file_list:
        logger.error("No valid XPCS files found")
        return 1

    xpcs_file = xpcs_file_list[0]

    # Plot 2D SAXS data
    saxs_image_data = xpcs_file.saxs_2d
    if arguments.log_scale:
        saxs_image_data = xpcs_file.saxs_2d_log

    if saxs_image_data is None:
        print("Error: SAXS 2D data is not available")
        return

    image_plot = ax.imshow(saxs_image_data, origin='lower', aspect='auto')
    plt.colorbar(image_plot, ax=ax)
    ax.set_title(f'2D SAXS Pattern: {xpcs_file.label}')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')

    # Save figure
    plt.tight_layout()
    output_filename = arguments.outfile
    dots_per_inch = arguments.dpi
    plt.savefig(output_filename, dpi=dots_per_inch, bbox_inches='tight')
    logger.info(f"Saved 2D scattering plot to {output_filename}")
    plt.close()
    return 0


def plot_g2_function(arguments):
    """Plot G2 correlation functions from XPCS data.

    Args:
        arguments: Command line arguments containing path, q-range, output settings, etc.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    logger.info(f"Processing correlation function analysis for path: {arguments.path}")

    file_locator = DataFileLocator(arguments.path)
    file_locator.build()

    if not file_locator.source_files.input_list:
        logger.error("No HDF files found in the specified path")
        return 1

    # Add files to target list
    maximum_files = arguments.max_files
    files_to_process = (file_locator.source_files.input_list[:maximum_files]
                       if maximum_files else file_locator.source_files.input_list)
    file_locator.add_target(files_to_process)

    analysis_kernel = AnalysisKernel(arguments.path)

    # Get multi-tau correlation files
    xpcs_file_list = analysis_kernel.get_xf_list(filter_atype="Multitau")
    if not xpcs_file_list:
        logger.error("No Multi-tau XPCS files found")
        return 1

    # Create matplotlib figure with subplots
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for file_index, xpcs_file in enumerate(xpcs_file_list[:4]):  # Plot up to 4 files
        axis = axes[file_index]

        # Get G2 correlation data
        q_minimum = arguments.qmin
        q_maximum = arguments.qmax
        q_range = (q_minimum, q_maximum) if q_minimum is not None and q_maximum is not None else None
        q_values, time_elapsed, g2, g2_error, q_bin_labels = xpcs_file.get_g2_data(q_range=q_range)

        # Plot first q-bin
        if g2.shape[1] > 0:
            axis.errorbar(time_elapsed, g2[:, 0], yerr=g2_error[:, 0],
                         fmt='o-', markersize=3, capsize=2, label=q_bin_labels[0])
            axis.set_xscale('log')
            axis.set_xlabel('Time (s)')
            axis.set_ylabel('g2')
            axis.set_title(f'{xpcs_file.label}')
            axis.grid(True, alpha=0.3)
            axis.legend()

    # Remove empty subplots
    for file_index in range(len(xpcs_file_list), 4):
        figure.delaxes(axes[file_index])

    plt.tight_layout()
    output_filename = arguments.outfile
    dots_per_inch = arguments.dpi
    plt.savefig(output_filename, dpi=dots_per_inch, bbox_inches='tight')
    logger.info(f"Saved G2 correlation function plot to {output_filename}")
    plt.close()
    return 0


def plot_saxs1d(args):
    """Plot 1D radial scattering profiles"""
    logger.info(f"Processing SAXS 1D for path: {args.path}")

    fl = DataFileLocator(args.path)
    fl.build()

    if not fl.source_files.input_list:
        logger.error("No HDF files found in the specified path")
        return 1

    # Add files to target list
    files_to_process = fl.source_files.input_list[:args.max_files] if args.max_files else fl.source_files.input_list
    fl.add_target(files_to_process)

    vk = AnalysisKernel(args.path)

    # Get files
    xf_list = vk.get_xf_list()
    if not xf_list:
        logger.error("No valid XPCS files found")
        return 1

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get colors from matplotlib colormap
    try:
        # Try to get tab10 colormap
        cmap = plt.cm.get_cmap('tab10')
        colors = cmap(np.linspace(0, 1, len(xf_list)))
    except (AttributeError, ValueError):
        # Fallback to a basic color cycle if tab10 is not available
        basic_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors = [basic_colors[i % len(basic_colors)] for i in range(len(xf_list))]

    # Initialize labels with defaults
    xlabel, ylabel = "q (Å⁻¹)", "Intensity"

    for i, xf in enumerate(xf_list):
        # Get 1D scattering data
        q_range = (args.qmin, args.qmax) if args.qmin is not None and args.qmax is not None else None
        saxs_data = xf.get_saxs1d_data(qrange=q_range) if hasattr(xf, 'get_saxs1d_data') else None
        if saxs_data is None:
            print(f"Warning: Cannot get SAXS 1D data for {xf.label}")
            continue
        q, Iq, xlabel, ylabel = saxs_data

        # Plot first phi slice
        ax.plot(q, Iq[0], 'o-', color=colors[i], markersize=3,
               label=f'{xf.label}', alpha=0.8)

    if args.log_x:
        ax.set_xscale('log')
    if args.log_y:
        ax.set_yscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('SAXS 1D Profiles')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.outfile, dpi=args.dpi, bbox_inches='tight')
    logger.info(f"Saved SAXS 1D plot to {args.outfile}")
    plt.close()
    return 0


def plot_stability(args):
    """Plot beam stability analysis"""
    logger.info(f"Processing stability analysis for path: {args.path}")

    fl = DataFileLocator(args.path)
    fl.build()

    if not fl.source_files.input_list:
        logger.error("No HDF files found in the specified path")
        return 1

    files_to_process = fl.source_files.input_list[:1]  # Just one file for stability
    fl.add_target(files_to_process)

    vk = AnalysisKernel(args.path)

    xf_list = vk.get_xf_list()
    if not xf_list:
        logger.error("No valid XPCS files found")
        return 1

    xf = xf_list[0]

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get intensity vs time data
    Int_t = getattr(xf, 'Int_t', None)
    if Int_t is None:
        print("Error: Intensity vs time data is not available")
        return
    t_data, I_data = Int_t

    ax.plot(t_data, I_data, 'b-', linewidth=1, alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Intensity')
    ax.set_title(f'Beam Stability: {xf.label}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.outfile, dpi=args.dpi, bbox_inches='tight')
    logger.info(f"Saved stability plot to {args.outfile}")
    plt.close()
    return 0


def list_files(args):
    """List available HDF files in directory"""
    logger.info(f"Listing files in path: {args.path}")

    fl = DataFileLocator(args.path)
    fl.build()

    if not fl.source_files.input_list:
        print("No HDF files found in the specified path")
        return 1

    print(f"Found {len(fl.source_files.input_list)} HDF files:")
    for i, fname in enumerate(fl.source_files.input_list, 1):
        print(f"{i:3d}. {fname}")

    return 0


def create_parser():
    """Create comprehensive argument parser for XPCS analysis.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with all XPCS analysis commands and options.
    """
    parser = argparse.ArgumentParser(
        description="""XPCS Toolkit - Advanced X-ray Photon Correlation Spectroscopy Analysis

A comprehensive command-line tool for analyzing XPCS datasets from synchrotron
beamlines, with specialized support for the customized NeXus file format
developed at Argonne National Laboratory's Advanced Photon Source beamline 8-ID-I.

Supported Analysis Types:
  • Multi-tau correlation analysis (g2 functions)
  • Two-time correlation analysis (C2 functions)
  • Small-angle X-ray scattering (SAXS) visualization
  • Beam stability monitoring and quality assessment
  • Radial averaging and sector integration

File Format Support:
  • APS 8-ID-I NeXus format (primary)
  • Legacy HDF5 formats (backward compatibility)
  • Automatic format detection and validation

Output Options:
  • High-resolution PNG, PDF, SVG figures
  • Publication-quality plots with LaTeX labels
  • Configurable DPI and color schemes
  • Batch processing with consistent formatting""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DETAILED EXAMPLES:

File Discovery and Validation:
  %(prog)s list /beamline/data/2024-1/user_experiment/
    → List all XPCS files and validate format compatibility

2D SAXS Pattern Visualization:
  %(prog)s saxs2d /data/xpcs/ --outfile detector_pattern.png --log-scale
    → Generate logarithmic-scale 2D scattering pattern

  %(prog)s saxs2d /data/ --outfile pattern.pdf --dpi 300 --max-files 1
    → High-resolution PDF output of first file only

G2 Correlation Function Analysis:
  %(prog)s g2 /data/multitau/ --outfile correlation.png --qmin 0.005 --qmax 0.2
    → Plot g2 functions for specific q-range (typical soft matter)

  %(prog)s g2 /data/ --qmin 0.01 --qmax 0.5 --max-files 4 --dpi 150
    → Compare up to 4 files with hard matter q-range

1D Radial Scattering Profiles:
  %(prog)s saxs1d /data/ --outfile intensity_profile.png --log-x --log-y
    → Double-logarithmic plot for power-law analysis

  %(prog)s saxs1d /data/ --qmin 0.001 --qmax 1.0 --outfile sector_avg.svg
    → Vector graphics output with custom q-range

Beam Stability Assessment:
  %(prog)s stability /data/long_measurement/ --outfile beam_stability.png
    → Monitor intensity fluctuations during measurement

Batch Processing Workflows:
  find /beamline/data/ -name "*.hdf" -exec dirname {} \\; | sort -u | \\
  while read dir; do
    %(prog)s list "$dir"
    %(prog)s saxs2d "$dir" --outfile "$dir/pattern.png" --log-scale
    %(prog)s g2 "$dir" --outfile "$dir/g2.png" --qmin 0.01 --qmax 0.1
  done
    → Automated analysis of entire experimental campaign

SYNCHROTRON INTEGRATION:
  • Real-time analysis: Process data as it's collected
  • Remote operation: Run analysis from anywhere with network access
  • Automated pipelines: Integration with beamline control systems
  • Quality control: Immediate feedback on data quality
  • Archive processing: Batch analysis of stored datasets

For interactive analysis and advanced features, use the full XPCS Toolkit GUI.
For technical support: https://github.com/imewei/xpcs-toolkit
        """
    )

    parser.add_argument("--version", action="version", version=f"xpcs-toolkit {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    # Logging configuration options
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Set logging level (overrides --verbose)")
    parser.add_argument("--log-file", help="Path to log file (default: xpcs_toolkit.log)")
    parser.add_argument("--log-config", help="Path to logging configuration file (YAML or JSON)")
    parser.add_argument("--log-format", choices=["simple", "detailed", "json"],
                       help="Log format style (default: detailed)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List HDF files in directory")
    list_parser.add_argument("path", help="Path to directory containing HDF files")

    # SAXS 2D command
    saxs2d_parser = subparsers.add_parser("saxs2d", help="Plot 2D scattering patterns")
    saxs2d_parser.add_argument("path", help="Path to directory containing HDF files")
    saxs2d_parser.add_argument("--outfile", "-o", default="saxs2d.png", help="Output filename")
    saxs2d_parser.add_argument("--log-scale", action="store_true", help="Use log scale")
    saxs2d_parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    saxs2d_parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")

    # G2 command
    g2_parser = subparsers.add_parser("g2", help="Plot G2 correlation functions")
    g2_parser.add_argument("path", help="Path to directory containing HDF files")
    g2_parser.add_argument("--outfile", "-o", default="g2.png", help="Output filename")
    g2_parser.add_argument("--qmin", type=float, help="Minimum q value")
    g2_parser.add_argument("--qmax", type=float, help="Maximum q value")
    g2_parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    g2_parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")

    # SAXS 1D command
    saxs1d_parser = subparsers.add_parser("saxs1d", help="Plot 1D radial scattering profiles")
    saxs1d_parser.add_argument("path", help="Path to directory containing HDF files")
    saxs1d_parser.add_argument("--outfile", "-o", default="saxs1d.png", help="Output filename")
    saxs1d_parser.add_argument("--qmin", type=float, help="Minimum q value")
    saxs1d_parser.add_argument("--qmax", type=float, help="Maximum q value")
    saxs1d_parser.add_argument("--log-x", action="store_true", help="Use log scale for x-axis")
    saxs1d_parser.add_argument("--log-y", action="store_true", help="Use log scale for y-axis")
    saxs1d_parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    saxs1d_parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")

    # Stability command
    stability_parser = subparsers.add_parser("stability", help="Plot beam stability analysis")
    stability_parser.add_argument("path", help="Path to directory containing HDF files")
    stability_parser.add_argument("--outfile", "-o", default="stability.png", help="Output filename")
    stability_parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")

    return parser


def main():
    """Main entry point with comprehensive logging and error handling."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Configure logging based on CLI arguments
    if hasattr(args, 'log_config') and args.log_config:
        # Use external config file
        setup_logging(config_file=args.log_config)
    else:
        # Build config from CLI arguments
        log_level = getattr(args, 'log_level', None) or ("DEBUG" if args.verbose else "INFO")
        log_file = getattr(args, 'log_file', None)
        log_format = getattr(args, 'log_format', "detailed")

        from xpcs_toolkit.helper.logging_config import get_default_config
        config = get_default_config(
            level=log_level,
            file_path=log_file,
            format_style=log_format
        )
        setup_logging(config)

    # Log startup information
    context_logger = get_logger(__name__,
                               command=args.command,
                               xpcs_version=__version__)
    context_logger.info("XPCS Toolkit CLI started",
                       extra={"args": vars(args)})

    # Ensure path exists
    if hasattr(args, 'path'):
        if not os.path.exists(args.path):
            context_logger.error("Path does not exist", extra={"path": args.path})
            return 1
        if not os.path.isdir(args.path):
            context_logger.error("Path is not a directory", extra={"path": args.path})
            return 1

    # Execute command
    command_map = {
        'list': list_files,
        'saxs2d': plot_saxs_2d,
        'g2': plot_g2_function,
        'saxs1d': plot_saxs1d,
        'stability': plot_stability,
    }

    try:
        exit_code = command_map[args.command](args)
        if exit_code == 0:
            context_logger.info("Command completed successfully")
        else:
            context_logger.warning("Command completed with errors",
                                  extra={"exit_code": exit_code})
        return exit_code

    except KeyError:
        context_logger.error("Unknown command", extra={"command": args.command})
        return 1
    except Exception as e:
        context_logger.exception("Uncaught exception during command execution",
                               extra={"command": args.command})
        if args.verbose or getattr(args, 'log_level', None) == "DEBUG":
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
