"""
Headless CLI interface for pyXpcsViewer

This module provides command-line interface for headless operation of XPCS analysis
without GUI dependencies. It uses matplotlib for plot generation and saves figures
to disk.
"""
import argparse
import sys
import logging
import os
import matplotlib.pyplot as plt
import numpy as np

from xpcs_toolkit import __version__
from xpcs_toolkit.data_file_locator import DataFileLocator
from xpcs_toolkit.analysis_kernel import AnalysisKernel

logger = logging.getLogger(__name__)


def configure_logging(enable_verbose_output=False):
    """Configure logging with appropriate level and format.
    
    Args:
        enable_verbose_output: If True, enable DEBUG level logging
    """
    log_level = logging.DEBUG if enable_verbose_output else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


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
    
    if not file_locator.source:
        logger.error("No HDF files found in the specified path")
        return 1
    
    # Add files to target list
    maximum_files = arguments.max_files
    files_to_process = (file_locator.source.input_list[:maximum_files] 
                       if maximum_files else file_locator.source.input_list)
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
    
    if not file_locator.source:
        logger.error("No HDF files found in the specified path")
        return 1
    
    # Add files to target list
    maximum_files = arguments.max_files
    files_to_process = (file_locator.source.input_list[:maximum_files] 
                       if maximum_files else file_locator.source.input_list)
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
    
    if not fl.source:
        logger.error("No HDF files found in the specified path")
        return 1
    
    # Add files to target list
    files_to_process = fl.source.input_list[:args.max_files] if args.max_files else fl.source.input_list
    fl.add_target(files_to_process)
    
    vk = AnalysisKernel(args.path)
    
    # Get files
    xf_list = vk.get_xf_list()
    if not xf_list:
        logger.error("No valid XPCS files found")
        return 1
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(xf_list)))
    
    for i, xf in enumerate(xf_list):
        # Get 1D scattering data
        q_range = (args.qmin, args.qmax) if args.qmin is not None and args.qmax is not None else None
        q, Iq, xlabel, ylabel = xf.get_saxs_1d_data(q_range=q_range)
        
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
    
    if not fl.source:
        logger.error("No HDF files found in the specified path")
        return 1
    
    files_to_process = fl.source.input_list[:1]  # Just one file for stability
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
    t_data, I_data = xf.Int_t
    
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
    
    if not fl.source:
        print("No HDF files found in the specified path")
        return 1
    
    print(f"Found {len(fl.source)} HDF files:")
    for i, fname in enumerate(fl.source.input_list, 1):
        print(f"{i:3d}. {fname}")
    
    return 0


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="XPCS Toolkit: Headless XPCS analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list /path/to/hdf/files
  %(prog)s saxs2d /path/to/hdf/files --outfile saxs2d.png
  %(prog)s g2 /path/to/hdf/files --outfile g2.png --qmin 0.01 --qmax 0.1
  %(prog)s saxs1d /path/to/hdf/files --outfile saxs1d.png --log-x --log-y
        """
    )
    
    parser.add_argument("--version", action="version", version=f"xpcs-toolkit {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
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
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    configure_logging(args.verbose)
    
    # Ensure path exists
    if hasattr(args, 'path'):
        if not os.path.exists(args.path):
            logger.error(f"Path does not exist: {args.path}")
            return 1
        if not os.path.isdir(args.path):
            logger.error(f"Path is not a directory: {args.path}")
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
        return command_map[args.command](args)
    except KeyError:
        logger.error(f"Unknown command: {args.command}")
        return 1
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
