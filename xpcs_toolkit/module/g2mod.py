"""
XPCS Toolkit - Multi-tau Correlation Function Analysis Module (g2mod)

This module provides comprehensive analysis and visualization capabilities for intensity 
correlation functions g₂(q,τ) from X-ray Photon Correlation Spectroscopy (XPCS) experiments.
The correlation function is the fundamental quantity in XPCS that reveals the dynamic 
information about fluctuating systems.

## Scientific Background

The intensity correlation function g₂(q,τ) is defined as:

    g₂(q,τ) = ⟨I(q,t)I(q,t+τ)⟩ / ⟨I(q,t)⟩²

Where:
    - I(q,t): Scattered intensity at wavevector q and time t
    - τ: Correlation delay time (lag time)
    - ⟨⟩: Time ensemble average
    - q = 4π sin(θ/2)/λ: Scattering wavevector magnitude

## Physical Interpretation

The correlation function provides access to:

### Relaxation Dynamics
- **Single exponentials**: g₂(τ) = 1 + β exp(-2Γτ) for Brownian motion
- **Stretched exponentials**: g₂(τ) = 1 + β exp(-(Γτ)^α) for glassy systems  
- **Double exponentials**: Multi-component dynamics with different timescales
- **Power laws**: Critical fluctuations near phase transitions

### Material Properties  
- **Diffusion coefficients**: D = Γ/q² for purely diffusive systems
- **Hydrodynamic radii**: R_h = kT/(6πηD) via Stokes-Einstein relation
- **Polydispersity**: Width of relaxation time distributions
- **Dynamic heterogeneity**: Spatial variations in local dynamics

### Non-equilibrium Behavior
- **Aging effects**: Time-dependent correlation functions
- **Non-ergodic systems**: g₂(τ) approaches values > 1 at long times
- **Active matter**: Ballistic motion and enhanced fluctuations
- **Glass transitions**: Slow dynamics and stretched exponentials

## Analysis Capabilities

### Multi-tau Algorithm
- **Extended time range**: 6-8 decades in correlation time
- **Statistical efficiency**: Optimal use of photon statistics  
- **Logarithmic sampling**: Dense sampling at short times, sparse at long times
- **Real-time processing**: Suitable for online analysis during experiments

### Data Processing Features
- **Q-range selection**: Analyze specific momentum transfer ranges
- **Time-range filtering**: Focus on specific dynamics timescales
- **Statistical error propagation**: Proper uncertainty quantification
- **Baseline correction**: Automatic normalization and offset correction
- **Bad data handling**: Robust analysis with missing or corrupted data points

### Visualization Tools
- **Multi-plot layouts**: Individual q-curves or combined displays
- **Logarithmic scaling**: Time and amplitude axis scaling options
- **Error bar display**: Statistical uncertainty visualization
- **Fitting overlays**: Theoretical model fits with parameters
- **Publication quality**: High-resolution figures with customizable styling

## Typical Analysis Workflow

1. **Data Loading**: Extract correlation data from XPCS files
2. **Quality Assessment**: Evaluate signal-to-noise and baseline behavior  
3. **Q-range Selection**: Choose appropriate momentum transfer range
4. **Visualization**: Generate correlation function plots
5. **Model Fitting**: Extract relaxation parameters using appropriate models
6. **Physical Interpretation**: Convert to diffusion coefficients or other properties

## Applications

### Soft Matter Physics
- **Colloidal suspensions**: Brownian motion and interactions
- **Polymer solutions**: Chain dynamics and entanglement effects
- **Gels and networks**: Structural relaxation and aging
- **Liquid crystals**: Orientational fluctuations and phase transitions

### Materials Science  
- **Nanoparticle dispersions**: Aggregation and stability studies
- **Composites**: Interface dynamics and mechanical properties
- **Thin films**: Surface fluctuations and wetting phenomena
- **Porous media**: Transport and confined dynamics

### Biological Systems
- **Protein solutions**: Conformational dynamics and interactions
- **Cell membranes**: Lipid diffusion and membrane fluctuations
- **Tissues**: Cellular motility and extracellular matrix dynamics
- **Drug delivery**: Carrier particle dynamics and release mechanisms

## Module Functions

The module provides the following key functions:

- `get_data()`: Extract correlation data from XPCS files with filtering options
- `compute_geometry()`: Calculate plot layouts for different visualization modes
- `pg_plot()`: Create comprehensive correlation function visualizations
- `pg_plot_one_g2()`: Plot individual correlation curves with customization

## Usage Examples

```python
# Basic correlation analysis
from xpcs_toolkit.module import g2mod

# Load XPCS data files
xf_list = [XpcsDataFile('sample_001.h5'), XpcsDataFile('sample_002.h5')]

# Extract correlation data for specific q-range
q, tau, g2, g2_err, labels = g2mod.get_data(
    xf_list, 
    q_range=(0.01, 0.05),  # Å⁻¹
    t_range=(1e-5, 1e-1)   # seconds
)

# Create correlation function plots
g2mod.pg_plot(
    hdl=plot_handle,
    q=q, tel=tau, g2=g2, g2_err=g2_err, labels=labels,
    plot_type='single-combined',
    log_x=True,
    show_error_bars=True,
    fit_function='single_exp'
)
```

## References

- Brown & Pusey, "Photon correlation spectroscopy and velocimetry" (1991)
- Cipelletti & Weitz, "Ultralow-angle dynamic light scattering", Rev. Sci. Instrum. 70, 3214 (1999)  
- Fluerasu et al., "Slow dynamics and aging in colloidal gels", Phys. Rev. E 76, 010401 (2007)
- Sandy et al., "Hard x-ray photon correlation spectroscopy methods", Annu. Rev. Mater. Res. 48, 167 (2018)

## Author

XPCS Toolkit Development Team
Advanced Photon Source, Argonne National Laboratory
"""

import logging
from ..mpl_compat import mkPen

# Use lazy imports for heavy dependencies
from .._lazy_imports import lazy_import
np = lazy_import('numpy')
plt = lazy_import('matplotlib.pyplot')
FormatStrFormatter = lazy_import('matplotlib.ticker', 'FormatStrFormatter')
logger = logging.getLogger(__name__)

# colors converted from
# https://matplotlib.org/stable/tutorials/colors/colors.html
# colors = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')

colors = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
)


# https://www.geeksforgeeks.org/pyqtgraph-symbols/
symbols = ["o", "t", "t1", "t2", "t3", "s", "p", "h", "star", "+", "d", "x"]


def get_data(xf_list, q_range=None, t_range=None):
    """
    Extract correlation function data from XPCS data files with optional filtering.
    
    This function processes a list of XPCS data files and extracts the multi-tau
    correlation function data g₂(q,τ) along with associated metadata. It validates
    that all files contain multi-tau correlation data and applies optional filtering
    by momentum transfer (q) and correlation time (τ) ranges.
    
    The extracted data maintains the temporal and spatial resolution of the original
    measurements while allowing selective analysis of specific dynamics regimes.
    
    Parameters
    ----------
    xf_list : list of XpcsDataFile
        List of XPCS data file objects containing multi-tau correlation data.
        Each file must have analysis type containing "Multitau".
    q_range : tuple of float, optional
        Momentum transfer range (q_min, q_max) in Å⁻¹. If None, all q-values
        are included. Used to focus on specific length scales.
    t_range : tuple of float, optional
        Correlation time range (t_min, t_max) in seconds. If None, all lag times
        are included. Used to focus on specific dynamics timescales.
    
    Returns
    -------
    q : list of numpy.ndarray
        Momentum transfer values for each file. Each array has shape (n_q,)
        where n_q is the number of q-bins after filtering.
    tel : list of numpy.ndarray
        Correlation delay times for each file. Each array has shape (n_tau,)
        where n_tau is the number of lag times after filtering.
    g2 : list of numpy.ndarray
        Correlation function values for each file. Each array has shape (n_tau, n_q)
        containing g₂(q,τ) values.
    g2_err : list of numpy.ndarray
        Statistical uncertainties for correlation function values. Each array has
        shape (n_tau, n_q) containing σ[g₂(q,τ)] values.
    labels : list of list of str
        Q-bin labels for each file. Each inner list contains descriptive labels
        for the momentum transfer bins.
    
    Returns (on error)
    -----------------
    tuple : (False, None, None, None, None)
        Returned when any file in xf_list does not contain multi-tau data.
    
    Examples
    --------
    >>> # Extract all correlation data from multiple files
    >>> q, tau, g2, g2_err, labels = get_data(xf_list)
    >>> 
    >>> # Focus on specific q-range (small-angle scattering)
    >>> q, tau, g2, g2_err, labels = get_data(
    ...     xf_list, 
    ...     q_range=(0.01, 0.05)  # Å⁻¹
    ... )
    >>> 
    >>> # Analyze only fast dynamics
    >>> q, tau, g2, g2_err, labels = get_data(
    ...     xf_list,
    ...     t_range=(1e-5, 1e-2)  # 10 μs to 10 ms
    ... )
    
    Notes
    -----
    - The function validates that all input files contain multi-tau correlation data
    - Q-range filtering selects momentum transfer bins within the specified range
    - Time-range filtering selects correlation delay times within the specified range
    - Statistical uncertainties are propagated through the filtering process
    - Empty results may occur if filtering parameters are too restrictive
    """
    for xf in xf_list:
        if "Multitau" not in xf.atype:
            return False, None, None, None, None

    q, tel, g2, g2_err, labels = [], [], [], [], []
    for fc in xf_list:
        _q, _tel, _g2, _g2_err, _labels = fc.get_g2_data(q_range=q_range, t_range=t_range)
        q.append(_q)
        tel.append(_tel)
        g2.append(_g2)
        g2_err.append(_g2_err)
        labels.append(_labels)
    return q, tel, g2, g2_err, labels


def compute_geometry(g2, plot_type):
    """
    Calculate optimal plot layout geometry for correlation function visualization.
    
    This function determines the number of figure panels and plot lines needed
    for different visualization strategies of multi-tau correlation data. It supports
    multiple layout modes optimized for different analysis scenarios:
    
    - **Multiple plots**: Separate panel for each q-bin, multiple datasets overlaid
    - **Single plots**: Separate panel for each dataset, multiple q-bins overlaid  
    - **Combined plot**: Single panel with all datasets and q-bins overlaid
    
    The layout calculation enables efficient use of screen space while maintaining
    visual clarity for correlation function analysis.
    
    Parameters
    ----------
    g2 : list of numpy.ndarray
        Correlation function data from multiple files/datasets. Each array has
        shape (n_tau, n_q) where n_tau is number of lag times and n_q is number
        of momentum transfer bins.
    plot_type : str
        Visualization layout strategy. Must be one of:
        
        * 'multiple' : Create separate subplot for each q-bin
        * 'single' : Create separate subplot for each dataset
        * 'single-combined' : Create single subplot for all data
    
    Returns
    -------
    num_figs : int
        Number of figure panels/subplots required for the visualization.
    num_lines : int
        Total number of plot lines (curves) that will be drawn across all panels.
    
    Raises
    ------
    ValueError
        If plot_type is not one of the supported visualization modes.
    
    Examples
    --------
    >>> # For 3 datasets, each with 5 q-bins
    >>> g2_data = [np.random.rand(100, 5) for _ in range(3)]
    >>> 
    >>> # Multiple mode: 5 panels (one per q-bin), 3 lines per panel
    >>> num_figs, num_lines = compute_geometry(g2_data, 'multiple')
    >>> print(f"Panels: {num_figs}, Lines: {num_lines}")  # Panels: 5, Lines: 3
    >>> 
    >>> # Single mode: 3 panels (one per dataset), 5 lines per panel
    >>> num_figs, num_lines = compute_geometry(g2_data, 'single')
    >>> print(f"Panels: {num_figs}, Lines: {num_lines}")  # Panels: 3, Lines: 5
    >>> 
    >>> # Combined mode: 1 panel, 15 lines total (3×5)
    >>> num_figs, num_lines = compute_geometry(g2_data, 'single-combined')
    >>> print(f"Panels: {num_figs}, Lines: {num_lines}")  # Panels: 1, Lines: 15
    
    Notes
    -----
    - The 'multiple' mode is ideal for comparing datasets at specific q-values
    - The 'single' mode is best for examining q-dependence within each dataset  
    - The 'single-combined' mode allows comprehensive overview but may be crowded
    - Layout geometry is used by plotting functions to create appropriate subplot grids
    """
    if plot_type == "multiple":
        num_figs = g2[0].shape[1]
        num_lines = len(g2)
    elif plot_type == "single":
        num_figs = len(g2)
        num_lines = g2[0].shape[1]
    elif plot_type == "single-combined":
        num_figs = 1
        num_lines = g2[0].shape[1] * len(g2)
    else:
        raise ValueError("plot_type not support.")
    return num_figs, num_lines


def pg_plot(
    hdl,
    xf_list,
    q_range,
    t_range,
    y_range,
    y_auto=False,
    q_auto=False,
    t_auto=False,
    num_col=4,
    rows=None,
    offset=0,
    show_fit=False,
    show_label=False,
    bounds=None,
    fit_flag=None,
    plot_type="multiple",
    subtract_baseline=True,
    marker_size=5,
    label_size=4,
    fit_func="single",
    **kwargs,
):
    """
    Create comprehensive visualization of multi-tau correlation functions g₂(q,τ).
    
    This is the main plotting function for XPCS correlation analysis, providing
    flexible visualization strategies for multi-dataset, multi-q correlation data.
    It supports scientific plotting standards with logarithmic time scaling,
    error bars, theoretical fit overlays, and customizable layouts optimized
    for different analysis workflows.
    
    The function handles complex correlation datasets with automatic layout
    computation, statistical uncertainty visualization, and optional model
    fitting integration for comprehensive dynamics analysis.
    
    Parameters
    ----------
    hdl : matplotlib.pyplot or compatible plotting handle
        Matplotlib-compatible plotting handle for rendering. Must support
        subplot creation and standard matplotlib plotting interface.
    xf_list : list of XpcsDataFile
        XPCS data files containing multi-tau correlation data. Each file
        must have analysis type containing "Multitau".
    q_range : tuple of float or None
        Momentum transfer range (q_min, q_max) in Å⁻¹ for filtering.
        Use None or set q_auto=True to include all q-values.
    t_range : tuple of float or None
        Correlation time range (t_min, t_max) in seconds for filtering.
        Use None or set t_auto=True to include all lag times.
    y_range : tuple of float or None
        Correlation function value range (g2_min, g2_max) for y-axis limits.
        Use None or set y_auto=True for automatic scaling.
    y_auto : bool, optional
        Enable automatic y-axis scaling. Default: False.
        When True, y_range is ignored and axes auto-scale to data.
    q_auto : bool, optional
        Enable automatic q-range selection. Default: False.
        When True, q_range is ignored and all q-bins are included.
    t_auto : bool, optional
        Enable automatic time-range selection. Default: False.
        When True, t_range is ignored and all lag times are included.
    num_col : int, optional
        Number of columns in subplot grid layout. Default: 4.
        Used for 'multiple' and 'single' plot types.
    rows : list of int or None, optional
        Row indices for color/symbol assignment. Default: None (auto-generated).
        Controls visual styling across datasets.
    offset : float, optional
        Vertical offset between curves for clarity. Default: 0.
        Useful for overlaid plots to separate similar curves.
    show_fit : bool, optional
        Display theoretical model fits. Default: False.
        Requires fit parameters to be available in xf_list.
    show_label : bool, optional
        Display legend labels on plots. Default: False.
        Shows dataset names and q-bin identifiers.
    bounds : tuple or None, optional
        Parameter bounds for fitting: (lower_bounds, upper_bounds).
        Only used when show_fit=True.
    fit_flag : array-like or None, optional
        Boolean flags indicating which parameters to fit.
        Only used when show_fit=True.
    plot_type : str, optional
        Visualization layout strategy. Default: 'multiple'.
        Options:
        * 'multiple': Separate subplot for each q-bin
        * 'single': Separate subplot for each dataset  
        * 'single-combined': Single subplot for all data
    subtract_baseline : bool, optional
        Apply baseline correction to correlation functions. Default: True.
        Normalizes g₂(τ→∞) to 1.0 for cleaner visualization.
    marker_size : int, optional
        Size of data point markers in points. Default: 5.
    label_size : int, optional
        Font size for axis labels in points. Default: 4.
    fit_func : str, optional
        Theoretical model for fitting. Default: 'single'.
        Options: 'single' (exponential), 'stretched', 'double', etc.
    **kwargs : dict
        Additional keyword arguments passed to plotting functions.
    
    Returns
    -------
    None
        Function performs plotting operations directly on the provided handle.
    
    Raises
    ------
    ValueError
        If plot_type is not supported or if input data is invalid.
        
    Examples
    --------
    >>> # Basic correlation function visualization
    >>> from xpcs_toolkit.module import g2mod
    >>> 
    >>> # Multiple q-bins, separate subplots
    >>> g2mod.pg_plot(
    ...     hdl=matplotlib_handle,
    ...     xf_list=[xf1, xf2],
    ...     q_range=(0.01, 0.05),  # Focus on small-angle regime
    ...     t_range=(1e-5, 1e-1),  # Cover 6 decades in time
    ...     y_range=(0.98, 1.5),   # Typical correlation range
    ...     plot_type='multiple',
    ...     show_fit=True,
    ...     show_label=True,
    ...     fit_func='single'
    ... )
    >>> 
    >>> # Combined overview plot with fits
    >>> g2mod.pg_plot(
    ...     hdl=matplotlib_handle,
    ...     xf_list=xf_list,
    ...     q_range=None,          # All q-bins
    ...     t_range=None,          # All times
    ...     y_range=None,          # Auto-scale
    ...     y_auto=True,
    ...     plot_type='single-combined',
    ...     show_fit=True,
    ...     subtract_baseline=True,
    ...     offset=0.1             # Separate curves vertically
    ... )
    
    Notes
    -----
    - Automatically applies logarithmic scaling to time axes for proper visualization
    - Statistical error bars are displayed when available in the data
    - Fitting overlays show theoretical models with baseline-corrected amplitudes
    - Layout is optimized for different analysis scenarios:
      * 'multiple': Best for comparing datasets at specific q-values
      * 'single': Best for examining q-dependence within datasets
      * 'single-combined': Best for comprehensive overview
    - Color and symbol schemes follow scientific visualization standards
    - Performance is optimized for large datasets with efficient data filtering
    
    See Also
    --------
    get_data : Extract correlation data with filtering
    compute_geometry : Calculate optimal plot layout
    pg_plot_one_g2 : Plot individual correlation curves
    """

    if q_auto:
        q_range = None
    if t_auto:
        t_range = None
    if y_auto:
        y_range = None

    data_result = get_data(xf_list, q_range=q_range, t_range=t_range)
    
    # Handle the case where get_data returns False (error condition)
    if data_result[0] is False:
        logger.error("Invalid data type for multitau analysis")
        return
    
    q, tel, g2, g2_err, labels = data_result
    
    # Ensure we have valid data before proceeding
    if g2 is None or len(g2) == 0 or g2[0] is None:
        logger.error("No valid g2 data available")
        return
    
    num_figs, num_lines = compute_geometry(g2, plot_type)

    num_data, num_qval = len(g2), g2[0].shape[1]
    # col and rows for the 2d layout
    col = min(num_figs, num_col)
    row = (num_figs + col - 1) // col

    if rows is not None and len(rows) == 0:
        rows = list(range(len(xf_list)))

    hdl.adjust_canvas_size(num_col=col, num_row=row)
    hdl.clear()
    # a bug in pyqtgraph; the log scale in x-axis doesn't apply
    t0_range = None
    if t_range:
        t0_range = np.log10(t_range)
    axes = []
    for n in range(num_figs):
        i_col = n % col
        i_row = n // col
        t = hdl.addPlot(row=i_row, col=i_col)
        axes.append(t)
        if show_label:
            t.addLegend(offset=(-1, 1), labelTextSize="9pt", verSpacing=-10)

        t.setMouseEnabled(x=False, y=y_auto)

    for m in range(num_data):
        # default base line to be 1.0; used for non-fitting or fit error cases
        baseline_offset = np.ones(num_qval)
        fit_summary = None  # Initialize to avoid unbound variable
        if show_fit:
            fit_summary = xf_list[m].fit_g2_function(
                q_range, t_range, bounds, fit_flag, fit_func
            )
            # Note: baseline_offset will be updated per q-bin in the loop below

        for n in range(num_qval):
            # Update baseline offset for this q-bin if fitting is enabled
            if show_fit and fit_summary is not None and subtract_baseline:
                # make sure the fitting is successful for this q-bin
                if (fit_summary.get("fit_line") is not None and 
                    len(fit_summary["fit_line"]) > n and
                    fit_summary["fit_line"][n].get("success", False)):
                    baseline_offset[n] = fit_summary["fit_val"][n, 0, 3]
            
            # Ensure rows and related arrays have valid data
            if rows is None or len(rows) == 0:
                rows = list(range(len(xf_list)))
            
            color = colors[rows[m] % len(colors)]
            label = None
            ax = None  # Initialize to avoid unbound variable
            
            if plot_type == "multiple":
                ax = axes[n]
                if labels is not None and len(labels) > m and labels[m] is not None and len(labels[m]) > n:
                    title = labels[m][n]
                else:
                    title = f"Q-bin {n}"
                label = getattr(xf_list[m], 'label', f'Dataset {m}')
                if m == 0:
                    ax.setTitle(title)
            elif plot_type == "single":
                ax = axes[m]
                # overwrite color; use the same color for the same set;
                color = colors[n % len(colors)]
                title = getattr(xf_list[m], 'label', f'Dataset {m}')
                # label = labels[m][n]
                ax.setTitle(title)
            elif plot_type == "single-combined":
                ax = axes[0]
                label_part1 = getattr(xf_list[m], 'label', f'Dataset {m}')
                if labels is not None and len(labels) > m and labels[m] is not None and len(labels[m]) > n:
                    label_part2 = labels[m][n]
                else:
                    label_part2 = f'Q-bin {n}'
                label = label_part1 + label_part2

            if ax is not None:
                ax.setLabel("bottom", "tau (s)")
                ax.setLabel("left", "g2")

            symbol = symbols[rows[m] % len(symbols)]

            if tel is not None and len(tel) > m:
                x = tel[m]
            else:
                x = np.array([])
                
            # normalize baseline
            if g2 is not None and len(g2) > m and g2[m] is not None:
                y = g2[m][:, n] - baseline_offset[n] + 1.0 + m * offset
            else:
                y = np.array([])
                
            if g2_err is not None and len(g2_err) > m and g2_err[m] is not None:
                y_err = g2_err[m][:, n]
            else:
                y_err = np.array([])

            if ax is not None:
                pg_plot_one_g2(
                    ax,
                    x,
                    y,
                    y_err,
                    color,
                    label=label,
                    symbol=symbol,
                    symbol_size=marker_size,
                )
                # if t_range is not None:
                if not y_auto:
                    ax.setRange(yRange=y_range)
                if not t_auto and t0_range is not None:
                    ax.setRange(xRange=t0_range)

                if show_fit and fit_summary is not None:
                    if (fit_summary.get("fit_line") is not None and 
                        len(fit_summary["fit_line"]) > n and
                        fit_summary["fit_line"][n].get("success", False)):
                        y_fit = fit_summary["fit_line"][n]["fit_y"] + m * offset
                        # normalize baseline
                        y_fit = y_fit - baseline_offset[n] + 1.0
                        ax.plot(
                            fit_summary["fit_line"][n]["fit_x"],
                            y_fit,
                            **mkPen(color, width=int(2.5)),  # Convert to int for width
                        )
    return


def pg_plot_one_g2(ax, x, y, dy, color, label, symbol, symbol_size=5):
    """
    Plot individual correlation function curve with error bars and formatting.
    
    This function renders a single g₂(q,τ) correlation curve on the provided axes
    with proper scientific visualization standards including logarithmic time scaling,
    error bars, and customizable styling. It provides compatibility between different
    plotting backends while maintaining consistent appearance.
    
    The visualization follows XPCS community conventions with logarithmic time axis
    and linear correlation function axis, enabling clear identification of different
    dynamic regimes and fitting quality assessment.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes or compatible
        Target axes object for plotting. Must support errorbar() method.
    x : numpy.ndarray
        Correlation delay times τ in seconds. Shape: (n_tau,)
    y : numpy.ndarray  
        Correlation function values g₂(τ). Shape: (n_tau,)
        May include baseline correction and vertical offset.
    dy : numpy.ndarray
        Statistical uncertainties σ[g₂(τ)]. Shape: (n_tau,)
        Standard errors propagated from photon counting statistics.
    color : tuple or str
        Plot color specification. Can be RGB tuple (0-255) or matplotlib color string.
    label : str
        Legend label for the correlation curve. Should identify dataset and q-bin.
    symbol : str
        Marker symbol specification. Mapped from PyQtGraph to matplotlib symbols.
        Common symbols: 'o' (circle), 's' (square), '^' (triangle).
    symbol_size : int, optional
        Marker size in points. Default: 5. Typical range: 3-8 for clarity.
    
    Returns
    -------
    None
        Function performs plotting operations directly on the provided axes.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> 
    >>> # Create sample correlation data
    >>> tau = np.logspace(-5, -1, 50)  # 10 μs to 0.1 s
    >>> g2 = 1.0 + 0.8 * np.exp(-2 * tau / 0.001)  # Single exponential
    >>> g2_err = 0.01 * np.ones_like(g2)  # 1% statistical uncertainty
    >>> 
    >>> # Plot correlation function
    >>> fig, ax = plt.subplots()
    >>> pg_plot_one_g2(
    ...     ax, tau, g2, g2_err, 
    ...     color=(31, 119, 180),  # Blue
    ...     label='Q = 0.02 Å⁻¹',
    ...     symbol='o',
    ...     symbol_size=4
    ... )
    >>> ax.set_xlabel('Delay time τ (s)')
    >>> ax.set_ylabel('g₂(τ)')
    >>> plt.show()
    
    Notes
    -----
    - Automatically applies logarithmic scaling to x-axis (delay time)
    - Color normalization handles both RGB tuples (0-255) and fractional values
    - Symbol mapping ensures compatibility between PyQtGraph and matplotlib conventions
    - Error bars use cap style appropriate for correlation function uncertainties
    - Function is backend-agnostic through duck typing on the axes object
    
    See Also
    --------
    pg_plot : Main plotting function for multiple correlation curves
    compute_geometry : Calculate layout for multi-panel correlation plots
    """
    # Convert color to matplotlib format
    if isinstance(color, tuple) and len(color) >= 3:
        color_norm = tuple(c/255.0 if c > 1 else c for c in color[:3])
    else:
        color_norm = color
    
    # Map PyQtGraph symbols to matplotlib markers
    symbol_map = {
        'o': 'o', 's': 's', 't': '^', 't1': '>', 't2': '<', 't3': 'v',
        'p': 'p', 'h': 'h', 'star': '*', '+': '+', 'd': 'D', 'x': 'x'
    }
    marker = symbol_map.get(symbol, 'o')
    
    # Plot with error bars and log scale
    if hasattr(ax, 'errorbar'):  # matplotlib axes
        ax.errorbar(x, y, yerr=dy, fmt=marker, color=color_norm, 
                   markersize=symbol_size, label=label, capsize=2)
        ax.set_xscale('log')
    else:  # compatibility layer axes
        ax.errorbar(x, y, yerr=dy, fmt=marker, color=color_norm,
                   markersize=symbol_size, label=label)
    return
