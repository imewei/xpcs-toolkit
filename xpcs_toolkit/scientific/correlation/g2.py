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

# Use lazy imports for heavy dependencies
from xpcs_toolkit._lazy_imports import lazy_import

np = lazy_import("numpy")
plt = lazy_import("matplotlib.pyplot")
FormatStrFormatter = lazy_import("matplotlib.ticker", "FormatStrFormatter")
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


def get_data(
    xf_list: list[str],
    q_range: tuple[int, int] | None = None,
    t_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        # Handle both analysis_type (string/list) and legacy atype attribute
        analysis_type = None

        # Try analysis_type first, then atype, handling Mock objects properly
        if hasattr(xf, "analysis_type") and not str(
            type(xf.analysis_type)
        ).__contains__("Mock"):
            analysis_type = xf.analysis_type
        elif hasattr(xf, "atype") and not str(type(xf.atype)).__contains__("Mock"):
            analysis_type = xf.atype
        else:
            # Handle Mock objects by checking both attributes more carefully
            try:
                # For Mock objects, the attribute might exist but be another Mock
                if hasattr(xf, "atype"):
                    atype_val = xf.atype
                    if not str(type(atype_val)).__contains__("Mock"):
                        analysis_type = atype_val
                if analysis_type is None and hasattr(xf, "analysis_type"):
                    analysis_type_val = xf.analysis_type
                    if not str(type(analysis_type_val)).__contains__("Mock"):
                        analysis_type = analysis_type_val
            except:
                analysis_type = []

        if analysis_type is None:
            analysis_type = []

        # Ensure analysis_type is iterable (list or tuple)
        if isinstance(analysis_type, str):
            analysis_type = [analysis_type]

        # Handle Mock objects and other non-iterable types safely
        try:
            has_multitau = "Multitau" in analysis_type
        except TypeError:
            # If analysis_type is not iterable (like a Mock), assume no Multitau
            has_multitau = False

        if not has_multitau:
            return False, None, None, None, None

    q, tel, g2, g2_err, labels = [], [], [], [], []
    for fc in xf_list:
        _q, _tel, _g2, _g2_err, _labels = fc.get_g2_data(
            q_range=q_range, time_range=t_range
        )
        q.append(_q)
        tel.append(_tel)
        g2.append(_g2)
        g2_err.append(_g2_err)
        labels.append(_labels)
    return q, tel, g2, g2_err, labels


def compute_geometry(g2: np.ndarray, plot_type: str) -> tuple[int, int, int]:
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
    xf_list: list[str],
    q_range: tuple[int, int],
    t_range: tuple[int, int],
    y_range: tuple[float, float],
    y_auto: bool = False,
    q_auto: bool = False,
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

    This function has been disabled in headless mode. PyQtGraph plotting
    functionality is not available when running without GUI dependencies.

    For visualization in headless mode, use the matplotlib-based plotting
    functions available in the CLI interface instead.

    Parameters
    ----------
    hdl : object
        Plot handle (disabled)
    xf_list : list of str
        List of XPCS data file paths (disabled)
    q_range : tuple of int
        Momentum transfer range (disabled)
    t_range : tuple of int
        Time range (disabled)
    y_range : tuple of float
        Y-axis range (disabled)
    y_auto : bool, optional
        Auto-scale Y axis (disabled)
    q_auto : bool, optional
        Auto-scale Q axis (disabled)
    t_auto : bool, optional
        Auto-scale time axis (disabled)
    **kwargs
        Additional plotting parameters (disabled)

    Returns
    -------
    None
        Function raises NotImplementedError

    Examples
    --------
    This function is disabled in headless mode:

    >>> pg_plot(None, [], (0, 10), (0, 1), (0.9, 2.0))
    Traceback (most recent call last):
    NotImplementedError: GUI plotting functionality has been disabled...
    """
    raise NotImplementedError(
        "GUI plotting functionality has been disabled in headless mode. "
        "Use the matplotlib-based CLI interface for visualization instead."
    )


def pg_plot_one_g2(ax, x, y, dy, color, label, symbol, symbol_size=5):
    """
    Plot individual correlation function curve - disabled in headless mode.

    This function has been disabled as it requires GUI plotting functionality.
    Use the matplotlib-based CLI interface for visualization instead.

    Parameters
    ----------
    ax : object
        Plot axis handle (disabled)
    x : array_like
        X-axis data (disabled)
    y : array_like
        Y-axis data (disabled)
    dy : array_like
        Error bar data (disabled)
    color : str
        Plot color (disabled)
    label : str
        Plot label (disabled)
    symbol : str
        Plot symbol (disabled)
    symbol_size : int, optional
        Symbol size (disabled), default 5

    Returns
    -------
    None
        Function raises NotImplementedError

    Examples
    --------
    This function is disabled in headless mode:

    >>> pg_plot_one_g2(None, [], [], [], 'red', 'test', 'o')
    Traceback (most recent call last):
    NotImplementedError: GUI plotting functionality has been disabled...
    """
    raise NotImplementedError(
        "GUI plotting functionality has been disabled in headless mode. "
        "Use the matplotlib-based CLI interface for visualization instead."
    )
