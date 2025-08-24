"""
XPCS Toolkit - Two-Time Correlation Analysis Module (twotime)

This module provides specialized analysis and visualization capabilities for 
two-time correlation functions in X-ray Photon Correlation Spectroscopy (XPCS).
Two-time correlation analysis is essential for studying non-stationary, aging,
and non-equilibrium systems where the dynamics depend on both measurement
time and delay time.

## Scientific Background

Two-time correlation functions extend the standard equal-time analysis to 
capture time-dependent behavior in dynamic systems. The two-time correlation
function is defined as:

    C(t₁, t₂) = ⟨I(t₁)I(t₂)⟩ / ⟨I(t₁)⟩⟨I(t₂)⟩

Where:
    - t₁, t₂: Absolute measurement times
    - I(t): Scattered intensity at time t
    - C(t₁, t₂): Two-time correlation function

## Physical Information Accessible

### Stationary vs Non-stationary Dynamics
- **Stationary systems**: C(t₁, t₂) = g₂(|t₂-t₁|) depends only on time difference
- **Non-stationary systems**: C(t₁, t₂) depends on both t₁ and t₂ independently
- **Aging behavior**: Correlation functions depend on waiting time since preparation

### Time-dependent Phenomena
- **Structural evolution**: Real-time monitoring of phase transitions
- **Gelation processes**: Sol-gel transitions and network formation
- **Crystallization**: Nucleation and growth dynamics
- **Coarsening**: Domain growth and Ostwald ripening
- **Glassy dynamics**: Aging and rejuvenation in supercooled liquids

### System Characterization
- **Ergodicity breaking**: Identification of non-equilibrium behavior
- **Dynamic heterogeneity**: Spatial and temporal variations in dynamics
- **Memory effects**: Dependence on sample history and preparation
- **Thermal equilibration**: Approach to steady-state behavior

## Analysis Capabilities

### Two-Time Correlation Maps
- **Full correlation matrix**: Complete C(t₁, t₂) visualization
- **Diagonal analysis**: Extract standard correlation functions
- **Off-diagonal features**: Identify non-stationary behavior
- **Time evolution**: Monitor changes in correlation structure

### Visualization Features
- **Color-coded intensity maps**: Intuitive representation of correlation strength
- **Logarithmic and linear scaling**: Optimize contrast for different time regimes
- **Interactive region selection**: Extract correlations from specific time windows
- **Real-time updates**: Monitor evolving systems during measurement

### Data Processing
- **Automatic cropping**: Focus on relevant time ranges
- **Background correction**: Remove systematic instrumental effects
- **Statistical weighting**: Account for photon counting uncertainties
- **Temporal binning**: Optimize time resolution vs statistics

## Typical Analysis Workflow

1. **Data Loading**: Import two-time correlation data from XPCS files
2. **Visualization Setup**: Configure display parameters and scaling
3. **Region Selection**: Choose time windows for detailed analysis
4. **Correlation Extraction**: Extract g₂(τ) functions from selected regions
5. **Stationarity Assessment**: Check for time-translation invariance
6. **Dynamic Analysis**: Interpret correlation patterns and evolution

## Applications

### Soft Matter Physics
- **Polymer melts**: Reptation dynamics and entanglement effects
- **Colloid-polymer mixtures**: Depletion interactions and phase behavior
- **Active matter**: Non-equilibrium dynamics in driven systems
- **Granular materials**: Jamming and unjamming transitions

### Materials Science
- **Glass-forming liquids**: Dynamic heterogeneity and cooperative motion
- **Crystallizing systems**: Nucleation kinetics and growth mechanisms
- **Phase-separating alloys**: Spinodal decomposition and coarsening
- **Nanocomposites**: Filler network formation and percolation

### Biological Systems
- **Living tissues**: Cell migration and tissue remodeling
- **Protein aggregation**: Amyloid formation and fibril growth
- **Membrane dynamics**: Lipid diffusion and phase separation
- **Bacterial colonies**: Collective behavior and biofilm formation

### Industrial Processes
- **Curing reactions**: Cross-linking kinetics in thermosets
- **Crystallization control**: Pharmaceutical polymorph selection
- **Coating formation**: Film evolution and defect healing
- **Emulsion stability**: Droplet coalescence and creaming

## Module Functions

The module provides the following key functions:

- `plot_twotime()`: Main two-time correlation visualization function
- `plot_twotime_g2()`: Extract and plot correlation functions from selected regions

## Usage Examples

```python
# Two-time correlation analysis
from xpcs_toolkit.module import twotime

# Visualize two-time correlation map
twotime.plot_twotime(
    xfile=aging_data,
    hdl=matplotlib_handles,
    scale='log',           # Logarithmic intensity scaling
    auto_crop=True,        # Focus on relevant time range
    cmap='viridis',        # Scientific colormap
    autolevel=True,        # Automatic contrast adjustment
    correct_diag=False     # Diagonal correction for artifacts
)

# Extract correlation functions from specific time regions
twotime.plot_twotime_g2(plot_handles, c2_results)
```

## References

- Cipelletti et al., "Universal aging features in the restructuring of fractal colloidal gels" (2000)
- Bandyopadhyay et al., "Speckle-visibility spectroscopy: A tool to study time-varying dynamics" (2005)
- Fluerasu et al., "Slow dynamics and aging in colloidal gels studied by x-ray photon correlation spectroscopy" (2007)
- Madsen et al., "Beyond simple exponential correlation functions and equilibrium dynamics in x-ray photon correlation spectroscopy" (2010)

## Author

XPCS Toolkit Development Team
Advanced Photon Source, Argonne National Laboratory
"""

import numpy as np
from ..mpl_compat import mkPen
import matplotlib.pyplot as plt

PG_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot_twotime(
    xfile,
    hdl,
    scale="log",
    auto_crop=True,
    highlight_xy=None,
    cmap="jet",
    vmin=None,
    vmax=None,
    autolevel=True,
    correct_diag=False,
    selection=0,
):
    """
    Visualize two-time correlation functions and extract dynamic information.
    
    This function creates comprehensive visualizations of two-time correlation data,
    including correlation maps, SAXS patterns, and extracted correlation functions.
    It provides tools for analyzing non-stationary dynamics, aging behavior, and
    time-dependent phenomena in XPCS measurements.
    
    The visualization includes multiple panels showing different aspects of the
    two-time correlation analysis: the full correlation matrix C(t₁,t₂), associated
    SAXS scattering patterns, and extracted correlation functions g₂(τ).
    
    Parameters
    ----------
    xfile : XpcsDataFile
        XPCS data file containing two-time correlation data. Must have analysis
        type containing "Twotime" and appropriate two-time correlation datasets.
    hdl : dict
        Dictionary of plot widget handles for different visualization panels.
        Expected keys: 'saxs', 'dqmap', 'tt', 'c2g2' for different display components.
    scale : str, optional
        Intensity scaling for correlation map display. Default: 'log'.
        Options: 'log' (logarithmic), 'linear' (linear intensity scaling).
    auto_crop : bool, optional
        Automatically crop correlation map to relevant time range. Default: True.
        Removes empty regions and focuses on data with significant correlations.
    highlight_xy : tuple, optional
        Coordinates (x, y) to highlight in the correlation map. Default: None.
        Used for marking specific time points or regions of interest.
    cmap : str, optional
        Colormap name for correlation map visualization. Default: 'jet'.
        Options: 'viridis', 'plasma', 'hot', 'jet', etc. (matplotlib colormaps).
    vmin : float, optional
        Minimum value for color scale. Default: None (automatic).
        Manual control over correlation intensity range display.
    vmax : float, optional
        Maximum value for color scale. Default: None (automatic).
        Manual control over correlation intensity range display.
    autolevel : bool, optional
        Enable automatic level adjustment for optimal contrast. Default: True.
        When False, uses vmin/vmax values or percentile-based scaling.
    correct_diag : bool, optional
        Apply diagonal correction to remove instrumental artifacts. Default: False.
        Useful for removing systematic effects along the t₁=t₂ diagonal.
    selection : int, optional
        Index for region selection in correlation analysis. Default: 0.
        Used for extracting correlations from specific spatial or temporal regions.
    
    Returns
    -------
    None
        Function performs visualization operations directly on the provided handles.
        Updates multiple display panels with correlation maps and extracted functions.
    
    Raises
    ------
    AssertionError
        If the input file does not contain "Twotime" in its analysis type.
    
    Examples
    --------
    >>> # Basic two-time correlation visualization
    >>> plot_twotime(
    ...     xfile=aging_data,
    ...     hdl=matplotlib_handles,
    ...     scale='log',
    ...     auto_crop=True,
    ...     cmap='viridis'
    ... )
    >>> 
    >>> # Custom intensity range for detailed analysis
    >>> plot_twotime(
    ...     xfile=gelation_data,
    ...     hdl=matplotlib_handles,
    ...     scale='linear',
    ...     autolevel=False,
    ...     vmin=0.98,
    ...     vmax=1.5,
    ...     correct_diag=True
    ... )
    
    Notes
    -----
    - Two-time correlation maps reveal non-stationary dynamics through off-diagonal features
    - Diagonal elements correspond to standard equal-time correlation functions
    - Aging systems show characteristic triangular or wedge-shaped correlation patterns
    - Color intensity represents correlation strength: darker = stronger correlation
    - Interactive selection allows extraction of correlation functions from specific regions
    
    See Also
    --------
    plot_twotime_g2 : Extract and plot correlation functions from correlation maps
    """
    if "Twotime" not in xfile.atype:
        raise AssertionError(f"Input file has analysis type '{xfile.atype}' but requires 'Twotime' for two-time correlation analysis")

    # display dqmap and saxs
    dqmap_disp, saxs, selection_xy = xfile.get_twotime_maps(
        scale=scale,
        auto_crop=auto_crop,
        highlight_xy=highlight_xy,
        selection=selection,
    )

    if selection_xy is not None:
        selection = selection_xy

    hdl["saxs"].setImage(np.flipud(saxs))
    hdl["dqmap"].setImage(dqmap_disp)

    c2_result = xfile.get_twotime_c2(selection=selection, correct_diag=correct_diag)
    if c2_result is None:
        return None

    c2, delta_t = c2_result["c2_mat"], c2_result["delta_t"]

    hdl["tt"].imageItem.setScale(delta_t)
    hdl["tt"].setImage(c2, autoRange=True)

    # cmap = matplotlib colormap (compatibility stub)
    # cmap = pg.colormap.getFromMatplotlib(cmap)
    hdl["tt"].setColorMap(cmap)
    hdl["tt"].ui.histogram.setHistogramRange(mn=0, mx=3)
    if not autolevel and vmin is not None and vmax is not None:
        hdl["tt"].setLevels(min=vmin, max=vmax)
    else:
        vmin, vmax = np.percentile(c2, [0.5, 99.5])
        hdl["tt"].setLevels(min=vmin, max=vmax)
    plot_twotime_g2(hdl, c2_result)


def plot_twotime_g2(hdl, c2_result):
    """
    Extract and visualize correlation functions from two-time correlation analysis.
    
    This function processes two-time correlation results to extract standard
    correlation functions g₂(τ) from selected regions or time windows. It displays
    both full and partial correlation functions, enabling comparison between
    different temporal regimes or spatial regions within the sample.
    
    The visualization helps identify time-dependent changes in dynamics and
    assess the stationarity of the system by comparing correlation functions
    extracted from different time periods.
    
    Parameters
    ----------
    hdl : dict
        Dictionary containing plot widget handles. Must include 'c2g2' key
        for the correlation function display panel.
    c2_result : dict
        Dictionary containing two-time correlation analysis results with keys:
        - 'g2_full': Full correlation function array
        - 'g2_partial': Array of partial correlation functions from different regions
        - 'acquire_period': Time resolution of the measurement (seconds)
        - Additional metadata from two-time analysis
    
    Returns
    -------
    None
        Function performs plotting operations directly on the provided handle.
        Updates the correlation function display with multiple curves.
    
    Examples
    --------
    >>> # Extract and plot correlation functions
    >>> c2_results = xfile.get_twotime_c2(selection=0)
    >>> plot_twotime_g2(display_handles, c2_results)
    
    Notes
    -----
    - Full correlation function represents average dynamics over entire measurement
    - Partial correlation functions show dynamics from specific time windows
    - Differences between partial functions indicate non-stationary behavior
    - Logarithmic time axis reveals dynamics across multiple decades
    - Color coding distinguishes between different correlation functions
    
    See Also
    --------
    plot_twotime : Main two-time correlation visualization function
    """
    g2_full, g2_partial = c2_result["g2_full"], c2_result["g2_partial"]

    hdl["c2g2"].clear()
    hdl["c2g2"].setLabel("left", "g2")
    hdl["c2g2"].setLabel("bottom", "t (s)")
    acquire_period = c2_result["acquire_period"]

    xaxis = np.arange(g2_full.size) * acquire_period
    hdl["c2g2"].plot(
        x=xaxis[1:],
        y=g2_full[1:],
        **mkPen(color=PG_COLORS[-1], width=4),
        label="g2_full",
    )
    for n in range(g2_partial.shape[0]):
        xaxis = np.arange(g2_partial.shape[1]) * acquire_period
        hdl["c2g2"].plot(
            x=xaxis[1:],
            y=g2_partial[n][1:],
            **mkPen(color=PG_COLORS[n], width=1),
            label=f"g2_partial_{n}",
        )
    hdl["c2g2"].setLogMode(x=True, y=False)
    hdl["c2g2"].autoRange()
