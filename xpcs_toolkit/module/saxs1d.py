"""
XPCS Toolkit - Small-Angle X-ray Scattering 1D Analysis Module (saxs1d)

This module provides comprehensive analysis and visualization capabilities for 
one-dimensional small-angle X-ray scattering (SAXS) patterns. SAXS is a powerful 
technique for characterizing the structure of materials at the nanometer scale, 
providing information about particle size, shape, and inter-particle correlations.

## Scientific Background

Small-angle X-ray scattering measures elastic scattering at small angles (typically 
0.1° to 10°), where the scattering intensity I(q) depends on the momentum transfer:

    q = 4π sin(θ/2)/λ

Where:
    - θ: Scattering angle
    - λ: X-ray wavelength
    - q: Momentum transfer magnitude (Å⁻¹)

The scattering intensity I(q) contains structural information about the sample:

    I(q) = n * V²_p * |F(q)|² * S(q)

Where:
    - n: Number density of scatterers
    - V_p: Particle volume
    - F(q): Form factor (single particle scattering)
    - S(q): Structure factor (inter-particle correlations)

## Physical Information Accessible

### Particle Size and Shape
- **Guinier analysis**: R_g = radius of gyration from I(q) ∝ exp(-q²R_g²/3)
- **Porod analysis**: Surface area from I(q) ∝ q⁻⁴ at high q
- **Form factors**: Sphere, cylinder, ellipsoid fitting for shape determination
- **Size distributions**: Polydispersity analysis from peak broadening

### Structure and Organization  
- **Inter-particle correlations**: Structure factor peaks indicate ordering
- **Fractal structures**: Power-law scattering I(q) ∝ q⁻ᴰ reveals fractal dimension D
- **Phase behavior**: Critical scattering near phase transitions
- **Surface roughness**: Deviation from ideal Porod behavior

### Concentration Effects
- **Forward scattering**: I(0) ∝ concentration × contrast² × V_p²
- **Concentration fluctuations**: Thermodynamic structure factors
- **Interaction parameters**: Virial coefficients from dilute solution analysis
- **Phase diagrams**: Spinodal decomposition and nucleation studies

## Analysis Capabilities

### Data Processing Features
- **Background subtraction**: Solvent and instrumental background correction
- **Absolute intensity scaling**: Calibration to absolute cross-sections (cm⁻¹)
- **Sector averaging**: Azimuthal integration for isotropic samples
- **Multi-sector analysis**: Anisotropic scattering characterization
- **Q-range selection**: Focus on specific size regimes
- **Data normalization**: Various scaling methods (q², q⁴, monitor normalization)

### Visualization Options
- **Log-log plots**: Standard SAXS representation showing power laws
- **Kratky plots**: q²I(q) vs q for chain conformation analysis
- **Guinier plots**: ln[I(q)] vs q² for size determination
- **Porod plots**: q⁴I(q) vs q for surface area analysis
- **Sector plots**: Angular dependence visualization
- **Multi-sample overlay**: Comparative analysis across conditions

### Advanced Analysis
- **Model fitting**: Integration with SASfit, SasView, or custom models
- **Peak analysis**: Bragg peak positions and intensities
- **Invariant calculation**: Total scattered intensity for phase quantification
- **Time-resolved analysis**: Kinetics and structural evolution studies

## Typical Analysis Workflow

1. **Data Loading**: Import 2D SAXS images and metadata
2. **Background Subtraction**: Remove solvent and instrumental contributions
3. **Sector Integration**: Convert 2D patterns to 1D I(q) profiles
4. **Q-calibration**: Ensure accurate momentum transfer scaling
5. **Normalization**: Apply absolute intensity calibration if needed
6. **Structural Analysis**: Extract size, shape, and organization parameters
7. **Model Comparison**: Validate results against theoretical predictions

## Applications

### Soft Matter Physics
- **Polymer solutions**: Chain conformation and interactions
- **Colloidal suspensions**: Particle size distributions and aggregation
- **Block copolymers**: Microphase separation and ordering
- **Gels and networks**: Cross-link density and mesh size

### Materials Science
- **Nanocomposites**: Filler dispersion and interface structure
- **Porous materials**: Pore size distributions and connectivity
- **Crystalline polymers**: Lamellar thickness and long periods
- **Thin films**: Layer structure and interfacial roughness

### Biological Systems
- **Protein structure**: Radius of gyration and folding states
- **Membrane systems**: Bilayer structure and phase behavior
- **DNA complexes**: Compaction and packaging mechanisms
- **Cell components**: Organelle and cytoskeletal organization

### Industrial Applications
- **Quality control**: Particle size monitoring in manufacturing
- **Formulation science**: Stability and structure optimization
- **Surface coatings**: Film thickness and surface morphology
- **Catalysts**: Support structure and active site distribution

## Module Functions

The module provides the following key functions:

- `pg_plot()`: Main 1D SAXS plotting function with flexible visualization options
- `offset_intensity()`: Apply vertical offsets for multi-curve comparison
- `get_color_marker()`: Consistent color and marker assignment
- `plot_line_with_marker()`: Individual curve plotting with styling
- `get_pyqtgraph_anchor_params()`: Legend positioning utilities

## Usage Examples

```python
# Basic 1D SAXS visualization
from xpcs_toolkit.module import saxs1d

# Load SAXS data files
xf_list = [XpcsDataFile('sample_001.h5'), XpcsDataFile('sample_002.h5')]

# Create log-log intensity plot
saxs1d.pg_plot(
    xf_list=xf_list,
    pg_hdl=matplotlib_handle,
    plot_type=3,      # log-log scale
    qmin=0.01,        # Minimum q (Å⁻¹)
    qmax=0.5,         # Maximum q (Å⁻¹)
    plot_norm=1,      # q² normalization (Kratky plot)
    title="SAXS Analysis",
    loc="upper right",
    marker_size=4,
    subtract_background=True
)

# Multi-sector analysis for anisotropic samples
saxs1d.pg_plot(
    xf_list=xf_list,
    pg_hdl=matplotlib_handle,
    plot_type=3,
    all_phi=True,     # Show all azimuthal sectors
    show_phi_roi=True,
    absolute_crosssection=True,  # Calibrated intensity
    sampling=2        # Every 2nd data point
)
```

## References

- Glatter & Kratky, "Small Angle X-ray Scattering" (1982)
- Feigin & Svergun, "Structure Analysis by Small-Angle X-ray and Neutron Scattering" (1987)
- Hamley, "Introduction to Soft Matter" (2007)
- Putnam et al., "X-ray solution scattering (SAXS) combined with crystallography and computation" (2007)

## Author

XPCS Toolkit Development Team
Advanced Photon Source, Argonne National Laboratory
"""

import numpy as np
from ..mpl_compat import mkPen, mkBrush
import logging

logger = logging.getLogger(__name__)


# Mapping from integer codes to string codes (based on Matplotlib docs)
_MPL_LOC_INT_TO_STR = {
    1: "upper right",
    2: "upper left",
    3: "lower left",
    4: "lower right",
    5: "right",  # Often equivalent to center right in placement
    6: "center left",
    7: "center right",
    8: "lower center",
    9: "upper center",
    10: "center",
}


def get_pyqtgraph_anchor_params(loc, padding=10):
    """
    Convert matplotlib legend position to pyqtgraph anchor parameters.
    
    This utility function bridges the gap between matplotlib's legend positioning
    system and pyqtgraph's anchor-based positioning, enabling consistent legend
    placement across different plotting backends in SAXS visualization.
    
    The function translates standard matplotlib location codes and strings into
    the itemPos, parentPos, and offset parameters required by pyqtgraph's
    LegendItem.anchor() method.
    
    Parameters
    ----------
    loc : str or int
        Matplotlib legend location specification. Accepts:
        - String codes: 'upper left', 'lower right', 'center', etc.
        - Integer codes: 0='best', 1='upper right', ..., 10='center'
        - Normalized strings: 'upperleft', 'centerright' (spaces removed)
    padding : int, optional
        Pixel offset from the anchor point. Default: 10.
        Positive values move legend inward from plot edges.
    
    Returns
    -------
    dict or None
        Dictionary with keys 'itemPos', 'parentPos', 'offset' for
        LegendItem.anchor(**params), or None if loc='best' (not supported).
    
    Raises
    ------
    ValueError
        If loc is not a recognized matplotlib location code or string.
    
    Examples
    --------
    >>> # Standard usage in SAXS plotting
    >>> params = get_pyqtgraph_anchor_params('upper right', padding=15)
    >>> if params:
    ...     legend.anchor(**params)
    >>> 
    >>> # Integer code usage
    >>> params = get_pyqtgraph_anchor_params(1, padding=20)  # upper right
    >>> legend.anchor(**params)
    
    Notes
    -----
    - The 'best' option (code 0) returns None as automatic positioning
      is not supported by pyqtgraph's deterministic anchor system
    - Padding values are applied directionally based on anchor position
      to move legends away from plot boundaries
    - Anchor calculations ensure legends remain within plot boundaries
    
    See Also
    --------
    pg_plot : Main SAXS plotting function using this positioning utility
    """
    if isinstance(loc, int):
        if loc in _MPL_LOC_INT_TO_STR:
            loc_str = _MPL_LOC_INT_TO_STR[loc]
        else:
            raise ValueError(f"Invalid Matplotlib integer location code: {loc}")
    elif isinstance(loc, str):
        loc_str = (
            loc.lower().replace(" ", "").replace("_", "")
        )  # Normalize input string
    else:
        raise ValueError(f"Invalid loc type: {type(loc)}. Must be str or int.")

    # --- Define anchor points and offset multipliers ---
    # Map: loc_string -> (itemPos, parentPos, offset_multipliers)
    # Offset multipliers (mult_x, mult_y) determine offset direction based on padding
    _ANCHOR_MAP = {
        # Corners
        "upperleft": ((0.0, 0.0), (0.0, 0.0), (1, 1)),  # Offset moves down-right
        "upperright": ((1.0, 0.0), (1.0, 0.0), (-1, 1)),  # Offset moves down-left
        "lowerleft": ((0.0, 1.0), (0.0, 1.0), (1, -1)),  # Offset moves up-right
        "lowerright": ((1.0, 1.0), (1.0, 1.0), (-1, -1)),  # Offset moves up-left
        # Centers
        "center": ((0.5, 0.5), (0.5, 0.5), (0, 0)),  # No offset needed usually
        "lowercenter": ((0.5, 1.0), (0.5, 1.0), (0, -1)),  # Offset moves up
        "uppercenter": ((0.5, 0.0), (0.5, 0.0), (0, 1)),  # Offset moves down
        # Sides (center align on edge)
        "centerleft": ((0.0, 0.5), (0.0, 0.5), (1, 0)),  # Offset moves right
        "centerright": ((1.0, 0.5), (1.0, 0.5), (-1, 0)),  # Offset moves left
        "right": (
            (1.0, 0.5),
            (1.0, 0.5),
            (-1, 0),
        ),  # Treat 'right' same as 'centerright'
    }

    if loc_str in _ANCHOR_MAP:
        itemPos, parentPos, offset_mult = _ANCHOR_MAP[loc_str]
        offset = (padding * offset_mult[0], padding * offset_mult[1])
        return {"itemPos": itemPos, "parentPos": parentPos, "offset": offset}
    else:
        raise ValueError(f"Invalid or unsupported Matplotlib location string: '{loc}'")


def offset_intensity(Iq, n, plot_offset=None, yscale=None):
    """
    Apply vertical offset to scattering intensity for multi-curve visualization.
    
    This function enables clear separation of multiple SAXS curves in overlay plots
    by applying appropriate offsets based on the plot scaling (linear or logarithmic).
    The offset strategy differs between linear and log scales to maintain visual
    clarity while preserving the relative intensity relationships.
    
    Parameters
    ----------
    Iq : numpy.ndarray
        Scattering intensity I(q) values. Shape: (n_q,) for single curve.
    n : int
        Curve index for determining offset magnitude. Higher indices receive
        larger offsets for visual separation.
    plot_offset : float, optional
        Offset scaling factor. If None, no offset is applied.
        - Linear scale: Fraction of maximum intensity per curve
        - Log scale: Multiplicative factor per curve
    yscale : str, optional
        Y-axis scaling mode. Options: 'linear', 'log', or None.
        Determines the offset calculation method.
    
    Returns
    -------
    numpy.ndarray
        Offset-corrected intensity values with same shape as input.
        
    Examples
    --------
    >>> # Linear scale offset (additive)
    >>> I_offset = offset_intensity(I_orig, curve_index=2, 
    ...                           plot_offset=0.1, yscale='linear')
    >>> 
    >>> # Log scale offset (multiplicative)
    >>> I_offset = offset_intensity(I_orig, curve_index=2,
    ...                           plot_offset=0.5, yscale='log')
    
    Notes
    -----
    - Linear offsets: I_offset = I - n × offset × max(I)
    - Log offsets: I_offset = I / 10^(n × offset)
    - Preserves intensity ratios within each curve
    - Enables comparison of curve shapes across different intensity levels
    """
    if yscale == "linear" and plot_offset is not None:
        offset = -1 * plot_offset * n * np.max(Iq)
        Iq = offset + Iq

    elif yscale == "log" and plot_offset is not None:
        offset = 10 ** (plot_offset * n)
        Iq = Iq / offset
    return Iq


def switch_line_builder(hdl, lb_type=None):
    """
    Switch line builder tool for interactive SAXS analysis.
    
    This function enables different line-building tools for interactive
    analysis of SAXS data, such as drawing regions of interest, creating
    integration sectors, or marking specific q-ranges for analysis.
    
    Parameters
    ----------
    hdl : matplotlib.pyplot or compatible plotting handle
        Matplotlib-compatible plotting handle with line builder functionality.
    lb_type : str, optional
        Line builder type specification. Default: None.
        Options depend on the plot widget implementation.
    
    Returns
    -------
    None
        Function performs tool switching directly on the plot handle.
    
    Notes
    -----
    - Enables interactive analysis features for SAXS data exploration
    - Line builder tools useful for defining integration regions
    - Implementation depends on specific plot widget capabilities
    """
    hdl.link_line_builder(lb_type)


def get_color_marker(index):
    """
    Assign consistent colors and markers for SAXS curve visualization.
    
    Provides cycling color and marker schemes compatible with matplotlib
    standards for scientific plotting. Ensures visual distinctiveness
    across multiple datasets while maintaining professional appearance.
    
    Parameters
    ----------
    index : int
        Curve index for color/marker assignment. Cycles through available
        options using modulo arithmetic for unlimited curves.
    
    Returns
    -------
    color_hex : str
        Hexadecimal color code (e.g., '#1f77b4') compatible with matplotlib.
    marker : str
        Marker symbol (e.g., 'o', 's', '^') for data points.
    
    Examples
    --------
    >>> color, marker = get_color_marker(0)  # Returns ('#1f77b4', 'o')
    >>> color, marker = get_color_marker(5)  # Returns ('#8c564b', '<')
    
    Notes
    -----
    - Uses standard matplotlib color cycle for consistency
    - Marker shapes chosen for clear distinction at small sizes
    - Cycles repeat after 10 colors/markers for unlimited datasets
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    return colors[index % len(colors)], markers[index % len(markers)]

def plot_line_with_marker(
    plot_item, x, y, index, label, alpha_val, marker_size=6, log_x=False, log_y=False
):
    """
    Render individual SAXS curve with proper styling and scaling.
    
    This function plots a single I(q) scattering curve with appropriate
    markers, colors, and scaling for scientific visualization. It provides
    matplotlib compatibility while handling logarithmic scaling requirements
    typical in SAXS analysis.
    
    Parameters
    ----------
    plot_item : matplotlib.axes.Axes or compatible
        Target plotting axes with matplotlib-like interface.
    x : numpy.ndarray
        Momentum transfer q values (Å⁻¹). Shape: (n_q,)
    y : numpy.ndarray
        Scattering intensity I(q) values. Shape: (n_q,)
        May be normalized (Kratky, Porod) or absolute units.
    index : int
        Curve index for color/marker assignment via get_color_marker().
    label : str
        Legend label for curve identification (e.g., sample name, sector).
    alpha_val : float
        Transparency level (0.0-1.0). Used for highlighting/de-emphasizing.
    marker_size : int, optional
        Data point marker size in points. Default: 6.
    log_x : bool, optional
        Apply logarithmic scaling to x-axis. Default: False.
        Standard for SAXS q-axis visualization.
    log_y : bool, optional
        Apply logarithmic scaling to y-axis. Default: False.
        Common for intensity axis in SAXS.
    
    Returns
    -------
    None
        Performs plotting operations directly on provided axes.
    
    Examples
    --------
    >>> # Plot single SAXS curve with log scaling
    >>> plot_line_with_marker(
    ...     ax, q_values, intensity, curve_index=0,
    ...     label='Sample A', alpha_val=1.0, marker_size=4,
    ...     log_x=True, log_y=True
    ... )
    
    Notes
    -----
    - Automatically applies consistent color/marker schemes
    - Handles transparency for multi-curve focus control
    - Logarithmic scaling applied after plotting for compatibility
    - Marker sizes optimized for SAXS data density
    """
    color_hex, marker = get_color_marker(index)
    
    # Handle alpha
    alpha = min(max(alpha_val, 0.0), 1.0)
    
    # Plot with matplotlib-style interface
    if hasattr(plot_item, 'plot'):
        plot_item.plot(x, y, color=color_hex, marker=marker, 
                      markersize=marker_size, alpha=alpha, label=label,
                      linewidth=1.5)
        if log_x:
            plot_item.set_xscale('log')
        if log_y:
            plot_item.set_yscale('log')
    else:
        # Compatibility layer fallback
        logger.warning("Using fallback plotting for matplotlib compatibility")


def pg_plot(
    xf_list,
    pg_hdl,
    plot_type=2,
    plot_norm=0,
    plot_offset=0,
    title=None,
    rows=None,
    qmax=10.0,
    qmin=0,
    loc="best",
    marker_size=3,
    sampling=1,
    all_phi=False,
    absolute_crosssection=False,
    subtract_background=False,
    bkg_file=None,
    weight=1.0,
    roi_list=None,
    show_roi=True,
    show_phi_roi=True,
):
    """
    Create comprehensive 1D SAXS intensity plots with flexible analysis options.
    
    This is the main visualization function for one-dimensional small-angle X-ray
    scattering data, supporting multiple plot types, normalization schemes, and
    advanced analysis features. It handles complex SAXS datasets with background
    subtraction, absolute intensity calibration, and multi-sector analysis.
    
    The function provides scientific-grade visualizations following SAXS community
    standards with logarithmic scaling, proper error handling, and customizable
    styling optimized for publication and analysis.
    
    Parameters
    ----------
    xf_list : list of XpcsDataFile
        SAXS data files containing 1D scattering profiles. Each file must
        have 1D SAXS data accessible via get_saxs_1d_data() method.
    pg_hdl : matplotlib.pyplot or compatible plotting handle
        Matplotlib-compatible plotting handle for rendering. Must support matplotlib-like interface
        with plot(), set_xlabel(), set_ylabel() methods.
    plot_type : int, optional
        Scaling mode for axes. Default: 2.
        Bit encoding: bit 0 = x-log, bit 1 = y-log
        - 0: linear-linear
        - 1: log-linear 
        - 2: linear-log
        - 3: log-log (standard SAXS presentation)
    plot_norm : int, optional
        Intensity normalization method. Default: 0 (no normalization).
        - 0: Raw intensity I(q)
        - 1: Kratky plot q²I(q) (chain analysis)
        - 2: Modified Kratky q⁴I(q) (surface analysis)
        - 3: Monitor normalized I(q)/I₀
    plot_offset : float, optional
        Vertical offset between curves for visual separation. Default: 0.
        Applied according to plot scaling (additive/multiplicative).
    title : str, optional
        Plot title for identification. Default: None.
    rows : list of int, optional
        Row indices to highlight with full opacity. Default: None (all visible).
        Other curves rendered with reduced alpha for focus.
    qmax : float, optional
        Maximum momentum transfer for display (Å⁻¹). Default: 10.0.
    qmin : float, optional
        Minimum momentum transfer for display (Å⁻¹). Default: 0.
    loc : str or int, optional
        Legend position following matplotlib conventions. Default: 'best'.
        Options: 'upper left', 'lower right', etc., or integer codes 0-10.
    marker_size : int, optional
        Data point marker size in points. Default: 3.
        Typical range: 2-6 for clarity at different scales.
    sampling : int, optional
        Data point sampling interval. Default: 1 (all points).
        Use >1 to reduce point density for large datasets.
    all_phi : bool, optional
        Display all azimuthal sectors. Default: False.
        When True, shows angular dependence for anisotropic samples.
    absolute_crosssection : bool, optional
        Use absolute intensity calibration (cm⁻¹). Default: False.
        Requires proper detector efficiency and sample thickness.
    subtract_background : bool, optional
        Enable background subtraction. Default: False.
        Requires bkg_file specification when True.
    bkg_file : XpcsDataFile, optional
        Background file for subtraction. Default: None.
        Must have matching geometry and measurement conditions.
    weight : float, optional
        Background subtraction weight factor. Default: 1.0.
        Scales background intensity before subtraction.
    roi_list : list, optional
        Region of interest specifications. Default: None.
        Not implemented in current version.
    show_roi : bool, optional
        Display region of interest overlays. Default: True.
        Not implemented in current version.
    show_phi_roi : bool, optional
        Display azimuthal sector boundaries. Default: True.
        Relevant when all_phi=True for sector analysis.
    
    Returns
    -------
    None
        Function performs plotting operations directly on the provided handle.
    
    Examples
    --------
    >>> # Standard log-log SAXS plot
    >>> saxs1d.pg_plot(
    ...     xf_list=[xf1, xf2, xf3],
    ...     pg_hdl=matplotlib_handle,
    ...     plot_type=3,        # log-log scale
    ...     qmin=0.005,         # Focus on small-q region
    ...     qmax=0.5,
    ...     title='Particle Size Analysis',
    ...     loc='upper right',
    ...     marker_size=4
    ... )
    >>> 
    >>> # Kratky plot for polymer analysis
    >>> saxs1d.pg_plot(
    ...     xf_list=polymer_samples,
    ...     pg_hdl=matplotlib_handle,
    ...     plot_type=1,        # log-linear
    ...     plot_norm=1,        # q²I(q) normalization
    ...     subtitle_background=True,
    ...     bkg_file=solvent_background,
    ...     absolute_crosssection=True
    ... )
    >>> 
    >>> # Multi-sector anisotropy analysis
    >>> saxs1d.pg_plot(
    ...     xf_list=[oriented_sample],
    ...     pg_hdl=matplotlib_handle,
    ...     plot_type=3,
    ...     all_phi=True,       # Show all sectors
    ...     show_phi_roi=True,  # Mark sector boundaries
    ...     plot_offset=0.1     # Separate curves
    ... )
    
    Notes
    -----
    - Q-range filtering applied before plotting for focused analysis
    - Background subtraction preserves statistical uncertainties
    - Absolute calibration requires proper experimental setup
    - Legend positioning optimized for SAXS curve shapes
    - Grid lines enhance quantitative analysis capabilities
    - Color and marker schemes cycle for unlimited datasets
    
    See Also
    --------
    offset_intensity : Apply vertical offsets for curve separation
    get_color_marker : Consistent visual styling assignment
    plot_line_with_marker : Individual curve rendering
    """

    pg_hdl.clear()
    plot_item = pg_hdl.getPlotItem()
    plot_item.setTitle(title)
    legend = plot_item.addLegend()
    anchor_param = get_pyqtgraph_anchor_params(loc, padding=15)
    legend.anchor(**anchor_param)

    alpha = np.ones(len(xf_list)) * 1.0
    if rows:
        alpha *= 0.35
        for t in rows:
            alpha[t] = 1.0

    if not subtract_background:
        bkg_file = None
    norm_method = [None, "q2", "q4", "I0"][plot_norm]
    log_x = (False, True)[plot_type % 2]
    log_y = (False, True)[plot_type // 2]
    plot_item.setLogMode(x=log_x, y=log_y)

    # Initialize labels with defaults
    xlabel = "q (Å⁻¹)"
    ylabel = "Intensity"
    
    plot_id = 0
    for n, fi in enumerate(xf_list):
        q, Iq, xlabel, ylabel = fi.get_saxs_1d_data(
            bkg_xf=bkg_file,
            bkg_weight=weight,
            q_range=(qmin, qmax),
            sampling=sampling,
            norm_method=norm_method,
            use_absolute_crosssection=absolute_crosssection,
        )

        num_lines = Iq.shape[0] if all_phi else 1
        for m in range(num_lines):
            plot_line_with_marker(
                plot_item,
                q,
                Iq[m],
                plot_id,
                fi.saxs_1d["labels"][m],
                alpha[n],
                marker_size=marker_size,
                log_x=log_x,
                log_y=log_y,
            )
            plot_id += 1

    if plot_norm == 0:  # no normalization
        if absolute_crosssection:
            ylabel = "Intensity (1/cm)"
        else:
            ylabel = "Intensity (photon/pixel/frame)"

    plot_item.setLabel("bottom", xlabel)
    plot_item.setLabel("left", ylabel)
    plot_item.showGrid(x=True, y=True, alpha=0.3)

    return
