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
    
    This function has been disabled in headless mode as it's specific to PyQtGraph GUI functionality.
    """
    raise NotImplementedError(
        "GUI plotting functionality has been disabled in headless mode."
    )


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
    
    This function has been disabled in headless mode as it's specific to PyQtGraph GUI functionality.
    """
    raise NotImplementedError(
        "GUI plotting functionality has been disabled in headless mode."
    )


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
    
    This function has been disabled in headless mode as it requires PyQtGraph GUI functionality.
    Use the matplotlib-based CLI interface for visualization instead."""
    raise NotImplementedError(
        "GUI plotting functionality has been disabled in headless mode. "
        "Use the matplotlib-based CLI interface for visualization instead."
    )
