"""
XPCS Toolkit - Small-Angle X-ray Scattering 2D Analysis Module (saxs2d)

This module provides comprehensive visualization and analysis capabilities for
two-dimensional small-angle X-ray scattering (SAXS) patterns. 2D SAXS reveals
the complete scattering information, including structural anisotropy, preferred
orientations, and spatial correlations that are averaged out in 1D analysis.

## Scientific Background

Two-dimensional SAXS patterns contain the full angular and radial scattering
information, providing access to:

### Structural Anisotropy
- **Orientation distributions**: Fiber textures, liquid crystal alignment
- **Preferred directions**: Crystallographic textures, flow alignment
- **Symmetry analysis**: Crystal systems, molecular packing arrangements
- **Correlation lengths**: Anisotropic structure factors and form factors

### Scattering Geometry
The 2D detector records intensity I(q_x, q_y) where:
- q_x, q_y: Momentum transfer components in detector plane
- |q| = sqrt(q_x² + q_y²): Radial momentum transfer
- φ = atan2(q_y, q_x): Azimuthal angle

### Pattern Features
- **Powder rings**: Isotropic polycrystalline materials
- **Spot patterns**: Single crystal or large grain structures
- **Fiber patterns**: Uniaxial orientation (polymers, biomaterials)
- **Streak patterns**: Layer structures, interfaces, defects
- **Anisotropic scattering**: Oriented domains, flow effects

## Analysis Capabilities

### Pattern Visualization
- **Intensity scaling**: Linear, logarithmic, square-root representations
- **Dynamic range control**: Manual and automatic level adjustment
- **Colormap options**: Scientific colormaps for quantitative analysis
- **Geometric corrections**: Detector rotation and coordinate transforms
- **Center determination**: Beam center marking and adjustment

### Image Processing
- **Background subtraction**: Dark field and solvent background removal
- **Detector corrections**: Flat field, geometric distortion, polarization
- **Intensity calibration**: Absolute units with standard references
- **Noise reduction**: Statistical filtering while preserving features
- **Region masking**: Exclude detector artifacts and beam stops

### Quantitative Analysis
- **Radial integration**: Convert to 1D I(q) profiles with sector averaging
- **Azimuthal integration**: Extract I(φ) at fixed q for orientation analysis
- **Peak finding**: Locate Bragg reflections and determine positions
- **Texture analysis**: Quantify preferred orientation distributions
- **Correlation functions**: 2D structure factor analysis

## Typical Analysis Workflow

1. **Data Loading**: Import 2D detector images with metadata
2. **Geometry Setup**: Determine beam center and sample-detector distance
3. **Background Correction**: Subtract instrumental and solvent backgrounds
4. **Pattern Visualization**: Optimize display parameters for feature identification
5. **ROI Definition**: Define sectors or regions for quantitative analysis
6. **Integration**: Convert to 1D profiles or extract angular distributions
7. **Structure Analysis**: Interpret patterns in terms of sample structure

## Applications

### Soft Matter Physics
- **Liquid crystals**: Phase identification and orientation analysis
- **Block copolymers**: Domain structure and ordering transitions
- **Polymer films**: Molecular orientation and crystallinity
- **Membranes**: Bilayer structure and phase behavior

### Materials Science
- **Nanocomposites**: Filler orientation and dispersion characterization
- **Thin films**: Texture analysis and epitaxial relationships
- **Ceramics**: Grain orientation and porosity assessment
- **Metals**: Texture evolution during processing

### Biological Systems
- **Muscle fibers**: Sarcomere structure and contractile proteins
- **Cell walls**: Cellulose microfibril orientation in plants
- **Bone**: Collagen and hydroxyapatite arrangement
- **Membranes**: Lipid organization and protein incorporation

### Industrial Applications
- **Fiber production**: Monitor orientation development during spinning
- **Film manufacturing**: Control molecular alignment in packaging
- **Composite processing**: Optimize fiber-matrix interface properties
- **Quality control**: Detect processing defects and inhomogeneities

## Module Functions

The module provides the following key function:

- `plot()`: Main 2D SAXS visualization function with comprehensive display options

## Usage Examples

```python
# Basic 2D SAXS pattern visualization
from xpcs_toolkit.module import saxs2d

# Standard logarithmic intensity display
saxs2d.plot(
    xfile=saxs_data,
    pg_hdl=matplotlib_handle,
    plot_type='log',      # Logarithmic intensity scale
    cmap='viridis',       # Scientific colormap
    autolevel=True,       # Automatic intensity range
    autorange=True        # Fit pattern to display
)

# Custom intensity range for detailed analysis
saxs2d.plot(
    xfile=saxs_data,
    pg_hdl=matplotlib_handle,
    plot_type='linear',   # Linear intensity scale
    vmin=10,              # Minimum display intensity
    vmax=1000,            # Maximum display intensity
    rotate=False,         # No geometric rotation
    cmap='plasma'         # Alternative colormap
)
```

## References

- Als-Nielsen & McMorrow, "Elements of Modern X-ray Physics" (2001)
- Guinier, "X-Ray Diffraction in Crystals, Imperfect Crystals, and Amorphous Bodies" (1994)
- Birkholz, "Thin Film Analysis by X-Ray Scattering" (2006)
- Smilgies, "Scherrer grain-size analysis adapted to grazing-incidence scattering" (2009)

## Author

XPCS Toolkit Development Team
Advanced Photon Source, Argonne National Laboratory
"""

from typing import Any, Union


def plot(
    xfile: Any,
    pg_hdl: Any = None,
    plot_type: str = "log",
    cmap: str = "jet",
    rotate: bool = False,
    autolevel: bool = False,
    autorange: bool = False,
    vmin: Union[float, None] = None,
    vmax: Union[float, None] = None,
) -> Any:
    """
    Display 2D SAXS scattering patterns with comprehensive visualization options.

    This function provides scientific-grade visualization of two-dimensional
    small-angle X-ray scattering patterns, supporting various intensity scaling
    modes, colormaps, and display optimizations for quantitative analysis.

    The visualization preserves the spatial information crucial for anisotropy
    analysis while offering flexible display options for different sample types
    and analysis requirements.

    Parameters
    ----------
    xfile : XpcsDataFile
        SAXS data file containing 2D scattering pattern. Must have saxs_2d
        and saxs_2d_log attributes with intensity data and beam center (bcx, bcy).
    pg_hdl : matplotlib.pyplot or compatible plotting handle, optional
        Matplotlib-compatible image display handle for rendering. Must support
        imshow() method and colormap functionality. If None, returns rotate parameter only.
    plot_type : str, optional
        Intensity scaling mode. Default: 'log'.
        - 'log': Logarithmic scaling (highlights weak features)
        - 'linear': Linear scaling (preserves intensity ratios)
        - Others: Use linear scaling as fallback
    cmap : str, optional
        Colormap name for intensity visualization. Default: 'jet'.
        Options: 'viridis', 'plasma', 'inferno', 'jet', 'hot', 'cool', etc.
        Should be compatible with the display widget's colormap system.
    rotate : bool, optional
        Apply geometric rotation to pattern. Default: False.
        Used for coordinate system alignment or pattern orientation.
    autolevel : bool, optional
        Enable automatic intensity level adjustment. Default: False.
        When True, optimizes contrast based on pattern statistics.
    autorange : bool, optional
        Enable automatic zoom to fit pattern. Default: False.
        When True, scales display to show entire scattering pattern.
    vmin : float, optional
        Minimum intensity for display range. Default: None.
        Used for manual contrast control. Ignored if autolevel=True.
    vmax : float, optional
        Maximum intensity for display range. Default: None.
        Used for manual contrast control. Ignored if autolevel=True.

    Returns
    -------
    bool
        The rotate parameter value, indicating rotation state.
        This return is maintained for compatibility with calling code.

    Examples
    --------
    >>> # Standard logarithmic display for weak feature analysis
    >>> saxs2d.plot(
    ...     xfile=pattern_data,
    ...     pg_hdl=matplotlib_handle,
    ...     plot_type='log',
    ...     cmap='viridis',
    ...     autolevel=True,
    ...     autorange=True
    ... )
    >>>
    >>> # Linear scale with manual intensity range
    >>> saxs2d.plot(
    ...     xfile=pattern_data,
    ...     pg_hdl=matplotlib_handle,
    ...     plot_type='linear',
    ...     vmin=50,
    ...     vmax=2000,
    ...     cmap='jet'
    ... )
    >>>
    >>> # High-contrast display for powder patterns
    >>> saxs2d.plot(
    ...     xfile=powder_pattern,
    ...     pg_hdl=matplotlib_handle,
    ...     plot_type='log',
    ...     cmap='hot',
    ...     autolevel=False,
    ...     vmin=1,
    ...     vmax=10000
    ... )

    Notes
    -----
    - Logarithmic scaling enhances visibility of weak scattering features
    - Linear scaling preserves quantitative intensity relationships
    - Beam center position is marked when available (bcx, bcy attributes)
    - View range preservation enables zoomed analysis across parameter changes
    - Colormap choice affects quantitative interpretation and feature visibility
    - Manual intensity ranges override autolevel settings for precise control

    See Also
    --------
    xpcs_toolkit.module.saxs1d.pg_plot : 1D SAXS visualization after integration
    """
    # Return early if no plot handler provided
    if pg_hdl is None:
        return rotate

    center = (xfile.bcx, xfile.bcy)
    img = xfile.saxs_2d_log if plot_type == "log" else xfile.saxs_2d

    if cmap is not None and hasattr(pg_hdl, "set_colormap"):
        pg_hdl.set_colormap(cmap)

    prev_img = getattr(pg_hdl, "image", None)
    shape_changed = prev_img is None or prev_img.shape != img.shape
    do_autorange = autorange or shape_changed

    # Initialize view_range to None
    view_range = None

    # Save view range if keeping it and view attribute exists
    if not do_autorange and hasattr(pg_hdl, "view") and pg_hdl.view is not None:
        view_range = pg_hdl.view.viewRange()

    # Set new image if method exists
    if hasattr(pg_hdl, "setImage"):
        pg_hdl.setImage(img, autoLevels=autolevel, autoRange=do_autorange)

    # Restore view range if we have it and skipped auto-ranging
    if (
        not do_autorange
        and view_range is not None
        and hasattr(pg_hdl, "view")
        and pg_hdl.view is not None
    ):
        pg_hdl.view.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)

    # Restore levels if needed and method exists
    if (
        not autolevel
        and vmin is not None
        and vmax is not None
        and hasattr(pg_hdl, "setLevels")
    ):
        pg_hdl.setLevels(vmin, vmax)

    # Restore intensity levels (if needed) - removing duplicate code
    # The above condition already handles this case

    if center is not None and hasattr(pg_hdl, "add_roi"):
        pg_hdl.add_roi(sl_type="Center", center=center, label="Center")

    return rotate
