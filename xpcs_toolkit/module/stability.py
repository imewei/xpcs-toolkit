"""
XPCS Toolkit - Temporal Stability Analysis Module (stability)

This module provides comprehensive analysis capabilities for temporal stability 
assessment in X-ray scattering experiments. Stability analysis is crucial for
identifying sample degradation, beam damage effects, temperature fluctuations,
and other time-dependent phenomena that can affect data quality and interpretation.

## Scientific Background

Temporal stability analysis in X-ray scattering involves monitoring the evolution
of scattering patterns over time to ensure data quality and identify systematic
changes. Key aspects include:

### Radiation Damage Assessment
X-ray exposure can cause:
- **Structural changes**: Bond breaking, cross-linking, phase transitions
- **Mass loss**: Volatile component evaporation, decomposition products  
- **Chemical modifications**: Oxidation, reduction, radical formation
- **Morphological evolution**: Particle aggregation, surface reconstruction

### Thermal Effects
Temperature variations lead to:
- **Thermal expansion**: Lattice parameter changes, density variations
- **Phase transitions**: Order-disorder transitions, melting, crystallization
- **Molecular dynamics**: Increased atomic motion, conformational changes
- **Convection effects**: Sample position drift, density gradients

### Environmental Stability
External factors affecting measurements:
- **Humidity changes**: Hydration/dehydration effects, swelling
- **Mechanical vibrations**: Sample position instability, alignment drift
- **Beam fluctuations**: Intensity variations, position drift
- **Chemical reactions**: Oxidation, hydrolysis, polymerization

## Analysis Capabilities

### Temporal Pattern Evolution
- **Partial integration analysis**: Monitor specific q-ranges over time
- **Peak intensity tracking**: Follow Bragg peak evolution and degradation
- **Background evolution**: Assess changes in diffuse scattering
- **Profile shape analysis**: Detect peak broadening or sharpening

### Statistical Assessment
- **Variance analysis**: Quantify measurement reproducibility
- **Correlation analysis**: Identify systematic trends versus random fluctuations
- **Change detection**: Statistical tests for significant structural changes
- **Drift correction**: Compensate for systematic instrumental effects

### Quality Control Metrics
- **Signal stability**: Monitor integrated intensities over time
- **Noise characteristics**: Assess statistical quality evolution
- **Systematic trends**: Linear, exponential, or complex time dependencies
- **Threshold detection**: Identify when changes exceed acceptable limits

## Physical Interpretation

### Structural Stability Indicators
- **Invariant intensity**: Total scattered intensity conservation
- **Peak positions**: Lattice parameter constancy
- **Peak widths**: Crystalline domain size stability
- **Form factors**: Shape and size parameter consistency

### Kinetic Analysis
- **Rate constants**: Extract degradation or evolution kinetics
- **Activation energies**: Temperature-dependent process characterization
- **Mechanism identification**: Distinguish different degradation pathways
- **Lifetime estimation**: Predict sample viability for measurements

### Process Monitoring
- **Phase transitions**: Track order parameter evolution
- **Crystallization**: Monitor nucleation and growth processes
- **Aging effects**: Characterize long-term structural evolution
- **Reversibility**: Assess damage versus equilibrium changes

## Typical Analysis Workflow

1. **Data Collection**: Acquire time series of scattering patterns
2. **Pattern Extraction**: Extract 1D profiles or 2D pattern metrics
3. **Temporal Tracking**: Monitor key parameters versus time/dose
4. **Trend Analysis**: Identify systematic changes and fluctuations
5. **Statistical Testing**: Assess significance of observed changes
6. **Quality Assessment**: Determine data validity and useful time range
7. **Correction Application**: Apply drift corrections if appropriate

## Applications

### Soft Matter Physics
- **Polymer degradation**: UV, thermal, and radiation damage studies
- **Protein stability**: Unfolding kinetics and aggregation processes
- **Colloidal systems**: Particle coarsening and Ostwald ripening
- **Liquid crystals**: Phase stability and alignment decay

### Materials Science
- **Radiation effects**: Damage in nuclear materials and electronics
- **Thermal cycling**: Stress-induced microstructural changes
- **Corrosion studies**: Oxidation and degradation processes
- **Catalyst deactivation**: Active site evolution and poisoning

### Biological Systems
- **Membrane stability**: Lipid reorganization and protein denaturation
- **Tissue samples**: Degradation during measurement conditions
- **Drug formulations**: Stability assessment and shelf-life studies
- **Cellular structures**: Radiation sensitivity and repair mechanisms

### Industrial Applications
- **Quality control**: Monitor product stability during processing
- **Formulation development**: Assess stability of new materials
- **Aging studies**: Predict long-term performance and lifetime
- **Process optimization**: Identify stable operating conditions

## Module Functions

The module provides the following key function:

- `plot()`: Visualize temporal evolution of partial scattering profiles

## Usage Examples

```python
# Basic stability analysis visualization
from xpcs_toolkit.module import stability

# Plot temporal evolution of scattering profiles
stability.plot(
    fc=time_series_data,
    pg_hdl=matplotlib_handle,
    plot_type=3,           # log-log scale
    plot_norm=0,           # No normalization
    loc="upper right",     # Legend position
    title="Stability Analysis"
)

# Monitor specific q-range with normalization
stability.plot(
    fc=degradation_study,
    pg_hdl=matplotlib_handle,
    plot_type=1,           # log-linear scale
    plot_norm=2,           # q⁴ normalization (Porod analysis)
    loc="lower left"
)
```

## References

- Jeffries et al., "Radiation damage and dose limits in serial synchrotron crystallography" (2015)
- Garman, "Developments in X-ray crystallographic structure determination of biological macromolecules" (2014)  
- Murray & Garman, "Investigation of possible free-radical scavengers" (2002)
- Warkentin & Thorne, "A general method for hyperquenching protein crystals" (2007)

## Author

XPCS Toolkit Development Team
Advanced Photon Source, Argonne National Laboratory
"""

from .saxs1d import get_pyqtgraph_anchor_params, plot_line_with_marker
import numpy as np


def plot(
    fc,
    pg_hdl,
    plot_type=2,
    plot_norm=0,
    legend=None,
    title=None,
    loc="upper right",
    **kwargs,
):
    """
    Visualize temporal stability of partial scattering profiles over time.
    
    This function creates comprehensive plots showing the evolution of scattering
    intensity profiles over time or dose, enabling assessment of sample stability,
    radiation damage effects, and other time-dependent phenomena. Each partial
    profile represents a different time point or measurement condition.
    
    The visualization supports various scaling and normalization options optimized
    for different types of stability analysis, from radiation damage assessment
    to thermal stability studies.
    
    Parameters
    ----------
    fc : XpcsDataFile
        Data file containing temporal series of partial SAXS profiles.
        Must have saxs1d_partial data with time-resolved scattering patterns.
    pg_hdl : matplotlib.pyplot or compatible plotting handle
        Matplotlib-compatible plotting handle for rendering. Must support
        standard matplotlib plotting interface and subplot creation.
    plot_type : int, optional
        Scaling mode for axes. Default: 2 (linear-log).
        Bit encoding: bit 0 = x-log, bit 1 = y-log
        - 0: linear-linear
        - 1: log-linear (useful for q-range analysis)
        - 2: linear-log (standard for intensity evolution)
        - 3: log-log (power-law behavior identification)
    plot_norm : int, optional
        Intensity normalization method. Default: 0 (no normalization).
        - 0: Raw intensity I(q,t)
        - 1: Kratky plot q²I(q,t) (polymer chain analysis)
        - 2: Porod plot q⁴I(q,t) (surface area evolution)
        - 3: Monitor normalized I(q,t)/I₀(t) (beam fluctuation correction)
    legend : str, optional
        Legend specification. Default: None (automatic).
        Currently not used - automatic labeling applied.
    title : str, optional
        Plot title override. Default: None (uses fc.label).
        Useful for custom analysis descriptions.
    loc : str or int, optional
        Legend position following matplotlib conventions. Default: 'upper right'.
        Options: 'upper left', 'lower right', 'center', etc., or integer codes.
    **kwargs : dict
        Additional keyword arguments for future extensibility.
        Currently passed through but not used.
    
    Returns
    -------
    None
        Function performs plotting operations directly on the provided handle.
    
    Examples
    --------
    >>> # Standard stability analysis for radiation damage
    >>> stability.plot(
    ...     fc=time_series_data,
    ...     pg_hdl=matplotlib_handle,
    ...     plot_type=3,        # log-log scale
    ...     plot_norm=0,        # Raw intensities
    ...     title="Radiation Damage Assessment",
    ...     loc="upper right"
    ... )
    >>> 
    >>> # Kratky analysis for polymer chain stability
    >>> stability.plot(
    ...     fc=polymer_time_series,
    ...     pg_hdl=matplotlib_handle,
    ...     plot_type=1,        # log-linear scale  
    ...     plot_norm=1,        # q²I(q) normalization
    ...     title="Chain Conformation Evolution",
    ...     loc="lower left"
    ... )
    >>> 
    >>> # High-q surface analysis for particle degradation
    >>> stability.plot(
    ...     fc=nanoparticle_series,
    ...     pg_hdl=matplotlib_handle,
    ...     plot_type=2,        # linear-log scale
    ...     plot_norm=2,        # q⁴I(q) normalization
    ...     title="Surface Area Evolution"
    ... )
    
    Notes
    -----
    - Each curve represents a different time point in the measurement series
    - Curve labels use format 'p{n}' where n is the temporal index
    - Grid lines enhance quantitative comparison between time points
    - Logarithmic scaling helps identify different degradation regimes
    - Normalization choices highlight different physical aspects:
      * Raw intensity: Total scattering power evolution
      * Kratky: Chain flexibility and coil-to-globule transitions
      * Porod: Surface-to-volume ratio changes
      * Monitor normalization: Beam-corrected comparison
    
    See Also
    --------
    saxs1d.plot_line_with_marker : Individual curve plotting function
    saxs1d.get_pyqtgraph_anchor_params : Legend positioning utility
    """

    pg_hdl.clear()
    plot_item = pg_hdl.getPlotItem()

    plot_item.setTitle(fc.label)
    legend = plot_item.addLegend()
    anchor_param = get_pyqtgraph_anchor_params(loc, padding=15)
    legend.anchor(**anchor_param)

    norm_method = [None, "q2", "q4", "I0"][plot_norm]
    log_x = (False, True)[plot_type % 2]
    log_y = (False, True)[plot_type // 2]
    plot_item.setLogMode(x=log_x, y=log_y)

    q, Iqp, xlabel, ylabel = fc.get_saxs_1d_data(
        target="saxs1d_partial", norm_method=norm_method
    )
    for n in range(Iqp.shape[0]):
        plot_line_with_marker(
            plot_item,
            q,
            Iqp[n],
            n,
            f"p{n}",  # label
            1.0,  # alpha
            marker_size=6,
            log_x=log_x,
            log_y=log_y,
        )

    plot_item.setLabel("bottom", xlabel)
    plot_item.setLabel("left", ylabel)
    plot_item.showGrid(x=True, y=True, alpha=0.3)
