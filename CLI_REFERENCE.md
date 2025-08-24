# XPCS Toolkit - Command-Line Interface Reference

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)

**Complete reference for the XPCS Toolkit headless command-line interface for X-ray Photon Correlation Spectroscopy data analysis.**

## Table of Contents

- [Overview](#overview)
- [Installation & Setup](#installation--setup)
- [Basic Usage](#basic-usage)
- [Commands Reference](#commands-reference)
- [Data Formats](#data-formats)
- [Analysis Workflows](#analysis-workflows)
- [Examples Gallery](#examples-gallery)
- [Troubleshooting](#troubleshooting)

## Overview

The XPCS Toolkit provides a powerful command-line interface for headless operation of X-ray Photon Correlation Spectroscopy data analysis. Designed for:

- **Batch processing** of large datasets
- **Automated workflows** in synchrotron environments  
- **Integration** with beamline control systems
- **Remote analysis** capabilities
- **Reproducible** analysis protocols

### Key Features

✅ **Multi-format Support**: APS 8-ID-I NeXus, legacy HDF5, automatic format detection  
✅ **Comprehensive Analysis**: 2D SAXS, g2 correlation, 1D radial profiles, beam stability  
✅ **Batch Processing**: Automated analysis of multiple files  
✅ **High-quality Output**: PNG, PDF, SVG publication-ready figures  
✅ **Robust Logging**: Detailed processing logs and error reporting  
✅ **Performance Optimized**: Memory-efficient processing of large datasets  

## Installation & Setup

### Prerequisites

```bash
# Python 3.9+ required
python --version

# Install XPCS Toolkit
pip install xpcs-toolkit

# Verify installation
xpcs-toolkit --version
```

### Environment Configuration

```bash
# Optional: Set up logging directory
export XPCS_LOG_DIR="/path/to/logs"

# Optional: Configure default output directory
export XPCS_OUTPUT_DIR="/path/to/output"

# Optional: Set memory limits for large files
export XPCS_MEMORY_LIMIT="8GB"
```

## Basic Usage

### Command Structure

```bash
xpcs-toolkit <command> <data_path> [options]
```

### Quick Start Examples

```bash
# List files in directory
xpcs-toolkit list /path/to/data/

# Generate 2D SAXS patterns
xpcs-toolkit saxs2d /path/to/data/ --output results.png

# Analyze correlation functions
xpcs-toolkit g2 /path/to/data/ --qmin 0.01 --qmax 0.1

# Check beam stability
xpcs-toolkit stability /path/to/data/

# Generate 1D radial profiles
xpcs-toolkit saxs1d /path/to/data/ --log-scale
```

## Commands Reference

### 🔍 `list` - File Discovery and Validation

**Purpose**: Discover, validate, and summarize XPCS data files in a directory.

```bash
xpcs-toolkit list <directory> [options]
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--recursive` | flag | False | Search subdirectories recursively |
| `--validate` | flag | False | Validate file format and structure |
| `--summary` | flag | False | Show detailed file summary |
| `--format` | str | "table" | Output format: table, json, csv |

#### Examples

```bash
# Basic file listing
xpcs-toolkit list /beamline/data/experiment_2024/

# Recursive search with validation
xpcs-toolkit list /beamline/data/ --recursive --validate

# Export file list to JSON
xpcs-toolkit list /data/ --format json --summary > file_inventory.json

# Validate large dataset
xpcs-toolkit list /archive/2024/ --recursive --validate --summary
```

#### Output Format

```
📁 XPCS Data Files Summary
==========================
Directory: /beamline/data/experiment_2024/
Files found: 156
Valid XPCS files: 152
Total size: 2.3 GB

File Details:
┌─────────────────────────┬──────────┬─────────┬─────────────┐
│ Filename                │ Format   │ Size    │ Status      │
├─────────────────────────┼──────────┼─────────┼─────────────┤
│ sample_001.h5          │ NeXus    │ 15.2 MB │ Valid       │
│ sample_002.h5          │ NeXus    │ 16.1 MB │ Valid       │
│ calibration.h5         │ Legacy   │ 2.1 MB  │ Valid       │
│ background_001.h5      │ NeXus    │ 8.3 MB  │ Valid       │
└─────────────────────────┴──────────┴─────────┴─────────────┘
```

---

### 📊 `saxs2d` - 2D SAXS Pattern Visualization

**Purpose**: Generate publication-ready 2D small-angle X-ray scattering pattern visualizations.

```bash
xpcs-toolkit saxs2d <directory> [options]
```

#### Scientific Background

Small-angle X-ray scattering (SAXS) reveals structural information about materials on nanometer length scales. The 2D scattering patterns show:

- **Isotropic samples**: Circular scattering rings
- **Anisotropic samples**: Oriented scattering features
- **Particle size**: Inverse relationship with scattering angle
- **Sample structure**: Form factor and structure factor contributions

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output` | str | "saxs2d.png" | Output filename |
| `--log-scale` | flag | False | Use logarithmic intensity scaling |
| `--vmin` | float | auto | Minimum intensity value |
| `--vmax` | float | auto | Maximum intensity value |
| `--colormap` | str | "viridis" | Matplotlib colormap name |
| `--beam-center` | str | auto | Beam center coordinates "x,y" |
| `--q-range` | str | None | Q-range overlay "qmin,qmax" |
| `--detector-mask` | str | None | Path to detector mask file |
| `--title` | str | auto | Plot title |
| `--dpi` | int | 300 | Output resolution (DPI) |

#### Examples

```bash
# Basic 2D SAXS visualization
xpcs-toolkit saxs2d /data/sample_001/ --output sample_001_saxs2d.png

# High-resolution with logarithmic scaling
xpcs-toolkit saxs2d /data/sample_001/ \
  --log-scale \
  --output sample_001_log.png \
  --dpi 600

# Custom intensity range and colormap
xpcs-toolkit saxs2d /data/sample_001/ \
  --vmin 1e-6 \
  --vmax 1e-2 \
  --colormap plasma \
  --output sample_001_plasma.png

# Apply detector mask and Q-range overlay
xpcs-toolkit saxs2d /data/sample_001/ \
  --detector-mask /calibration/mask.h5 \
  --q-range "0.01,0.1" \
  --beam-center "512,512"

# Batch processing multiple files
for dir in /data/sample_*/; do
    xpcs-toolkit saxs2d "$dir" --output "${dir}/saxs2d_pattern.png"
done
```

#### Scientific Applications

- **Sample characterization**: Identify structural features and defects
- **Quality control**: Verify sample homogeneity and preparation
- **Real-time monitoring**: Track structural changes during experiments  
- **Publication figures**: Generate high-quality images for papers

---

### 📈 `g2` - Multi-tau Correlation Analysis

**Purpose**: Analyze intensity correlation functions to extract dynamic information from XPCS measurements.

```bash
xpcs-toolkit g2 <directory> [options]
```

#### Scientific Background

The intensity correlation function g₂(q,τ) is the fundamental quantity in XPCS:

```
g₂(q,τ) = ⟨I(q,t)I(q,t+τ)⟩ / ⟨I(q,t)⟩²
```

Where:
- **I(q,t)**: Scattered intensity at wavevector q and time t
- **τ**: Correlation delay time
- **⟨⟩**: Time average

This function reveals:
- **Relaxation times**: Characteristic timescales of sample dynamics
- **Dynamic heterogeneity**: Spatial variations in dynamics
- **Non-ergodic behavior**: Aging and glassy dynamics
- **Diffusion coefficients**: Through the Stokes-Einstein relation

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--qmin` | float | None | Minimum q-value for analysis (Å⁻¹) |
| `--qmax` | float | None | Maximum q-value for analysis (Å⁻¹) |
| `--output` | str | "g2_analysis.png" | Output filename |
| `--fit-range` | str | None | Fit range "tau_min,tau_max" (seconds) |
| `--fit-function` | str | "single_exp" | Fit function: single_exp, double_exp, stretched |
| `--log-x` | flag | False | Use logarithmic time axis |
| `--log-y` | flag | False | Use logarithmic g2 axis |
| `--error-bars` | flag | True | Show statistical error bars |
| `--export-data` | str | None | Export data to CSV file |
| `--tau-max` | float | None | Maximum correlation time (seconds) |

#### Examples

```bash
# Basic g2 analysis with automatic q-range
xpcs-toolkit g2 /data/dynamics_study/ --output g2_auto.png

# Analyze specific q-range with fitting
xpcs-toolkit g2 /data/dynamics_study/ \
  --qmin 0.01 \
  --qmax 0.05 \
  --fit-function single_exp \
  --fit-range "1e-5,1e-2" \
  --output g2_fitted.png

# High-quality analysis with data export
xpcs-toolkit g2 /data/dynamics_study/ \
  --qmin 0.008 \
  --qmax 0.12 \
  --log-x \
  --error-bars \
  --export-data g2_data.csv \
  --output g2_publication.png \
  --dpi 600

# Stretched exponential fitting for complex dynamics
xpcs-toolkit g2 /data/glass_transition/ \
  --fit-function stretched \
  --fit-range "1e-6,10" \
  --tau-max 100 \
  --output g2_stretched.png

# Multi-q analysis for diffusion measurement
xpcs-toolkit g2 /data/brownian_motion/ \
  --qmin 0.005 \
  --qmax 0.15 \
  --fit-function single_exp \
  --export-data diffusion_data.csv
```

#### Output Files

1. **Correlation plot**: Visualization of g₂(q,τ) vs τ for multiple q-values
2. **Fit parameters**: Relaxation times, amplitudes, and fit quality metrics
3. **Data export**: Raw correlation data and fit results in CSV format

#### Scientific Applications

- **Brownian motion**: Measure diffusion coefficients
- **Gelation studies**: Track gel formation dynamics
- **Glass transition**: Characterize slow dynamics near Tg
- **Active matter**: Study non-equilibrium fluctuations

---

### 🎯 `saxs1d` - 1D Radial Profile Analysis

**Purpose**: Generate 1D radial profiles from 2D scattering patterns through angular integration.

```bash
xpcs-toolkit saxs1d <directory> [options]
```

#### Scientific Background

1D radial averaging converts 2D scattering patterns I(qx,qy) into 1D profiles I(q):

```
I(q) = ∫₀²π I(q,φ) dφ / 2π
```

Where:
- **q = |q|**: Magnitude of scattering wavevector
- **φ**: Azimuthal angle
- **Integration**: Averages over all angles for isotropic samples

Benefits:
- **Improved statistics**: Better signal-to-noise ratio
- **Model fitting**: Direct comparison with theoretical form factors
- **Peak analysis**: Identify characteristic length scales
- **Time evolution**: Track structural changes over time

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output` | str | "saxs1d.png" | Output filename |
| `--log-x` | flag | False | Logarithmic q-axis |
| `--log-y` | flag | False | Logarithmic intensity axis |
| `--q-range` | str | None | Q-range for analysis "qmin,qmax" (Å⁻¹) |
| `--phi-range` | str | None | Angular range "phi_min,phi_max" (degrees) |
| `--background` | str | None | Background file for subtraction |
| `--normalize` | flag | False | Normalize to unit integral |
| `--smooth` | float | None | Gaussian smoothing sigma |
| `--export-data` | str | None | Export to CSV file |
| `--multi-phi` | flag | False | Generate multi-angle analysis |

#### Examples

```bash
# Basic 1D radial averaging
xpcs-toolkit saxs1d /data/sample_001/ --output sample_001_1d.png

# Logarithmic scaling for wide q-range
xpcs-toolkit saxs1d /data/sample_001/ \
  --log-x \
  --log-y \
  --output sample_001_log.png

# Specific q-range with background subtraction
xpcs-toolkit saxs1d /data/sample_001/ \
  --q-range "0.005,0.5" \
  --background /data/background.h5 \
  --output sample_001_corrected.png

# Angular sector analysis for anisotropic samples
xpcs-toolkit saxs1d /data/oriented_sample/ \
  --multi-phi \
  --phi-range "0,30" \
  --output oriented_sectors.png

# High-quality analysis with data export
xpcs-toolkit saxs1d /data/nanoparticles/ \
  --log-x \
  --log-y \
  --smooth 0.02 \
  --normalize \
  --export-data nanoparticles_1d.csv \
  --output nanoparticles_publication.png \
  --dpi 600

# Time series analysis
for file in /data/time_series/*.h5; do
    timestamp=$(basename "$file" .h5)
    xpcs-toolkit saxs1d "$file" \
      --log-x --log-y \
      --export-data "time_series_${timestamp}.csv"
done
```

#### Output Analysis

The 1D profiles reveal:

- **Power-law behavior**: I(q) ∝ q⁻ᵖ indicates fractal structures
- **Peak positions**: Characteristic length scales d = 2π/q
- **Peak widths**: Size distribution and structural order
- **Low-q upturn**: Large-scale aggregation or clustering

#### Scientific Applications

- **Particle sizing**: Extract size distributions from form factors
- **Structure analysis**: Identify liquid, crystalline, or glassy order
- **Phase transitions**: Monitor structural changes
- **Quality control**: Verify sample consistency and preparation

---

### 🔬 `stability` - Beam Stability Analysis

**Purpose**: Assess beam stability and data quality through statistical analysis of intensity fluctuations.

```bash
xpcs-toolkit stability <directory> [options]
```

#### Scientific Background

Beam stability is critical for XPCS measurements because:

1. **Intensity fluctuations**: Affect correlation function accuracy
2. **Beam position drift**: Causes artifacts in analysis
3. **Detector stability**: Influences noise characteristics
4. **Long-term stability**: Essential for extended measurements

Key metrics:
- **Relative standard deviation**: σ/⟨I⟩ for intensity stability
- **Drift analysis**: Linear trends over time
- **Stability windows**: Identify stable measurement periods
- **Quality flags**: Automated quality assessment

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output` | str | "stability.png" | Output filename |
| `--time-windows` | int | 10 | Number of time windows for analysis |
| `--metric` | str | "intensity" | Stability metric: intensity, position, detector |
| `--threshold` | float | 0.05 | Stability threshold (5% default) |
| `--export-data` | str | None | Export stability data to CSV |
| `--show-trends` | flag | True | Show linear trend analysis |

#### Examples

```bash
# Basic stability analysis
xpcs-toolkit stability /data/long_measurement/ --output stability.png

# Detailed stability with custom windows
xpcs-toolkit stability /data/long_measurement/ \
  --time-windows 20 \
  --threshold 0.02 \
  --show-trends \
  --output stability_detailed.png

# Export stability data for external analysis
xpcs-toolkit stability /data/experiment/ \
  --export-data stability_metrics.csv \
  --time-windows 15 \
  --metric intensity

# Multi-metric analysis
xpcs-toolkit stability /data/beamtime/ \
  --metric position \
  --threshold 0.01 \
  --output beam_position_stability.png
```

#### Interpretation Guide

**Good stability** (< 2% RSD):
- Reliable correlation functions
- Accurate dynamic measurements
- Suitable for publication

**Moderate stability** (2-5% RSD):
- Usable with increased error bars
- May require additional averaging
- Check for systematic trends

**Poor stability** (> 5% RSD):
- Data quality compromised
- Investigate experimental conditions
- Consider data exclusion

---

## Data Formats

### APS 8-ID-I NeXus Format

**Primary format** optimized for XPCS analysis:

```
experiment.h5
├── /exchange/
│   ├── g2                  # Multi-tau correlation [q_index, tau_index]
│   ├── tau                 # Correlation times [seconds]
│   ├── saxs_2d            # 2D scattering pattern [pixels]
│   ├── saxs_1d            # 1D radial profile [q_points]
│   ├── q_2d               # Q-map [pixels] 
│   ├── q_1d               # Q-values for 1D [Å⁻¹]
│   └── intensity_mean     # Average intensity [counts]
├── /measurement/
│   ├── instrument/        # Beamline configuration
│   │   ├── detector/      # Detector settings
│   │   ├── source/        # X-ray source parameters
│   │   └── monochromator/ # Energy settings
│   ├── sample/            # Sample information
│   │   ├── name           # Sample identifier
│   │   ├── temperature    # Sample temperature [K]
│   │   └── environment    # Sample environment
│   └── acquisition/       # Data collection parameters
│       ├── frame_time     # Frame exposure time [s]
│       ├── num_frames     # Total number of frames
│       └── start_time     # Measurement timestamp
```

### Legacy HDF5 Format

**Backward compatibility** for older datasets:
- Automatic format detection
- Data structure conversion
- Metadata preservation
- Seamless workflow integration

## Analysis Workflows

### 🔄 Complete Analysis Pipeline

```bash
#!/bin/bash
# Complete XPCS analysis workflow

DATA_DIR="/beamline/data/experiment_2024"
OUTPUT_DIR="/results/experiment_2024"
mkdir -p "$OUTPUT_DIR"

# 1. File discovery and validation
echo "Discovering and validating files..."
xpcs-toolkit list "$DATA_DIR" --recursive --validate > "$OUTPUT_DIR/file_inventory.txt"

# 2. Quality assessment - beam stability
echo "Analyzing beam stability..."
xpcs-toolkit stability "$DATA_DIR" \
  --time-windows 20 \
  --export-data "$OUTPUT_DIR/stability_metrics.csv" \
  --output "$OUTPUT_DIR/beam_stability.png"

# 3. 2D SAXS pattern visualization
echo "Generating 2D SAXS patterns..."
xpcs-toolkit saxs2d "$DATA_DIR" \
  --log-scale \
  --dpi 300 \
  --output "$OUTPUT_DIR/saxs2d_pattern.png"

# 4. 1D radial profile analysis
echo "Creating 1D radial profiles..."
xpcs-toolkit saxs1d "$DATA_DIR" \
  --log-x --log-y \
  --export-data "$OUTPUT_DIR/saxs1d_profile.csv" \
  --output "$OUTPUT_DIR/saxs1d_profile.png"

# 5. Correlation function analysis
echo "Analyzing correlation functions..."
xpcs-toolkit g2 "$DATA_DIR" \
  --qmin 0.01 --qmax 0.08 \
  --fit-function single_exp \
  --log-x \
  --export-data "$OUTPUT_DIR/g2_analysis.csv" \
  --output "$OUTPUT_DIR/g2_correlation.png"

echo "Analysis complete! Results saved to $OUTPUT_DIR"
```

### 🚀 Batch Processing Script

```bash
#!/bin/bash
# Batch process multiple experiments

EXPERIMENTS=("/data/exp_001" "/data/exp_002" "/data/exp_003")

for EXP_DIR in "${EXPERIMENTS[@]}"; do
    EXP_NAME=$(basename "$EXP_DIR")
    OUTPUT_DIR="/results/$EXP_NAME"
    mkdir -p "$OUTPUT_DIR"
    
    echo "Processing $EXP_NAME..."
    
    # Run complete analysis
    xpcs-toolkit saxs2d "$EXP_DIR" --output "$OUTPUT_DIR/saxs2d.png"
    xpcs-toolkit g2 "$EXP_DIR" --output "$OUTPUT_DIR/g2.png" --export-data "$OUTPUT_DIR/g2.csv"
    xpcs-toolkit saxs1d "$EXP_DIR" --output "$OUTPUT_DIR/saxs1d.png"
    xpcs-toolkit stability "$EXP_DIR" --output "$OUTPUT_DIR/stability.png"
    
    echo "Completed $EXP_NAME"
done
```

## Examples Gallery

### 📊 2D SAXS Patterns

```bash
# Isotropic liquid sample
xpcs-toolkit saxs2d /data/liquid_sample/ \
  --log-scale \
  --colormap viridis \
  --output liquid_isotropic.png

# Oriented polymer sample  
xpcs-toolkit saxs2d /data/polymer_oriented/ \
  --vmin 1e-5 --vmax 1e-2 \
  --colormap plasma \
  --output polymer_anisotropic.png

# Crystalline powder
xpcs-toolkit saxs2d /data/powder_diffraction/ \
  --colormap hot \
  --q-range "0.1,2.0" \
  --output powder_rings.png
```

### 📈 Correlation Analysis Examples

```bash
# Brownian motion - simple diffusion
xpcs-toolkit g2 /data/brownian_particles/ \
  --qmin 0.02 --qmax 0.10 \
  --fit-function single_exp \
  --log-x \
  --output brownian_g2.png

# Glass transition - stretched exponentials
xpcs-toolkit g2 /data/glass_former/ \
  --fit-function stretched \
  --fit-range "1e-4,100" \
  --log-x \
  --output glass_transition_g2.png

# Active matter - non-equilibrium dynamics
xpcs-toolkit g2 /data/active_particles/ \
  --qmin 0.005 --qmax 0.05 \
  --fit-function double_exp \
  --tau-max 1000 \
  --output active_matter_g2.png
```

### 📐 1D Profile Analysis

```bash
# Nanoparticle form factor analysis
xpcs-toolkit saxs1d /data/nanoparticles/ \
  --log-x --log-y \
  --q-range "0.01,1.0" \
  --smooth 0.01 \
  --output nanoparticles_form_factor.png

# Fractal aggregates - power law behavior
xpcs-toolkit saxs1d /data/aggregates/ \
  --log-x --log-y \
  --q-range "0.005,0.5" \
  --output fractal_aggregates.png

# Lamellar structure - peak analysis
xpcs-toolkit saxs1d /data/lamellar_phase/ \
  --q-range "0.02,0.3" \
  --output lamellar_peaks.png
```

## Troubleshooting

### Common Issues

#### ❌ "No HDF files found in the specified path"

**Cause**: Directory doesn't contain valid XPCS files

**Solutions**:
```bash
# Check file listing
xpcs-toolkit list /path/to/data/ --validate

# Verify file extensions
ls -la /path/to/data/*.h5

# Check file format
h5dump -H /path/to/data/file.h5
```

#### ❌ "Memory error during analysis"

**Cause**: Insufficient memory for large datasets

**Solutions**:
```bash
# Reduce memory usage
export XPCS_MEMORY_LIMIT="4GB"

# Process smaller q-ranges
xpcs-toolkit g2 /data/ --qmin 0.01 --qmax 0.05

# Use data chunking
xpcs-toolkit saxs2d /data/ --chunk-size 1000
```

#### ❌ "Correlation function fitting failed"

**Cause**: Poor signal-to-noise or inappropriate fit model

**Solutions**:
```bash
# Try different fit functions
xpcs-toolkit g2 /data/ --fit-function stretched

# Adjust fit range
xpcs-toolkit g2 /data/ --fit-range "1e-5,1e-2"

# Check data quality first
xpcs-toolkit stability /data/
```

### Performance Optimization

#### 🚀 Speed Up Analysis

```bash
# Use parallel processing
export XPCS_THREADS=8

# Enable memory mapping
export XPCS_MMAP=true

# Optimize chunking
export XPCS_CHUNK_SIZE=2048
```

#### 💾 Reduce Memory Usage

```bash
# Lazy loading
export XPCS_LAZY_LOADING=true

# Smaller data types
export XPCS_FLOAT_PRECISION=float32

# Garbage collection
export XPCS_GC_FREQUENCY=100
```

### Getting Help

#### 📚 Documentation

```bash
# Command help
xpcs-toolkit --help
xpcs-toolkit g2 --help

# Version information
xpcs-toolkit --version

# Detailed logging
xpcs-toolkit g2 /data/ --verbose
```

#### 🐛 Bug Reports

For bug reports and feature requests:
- **GitHub Issues**: https://github.com/imewei/xpcs-toolkit/issues
- **Discussions**: https://github.com/imewei/xpcs-toolkit/discussions
- **Email**: weichen@anl.gov

#### 📞 Support Channels

- **Community Forum**: Scientific discussions and usage questions
- **Documentation**: Complete API reference and tutorials
- **Example Gallery**: Real-world analysis examples
- **Video Tutorials**: Step-by-step analysis workflows

---

## Advanced Usage

### 🔧 Configuration Files

Create `~/.xpcs_config.yaml` for default settings:

```yaml
# XPCS Toolkit Configuration
default_output_dir: "/results"
default_colormap: "viridis"
log_level: "INFO"
memory_limit: "8GB"
parallel_workers: 4

# Analysis defaults
saxs2d:
  dpi: 300
  log_scale: true

g2:
  fit_function: "single_exp"
  error_bars: true
  log_x: true

saxs1d:
  log_x: true
  log_y: true
```

### 🐍 Python API Integration

```python
# Use CLI functions in Python scripts
from xpcs_toolkit.cli_headless import plot_saxs_2d, plot_g2_function
import argparse

# Create argument object
args = argparse.Namespace()
args.path = "/data/experiment/"
args.output = "result.png"
args.log_scale = True

# Run analysis
plot_saxs_2d(args)
plot_g2_function(args)
```

### 🔄 Workflow Automation

```bash
#!/bin/bash
# Automated daily processing script

# Configuration
DATA_ARCHIVE="/beamline/archive"
PROCESSING_DIR="/processing/daily"
RESULTS_DIR="/results/daily"

# Create directories
mkdir -p "$PROCESSING_DIR" "$RESULTS_DIR"

# Find new data files
find "$DATA_ARCHIVE" -name "*.h5" -mtime -1 > new_files.txt

# Process each file
while read -r file; do
    basename=$(basename "$file" .h5)
    
    # Quality check
    if xpcs-toolkit stability "$file" --threshold 0.03; then
        echo "Processing $basename - quality OK"
        
        # Full analysis
        xpcs-toolkit saxs2d "$file" --output "$RESULTS_DIR/${basename}_2d.png"
        xpcs-toolkit g2 "$file" --output "$RESULTS_DIR/${basename}_g2.png"
        xpcs-toolkit saxs1d "$file" --output "$RESULTS_DIR/${basename}_1d.png"
        
        # Archive processed file
        mv "$file" "$PROCESSING_DIR/"
    else
        echo "Skipping $basename - quality issues detected"
    fi
done < new_files.txt

# Generate daily report
echo "Daily processing complete: $(date)" > "$RESULTS_DIR/daily_report.txt"
ls -la "$RESULTS_DIR"/*.png >> "$RESULTS_DIR/daily_report.txt"
```

---

**For the most up-to-date documentation and examples, visit:**  
🌐 https://github.com/imewei/xpcs-toolkit

**Citation**: If you use XPCS Toolkit in your research, please cite:  
*Chen, Wei. "XPCS Toolkit: Advanced X-ray Photon Correlation Spectroscopy Analysis." Argonne National Laboratory (2024).*