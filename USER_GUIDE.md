# XPCS Toolkit User Guide

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://python.org)

**Complete user guide for X-ray Photon Correlation Spectroscopy data analysis with the XPCS Toolkit.**

## Table of Contents

- [Quick Start](#quick-start)
- [Command Line Interface](#command-line-interface)
- [Python API](#python-api)
- [Analysis Workflows](#analysis-workflows)
- [Examples Gallery](#examples-gallery)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

```bash
# Basic installation
pip install xpcs-toolkit

# Development installation  
git clone https://github.com/imewei/xpcs-toolkit.git
cd xpcs-toolkit
make dev
```

### First Steps

```bash
# Verify installation
xpcs-toolkit --version

# Quick help
xpcs-toolkit --help

# List files in a directory
xpcs-toolkit list /path/to/data/

# Generate quick analysis
xpcs-toolkit saxs2d /path/to/data/ --output results.png
```

---

## Command Line Interface

### Core Commands

#### 🔍 `list` - File Discovery
```bash
# Basic file listing
xpcs-toolkit list /path/to/data/

# Recursive search with validation
xpcs-toolkit list /path/to/data/ --recursive --validate

# Export inventory to JSON
xpcs-toolkit list /path/to/data/ --format json > inventory.json
```

#### 📊 `saxs2d` - 2D SAXS Visualization
```bash
# Basic 2D pattern
xpcs-toolkit saxs2d /path/to/data/ --output pattern.png

# High-resolution with log scaling
xpcs-toolkit saxs2d /path/to/data/ \
  --log-scale --dpi 600 --output high_res.png

# Custom intensity range
xpcs-toolkit saxs2d /path/to/data/ \
  --vmin 1e-6 --vmax 1e-2 --colormap plasma
```

#### 📈 `g2` - Correlation Analysis
```bash
# Basic correlation analysis
xpcs-toolkit g2 /path/to/data/ --output correlation.png

# Specific q-range with fitting
xpcs-toolkit g2 /path/to/data/ \
  --qmin 0.01 --qmax 0.05 \
  --fit-function single_exp --log-x

# Export data for external analysis
xpcs-toolkit g2 /path/to/data/ \
  --export-data correlation_data.csv \
  --fit-range "1e-5,1e-2"
```

#### 🎯 `saxs1d` - 1D Profiles
```bash
# Basic radial profile
xpcs-toolkit saxs1d /path/to/data/ --output profile.png

# Logarithmic scaling
xpcs-toolkit saxs1d /path/to/data/ \
  --log-x --log-y --output log_profile.png

# Background subtraction
xpcs-toolkit saxs1d /path/to/data/ \
  --background /path/to/background.h5 \
  --q-range "0.005,0.5"
```

#### 🔬 `stability` - Quality Assessment
```bash
# Basic stability analysis
xpcs-toolkit stability /path/to/data/ --output stability.png

# Detailed analysis with custom windows
xpcs-toolkit stability /path/to/data/ \
  --time-windows 20 --threshold 0.02 \
  --export-data stability_metrics.csv
```

### Complete Analysis Pipeline

```bash
#!/bin/bash
# Complete XPCS analysis workflow

DATA_DIR="/path/to/experiment"
OUTPUT_DIR="/path/to/results"
mkdir -p "$OUTPUT_DIR"

# 1. File validation
xpcs-toolkit list "$DATA_DIR" --validate > "$OUTPUT_DIR/files.txt"

# 2. Quality assessment
xpcs-toolkit stability "$DATA_DIR" \
  --export-data "$OUTPUT_DIR/stability.csv" \
  --output "$OUTPUT_DIR/stability.png"

# 3. SAXS analysis
xpcs-toolkit saxs2d "$DATA_DIR" \
  --log-scale --output "$OUTPUT_DIR/saxs2d.png"

xpcs-toolkit saxs1d "$DATA_DIR" \
  --log-x --log-y --output "$OUTPUT_DIR/saxs1d.png" \
  --export-data "$OUTPUT_DIR/saxs1d.csv"

# 4. Correlation analysis  
xpcs-toolkit g2 "$DATA_DIR" \
  --qmin 0.01 --qmax 0.08 --fit-function single_exp \
  --export-data "$OUTPUT_DIR/g2.csv" \
  --output "$OUTPUT_DIR/g2.png"

echo "Analysis complete! Results in $OUTPUT_DIR"
```

---

## Python API

### Core Classes

#### XpcsDataFile - Data Container
```python
from xpcs_toolkit import XpcsDataFile

# Load XPCS dataset
data = XpcsDataFile('/path/to/experiment.h5')

# Access data arrays
g2_function = data.g2              # Correlation function [n_q, n_tau]
tau_values = data.tau             # Delay times [seconds]
saxs_2d = data.saxs_2d           # 2D scattering pattern
saxs_1d = data.saxs_1d           # 1D radial profile
q_values = data.q_1d             # Q-values [Å⁻¹]

# Get experimental parameters
print(f"Sample: {data.sample_name}")
print(f"Analysis: {data.analysis_type}")
print(f"Q-range: {q_values.min():.4f} - {q_values.max():.4f} Å⁻¹")
print(f"Time range: {tau_values.min():.2e} - {tau_values.max():.2e} s")
```

#### AnalysisKernel - Analysis Engine
```python
from xpcs_toolkit import AnalysisKernel

# Initialize analysis kernel
kernel = AnalysisKernel('/path/to/data/')
kernel.build_file_list()

# Get file list
files = kernel.get_xf_list(rows=[0, 1, 2])  # First 3 files

# Run analyses
correlation_results = kernel.run_correlation_analysis(
    q_range=(0.01, 0.1),
    fit_function='single_exp'
)

saxs_results = kernel.run_saxs_analysis(
    log_scale=True,
    output_format='png'
)

stability_results = kernel.run_stability_analysis(
    time_windows=20
)

print(f"Processed {len(files)} files")
print(f"Stability score: {stability_results['overall_stability']:.3f}")
```

### Analysis Functions

#### Correlation Analysis
```python
from xpcs_toolkit.scientific.correlation import g2

# Extract correlation data
q, tel, g2_data, g2_err, labels = g2.get_data(
    xf_list, 
    q_range=(0.01, 0.05),
    time_range=(1e-5, 1e-1)
)

# Analyze data
print(f"Q-bins: {len(q[0])}")
print(f"Time points: {len(tel[0])}")
print(f"G2 shape: {g2_data[0].shape}")
```

#### SAXS Analysis
```python
from xpcs_toolkit.scientific.scattering import saxs_1d

# Generate 1D profiles
saxs_1d.pg_plot(xf_list, log_x=True, log_y=True)

# Get plotting colors
color, marker = saxs_1d.get_color_marker(index=0)
```

#### Data Quality
```python
# Validate data quality
quality = data.validate_data_quality()
print(f"Overall quality score: {quality['overall_score']:.3f}")

if quality['overall_score'] < 0.7:
    print("⚠️ Data quality issues detected:")
    for issue in quality['issues']:
        print(f"  - {issue}")
```

### Scientific Analysis Examples

#### Diffusion Coefficient Measurement
```python
import numpy as np
from xpcs_toolkit import XpcsDataFile
from xpcs_toolkit.utils.math.fitting import single_exp, fit_tau

# Load data
data = XpcsDataFile('brownian_particles.h5')

# Extract correlation data
q_vals = data.q_1d
tau_vals = data.tau
g2_vals = data.g2

# Fit each q-bin
diffusion_coeffs = []
for q_idx, q in enumerate(q_vals):
    if q > 0.01 and q < 0.1:  # Select q-range
        g2_q = g2_vals[q_idx, :]
        
        # Fit single exponential
        fit_results = fit_tau(g2_q, tau_vals, fit_range=(1e-5, 1e-2))
        
        if fit_results['success']:
            gamma = fit_results['gamma']
            D = gamma / (q**2)  # D = Γ/q²
            diffusion_coeffs.append(D)

# Average diffusion coefficient
D_avg = np.mean(diffusion_coeffs)
D_std = np.std(diffusion_coeffs)

print(f"Diffusion coefficient: {D_avg:.2e} ± {D_std:.2e} m²/s")

# Calculate hydrodynamic radius (Stokes-Einstein)
kT = 4.11e-21  # J at 298K
eta = 0.89e-3  # Pa·s (water viscosity)
R_h = kT / (6 * np.pi * eta * D_avg)

print(f"Hydrodynamic radius: {R_h * 1e9:.1f} nm")
```

---

## Analysis Workflows

### Batch Processing
```python
from pathlib import Path
import pandas as pd

def process_experiment_series(data_dir, output_dir):
    """Process multiple XPCS experiments in batch."""
    
    results = []
    
    for h5_file in Path(data_dir).glob('*.h5'):
        try:
            # Load data
            data = XpcsDataFile(str(h5_file))
            
            # Quick analysis
            quality = data.validate_data_quality()
            
            # Extract key metrics
            result = {
                'filename': h5_file.name,
                'sample_name': getattr(data, 'sample_name', 'unknown'),
                'analysis_type': data.analysis_type,
                'q_min': data.q_1d.min(),
                'q_max': data.q_1d.max(),
                'tau_min': data.tau.min(),
                'tau_max': data.tau.max(),
                'quality_score': quality['overall_score'],
                'data_points': data.g2.size
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"⚠️ Failed to process {h5_file.name}: {e}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/batch_analysis.csv", index=False)
    
    return df

# Usage
results_df = process_experiment_series('/data/experiment/', '/results/')
print(f"Processed {len(results_df)} files successfully")
```

### Temperature Series Analysis
```python
def analyze_temperature_series(file_pattern, temperatures):
    """Analyze diffusion vs temperature."""
    
    import matplotlib.pyplot as plt
    
    diffusion_data = []
    
    for temp, filename in zip(temperatures, file_pattern):
        data = XpcsDataFile(filename)
        
        # Extract diffusion coefficient
        # (using previous diffusion analysis code)
        D = extract_diffusion_coefficient(data)
        
        diffusion_data.append({'T': temp, 'D': D, 'file': filename})
    
    # Arrhenius analysis
    df = pd.DataFrame(diffusion_data)
    df['1000/T'] = 1000 / df['T']
    df['ln_D'] = np.log(df['D'])
    
    # Linear fit
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['1000/T'], df['ln_D']
    )
    
    # Activation energy
    E_a = -slope * 8.314  # J/mol (R = 8.314 J/mol/K)
    
    print(f"Activation energy: {E_a:.0f} J/mol")
    print(f"R² = {r_value**2:.3f}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df['1000/T'], df['ln_D'], 'o-')
    plt.xlabel('1000/T (K⁻¹)')
    plt.ylabel('ln(D)')
    plt.title(f'Arrhenius Plot (Ea = {E_a:.0f} J/mol)')
    plt.grid(True)
    plt.savefig('arrhenius_plot.png', dpi=300, bbox_inches='tight')
    
    return df
```

---

## Examples Gallery

### 2D SAXS Patterns

```bash
# Isotropic liquid sample
xpcs-toolkit saxs2d /data/liquid/ \
  --log-scale --colormap viridis --output liquid.png

# Oriented fiber sample  
xpcs-toolkit saxs2d /data/fiber/ \
  --colormap plasma --q-range "0.1,2.0" --output fiber.png

# Powder diffraction
xpcs-toolkit saxs2d /data/powder/ \
  --colormap hot --output powder.png
```

### Correlation Analysis

```bash
# Simple Brownian motion
xpcs-toolkit g2 /data/brownian/ \
  --qmin 0.02 --qmax 0.10 --fit-function single_exp \
  --log-x --output brownian_g2.png

# Glass transition dynamics
xpcs-toolkit g2 /data/glass/ \
  --fit-function stretched --fit-range "1e-4,100" \
  --log-x --output glass_g2.png

# Two-timescale system
xpcs-toolkit g2 /data/complex/ \
  --fit-function double_exp --tau-max 1000 \
  --output complex_g2.png
```

### 1D Profile Analysis

```bash
# Nanoparticle form factor
xpcs-toolkit saxs1d /data/nanoparticles/ \
  --log-x --log-y --smooth 0.01 \
  --q-range "0.01,1.0" --output particles_1d.png

# Fractal aggregates
xpcs-toolkit saxs1d /data/aggregates/ \
  --log-x --log-y --q-range "0.005,0.5" \
  --output fractal_1d.png
```

---

## Troubleshooting

### Common Issues

#### ❌ "No HDF files found"
**Problem**: Directory doesn't contain valid XPCS files

**Solutions**:
```bash
# Check file listing
xpcs-toolkit list /path/to/data/ --validate

# Verify file format
h5dump -H /path/to/data/file.h5
```

#### ❌ "Memory error during analysis"
**Problem**: Large dataset exceeds available memory

**Solutions**:
```bash
# Set memory limits
export XPCS_MEMORY_LIMIT="4GB"

# Process smaller q-ranges
xpcs-toolkit g2 /data/ --qmin 0.01 --qmax 0.05

# Use data export instead of plotting
xpcs-toolkit g2 /data/ --export-data results.csv
```

#### ❌ "Fitting failed"
**Problem**: Poor data quality or wrong model

**Solutions**:
```bash
# Check data quality first
xpcs-toolkit stability /data/

# Try different fit functions
xpcs-toolkit g2 /data/ --fit-function stretched

# Adjust fit range
xpcs-toolkit g2 /data/ --fit-range "1e-5,1e-2"
```

#### ❌ "Import errors"
**Problem**: Package not properly installed

**Solutions**:
```bash
# Verify installation
pip show xpcs-toolkit

# Reinstall if needed
pip install --upgrade --force-reinstall xpcs-toolkit

# Check Python path
python -c "import xpcs_toolkit; print(xpcs_toolkit.__file__)"
```

### Performance Tips

#### Speed Up Analysis
```bash
# Use parallel processing
export XPCS_THREADS=8

# Enable memory mapping
export XPCS_MMAP=true

# Optimize chunking
export XPCS_CHUNK_SIZE=2048
```

#### Memory Optimization
```bash
# Enable lazy loading
export XPCS_LAZY_LOADING=true

# Use smaller precision
export XPCS_FLOAT_PRECISION=float32
```

### Getting Help

#### Documentation
```bash
# Command help
xpcs-toolkit --help
xpcs-toolkit g2 --help

# Version info
xpcs-toolkit --version
```

#### Debugging
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check data structure
with h5py.File('data.h5', 'r') as f:
    print(list(f.keys()))
```

#### Support Resources
- **GitHub Issues**: [Report bugs and request features](https://github.com/imewei/xpcs-toolkit/issues)
- **Discussions**: [Ask questions and share tips](https://github.com/imewei/xpcs-toolkit/discussions)
- **Email**: weichen@anl.gov
- **Documentation**: See [SCIENTIFIC_BACKGROUND.md](SCIENTIFIC_BACKGROUND.md) for theory

---

**For advanced topics see:**
- 📚 [FILE_FORMAT_GUIDE.md](FILE_FORMAT_GUIDE.md) - Detailed file format reference
- 🧪 [SCIENTIFIC_BACKGROUND.md](SCIENTIFIC_BACKGROUND.md) - Theory and methods
- ⚙️ [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Contributing and development