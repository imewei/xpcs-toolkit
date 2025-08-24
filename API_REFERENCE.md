# XPCS Toolkit - API Reference

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)

**Complete API reference for the XPCS Toolkit - Advanced X-ray Photon Correlation Spectroscopy Analysis Framework**

## Table of Contents

- [Overview](#overview)
- [Core Classes](#core-classes)
- [Analysis Modules](#analysis-modules)
- [File I/O Components](#file-io-components)  
- [Helper Utilities](#helper-utilities)
- [Data Structures](#data-structures)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Overview

The XPCS Toolkit provides a comprehensive Python API for X-ray Photon Correlation Spectroscopy data analysis. The architecture is designed for:

- **Scientific Accuracy**: Rigorous implementation of XPCS theory and methods
- **Performance**: Optimized for large-scale synchrotron datasets  
- **Flexibility**: Modular design supporting diverse experimental configurations
- **Extensibility**: Clean interfaces for custom analysis development
- **Reproducibility**: Consistent analysis protocols and detailed logging

### Architecture Overview

```python
# Core Architecture
xpcs_toolkit/
├── 📁 Core Classes
│   ├── XpcsDataFile      # Primary data container and loader
│   ├── AnalysisKernel    # Main analysis orchestrator  
│   └── DataFileLocator   # File discovery and management
├── 📁 Analysis Modules
│   ├── g2mod            # Multi-tau correlation analysis
│   ├── saxs1d           # 1D radial profile analysis
│   ├── saxs2d           # 2D scattering visualization
│   ├── stability       # Beam stability monitoring
│   ├── twotime          # Two-time correlation analysis
│   └── average_toolbox  # Data averaging utilities
├── 📁 File I/O
│   ├── hdf_reader       # HDF5/NeXus file handlers
│   ├── ftype_utils      # Format detection utilities
│   └── qmap_utils       # Q-space mapping tools
└── 📁 Utilities
    ├── helper/          # Data models and utilities
    ├── mpl_compat       # Matplotlib compatibility
    └── logging/         # Advanced logging system
```

---

## Core Classes

### 🔬 XpcsDataFile

**Primary data container for XPCS datasets with comprehensive analysis capabilities.**

```python
from xpcs_toolkit import XpcsDataFile

# Initialize from HDF5 file
data = XpcsDataFile('/path/to/experiment.h5')
```

#### Class Definition

```python
class XpcsDataFile:
    """
    Comprehensive XPCS Dataset Handler for NeXus format files.
    
    Provides complete functionality for loading, analyzing, and manipulating
    X-ray Photon Correlation Spectroscopy datasets from APS 8-ID-I.
    """
```

#### Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `g2` | `np.ndarray` | Multi-tau correlation function [q_index, tau_index] |
| `tau` | `np.ndarray` | Correlation delay times in seconds |
| `saxs_2d` | `np.ndarray` | 2D scattering pattern [pixels_y, pixels_x] |
| `saxs_1d` | `np.ndarray` | 1D radial intensity profile [q_points] |
| `q_2d` | `np.ndarray` | Q-map for 2D detector [pixels_y, pixels_x] |
| `q_1d` | `np.ndarray` | Q-values for 1D profile [Å⁻¹] |
| `intensity_mean` | `np.ndarray` | Time-averaged intensity [counts] |
| `analysis_type` | `str` | Type of analysis: 'Multitau' or 'Twotime' |

#### Scientific Methods

```python
# Data Access Methods
def get_correlation_data(self, q_range: Tuple[float, float] = None) -> Dict:
    """
    Extract correlation function data for specified q-range.
    
    Args:
        q_range: Tuple of (q_min, q_max) in Å⁻¹
        
    Returns:
        Dict containing g2, tau, and q_values arrays
        
    Example:
        >>> data = XpcsDataFile('experiment.h5')
        >>> corr = data.get_correlation_data(q_range=(0.01, 0.05))
        >>> print(f"Q-range: {corr['q_values'].min():.3f} - {corr['q_values'].max():.3f}")
    """

def get_saxs_profile(self, phi_range: Tuple[float, float] = None) -> Dict:
    """
    Extract 1D SAXS profile with optional angular integration.
    
    Args:
        phi_range: Angular range for sector integration in degrees
        
    Returns:
        Dict containing q_values and intensity arrays
    """

def get_detector_geometry(self) -> Dict:
    """
    Get complete detector geometry and calibration parameters.
    
    Returns:
        Dict with detector distance, pixel size, beam center, wavelength
    """

def validate_data_quality(self) -> Dict:
    """
    Perform comprehensive data quality assessment.
    
    Returns:
        Dict with quality metrics and recommendations
    """
```

#### Usage Examples

```python
# Basic data loading and exploration
data = XpcsDataFile('/beamline/data/sample_001.h5')

# Check available data
print(f"Analysis type: {data.analysis_type}")
print(f"Dataset shape: g2={data.g2.shape}, SAXS 2D={data.saxs_2d.shape}")
print(f"Q-range: {data.q_1d.min():.4f} to {data.q_1d.max():.4f} Å⁻¹")

# Extract correlation data for specific q-range
correlation = data.get_correlation_data(q_range=(0.02, 0.08))
g2_subset = correlation['g2']
tau_values = correlation['tau'] 
q_selected = correlation['q_values']

# Get SAXS profile
saxs_profile = data.get_saxs_profile()
intensity_1d = saxs_profile['intensity']
q_1d = saxs_profile['q_values']

# Quality assessment
quality = data.validate_data_quality()
print(f"Data quality score: {quality['overall_score']:.2f}")
```

---

### 🧮 AnalysisKernel

**Main analysis orchestrator coordinating multiple analysis modules.**

```python
from xpcs_toolkit import AnalysisKernel

# Initialize analysis kernel
kernel = AnalysisKernel('/path/to/data/directory')
```

#### Class Definition

```python
class AnalysisKernel(DataFileLocator):
    """
    Advanced XPCS Data Analysis Engine.
    
    Coordinates multiple specialized analysis modules to provide complete 
    analysis workflows for both multi-tau and two-time correlation experiments.
    """
```

#### Core Analysis Methods

```python
def build_file_list(self) -> None:
    """
    Discover and validate XPCS files in the specified directory.
    
    Automatically detects file formats and builds internal file inventory
    with metadata extraction and quality pre-assessment.
    """

def get_xf_list(self, rows: List[int] = None) -> List[XpcsDataFile]:
    """
    Get list of XpcsDataFile objects for analysis.
    
    Args:
        rows: List of file indices to load (None for all files)
        
    Returns:
        List of loaded XpcsDataFile objects
    """

def run_correlation_analysis(self, 
                           q_range: Tuple[float, float] = None,
                           fit_function: str = 'single_exp') -> Dict:
    """
    Execute comprehensive correlation function analysis.
    
    Args:
        q_range: Q-range for analysis (Å⁻¹)
        fit_function: Fitting model ('single_exp', 'double_exp', 'stretched')
        
    Returns:
        Dict with correlation functions, fit parameters, and quality metrics
    """

def run_saxs_analysis(self, 
                     log_scale: bool = True,
                     output_format: str = 'png') -> Dict:
    """
    Execute comprehensive SAXS analysis workflow.
    
    Args:
        log_scale: Use logarithmic intensity scaling
        output_format: Output file format for plots
        
    Returns:
        Dict with 2D patterns, 1D profiles, and analysis results
    """

def run_stability_analysis(self, 
                          time_windows: int = 10,
                          metrics: List[str] = None) -> Dict:
    """
    Assess beam stability and data quality over measurement time.
    
    Args:
        time_windows: Number of time segments for analysis
        metrics: List of stability metrics to calculate
        
    Returns:
        Dict with stability metrics and quality assessment
    """
```

#### Workflow Examples

```python
# Complete analysis workflow
kernel = AnalysisKernel('/beamline/data/experiment/')
kernel.build_file_list()

# Get data files
xf_list = kernel.get_xf_list(rows=[0, 1, 2])  # First 3 files

# Correlation analysis
corr_results = kernel.run_correlation_analysis(
    q_range=(0.01, 0.1),
    fit_function='single_exp'
)

# SAXS analysis  
saxs_results = kernel.run_saxs_analysis(
    log_scale=True,
    output_format='png'
)

# Stability assessment
stability = kernel.run_stability_analysis(time_windows=20)

print(f"Processed {len(xf_list)} files")
print(f"Relaxation time range: {corr_results['tau_range']}")
print(f"Stability score: {stability['overall_stability']:.3f}")
```

---

### 📁 DataFileLocator

**Intelligent file discovery and management system.**

```python
from xpcs_toolkit import DataFileLocator

# Initialize file locator
locator = DataFileLocator('/path/to/data')
```

#### Class Definition

```python
class DataFileLocator:
    """
    Intelligent XPCS file discovery and management system.
    
    Provides automated file discovery, format validation, and metadata
    extraction for XPCS datasets with caching and performance optimization.
    """
```

#### File Management Methods

```python
def build_file_inventory(self) -> Dict:
    """
    Build comprehensive inventory of XPCS files in directory.
    
    Returns:
        Dict with file lists, metadata, and validation results
    """

def validate_file_format(self, filename: str) -> Dict:
    """
    Validate XPCS file format and extract basic metadata.
    
    Args:
        filename: Path to XPCS file
        
    Returns:
        Dict with format validation and metadata
    """

def get_file_summary(self) -> Dict:
    """
    Get summary statistics for discovered files.
    
    Returns:
        Dict with file counts, sizes, and format distribution
    """

def filter_files(self, 
                criteria: Dict = None,
                date_range: Tuple[str, str] = None) -> List[str]:
    """
    Filter files based on specified criteria.
    
    Args:
        criteria: Dict with filtering parameters
        date_range: Tuple of (start_date, end_date) strings
        
    Returns:
        List of filtered file paths
    """
```

---

## Analysis Modules

### 📊 g2mod - Multi-tau Correlation Analysis

**Advanced correlation function analysis with fitting capabilities.**

```python
from xpcs_toolkit.module import g2mod

# Extract correlation data
g2_data = g2mod.get_data(xf_list, q_range=(0.01, 0.05))

# Create correlation plots
g2mod.pg_plot(g2_data, log_x=True, fit_function='single_exp')
```

#### Key Functions

```python
def get_data(xf_list: List[XpcsDataFile], 
            q_range: Tuple[float, float] = None,
            tau_range: Tuple[float, float] = None) -> Dict:
    """
    Extract correlation function data from XPCS files.
    
    Args:
        xf_list: List of XpcsDataFile objects
        q_range: Q-range for analysis (Å⁻¹)
        tau_range: Time range for analysis (seconds)
        
    Returns:
        Dict with correlation data, errors, and metadata
    """

def fit_correlation(g2_data: np.ndarray, 
                   tau_data: np.ndarray,
                   fit_function: str = 'single_exp',
                   fit_range: Tuple[float, float] = None) -> Dict:
    """
    Fit correlation functions with specified model.
    
    Args:
        g2_data: Correlation function array
        tau_data: Correlation times array  
        fit_function: Model ('single_exp', 'double_exp', 'stretched')
        fit_range: Time range for fitting
        
    Returns:
        Dict with fit parameters, errors, and quality metrics
    """

def pg_plot(g2_data: Dict, 
           log_x: bool = True,
           log_y: bool = False,
           show_fits: bool = True) -> None:
    """
    Generate publication-quality correlation function plots.
    
    Args:
        g2_data: Correlation data from get_data()
        log_x: Use logarithmic time axis
        log_y: Use logarithmic g2 axis  
        show_fits: Display fitted curves
    """
```

### 📈 saxs1d - 1D Radial Profile Analysis

**Comprehensive 1D SAXS profile analysis and visualization.**

```python
from xpcs_toolkit.module import saxs1d

# Generate 1D profiles
saxs1d.pg_plot(xf_list, log_x=True, log_y=True)

# Get color/marker for plotting
color, marker = saxs1d.get_color_marker(index=0)
```

#### Key Functions

```python
def pg_plot(xf_list: List[XpcsDataFile],
           log_x: bool = False,
           log_y: bool = False, 
           q_range: Tuple[float, float] = None) -> None:
    """
    Generate 1D SAXS profile plots.
    
    Args:
        xf_list: List of XpcsDataFile objects
        log_x: Use logarithmic q-axis
        log_y: Use logarithmic intensity axis
        q_range: Q-range for display
    """

def get_color_marker(index: int) -> Tuple[str, str]:
    """
    Get color and marker style for plotting.
    
    Args:
        index: Plot series index
        
    Returns:
        Tuple of (color_string, marker_string)
    """

def radial_average(saxs_2d: np.ndarray, 
                  q_map: np.ndarray,
                  phi_range: Tuple[float, float] = None) -> Dict:
    """
    Perform radial averaging of 2D scattering patterns.
    
    Args:
        saxs_2d: 2D scattering pattern
        q_map: Q-space mapping
        phi_range: Angular range for sector averaging
        
    Returns:
        Dict with q_values and averaged intensity
    """
```

### 🔍 stability - Beam Stability Analysis

**Statistical analysis of beam stability and data quality.**

```python
from xpcs_toolkit.module import stability

# Analyze beam stability
stability.plot(xf_list, time_windows=10, metric='intensity')
```

#### Key Functions  

```python
def plot(xf_list: List[XpcsDataFile],
         time_windows: int = 10,
         metric: str = 'intensity',
         threshold: float = 0.05) -> Dict:
    """
    Analyze and plot beam stability metrics.
    
    Args:
        xf_list: List of XpcsDataFile objects
        time_windows: Number of time segments
        metric: Stability metric ('intensity', 'position', 'detector')
        threshold: Stability threshold for quality assessment
        
    Returns:
        Dict with stability metrics and quality flags
    """

def calculate_stability_metrics(intensity_data: np.ndarray,
                              time_windows: int = 10) -> Dict:
    """
    Calculate comprehensive stability metrics.
    
    Args:
        intensity_data: Time series intensity data
        time_windows: Number of analysis windows
        
    Returns:
        Dict with RSD, drift, and stability statistics
    """
```

### 🌊 twotime - Two-time Correlation Analysis

**Advanced two-time correlation visualization and analysis.**

```python
from xpcs_toolkit.module import twotime

# Two-time correlation plot
twotime.plot_twotime(xf_list, q_index=5, age_time=100)
```

---

## File I/O Components

### 📄 hdf_reader - HDF5/NeXus File Handlers

**Robust HDF5 and NeXus format file reading with error handling.**

```python
from xpcs_toolkit.fileIO.hdf_reader import get_abs_cs_scale, get_analysis_type

# Get absolute cross-section scaling
scale_factor = get_abs_cs_scale('experiment.h5')

# Determine analysis type
analysis_type = get_analysis_type('experiment.h5')
```

#### Key Functions

```python
def get_abs_cs_scale(filename: str, 
                    file_type: str = None) -> float:
    """
    Extract absolute cross-section scaling factor.
    
    Args:
        filename: Path to HDF5 file
        file_type: File format type (auto-detected if None)
        
    Returns:
        Scaling factor for absolute intensity calibration
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If scaling data not found
    """

def get_analysis_type(filename: str) -> str:
    """
    Determine XPCS analysis type from file metadata.
    
    Args:
        filename: Path to HDF5 file
        
    Returns:
        Analysis type string ('Multitau' or 'Twotime')
    """

def read_detector_geometry(filename: str) -> Dict:
    """
    Read complete detector geometry parameters.
    
    Args:
        filename: Path to HDF5 file
        
    Returns:
        Dict with detector distance, pixel size, beam center
    """
```

### 🔍 ftype_utils - Format Detection

**Intelligent file format detection and validation.**

```python
from xpcs_toolkit.fileIO.ftype_utils import get_ftype, isNeXusFile

# Detect file format
file_format = get_ftype('experiment.h5')

# Check if NeXus format
is_nexus = isNeXusFile('experiment.h5')
```

---

## Helper Utilities

### 📋 listmodel - Data Models

**Qt-compatible data models for GUI applications.**

```python
from xpcs_toolkit.helper.listmodel import ListDataModel, TableDataModel

# Create list model
data_list = ['file1.h5', 'file2.h5', 'file3.h5']
list_model = ListDataModel(data_list)

# Create table model
table_model = TableDataModel()
```

### 📊 fitting - Scientific Fitting Functions

**Advanced fitting functions for XPCS data analysis.**

```python
from xpcs_toolkit.helper.fitting import single_exp, fit_tau

# Single exponential function
y_fit = single_exp(x_data, tau=2.5, bkg=0.1, cts=1.0)

# Fit correlation function
fit_results = fit_tau(g2_data, tau_data, fit_range=(1e-5, 1e-1))
```

---

## Data Structures

### 🗂️ Standard Data Formats

**Consistent data structures used throughout the toolkit.**

#### Correlation Data Structure
```python
correlation_data = {
    'g2': np.ndarray,           # Correlation function [q_index, tau_index]
    'tau': np.ndarray,          # Correlation times [seconds]
    'q_values': np.ndarray,     # Q-values [Å⁻¹]
    'errors': np.ndarray,       # Statistical errors
    'metadata': Dict            # Analysis parameters and timestamps
}
```

#### SAXS Data Structure
```python
saxs_data = {
    'saxs_2d': np.ndarray,      # 2D scattering pattern
    'saxs_1d': np.ndarray,      # 1D radial profile  
    'q_2d': np.ndarray,         # 2D Q-map
    'q_1d': np.ndarray,         # 1D Q-values
    'intensity_mean': np.ndarray, # Average intensity
    'geometry': Dict            # Detector geometry parameters
}
```

#### Analysis Results Structure
```python
analysis_results = {
    'fit_parameters': Dict,     # Fitted parameter values
    'fit_errors': Dict,         # Parameter uncertainties
    'fit_quality': Dict,        # R², chi², residuals
    'data_quality': Dict,       # SNR, statistics, flags
    'processing_info': Dict     # Timestamps, versions, settings
}
```

---

## Error Handling

### 🚨 Exception Hierarchy

```python
# Custom exceptions for specific error conditions
class XpcsError(Exception):
    """Base exception for XPCS Toolkit errors."""
    
class FileFormatError(XpcsError):
    """Invalid or unsupported file format."""
    
class DataQualityError(XpcsError):
    """Data quality below acceptable threshold."""
    
class AnalysisError(XpcsError):
    """Analysis computation failed."""
    
class CalibrationError(XpcsError):
    """Detector calibration or geometry error."""
```

### 🛡️ Error Handling Best Practices

```python
# Robust file loading with error handling
try:
    data = XpcsDataFile('experiment.h5')
except FileNotFoundError:
    print("File not found - check path")
except FileFormatError:
    print("Invalid file format - use NeXus or legacy HDF5")
except DataQualityError as e:
    print(f"Data quality issue: {e}")
    # Continue with reduced analysis or skip file
    
# Graceful analysis failure handling
try:
    correlation = kernel.run_correlation_analysis(q_range=(0.01, 0.1))
except AnalysisError as e:
    print(f"Correlation analysis failed: {e}")
    # Try alternative analysis or adjust parameters
```

---

## Best Practices

### 🎯 Performance Optimization

```python
# Efficient data loading for large datasets
data = XpcsDataFile('large_dataset.h5')

# Use lazy loading to minimize memory usage
data.enable_lazy_loading(chunk_size=1000)

# Load only required data fields
correlation_data = data.get_correlation_data(
    q_range=(0.02, 0.08),     # Specific q-range only
    tau_range=(1e-5, 1e-1)    # Specific time range only
)

# Batch processing with memory management
kernel = AnalysisKernel('/large_dataset/')
kernel.build_file_list()

# Process files in batches to manage memory
batch_size = 10
file_count = len(kernel.get_file_list())

for i in range(0, file_count, batch_size):
    batch_files = kernel.get_xf_list(rows=list(range(i, min(i+batch_size, file_count))))
    
    # Process batch
    results = kernel.run_correlation_analysis()
    
    # Save results and clear memory
    save_results(results)
    del batch_files, results
```

### 🔬 Scientific Accuracy

```python
# Proper error propagation
correlation = data.get_correlation_data(include_errors=True)
g2_values = correlation['g2']
g2_errors = correlation['errors']

# Statistical significance testing
if np.mean(g2_errors) / np.mean(g2_values) > 0.1:
    warnings.warn("Statistical errors > 10% - consider longer measurement")

# Proper normalization and baseline correction
g2_corrected = (g2_values - baseline) / normalization_factor

# Quality assessment before analysis
quality = data.validate_data_quality()
if quality['overall_score'] < 0.7:
    warnings.warn(f"Data quality score {quality['overall_score']:.2f} < 0.7")
```

### 📊 Reproducible Analysis

```python
# Always log analysis parameters
analysis_params = {
    'q_range': (0.01, 0.1),
    'fit_function': 'single_exp',
    'fit_range': (1e-5, 1e-2),
    'timestamp': datetime.now().isoformat(),
    'toolkit_version': xpcs_toolkit.__version__
}

# Save complete analysis record
results = {
    'data': correlation_results,
    'parameters': analysis_params,
    'quality_metrics': quality_assessment,
    'processing_log': processing_log
}

# Export for external analysis
np.savez('analysis_results.npz', **results)

# Generate analysis report
generate_analysis_report(results, output_file='analysis_report.pdf')
```

---

## Advanced Usage Examples

### 🚀 Complete Analysis Pipeline

```python
import numpy as np
from xpcs_toolkit import XpcsDataFile, AnalysisKernel
from xpcs_toolkit.module import g2mod, saxs1d, stability

# Initialize analysis pipeline
def run_complete_analysis(data_directory: str, output_directory: str):
    """
    Execute complete XPCS analysis pipeline.
    
    Args:
        data_directory: Path to XPCS data files
        output_directory: Path for analysis results
    """
    
    # Initialize analysis kernel
    kernel = AnalysisKernel(data_directory)
    kernel.build_file_list()
    
    # Quality assessment
    print("Assessing data quality...")
    stability_results = stability.plot(
        kernel.get_xf_list(), 
        time_windows=20,
        metric='intensity'
    )
    
    # Filter high-quality files
    quality_threshold = 0.8
    high_quality_files = [
        f for f, score in zip(kernel.get_file_list(), stability_results['quality_scores'])
        if score > quality_threshold
    ]
    
    print(f"Selected {len(high_quality_files)} high-quality files")
    
    # SAXS analysis
    print("Analyzing SAXS patterns...")
    saxs_results = kernel.run_saxs_analysis(
        log_scale=True,
        output_format='png'
    )
    
    # Correlation analysis with multiple q-ranges
    q_ranges = [(0.005, 0.02), (0.02, 0.05), (0.05, 0.1)]
    correlation_results = {}
    
    for i, q_range in enumerate(q_ranges):
        print(f"Analyzing q-range {q_range} Å⁻¹...")
        
        corr_data = kernel.run_correlation_analysis(
            q_range=q_range,
            fit_function='single_exp'
        )
        
        correlation_results[f'q_range_{i}'] = corr_data
    
    # Generate comprehensive report
    generate_analysis_report({
        'saxs': saxs_results,
        'correlation': correlation_results,
        'stability': stability_results,
        'parameters': {
            'data_directory': data_directory,
            'analysis_date': datetime.now().isoformat(),
            'toolkit_version': xpcs_toolkit.__version__
        }
    }, output_path=f"{output_directory}/analysis_report.pdf")
    
    print(f"Analysis complete. Results saved to {output_directory}")

# Execute analysis
run_complete_analysis('/beamline/data/experiment_2024/', '/results/experiment_2024/')
```

### 🔄 Batch Processing with Progress Monitoring

```python
from tqdm import tqdm
import multiprocessing as mp

def process_single_file(file_path: str) -> Dict:
    """Process single XPCS file."""
    try:
        data = XpcsDataFile(file_path)
        
        # Quick quality check
        quality = data.validate_data_quality()
        if quality['overall_score'] < 0.5:
            return {'status': 'low_quality', 'file': file_path}
        
        # Extract key results
        correlation = data.get_correlation_data(q_range=(0.01, 0.1))
        saxs_profile = data.get_saxs_profile()
        
        return {
            'status': 'success',
            'file': file_path,
            'correlation': correlation,
            'saxs': saxs_profile,
            'quality': quality
        }
        
    except Exception as e:
        return {'status': 'error', 'file': file_path, 'error': str(e)}

def batch_process_files(file_list: List[str], n_workers: int = 4):
    """Process multiple files in parallel with progress monitoring."""
    
    print(f"Processing {len(file_list)} files with {n_workers} workers...")
    
    with mp.Pool(n_workers) as pool:
        # Create progress bar
        with tqdm(total=len(file_list), desc="Processing files") as pbar:
            
            # Submit all jobs
            results = []
            for file_path in file_list:
                result = pool.apply_async(
                    process_single_file, 
                    (file_path,),
                    callback=lambda x: pbar.update(1)
                )
                results.append(result)
            
            # Collect results
            processed_results = [r.get() for r in results]
    
    # Analyze processing results
    successful = sum(1 for r in processed_results if r['status'] == 'success')
    low_quality = sum(1 for r in processed_results if r['status'] == 'low_quality')  
    errors = sum(1 for r in processed_results if r['status'] == 'error')
    
    print(f"Processing complete:")
    print(f"  Successful: {successful}")
    print(f"  Low quality: {low_quality}")
    print(f"  Errors: {errors}")
    
    return processed_results
```

---

**For complete documentation and examples:**
🌐 https://github.com/imewei/xpcs-toolkit

**Scientific References:**
📚 https://github.com/imewei/xpcs-toolkit/blob/main/SCIENTIFIC_BACKGROUND.md