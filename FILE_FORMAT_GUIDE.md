# XPCS Toolkit - File Format Guide

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)

**Comprehensive guide to XPCS data file formats, structures, and conversion utilities.**

## Table of Contents

- [Overview](#overview)
- [APS 8-ID-I NeXus Format](#aps-8-id-i-nexus-format)
- [Legacy HDF5 Format](#legacy-hdf5-format)
- [Data Structure Details](#data-structure-details)
- [Format Detection](#format-detection)
- [Conversion Utilities](#conversion-utilities)
- [Quality Validation](#quality-validation)
- [Best Practices](#best-practices)

## Overview

The XPCS Toolkit supports multiple file formats used in X-ray Photon Correlation Spectroscopy experiments, with automatic format detection and seamless data loading. The primary formats are:

### 🎯 **Primary Format: APS 8-ID-I NeXus**
- **Purpose**: Production format for Advanced Photon Source beamline 8-ID-I
- **Structure**: Hierarchical NeXus/HDF5 with standardized metadata
- **Features**: Complete experimental parameters, calibration data, quality metrics
- **Optimization**: Memory-efficient storage with compression and chunking

### 🔄 **Legacy Format: HDF5**
- **Purpose**: Backward compatibility with older datasets
- **Structure**: Custom HDF5 layouts from previous analysis software
- **Features**: Automatic migration to modern format structures  
- **Support**: Full conversion and analysis capabilities

### 🔍 **Format Detection**
- **Automatic**: Format detection based on file structure and metadata
- **Fallback**: Graceful handling of non-standard or corrupted files
- **Validation**: Comprehensive format validation and quality assessment
- **Reporting**: Detailed format compatibility reports

---

## APS 8-ID-I NeXus Format

The standardized NeXus format developed at Argonne National Laboratory's Advanced Photon Source provides a complete, self-contained description of XPCS experiments.

### 📁 File Structure Overview

```
experiment_001.h5                    # Root HDF5 container
├── 📁 /exchange/                   # Primary analysis data
│   ├── 🔢 g2                      # Multi-tau correlation functions
│   ├── 🔢 tau                     # Correlation delay times  
│   ├── 🔢 g2_fit_parameters       # Fitted relaxation parameters
│   ├── 🔢 saxs_2d                 # 2D scattering patterns
│   ├── 🔢 saxs_1d                 # 1D radial profiles
│   ├── 🔢 q_2d                    # 2D momentum transfer map
│   ├── 🔢 q_1d                    # 1D momentum transfer values
│   ├── 🔢 intensity_mean          # Time-averaged intensity
│   ├── 🔢 intensity_std           # Intensity standard deviation
│   ├── 🔢 c2                      # Two-time correlation (if available)
│   └── 🔢 abs_cross_section       # Absolute calibration factor
├── 📁 /measurement/                # Experimental conditions
│   ├── 📁 instrument/             # Beamline configuration
│   │   ├── 📁 detector/           # Detector parameters
│   │   │   ├── 🏷️ name           # Detector model/ID
│   │   │   ├── 🔢 distance        # Sample-detector distance [mm]
│   │   │   ├── 🔢 pixel_size_x    # Pixel size [mm]
│   │   │   ├── 🔢 pixel_size_y    # Pixel size [mm] 
│   │   │   ├── 🔢 beam_center_x   # Beam center [pixels]
│   │   │   ├── 🔢 beam_center_y   # Beam center [pixels]
│   │   │   ├── 🔢 efficiency      # Detective quantum efficiency
│   │   │   └── 🔢 mask            # Bad pixel mask
│   │   ├── 📁 source/             # X-ray source parameters
│   │   │   ├── 🔢 energy          # Photon energy [keV]
│   │   │   ├── 🔢 wavelength      # X-ray wavelength [Å]  
│   │   │   ├── 🔢 flux            # Photon flux [photons/s]
│   │   │   └── 🏷️ coherence_time # Source coherence time [s]
│   │   ├── 📁 monochromator/      # Energy selection
│   │   │   ├── 🔢 energy          # Selected energy [keV]
│   │   │   └── 🔢 bandwidth       # Energy bandwidth [ΔE/E]
│   │   └── 📁 attenuator/         # Beam attenuation
│   │       ├── 🔢 transmission    # Transmission factor
│   │       └── 🏷️ material       # Attenuator material
│   ├── 📁 sample/                 # Sample information
│   │   ├── 🏷️ name               # Sample identifier
│   │   ├── 🏷️ description        # Sample description
│   │   ├── 🔢 temperature         # Sample temperature [K]
│   │   ├── 🔢 pressure            # Sample pressure [bar]
│   │   ├── 🏷️ environment        # Sample environment details
│   │   ├── 🔢 concentration       # Sample concentration [mg/ml]
│   │   ├── 🏷️ solvent            # Solvent information  
│   │   └── 🔢 path_length         # Beam path length [mm]
│   ├── 📁 acquisition/            # Data collection parameters
│   │   ├── 🔢 frame_time          # Frame exposure time [s]
│   │   ├── 🔢 num_frames          # Total number of frames
│   │   ├── 🔢 total_time          # Total measurement time [s]
│   │   ├── 🏷️ start_time         # Measurement start timestamp
│   │   ├── 🏷️ end_time           # Measurement end timestamp
│   │   ├── 🔢 deadtime_per_frame  # Detector deadtime [s]
│   │   └── 🏷️ acquisition_mode   # Collection mode (continuous/triggered)
│   └── 📁 analysis/               # Analysis parameters
│       ├── 🏷️ analysis_type      # 'Multitau' or 'Twotime'
│       ├── 🔢 q_min               # Minimum q-value [Å⁻¹]
│       ├── 🔢 q_max               # Maximum q-value [Å⁻¹] 
│       ├── 🔢 tau_min             # Minimum correlation time [s]
│       ├── 🔢 tau_max             # Maximum correlation time [s]
│       ├── 🏷️ correlation_method # Algorithm used for correlation
│       ├── 🏷️ processing_date    # Analysis timestamp
│       └── 🏷️ software_version   # Analysis software version
└── 📁 /quality/                   # Data quality metrics
    ├── 🔢 stability_metrics       # Beam stability assessment
    ├── 🔢 signal_to_noise         # SNR for each q-bin
    ├── 🔢 count_rate_stability    # Count rate time series
    ├── 🔢 detector_uniformity     # Spatial uniformity map
    └── 🏷️ quality_flags          # Automated quality assessment
```

### 📊 Dataset Specifications

#### Core Analysis Data

| Dataset | Shape | Type | Units | Description |
|---------|--------|------|-------|-------------|
| `/exchange/g2` | `[n_q, n_tau]` | float64 | dimensionless | Multi-tau intensity correlation g₂(q,τ) |
| `/exchange/tau` | `[n_tau]` | float64 | seconds | Correlation delay times |
| `/exchange/saxs_2d` | `[det_y, det_x]` | float32 | counts | 2D scattering pattern |
| `/exchange/saxs_1d` | `[n_q]` | float32 | counts | Radially averaged intensity I(q) |
| `/exchange/q_2d` | `[det_y, det_x]` | float32 | Å⁻¹ | 2D momentum transfer magnitude map |
| `/exchange/q_1d` | `[n_q]` | float32 | Å⁻¹ | 1D momentum transfer values |

#### Two-time Correlation Data (if available)

| Dataset | Shape | Type | Units | Description |
|---------|--------|------|-------|-------------|
| `/exchange/c2` | `[n_q, n_t1, n_t2]` | float32 | dimensionless | Two-time correlation C(q,t₁,t₂) |
| `/exchange/age_times` | `[n_t1]` | float64 | seconds | Age time points t₁ |
| `/exchange/delay_times` | `[n_t2]` | float64 | seconds | Delay time points t₂ |

#### Quality and Calibration Data

| Dataset | Shape | Type | Units | Description |
|---------|--------|------|-------|-------------|
| `/exchange/intensity_mean` | `[det_y, det_x]` | float32 | counts | Time-averaged detector image |
| `/exchange/intensity_std` | `[det_y, det_x]` | float32 | counts | Intensity standard deviation |
| `/exchange/abs_cross_section` | scalar | float64 | cm⁻¹ | Absolute scattering cross-section |
| `/measurement/detector/mask` | `[det_y, det_x]` | bool | - | Bad pixel mask (True = good pixel) |

### 🔬 Scientific Metadata

#### Detector Geometry Parameters

```python
# Example detector parameters from NeXus file
detector_params = {
    'distance': 5000.0,          # mm, sample-to-detector distance
    'pixel_size_x': 0.075,       # mm, horizontal pixel size  
    'pixel_size_y': 0.075,       # mm, vertical pixel size
    'beam_center_x': 1043.2,     # pixels, horizontal beam center
    'beam_center_y': 1024.7,     # pixels, vertical beam center
    'efficiency': 0.85,          # detective quantum efficiency
    'name': 'Rigaku_4M'         # detector model identifier
}
```

#### X-ray Source Parameters

```python
# Example source parameters
source_params = {
    'energy': 7.35,              # keV, photon energy
    'wavelength': 1.687,         # Å, X-ray wavelength  
    'flux': 1.2e12,             # photons/s, incident flux
    'coherence_time': 2.5e-9,    # s, temporal coherence
    'coherence_length': 2.1e-6   # m, transverse coherence length
}
```

#### Sample Information

```python
# Example sample metadata
sample_info = {
    'name': 'silica_nanoparticles_pH7',
    'description': '50nm silica spheres in water',
    'temperature': 298.15,       # K, sample temperature
    'pressure': 1.013,          # bar, ambient pressure  
    'concentration': 0.5,       # mg/ml, particle concentration
    'solvent': 'deionized_water',
    'path_length': 1.0,         # mm, sample thickness
    'viscosity': 0.89e-3        # Pa·s, solvent viscosity
}
```

### 📈 Data Access Examples

#### Loading NeXus File

```python
from xpcs_toolkit import XpcsDataFile
import h5py

# Load complete XPCS dataset
data = XpcsDataFile('experiment_001.h5')

# Access correlation data  
g2_function = data.g2              # Shape: [n_q, n_tau]
tau_values = data.tau             # Shape: [n_tau]
q_values = data.q_1d             # Shape: [n_q]

# Access SAXS data
saxs_2d = data.saxs_2d           # Shape: [det_y, det_x]
saxs_1d = data.saxs_1d           # Shape: [n_q]

# Get experimental parameters
detector_distance = data.detector_distance  # mm
beam_energy = data.beam_energy              # keV
sample_temp = data.sample_temperature       # K

print(f"Dataset: {data.sample_name}")
print(f"Analysis type: {data.analysis_type}")
print(f"Q-range: {q_values.min():.4f} - {q_values.max():.4f} Å⁻¹")
print(f"Time range: {tau_values.min():.2e} - {tau_values.max():.2e} s")
```

#### Direct HDF5 Access

```python
import h5py
import numpy as np

# Direct access to HDF5 file for custom analysis
with h5py.File('experiment_001.h5', 'r') as f:
    # Read correlation data
    g2_data = f['/exchange/g2'][:]           # Load full array
    tau_data = f['/exchange/tau'][:]
    
    # Read detector geometry
    det_distance = f['/measurement/instrument/detector/distance'][()]
    pixel_size = f['/measurement/instrument/detector/pixel_size_x'][()]
    beam_center = (
        f['/measurement/instrument/detector/beam_center_x'][()],
        f['/measurement/instrument/detector/beam_center_y'][()]
    )
    
    # Read sample information
    sample_name = f['/measurement/sample/name'][()].decode('utf-8')
    temperature = f['/measurement/sample/temperature'][()]
    
    # Read quality metrics
    snr_data = f['/quality/signal_to_noise'][:]
    stability = f['/quality/stability_metrics'][:]
    
    print(f"Loaded {sample_name}")
    print(f"Data shape: g2={g2_data.shape}, SNR shape={snr_data.shape}")
    print(f"Detector: {det_distance}mm distance, {pixel_size}mm pixels")
```

### 🔍 Quality Assessment

#### Automated Quality Metrics

```python
# Quality validation for NeXus files
def validate_nexus_quality(filename: str) -> Dict:
    """
    Comprehensive quality assessment for NeXus XPCS files.
    
    Args:
        filename: Path to NeXus HDF5 file
        
    Returns:
        Dict with quality metrics and recommendations
    """
    quality_report = {}
    
    with h5py.File(filename, 'r') as f:
        # Check data completeness
        required_datasets = [
            '/exchange/g2', '/exchange/tau', '/exchange/saxs_2d',
            '/measurement/instrument/detector/distance',
            '/measurement/source/energy'
        ]
        
        missing_datasets = []
        for dataset in required_datasets:
            if dataset not in f:
                missing_datasets.append(dataset)
        
        quality_report['data_completeness'] = {
            'missing_datasets': missing_datasets,
            'completeness_score': 1.0 - len(missing_datasets) / len(required_datasets)
        }
        
        # Statistical quality assessment
        if '/exchange/g2' in f:
            g2_data = f['/exchange/g2'][:]
            
            # Signal-to-noise ratio
            g2_mean = np.mean(g2_data)
            g2_std = np.std(g2_data)
            snr = g2_mean / g2_std if g2_std > 0 else 0
            
            # Check for reasonable g2 values (should be >= 1 for t=0)
            g2_baseline = np.mean(g2_data[:, -10:])  # Last 10 time points
            
            quality_report['statistical_quality'] = {
                'signal_to_noise_ratio': float(snr),
                'g2_baseline': float(g2_baseline),
                'baseline_reasonable': 0.9 <= g2_baseline <= 1.1
            }
        
        # Detector quality
        if '/exchange/saxs_2d' in f:
            saxs_data = f['/exchange/saxs_2d'][:]
            
            # Check for detector artifacts
            hot_pixels = np.sum(saxs_data > np.percentile(saxs_data, 99.9))
            dead_pixels = np.sum(saxs_data == 0)
            total_pixels = saxs_data.size
            
            quality_report['detector_quality'] = {
                'hot_pixel_fraction': hot_pixels / total_pixels,
                'dead_pixel_fraction': dead_pixels / total_pixels,
                'detector_health_score': 1.0 - (hot_pixels + dead_pixels) / total_pixels
            }
        
        # Overall quality score
        scores = []
        if 'data_completeness' in quality_report:
            scores.append(quality_report['data_completeness']['completeness_score'])
        if 'statistical_quality' in quality_report:
            scores.append(1.0 if quality_report['statistical_quality']['baseline_reasonable'] else 0.5)
        if 'detector_quality' in quality_report:
            scores.append(quality_report['detector_quality']['detector_health_score'])
        
        quality_report['overall_score'] = np.mean(scores) if scores else 0.0
        
        # Recommendations
        recommendations = []
        if quality_report['overall_score'] < 0.7:
            recommendations.append("Overall data quality is below recommended threshold")
        
        if quality_report.get('statistical_quality', {}).get('signal_to_noise_ratio', 0) < 10:
            recommendations.append("Low signal-to-noise ratio - consider longer measurement time")
            
        if quality_report.get('detector_quality', {}).get('hot_pixel_fraction', 0) > 0.001:
            recommendations.append("High hot pixel fraction - check detector calibration")
        
        quality_report['recommendations'] = recommendations
    
    return quality_report

# Usage example
quality = validate_nexus_quality('experiment_001.h5')
print(f"Overall quality score: {quality['overall_score']:.3f}")
for rec in quality['recommendations']:
    print(f"⚠️  {rec}")
```

---

## Legacy HDF5 Format

The legacy HDF5 format provides backward compatibility with older XPCS analysis software and datasets collected before the NeXus standardization.

### 📁 Legacy Structure

```
legacy_data.h5                      # Legacy HDF5 file
├── 🔢 /g2                         # Correlation functions (variable layout)
├── 🔢 /tau                        # Time delays  
├── 🔢 /Iqphi                      # 2D scattering data
├── 🔢 /Iq                         # 1D radial profile
├── 🔢 /qxy                        # 2D q-map (if available)
├── 🔢 /qr                         # 1D q-values
├── 📁 /xpcs/                      # Analysis metadata (optional)
└── 📁 /setup/                     # Experimental setup (optional)
```

### 🔄 Format Conversion

The XPCS Toolkit automatically handles legacy format conversion:

```python
from xpcs_toolkit import XpcsDataFile
from xpcs_toolkit.fileIO.hdf_reader import convert_legacy_format

# Automatic conversion during loading
data = XpcsDataFile('legacy_data.h5')  # Automatically detects and converts

# Check if conversion was applied
if hasattr(data, '_format_converted'):
    print("Legacy format automatically converted")
    print(f"Original format: {data._original_format}")
    print(f"Conversion method: {data._conversion_method}")

# Manual conversion to NeXus format
convert_legacy_format(
    input_file='legacy_data.h5',
    output_file='converted_nexus.h5',
    preserve_metadata=True
)
```

### 🔧 Legacy Format Detection

```python
from xpcs_toolkit.fileIO.ftype_utils import get_ftype, isLegacyFile

# Detect file format
file_format = get_ftype('data_file.h5')
print(f"Detected format: {file_format}")

# Check if legacy format
is_legacy = isLegacyFile('data_file.h5')
if is_legacy:
    print("Legacy HDF5 format detected - will be converted automatically")

# Format-specific handling
if file_format == 'nexus':
    # Use optimized NeXus loader
    data = XpcsDataFile('data_file.h5', format='nexus')
elif file_format == 'legacy':
    # Use legacy converter
    data = XpcsDataFile('data_file.h5', format='legacy')
else:
    # Auto-detect format
    data = XpcsDataFile('data_file.h5')
```

---

## Data Structure Details

### 🧮 Correlation Function Storage

#### Multi-tau Correlation Format

```python
# Standard g2 correlation function storage
g2_shape = [n_q_bins, n_tau_points]  # Typical: [100, 200]

# Multi-tau algorithm produces logarithmically spaced time points
tau_values = np.logspace(-6, 2, 200)  # 1 μs to 100 s

# G2 values should approach 1.0 at long times for ergodic systems
g2_data[q_index, :] = correlation_function_for_q_bin

# Statistical errors (when available)
g2_errors_shape = [n_q_bins, n_tau_points]  # Same as g2 data
```

#### Two-time Correlation Format  

```python
# Two-time correlation C(q, t1, t2) storage
c2_shape = [n_q_bins, n_t1_points, n_t2_points]  # Typical: [20, 100, 100]

# Age times t1 (when correlation starts)
age_times = np.linspace(0, total_measurement_time, n_t1_points)

# Delay times t2 (correlation duration)  
delay_times = np.logspace(-6, 3, n_t2_points)  # 1 μs to 1000 s

# Two-time data access
c2_for_q = c2_data[q_index, :, :]  # Shape: [n_t1, n_t2]
```

### 🎯 Q-space Mapping

#### 2D Q-map Calculation

```python
def calculate_q_map(detector_distance: float,
                   pixel_size: float, 
                   beam_center: Tuple[float, float],
                   wavelength: float,
                   detector_shape: Tuple[int, int]) -> np.ndarray:
    """
    Calculate 2D momentum transfer map for detector geometry.
    
    Args:
        detector_distance: Sample-to-detector distance [mm]
        pixel_size: Detector pixel size [mm]  
        beam_center: Beam center coordinates [pixels]
        wavelength: X-ray wavelength [Å]
        detector_shape: Detector dimensions [pixels]
        
    Returns:
        2D array of q-magnitudes [Å⁻¹]
    """
    
    # Create pixel coordinate arrays
    ny, nx = detector_shape
    y_pixels, x_pixels = np.mgrid[0:ny, 0:nx]
    
    # Convert to physical coordinates [mm]
    x_phys = (x_pixels - beam_center[0]) * pixel_size
    y_phys = (y_pixels - beam_center[1]) * pixel_size
    
    # Calculate scattering angles
    theta = np.arctan(np.sqrt(x_phys**2 + y_phys**2) / detector_distance)
    
    # Convert to q-space [Å⁻¹]
    q_magnitude = 4 * np.pi * np.sin(theta / 2) / wavelength
    
    return q_magnitude

# Example usage
q_map = calculate_q_map(
    detector_distance=5000.0,  # mm
    pixel_size=0.075,         # mm  
    beam_center=(1043, 1025), # pixels
    wavelength=1.687,         # Å
    detector_shape=(2048, 2048)
)

print(f"Q-range: {q_map.min():.4f} - {q_map.max():.4f} Å⁻¹")
```

#### 1D Q-binning

```python
def create_q_binning(q_map: np.ndarray, 
                    n_bins: int = 100,
                    q_range: Tuple[float, float] = None) -> Dict:
    """
    Create 1D q-binning from 2D q-map.
    
    Args:
        q_map: 2D momentum transfer map
        n_bins: Number of radial bins
        q_range: Q-range limits [Å⁻¹]
        
    Returns:
        Dict with q_values, bin_edges, and pixel_indices
    """
    
    # Determine q-range
    if q_range is None:
        q_min = np.percentile(q_map[q_map > 0], 1)   # Avoid q=0 
        q_max = np.percentile(q_map, 99)             # Avoid detector edges
    else:
        q_min, q_max = q_range
    
    # Create logarithmic binning (typical for SAXS)
    q_bin_edges = np.logspace(np.log10(q_min), np.log10(q_max), n_bins + 1)
    q_bin_centers = np.sqrt(q_bin_edges[:-1] * q_bin_edges[1:])  # Geometric mean
    
    # Assign pixels to q-bins
    q_indices = np.digitize(q_map, q_bin_edges) - 1
    q_indices = np.clip(q_indices, 0, n_bins - 1)
    
    return {
        'q_values': q_bin_centers,
        'q_bin_edges': q_bin_edges,
        'q_indices': q_indices,
        'n_bins': n_bins
    }
```

---

## Format Detection

### 🔍 Automatic Detection Algorithm

```python
from xpcs_toolkit.fileIO.ftype_utils import detect_file_format

def detect_file_format(filename: str) -> Dict:
    """
    Comprehensive file format detection for XPCS data files.
    
    Args:
        filename: Path to HDF5 file
        
    Returns:
        Dict with format type, confidence, and metadata
    """
    
    detection_result = {
        'format': 'unknown',
        'confidence': 0.0,
        'features': {},
        'recommendations': []
    }
    
    try:
        with h5py.File(filename, 'r') as f:
            # Check for NeXus format signatures
            nexus_features = {
                'has_exchange_group': '/exchange' in f,
                'has_measurement_group': '/measurement' in f,
                'has_quality_group': '/quality' in f,
                'has_g2_dataset': '/exchange/g2' in f,
                'has_saxs_2d': '/exchange/saxs_2d' in f,
                'has_detector_params': '/measurement/instrument/detector' in f
            }
            
            nexus_score = sum(nexus_features.values()) / len(nexus_features)
            
            # Check for legacy format signatures  
            legacy_features = {
                'has_root_g2': '/g2' in f,
                'has_root_tau': '/tau' in f, 
                'has_iqphi': '/Iqphi' in f,
                'has_iq': '/Iq' in f,
                'has_qr': '/qr' in f
            }
            
            legacy_score = sum(legacy_features.values()) / len(legacy_features)
            
            # Format decision
            if nexus_score > 0.7:
                detection_result['format'] = 'nexus'
                detection_result['confidence'] = nexus_score
                detection_result['features'] = nexus_features
                
                if nexus_score < 1.0:
                    detection_result['recommendations'].append(
                        "Incomplete NeXus format - some standard groups missing"
                    )
                    
            elif legacy_score > 0.6:
                detection_result['format'] = 'legacy'  
                detection_result['confidence'] = legacy_score
                detection_result['features'] = legacy_features
                detection_result['recommendations'].append(
                    "Legacy format detected - consider converting to NeXus"
                )
                
            else:
                detection_result['format'] = 'custom'
                detection_result['confidence'] = max(nexus_score, legacy_score)
                detection_result['recommendations'].append(
                    "Non-standard format - manual inspection required"
                )
                
            # Additional metadata extraction
            detection_result['file_size'] = os.path.getsize(filename)
            
            # Check for analysis type indicators
            if '/exchange/g2' in f:
                g2_shape = f['/exchange/g2'].shape
                if len(g2_shape) == 2:
                    detection_result['analysis_type'] = 'multitau'
                elif len(g2_shape) == 3:
                    detection_result['analysis_type'] = 'twotime'
                    
    except Exception as e:
        detection_result['error'] = str(e)
        detection_result['recommendations'].append(
            f"File access error: {e}"
        )
    
    return detection_result

# Usage example
format_info = detect_file_format('experiment.h5')
print(f"Detected format: {format_info['format']} (confidence: {format_info['confidence']:.2f})")

for rec in format_info['recommendations']:
    print(f"💡 {rec}")
```

---

## Conversion Utilities

### 🔄 Legacy to NeXus Conversion

```python
from xpcs_toolkit.fileIO.format_converter import LegacyToNexusConverter

def convert_legacy_to_nexus(input_file: str, output_file: str, **options) -> Dict:
    """
    Convert legacy HDF5 format to standardized NeXus format.
    
    Args:
        input_file: Path to legacy HDF5 file
        output_file: Path for converted NeXus file  
        **options: Conversion options
        
    Returns:
        Dict with conversion status and metadata
    """
    
    converter = LegacyToNexusConverter()
    
    conversion_options = {
        'preserve_metadata': options.get('preserve_metadata', True),
        'add_quality_metrics': options.get('add_quality_metrics', True),
        'optimize_storage': options.get('optimize_storage', True),
        'validate_output': options.get('validate_output', True)
    }
    
    try:
        # Load legacy data
        legacy_data = converter.load_legacy_file(input_file)
        
        # Map to NeXus structure
        nexus_structure = converter.map_to_nexus(legacy_data)
        
        # Add missing metadata with reasonable defaults
        nexus_structure = converter.add_default_metadata(
            nexus_structure, 
            source_file=input_file
        )
        
        # Calculate quality metrics
        if conversion_options['add_quality_metrics']:
            quality_metrics = converter.calculate_quality_metrics(nexus_structure)
            nexus_structure['/quality'] = quality_metrics
        
        # Write NeXus file
        converter.write_nexus_file(output_file, nexus_structure)
        
        # Validate output
        if conversion_options['validate_output']:
            validation = converter.validate_nexus_output(output_file)
            if not validation['valid']:
                raise ValueError(f"Output validation failed: {validation['errors']}")
        
        return {
            'status': 'success',
            'input_format': 'legacy',
            'output_format': 'nexus',
            'data_preserved': True,
            'metadata_enhanced': True,
            'file_size_original': os.path.getsize(input_file),
            'file_size_converted': os.path.getsize(output_file)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e),
            'recommendations': converter.get_error_recommendations(e)
        }

# Batch conversion example
def batch_convert_directory(input_dir: str, output_dir: str):
    """Convert all legacy files in directory to NeXus format."""
    
    from pathlib import Path
    from tqdm import tqdm
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all HDF5 files
    h5_files = list(input_path.glob('*.h5')) + list(input_path.glob('*.hdf'))
    
    conversion_results = []
    
    for input_file in tqdm(h5_files, desc="Converting files"):
        # Check if legacy format
        if isLegacyFile(str(input_file)):
            output_file = output_path / f"{input_file.stem}_nexus.h5"
            
            result = convert_legacy_to_nexus(
                str(input_file), 
                str(output_file),
                preserve_metadata=True,
                add_quality_metrics=True
            )
            
            result['input_file'] = str(input_file)
            result['output_file'] = str(output_file)
            conversion_results.append(result)
    
    # Summary report
    successful = sum(1 for r in conversion_results if r['status'] == 'success')
    failed = len(conversion_results) - successful
    
    print(f"\nConversion Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    # Report failures
    for result in conversion_results:
        if result['status'] == 'error':
            print(f"  ❌ {result['input_file']}: {result['error_message']}")
    
    return conversion_results
```

---

## Quality Validation

### ✅ Comprehensive Quality Checks

```python
from xpcs_toolkit.fileIO.quality_validator import XpcsFileValidator

class XpcsFileValidator:
    """Comprehensive XPCS file format and quality validator."""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
    
    def validate_file(self, filename: str, 
                     strict_mode: bool = False) -> Dict:
        """
        Complete file validation including format, data quality, and metadata.
        
        Args:
            filename: Path to XPCS file
            strict_mode: Enable strict validation rules
            
        Returns:
            Dict with comprehensive validation results
        """
        
        validation_result = {
            'filename': filename,
            'validation_time': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'checks': {},
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:
            # Format structure validation
            format_check = self._validate_format_structure(filename)
            validation_result['checks']['format'] = format_check
            
            # Data integrity validation
            integrity_check = self._validate_data_integrity(filename)
            validation_result['checks']['integrity'] = integrity_check
            
            # Scientific validity validation
            science_check = self._validate_scientific_validity(filename)
            validation_result['checks']['science'] = science_check
            
            # Metadata completeness validation
            metadata_check = self._validate_metadata_completeness(filename)
            validation_result['checks']['metadata'] = metadata_check
            
            # Performance optimization validation
            performance_check = self._validate_performance_optimization(filename)
            validation_result['checks']['performance'] = performance_check
            
            # Calculate overall status
            validation_result['overall_status'] = self._calculate_overall_status(
                validation_result['checks'], 
                strict_mode
            )
            
            # Generate recommendations
            validation_result['recommendations'] = self._generate_recommendations(
                validation_result['checks']
            )
            
        except Exception as e:
            validation_result['overall_status'] = 'error'
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _validate_format_structure(self, filename: str) -> Dict:
        """Validate HDF5/NeXus format structure."""
        
        format_result = {
            'status': 'unknown',
            'format_type': 'unknown',
            'required_groups': [],
            'missing_groups': [],
            'extra_groups': [],
            'hdf5_valid': False
        }
        
        try:
            with h5py.File(filename, 'r') as f:
                format_result['hdf5_valid'] = True
                
                # Detect format type
                if '/exchange' in f and '/measurement' in f:
                    format_result['format_type'] = 'nexus'
                    required_groups = [
                        '/exchange', '/measurement', 
                        '/measurement/instrument/detector',
                        '/measurement/sample'
                    ]
                elif '/g2' in f and '/tau' in f:
                    format_result['format_type'] = 'legacy'
                    required_groups = ['/g2', '/tau', '/Iq']
                else:
                    format_result['format_type'] = 'custom'
                    required_groups = []
                
                # Check required groups
                format_result['required_groups'] = required_groups
                format_result['missing_groups'] = [
                    group for group in required_groups if group not in f
                ]
                
                # Overall format status
                if len(format_result['missing_groups']) == 0:
                    format_result['status'] = 'valid'
                elif len(format_result['missing_groups']) < len(required_groups) / 2:
                    format_result['status'] = 'partial'
                else:
                    format_result['status'] = 'invalid'
                    
        except OSError:
            format_result['hdf5_valid'] = False
            format_result['status'] = 'invalid'
        
        return format_result
    
    def _validate_data_integrity(self, filename: str) -> Dict:
        """Validate data array integrity and consistency."""
        
        integrity_result = {
            'status': 'unknown',
            'arrays_valid': {},
            'shape_consistency': True,
            'value_ranges': {},
            'nan_inf_check': {}
        }
        
        with h5py.File(filename, 'r') as f:
            # Check common data arrays
            test_arrays = {
                'g2': ['/exchange/g2', '/g2'],
                'tau': ['/exchange/tau', '/tau'], 
                'saxs_2d': ['/exchange/saxs_2d', '/Iqphi'],
                'q_values': ['/exchange/q_1d', '/qr']
            }
            
            for array_name, possible_paths in test_arrays.items():
                for path in possible_paths:
                    if path in f:
                        dataset = f[path]
                        
                        # Basic validity checks
                        integrity_result['arrays_valid'][array_name] = {
                            'exists': True,
                            'readable': True,
                            'shape': dataset.shape,
                            'dtype': str(dataset.dtype)
                        }
                        
                        # Load small sample for value checks
                        if dataset.size > 1e6:  # Large array - sample
                            sample_data = dataset[::max(1, dataset.size // 1000)]
                        else:
                            sample_data = dataset[:]
                        
                        # Check for NaN/Inf values
                        if np.issubdtype(sample_data.dtype, np.floating):
                            nan_count = np.sum(np.isnan(sample_data))
                            inf_count = np.sum(np.isinf(sample_data))
                            
                            integrity_result['nan_inf_check'][array_name] = {
                                'nan_count': int(nan_count),
                                'inf_count': int(inf_count),
                                'total_checked': len(sample_data)
                            }
                        
                        # Value range checks
                        integrity_result['value_ranges'][array_name] = {
                            'min': float(np.min(sample_data)),
                            'max': float(np.max(sample_data)),
                            'mean': float(np.mean(sample_data))
                        }
                        
                        break
            
            # Overall integrity status
            has_critical_data = 'g2' in integrity_result['arrays_valid']
            has_nan_inf = any(
                check.get('nan_count', 0) + check.get('inf_count', 0) > 0 
                for check in integrity_result['nan_inf_check'].values()
            )
            
            if has_critical_data and not has_nan_inf:
                integrity_result['status'] = 'valid'
            elif has_critical_data:
                integrity_result['status'] = 'warning'
            else:
                integrity_result['status'] = 'invalid'
        
        return integrity_result

# Usage examples
validator = XpcsFileValidator()

# Single file validation
result = validator.validate_file('experiment.h5', strict_mode=True)
print(f"Validation status: {result['overall_status']}")

for warning in result['warnings']:
    print(f"⚠️  {warning}")

for error in result['errors']:
    print(f"❌ {error}")

for rec in result['recommendations']:
    print(f"💡 {rec}")
```

---

## Best Practices

### ✅ File Format Best Practices

#### 🎯 **Data Production Guidelines**

1. **Use NeXus Format**: Always generate data in standardized NeXus format
2. **Complete Metadata**: Include all experimental parameters and calibration data
3. **Quality Metrics**: Embed automated quality assessment in data files
4. **Compression**: Use appropriate HDF5 compression for large datasets
5. **Chunking**: Optimize chunk sizes for typical access patterns

#### 🔧 **Data Analysis Guidelines**

```python
# Best practice data loading
def load_xpcs_data_safely(filename: str) -> XpcsDataFile:
    """Load XPCS data with comprehensive error handling and validation."""
    
    try:
        # Validate file format first
        format_info = detect_file_format(filename)
        
        if format_info['confidence'] < 0.7:
            warnings.warn(f"Low confidence format detection: {format_info['confidence']:.2f}")
        
        # Load data with appropriate method
        if format_info['format'] == 'nexus':
            data = XpcsDataFile(filename, format='nexus')
        elif format_info['format'] == 'legacy':
            warnings.warn("Loading legacy format - consider converting to NeXus")
            data = XpcsDataFile(filename, format='legacy')
        else:
            # Try auto-detection
            data = XpcsDataFile(filename)
        
        # Validate data quality
        quality = data.validate_data_quality()
        
        if quality['overall_score'] < 0.5:
            warnings.warn(f"Low data quality score: {quality['overall_score']:.2f}")
            
        # Log successful loading
        logging.info(f"Loaded {filename}: {data.analysis_type} analysis, "
                    f"Q-range {data.q_1d.min():.3f}-{data.q_1d.max():.3f} Å⁻¹")
        
        return data
        
    except Exception as e:
        logging.error(f"Failed to load {filename}: {str(e)}")
        raise

# Best practice batch processing
def process_xpcs_files(file_list: List[str]) -> Dict:
    """Process multiple XPCS files with robust error handling."""
    
    results = {
        'successful': [],
        'failed': [],
        'warnings': [],
        'summary': {}
    }
    
    for filename in file_list:
        try:
            # Load and validate
            data = load_xpcs_data_safely(filename)
            
            # Quick analysis
            if data.analysis_type == 'Multitau':
                correlation = data.get_correlation_data()
                analysis_result = {
                    'filename': filename,
                    'type': 'multitau',
                    'q_range': (data.q_1d.min(), data.q_1d.max()),
                    'tau_range': (data.tau.min(), data.tau.max()),
                    'data_points': data.g2.size
                }
            else:
                analysis_result = {
                    'filename': filename,
                    'type': data.analysis_type.lower(),
                    'status': 'loaded'
                }
            
            results['successful'].append(analysis_result)
            
        except Exception as e:
            results['failed'].append({
                'filename': filename,
                'error': str(e)
            })
    
    # Generate summary
    results['summary'] = {
        'total_files': len(file_list),
        'successful': len(results['successful']),
        'failed': len(results['failed']),
        'success_rate': len(results['successful']) / len(file_list) * 100
    }
    
    return results
```

#### 💾 **Storage Optimization**

```python
# Optimize HDF5 storage for XPCS data
def create_optimized_nexus_file(output_file: str, 
                               data_arrays: Dict, 
                               metadata: Dict) -> None:
    """
    Create NeXus file with optimal storage settings for XPCS data.
    
    Args:
        output_file: Output filename
        data_arrays: Dict with array data
        metadata: Dict with experimental metadata
    """
    
    with h5py.File(output_file, 'w') as f:
        # Create groups
        exchange_group = f.create_group('exchange')
        measurement_group = f.create_group('measurement')
        quality_group = f.create_group('quality')
        
        # Optimized storage for different data types
        storage_configs = {
            'g2': {
                'compression': 'gzip',
                'compression_opts': 6,
                'chunks': True,  # Auto-chunking
                'fletcher32': True  # Checksum
            },
            'saxs_2d': {
                'compression': 'lzf',  # Fast compression for large arrays
                'chunks': True,
                'shuffle': True  # Improve compression
            },
            'tau': {
                'compression': 'gzip',
                'compression_opts': 9  # High compression for small arrays
            }
        }
        
        # Store data arrays with optimal settings
        for array_name, array_data in data_arrays.items():
            if array_name in storage_configs:
                config = storage_configs[array_name]
            else:
                # Default configuration
                config = {'compression': 'gzip', 'compression_opts': 6}
            
            # Determine appropriate group
            if array_name in ['g2', 'tau', 'saxs_2d', 'saxs_1d', 'q_2d', 'q_1d']:
                group = exchange_group
            elif array_name.startswith('quality_'):
                group = quality_group
            else:
                group = f  # Root level
            
            # Create dataset with optimization
            dataset = group.create_dataset(
                array_name.replace('quality_', ''),
                data=array_data,
                **config
            )
            
            # Add units and descriptions
            if array_name == 'g2':
                dataset.attrs['units'] = 'dimensionless'
                dataset.attrs['description'] = 'Multi-tau intensity correlation function'
            elif array_name == 'tau':
                dataset.attrs['units'] = 's'
                dataset.attrs['description'] = 'Correlation delay times'
            elif array_name in ['q_2d', 'q_1d']:
                dataset.attrs['units'] = 'Å⁻¹'
                dataset.attrs['description'] = 'Momentum transfer magnitude'
        
        # Store metadata efficiently
        for group_path, group_metadata in metadata.items():
            if group_path.startswith('/'):
                group_path = group_path[1:]  # Remove leading slash
            
            # Navigate to or create group
            current_group = f
            for subgroup in group_path.split('/'):
                if subgroup not in current_group:
                    current_group = current_group.create_group(subgroup)
                else:
                    current_group = current_group[subgroup]
            
            # Store metadata as attributes
            for key, value in group_metadata.items():
                if isinstance(value, str):
                    current_group.attrs[key] = value.encode('utf-8')
                elif isinstance(value, (int, float, bool)):
                    current_group.attrs[key] = value
                elif isinstance(value, np.ndarray) and value.size < 100:
                    current_group.attrs[key] = value
                else:
                    # Store large arrays as datasets
                    current_group.create_dataset(key, data=value)
        
        # File-level attributes
        f.attrs['format'] = 'NeXus_XPCS_v1.0'
        f.attrs['created'] = datetime.now().isoformat()
        f.attrs['creator'] = 'XPCS Toolkit'
        
    print(f"Optimized NeXus file created: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1e6:.1f} MB")
```

---

**For additional file format documentation:**
📚 https://github.com/imewei/xpcs-toolkit/tree/main/docs/file_formats

**NeXus Standard Reference:**
🌐 https://www.nexusformat.org/