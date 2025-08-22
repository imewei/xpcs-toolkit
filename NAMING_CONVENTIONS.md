# XPCS Toolkit - Naming Conventions & Refactoring Plan

## Overview

This document outlines the naming conventions and refactoring plan to improve code readability throughout the XPCS Toolkit.

## Core Naming Principles

### 1. Use Descriptive Names
- **Before**: `xf`, `fl`, `vk` 
- **After**: `xpcs_file`, `file_locator`, `kernel`

### 2. Keep Scientific Terms, Improve Generic Abbreviations  
- **Before**: `g2`, `saxs`, `Iq`, `t_el`
- **After**: `g2`, `saxs`, `intensity_q`, `time_elapsed`

### 3. Use Consistent Function Names
- **Before**: `plot_saxs2d()`, `get_g2_data()`, `load_data()`
- **After**: `plot_saxs_2d()`, `get_g2_data()`, `load_dataset()`

### 4. Clear Parameter Names
- **Before**: `qrange`, `trange`, `args`
- **After**: `q_range`, `time_range`, `arguments`

### 5. Descriptive Class Names
- **Before**: `XpcsFile` (unclear purpose)
- **After**: `XpcsDataFile`

## Specific Refactoring Mappings

### Core Classes
```python
# Old → New
XpcsFile → XpcsDataFile
ViewerKernel → AnalysisKernel  
FileLocator → DataFileLocator
```

### Common Variables
```python
# Scientific/Technical Terms (Keep Established Terms)
g2 → g2
saxs → saxs  
Iq → intensity_q
qmap → q_space_map
dqmap → dynamic_q_map
sqmap → static_q_map
t_el → time_elapsed
tau → tau
phi → phi

# Generic Abbreviations  
xf → xpcs_file
fl → file_locator
vk → kernel / analysis_kernel
args → arguments
fname → filename  # ✅ COMPLETED
ftype → file_type  # ✅ COMPLETED
fstr → filter_string
qrange → q_range  # ✅ COMPLETED
qindex → q_index  # ✅ COMPLETED
```

### Function Names
```python
# Analysis Functions
plot_saxs2d() → plot_saxs_2d()
plot_saxs1d() → plot_saxs_1d() 
plot_g2() → plot_g2_function()
get_g2_data() → get_g2_data()
get_saxs1d_data() → get_saxs_1d_data()
fit_g2() → fit_g2_function()
fit_tauq() → fit_tau_vs_q()

# File Operations
load_data() → load_dataset()
get_hdf_info() → get_hdf_metadata()
create_id() → create_identifier()

# Utility Functions  
setup_logging() → configure_logging()
list_files() → list_data_files()
```

### Module-Specific Conventions

#### CLI Module (`cli_headless.py`)
```python
# Function Names
plot_saxs2d() → plot_saxs_2d()
plot_g2() → plot_g2_function()
plot_stability() → analyze_stability()
list_files() → list_data_files()
create_parser() → create_argument_parser()

# Variable Names  
saxs2d_parser → saxs_2d_parser
g2_parser → g2_parser
outfile → output_filename
dpi → dots_per_inch
max_files → maximum_file_count
```

#### Analysis Modules (`module/`)
```python
# Keep existing module names but improve internal naming:
# g2mod.py (keep name, improve internal variables)
# saxs1d.py (keep name, improve internal variables)  
# saxs2d.py (keep name, improve internal variables)
# twotime.py (keep name, improve internal variables)
# tauq.py (keep name, improve internal variables)
```

#### File I/O (`fileIO/`)
```python
# Functions
get() → get_hdf_data()
get_analysis_type() → get_analysis_type_from_file()
read_metadata_to_dict() → read_hdf_metadata()

# Variables in functions
ftype → file_type
fields → data_fields  
ret → result_data
```

### Constants and Configuration
```python
# Physical Constants
X_energy → xray_energy
det_dist → detector_distance  
pixel_size → detector_pixel_size
bcx, bcy → beam_center_x, beam_center_y

# Analysis Parameters
qmin, qmax → q_minimum, q_maximum
tmin, tmax → time_minimum, time_maximum
```

## Implementation Strategy

### Phase 1: Core Classes (Priority 1)
1. Refactor `XpcsFile` class with improved variable names
2. Update `ViewerKernel` to `AnalysisKernel` with better method names
3. Improve `FileLocator` to `DataFileLocator`

### Phase 2: CLI Interface (Priority 1) 
1. Refactor `cli_headless.py` with descriptive function and variable names
2. Update command-line argument names for clarity
3. Improve error messages and logging

### Phase 3: Analysis Modules (Priority 2)
1. Refactor each module in `module/` directory
2. Update scientific variable names to be more descriptive
3. Improve function documentation

### Phase 4: File I/O (Priority 2)
1. Refactor `fileIO/` modules with consistent naming
2. Update function signatures for clarity
3. Improve error handling messages

### Phase 5: Helper Modules (Priority 3)
1. Refactor `helper/` modules 
2. Fix naming inconsistencies
3. Update utility functions

### Phase 6: Tests & Documentation (Priority 3)
1. Update test files to match new names
2. Update all docstrings
3. Create migration guide for users

## Backwards Compatibility

### Deprecated Names
- Keep old method names as deprecated aliases
- Add deprecation warnings for old names
- Provide clear migration path

### Example:
```python
class XpcsDataFile:
    def get_correlation_data(self, q_range=None):
        """New descriptive method name"""
        pass
    
    def get_g2_data(self, qrange=None):  
        """Deprecated: Use get_correlation_data() instead"""
        warnings.warn("get_g2_data is deprecated, use get_correlation_data", 
                     DeprecationWarning)
        return self.get_correlation_data(q_range=qrange)
```

## Code Quality Improvements

### 1. Type Hints
Add type hints for better code clarity:
```python
def get_correlation_data(self, 
                        q_range: Optional[Tuple[float, float]] = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
```

### 2. Documentation
Improve docstrings with clear parameter descriptions:
```python
def plot_scattering_2d(self, arguments: argparse.Namespace) -> int:
    """Plot 2D scattering patterns from XPCS data.
    
    Args:
        arguments: Command line arguments containing:
            - path: Directory containing HDF files
            - output_filename: Output file for the plot
            - log_scale: Whether to use logarithmic scaling
            - maximum_file_count: Maximum number of files to process
            
    Returns:
        Exit code (0 for success, 1 for error)
    """
```

### 3. Constants
Define clear constants:
```python
class XpcsConstants:
    DEFAULT_DPI = 150
    DEFAULT_FIGURE_WIDTH = 10
    DEFAULT_FIGURE_HEIGHT = 8
    LOG_SCALE_THRESHOLD = 1e-6
    MAX_CORRELATION_POINTS = 32678
```

## Benefits

1. **Improved Readability**: Code is self-documenting
2. **Better Maintainability**: Easier to understand and modify  
3. **Reduced Learning Curve**: New developers can understand code faster
4. **Fewer Bugs**: Clear names reduce confusion and errors
5. **Professional Quality**: Industry-standard naming conventions

## Timeline

- **Phase 1-2**: 2-3 days (Core functionality)
- **Phase 3-4**: 3-4 days (Analysis modules)  
- **Phase 5-6**: 2-3 days (Helpers, tests, docs)
- **Total**: ~1-2 weeks for complete refactoring

This refactoring will significantly improve the codebase quality while maintaining full backwards compatibility.
