# XPCS Toolkit Migration Guide

## Overview

This guide helps users migrate from the old XPCS Toolkit API to the new, refactored API. The new API provides better readability, consistency, and maintainability while preserving full backward compatibility.

## 🔄 Migration Strategy

**Good News**: Your existing code will continue to work unchanged! The refactoring maintains full backward compatibility with deprecation warnings to guide you toward the new API.

## 📋 What's Changed

### 1. Core Class Names

| Old Name | New Name | Status | Migration Required |
|----------|----------|--------|-------------------|
| `XpcsFile` | `XpcsDataFile` | ⚠️ Deprecated | Optional - both work |
| `ViewerKernel` | `AnalysisKernel` | ⚠️ Deprecated | Optional - both work |
| `FileLocator` | `DataFileLocator` | ⚠️ Deprecated | Optional - both work |

### 2. Parameter Names

| Old Parameter | New Parameter | Affected Functions | Status |
|--------------|---------------|-------------------|--------|
| `fname` | `filename` | `get_abs_cs_scale()`, `get_analysis_type()`, `get()` | ⚠️ Deprecated |
| `ftype` | `file_type` | `get_abs_cs_scale()`, `get_analysis_type()`, `get()` | ⚠️ Deprecated |
| `qrange` | `q_range` | `get_qbin_in_qrange()`, `get_g2_data()`, `get_saxs_1d_data()` | ⚠️ Deprecated |
| `qindex` | `q_index` | Internal variable in `qmap_utils.py` | ✅ Updated |

### 3. CLI Function Names

| Old Function | New Function | Status |
|-------------|-------------|--------|
| `plot_saxs2d()` | `plot_saxs_2d()` | ✅ Updated |
| `plot_g2()` | `plot_g2_function()` | ✅ Updated |

## 🚨 Deprecation Schedule

### Immediate (Current Release)
- All old names work with deprecation warnings
- New names are fully functional
- No breaking changes

### Future Releases (6-12 months)
- Deprecation warnings will become more prominent
- Documentation will primarily show new names
- Old names continue to work

### Long Term (12+ months)
- Consider removing deprecated names in major version bump
- Full migration path will be provided before any removal

## 💻 Migration Examples

### Example 1: Basic Class Usage

**Old Code (still works):**
```python
from xpcs_toolkit import XpcsFile, ViewerKernel, FileLocator

# This will show deprecation warnings but works fine
xf = XpcsFile('/path/to/file.hdf')
vk = ViewerKernel('/path/to/directory')
fl = FileLocator('/path/to/directory')
```

**New Code (recommended):**
```python
from xpcs_toolkit import XpcsDataFile, AnalysisKernel, DataFileLocator

# New names - no warnings
xf = XpcsDataFile('/path/to/file.hdf')
ak = AnalysisKernel('/path/to/directory')
dfl = DataFileLocator('/path/to/directory')
```

**Mixed Approach (pragmatic):**
```python
# Import both old and new for gradual migration
from xpcs_toolkit import XpcsFile, XpcsDataFile, AnalysisKernel

# Use new classes for new code
ak = AnalysisKernel('/path/to/directory')

# Keep old class names in existing code (until convenient to change)
xf = XpcsFile('/path/to/file.hdf')  # Will show warning
```

### Example 2: Parameter Name Changes

**Old Code (still works):**
```python
from xpcs_toolkit.fileIO.hdf_reader import get_abs_cs_scale, get_analysis_type

# Old parameter names - will show deprecation warnings
scale = get_abs_cs_scale(fname='data.hdf', ftype='nexus')
analysis_type = get_analysis_type(fname='data.hdf', ftype='nexus')
```

**New Code (recommended):**
```python
from xpcs_toolkit.fileIO.hdf_reader import get_abs_cs_scale, get_analysis_type

# New parameter names - no warnings
scale = get_abs_cs_scale(filename='data.hdf', file_type='nexus')
analysis_type = get_analysis_type(filename='data.hdf', file_type='nexus')
```

### Example 3: Q-range Parameters

**Old Code (still works):**
```python
# Old parameter name
q_values, time_elapsed, g2, g2_error, labels = xpcs_file.get_g2_data(qrange=(0.01, 0.1))
qbins, qvals = qmap.get_qbin_in_qrange(qrange=(0.01, 0.1))
```

**New Code (recommended):**
```python
# New parameter name
q_values, time_elapsed, g2, g2_error, labels = xpcs_file.get_g2_data(q_range=(0.01, 0.1))
qbins, qvals = qmap.get_qbin_in_qrange(q_range=(0.01, 0.1))
```

## 📝 Step-by-Step Migration

### Step 1: Update Imports (Optional)
Replace old class names with new ones in your imports:
```python
# Before
from xpcs_toolkit import XpcsFile, ViewerKernel, FileLocator

# After  
from xpcs_toolkit import XpcsDataFile, AnalysisKernel, DataFileLocator
```

### Step 2: Update Class Instantiation (Optional)
```python
# Before
xf = XpcsFile('/path/to/file.hdf')

# After
xf = XpcsDataFile('/path/to/file.hdf')
```

### Step 3: Update Parameter Names (Recommended)
```python
# Before
result = get_abs_cs_scale(fname='file.hdf', ftype='nexus')

# After
result = get_abs_cs_scale(filename='file.hdf', file_type='nexus')
```

### Step 4: Test Your Code
Run your code and check for deprecation warnings:
```python
import warnings
warnings.simplefilter('always')  # Show all warnings

# Your code here - check console for deprecation warnings
```

## 🔧 Automated Migration Tools

### Option 1: Simple Search and Replace
For quick migration, you can use find and replace in your IDE:

- `fname=` → `filename=`
- `ftype=` → `file_type=`
- `qrange=` → `q_range=`
- `XpcsFile` → `XpcsDataFile`
- `ViewerKernel` → `AnalysisKernel`
- `FileLocator` → `DataFileLocator`

### Option 2: Python Script for Batch Migration
```python
import re
import os
import glob

def migrate_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parameter name replacements
    content = re.sub(r'fname\s*=', 'filename=', content)
    content = re.sub(r'ftype\s*=', 'file_type=', content)  
    content = re.sub(r'qrange\s*=', 'q_range=', content)
    
    # Class name replacements
    content = re.sub(r'\bXpcsFile\b', 'XpcsDataFile', content)
    content = re.sub(r'\bViewerKernel\b', 'AnalysisKernel', content)
    content = re.sub(r'\bFileLocator\b', 'DataFileLocator', content)
    
    with open(filepath, 'w') as f:
        f.write(content)

# Migrate all Python files in your project
for filepath in glob.glob('**/*.py', recursive=True):
    migrate_file(filepath)
```

## ⚠️ Important Notes

### 1. No Breaking Changes
- All existing code will continue to work
- Deprecation warnings help guide migration
- Migration is completely optional

### 2. Gradual Migration Recommended
- Migrate new code to use new names immediately
- Update existing code gradually over time
- No rush - old names will be supported long-term

### 3. Import Aliases Still Work
```python
# These all work fine:
from xpcs_toolkit import XpcsDataFile as XF
from xpcs_toolkit import XpcsFile as XF  # deprecated but works
from xpcs_toolkit.xpcs_file import XpcsDataFile
```

### 4. Mixed Usage is Fine
```python
# You can mix old and new names during transition
old_style = XpcsFile('/path/file1.hdf')
new_style = XpcsDataFile('/path/file2.hdf')
```

## 🐛 Troubleshooting

### Issue: Too Many Deprecation Warnings
```python
# Temporarily suppress warnings during migration
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
```

### Issue: Import Errors
```python
# If imports fail, make sure you have the latest version
pip install --upgrade xpcs-toolkit
```

### Issue: Unclear Migration Path
Check the deprecation warning messages - they provide specific guidance:
```
DeprecationWarning: Parameter 'fname' is deprecated, use 'filename' instead
```

## 📞 Getting Help

### 1. Check the Logs
Deprecation warnings provide specific migration instructions.

### 2. Consult Documentation
- `NAMING_CONVENTIONS.md` - Complete naming standards
- `REFACTORING_PROGRESS.md` - Detailed progress report
- This `MIGRATION_GUIDE.md` - Migration instructions

### 3. Test Before and After
```python
# Test that your code works before migration
pytest tests/

# Make migration changes

# Test that everything still works after migration  
pytest tests/
```

## ✅ Benefits of Migration

### Immediate Benefits
- No deprecation warnings in your logs
- Code follows current best practices
- Better code readability

### Long-term Benefits  
- Future-proof your code
- Easier maintenance and debugging
- Improved team collaboration
- Better IDE support and autocomplete

## 📊 Migration Checklist

- [ ] **Read this migration guide completely**
- [ ] **Test existing code to ensure it works**
- [ ] **Update imports to new class names** (optional)
- [ ] **Update parameter names** (recommended)
- [ ] **Test migrated code thoroughly**
- [ ] **Update documentation and comments**
- [ ] **Train team members on new names**

## 🎯 Summary

The XPCS Toolkit refactoring provides:
- ✅ **Full backward compatibility** - your code keeps working
- ✅ **Improved readability** - clearer, more descriptive names
- ✅ **Better maintainability** - consistent naming conventions  
- ✅ **Gradual migration path** - change at your own pace
- ✅ **No breaking changes** - risk-free upgrade

**Recommendation**: Start using new names for new code, and migrate existing code gradually as convenient. There's no rush - the old API will be supported for the foreseeable future.
