# XPCS Toolkit Tutorials

This directory contains comprehensive Jupyter notebook tutorials for learning the XPCS Toolkit. The tutorials are designed to take you from basic concepts to advanced analysis techniques used in X-ray Photon Correlation Spectroscopy (XPCS) and Small-Angle X-ray Scattering (SAXS).

## Tutorial Overview

### 📚 Tutorial Series

1. **[01_getting_started.ipynb](01_getting_started.ipynb)** 
   - Introduction to XPCS Toolkit
   - Basic data loading and visualization
   - Understanding file formats
   - Command-line interface overview
   - *Prerequisites: None*
   - *Duration: 30-45 minutes*

2. **[02_correlation_analysis.ipynb](02_correlation_analysis.ipynb)**
   - Advanced XPCS correlation function analysis  
   - Multiple fitting models (single, stretched, double exponential)
   - Q-dependent dynamics analysis
   - Statistical assessment and model selection
   - *Prerequisites: Tutorial 01*
   - *Duration: 60-90 minutes*

3. **[03_saxs_analysis.ipynb](03_saxs_analysis.ipynb)**
   - Comprehensive SAXS structure analysis
   - Guinier analysis for particle sizes
   - Porod analysis for surface characterization
   - Structure factor analysis for interactions
   - 2D SAXS pattern analysis
   - *Prerequisites: Tutorial 01*
   - *Duration: 90-120 minutes*

### 🎯 Learning Path

**For XPCS Analysis:**
```
01_getting_started → 02_correlation_analysis
```

**For SAXS Analysis:**
```
01_getting_started → 03_saxs_analysis
```

**Complete Course:**
```
01_getting_started → 02_correlation_analysis → 03_saxs_analysis
```

## Prerequisites

### Software Requirements
- Python 3.8+
- Jupyter Notebook or JupyterLab
- XPCS Toolkit (installed)

### Required Python Packages
```bash
pip install numpy scipy matplotlib pandas
pip install jupyter ipython
```

### Recommended Setup
```bash
# Create dedicated environment
conda create -n xpcs-tutorials python=3.9
conda activate xpcs-tutorials

# Install requirements
pip install -e /path/to/xpcs_toolkit
pip install jupyter numpy scipy matplotlib pandas

# Launch Jupyter
jupyter notebook
```

## Quick Start

1. **Clone or download** the XPCS Toolkit repository
2. **Navigate** to the tutorials directory:
   ```bash
   cd xpcs_toolkit/tutorials/
   ```
3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
4. **Open** `01_getting_started.ipynb` to begin

## Tutorial Contents

### 01_getting_started.ipynb
- **Data Structures**: Understanding XPCS/SAXS file formats
- **Basic Loading**: Import data with `XpcsDataFile`
- **Visualization**: Create basic plots with matplotlib
- **Mock Data**: Generate synthetic data for practice
- **CLI Overview**: Command-line tools and batch processing
- **Navigation**: Toolkit architecture and module organization

**Key Learning Outcomes:**
- Load and inspect XPCS/SAXS data files
- Create basic visualizations
- Understand the toolkit's modular structure
- Use help and documentation effectively

### 02_correlation_analysis.ipynb  
- **Correlation Theory**: Mathematical foundations of g₂(q,τ)
- **Fitting Models**: Single, stretched, and double exponentials
- **Model Selection**: Statistical criteria and goodness-of-fit
- **Q-dependence**: Extract diffusion coefficients
- **Advanced Topics**: Non-ergodic systems, aging behavior
- **Parameter Extraction**: Physical quantities from fit parameters

**Key Learning Outcomes:**
- Fit multiple correlation function models
- Extract diffusion coefficients and particle sizes
- Assess data quality and statistical significance
- Handle complex relaxation behaviors

### 03_saxs_analysis.ipynb
- **SAXS Theory**: Form factors and structure factors
- **Guinier Analysis**: Radius of gyration determination
- **Porod Analysis**: Surface area characterization  
- **Interactions**: Hard sphere and electrostatic correlations
- **2D Analysis**: Anisotropic scattering patterns
- **Structure Characterization**: Size, shape, and organization

**Key Learning Outcomes:**
- Perform quantitative SAXS analysis
- Extract particle sizes and shapes
- Analyze inter-particle interactions
- Interpret 2D scattering patterns

## Data Files

### Synthetic Data
All tutorials include code to generate realistic synthetic data, so **no external data files are required**. The synthetic datasets simulate:
- XPCS correlation functions with realistic noise
- SAXS scattering profiles for various particle types
- 2D scattering patterns with different symmetries

### Using Your Own Data
Each tutorial includes examples of loading real data:
```python
# Load your XPCS data
xf = XpcsDataFile('your_data.h5')

# Extract correlation functions
q, tau, g2, g2_err, labels = xf.get_g2_data()

# Extract SAXS data  
q, I, xlabel, ylabel = xf.get_saxs_1d_data()
```

## Tips for Success

### 🔧 Technical Tips
- **Run all cells sequentially** for proper variable initialization
- **Restart kernel** if you encounter import errors
- **Check file paths** when loading your own data
- **Save notebooks** regularly to preserve your work

### 📈 Analysis Tips  
- **Start simple**: Begin with single exponential fits before advanced models
- **Check validity ranges**: Ensure qRg < 1.3 for Guinier, qR > 3 for Porod  
- **Visualize residuals**: Look for systematic deviations from fits
- **Compare models**: Use statistical criteria for model selection

### 🐛 Troubleshooting
- **Import errors**: Check XPCS Toolkit installation
- **Fitting failures**: Try different initial parameters or bounds
- **Empty plots**: Verify data ranges and scaling
- **Memory issues**: Reduce data size or use sampling

## Additional Resources

### Documentation
- **[API Reference](../API_REFERENCE.md)**: Complete function documentation
- **[Scientific Background](../SCIENTIFIC_BACKGROUND.md)**: Theoretical foundations
- **[File Format Guide](../FILE_FORMAT_GUIDE.md)**: Data structure specifications
- **[CLI Reference](../CLI_REFERENCE.md)**: Command-line usage

### Example Datasets
- Contact the development team for example datasets
- Check the project repository for sample files
- Generate synthetic data using tutorial code

### Getting Help
- **In-notebook help**: Use `help(function)` or `function?`
- **Documentation**: Access built-in docstrings
- **Community**: Project repository for issues and questions
- **Development team**: Contact information in main README

## Contributing

### Improve Tutorials
- **Fix errors**: Submit corrections or clarifications
- **Add examples**: Contribute additional analysis scenarios  
- **Enhance explanations**: Improve scientific or technical content
- **New tutorials**: Propose advanced topics or specialized applications

### Share Your Work
- **Case studies**: Share real-world applications
- **Tips and tricks**: Contribute analysis best practices
- **Extensions**: Build on tutorial concepts for specific research needs

## License

These tutorials are part of the XPCS Toolkit and are distributed under the same license terms. See the main repository for license details.

---

*Start your XPCS analysis journey with Tutorial 01: [Getting Started](01_getting_started.ipynb)* 🚀