# Contributing to XPCS Toolkit

[![Contributors](https://img.shields.io/github/contributors/imewei/xpcs-toolkit.svg)](https://github.com/imewei/xpcs-toolkit/graphs/contributors)
[![Good First Issues](https://img.shields.io/github/issues-good-first-issue/imewei/xpcs-toolkit.svg)](https://github.com/imewei/xpcs-toolkit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

We welcome contributions to the XPCS Toolkit! This document provides guidelines for contributing to the project.

## 🚀 Quick Start for Contributors

### Development Setup

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/xpcs-toolkit.git
cd xpcs-toolkit

# 3. Set up development environment
make venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
make dev

# 4. Verify setup
make status
make test/cli
```

### Making Your First Contribution

```bash
# 1. Create a feature branch
git checkout -b feature/your-feature-name

# 2. Make your changes
# ... edit files ...

# 3. Run quality checks
make format              # Auto-format code
make lint               # Check code quality
make test               # Run test suite
make coverage           # Check coverage

# 4. Commit your changes
git add .
git commit -m "Add: description of your changes"

# 5. Push and create pull request
git push origin feature/your-feature-name
# Then create PR on GitHub
```

## 🎯 Ways to Contribute

### 🐛 Bug Reports
- **Search existing issues** first to avoid duplicates
- **Use the bug report template** when creating new issues
- **Include minimal reproduction steps** and system information
- **Attach relevant files** (data files, screenshots, logs)

### ✨ Feature Requests
- **Check existing feature requests** to avoid duplicates
- **Describe the scientific use case** and why it's important
- **Provide implementation suggestions** if you have ideas
- **Consider contributing the implementation** yourself!

### 📝 Documentation Improvements
- **Fix typos** and grammatical errors
- **Add examples** for existing functionality
- **Improve docstrings** and type hints
- **Create tutorials** for common scientific workflows
- **Translate documentation** to other languages

### 🧪 Code Contributions
- **Bug fixes** - Fix reported issues
- **New features** - Add scientific analysis capabilities
- **Performance improvements** - Optimize algorithms
- **Test coverage** - Add tests for untested code
- **Refactoring** - Improve code organization

## 📋 Development Guidelines

### Code Style
- **Follow PEP 8** - Use `make format` for automatic formatting
- **Use type hints** - All public functions should have type annotations
- **Write docstrings** - Use NumPy/Google style with examples
- **Keep functions focused** - Single responsibility principle

### Example Function Style
```python
def analyze_correlation(
    g2_data: np.ndarray,
    tau_values: np.ndarray,
    q_range: Optional[Tuple[float, float]] = None,
    fit_function: str = "single_exp"
) -> Dict[str, Any]:
    """
    Analyze correlation function data with optional q-filtering.
    
    This function fits correlation functions to extract dynamics information
    from XPCS data, supporting multiple fitting models and q-range selection.
    
    Parameters
    ----------
    g2_data : np.ndarray
        Correlation function array with shape (n_tau, n_q)
    tau_values : np.ndarray
        Time delay values in seconds
    q_range : tuple of float, optional
        Q-range filter as (q_min, q_max) in Å⁻¹. If None, use all q-values.
    fit_function : str, default "single_exp"
        Fitting function name. Options: "single_exp", "stretched", "double_exp"
        
    Returns
    -------
    dict
        Analysis results containing:
        
        - **fit_params** : list of dict
            Fit parameters for each q-bin
        - **quality_metrics** : dict
            Goodness-of-fit statistics
        - **processed_data** : dict
            Filtered and processed data arrays
        
    Examples
    --------
    >>> import numpy as np
    >>> g2 = np.random.random((100, 10)) + 1  # Mock g2 data
    >>> tau = np.logspace(-6, 0, 100)
    >>> results = analyze_correlation(g2, tau, q_range=(0.01, 0.1))
    >>> print(f"Found {len(results['fit_params'])} successful fits")
    Found 8 successful fits
    
    >>> # Access fit results
    >>> for i, params in enumerate(results['fit_params']):
    ...     if params['success']:
    ...         print(f"Q-bin {i}: γ = {params['gamma']:.2e} Hz")
    """
    # Implementation here...
    return results
```

### Testing Requirements

#### Test Categories
All new features must include comprehensive tests:

1. **Unit Tests** - Test individual functions in isolation
2. **Integration Tests** - Test module interactions
3. **Performance Tests** - Benchmark critical algorithms
4. **Scientific Accuracy Tests** - Validate against analytical solutions

#### Test Structure
```python
import pytest
import numpy as np
from xpcs_toolkit import XpcsDataFile
from xpcs_toolkit.tests.fixtures import synthetic_data

class TestCorrelationAnalysis:
    """Test suite for correlation analysis functions."""
    
    @pytest.fixture
    def sample_correlation_data(self):
        """Generate sample correlation data for testing."""
        generator = synthetic_data.SyntheticXPCSDataGenerator()
        return generator.generate_brownian_motion_intensity(
            n_times=100, n_q_bins=5
        )
    
    def test_single_exponential_fit(self, sample_correlation_data):
        """Test fitting single exponential model to Brownian motion data."""
        intensity, q_vals, tau_vals = sample_correlation_data
        
        # Calculate g2 function
        g2_data = calculate_g2_function(intensity)
        
        # Fit single exponential
        results = analyze_correlation(
            g2_data, tau_vals, fit_function="single_exp"
        )
        
        assert results['success']
        assert len(results['fit_params']) == len(q_vals)
        
        # Verify physical reasonableness
        for params in results['fit_params']:
            if params['success']:
                assert params['gamma'] > 0  # Positive relaxation rate
                assert 0 < params['beta'] <= 1  # Physical beta range
    
    @pytest.mark.parametrize("q_range", [
        (0.01, 0.1),
        (0.005, 0.05),
        None
    ])
    def test_q_range_filtering(self, sample_correlation_data, q_range):
        """Test q-range filtering functionality."""
        intensity, q_vals, tau_vals = sample_correlation_data
        g2_data = calculate_g2_function(intensity)
        
        results = analyze_correlation(g2_data, tau_vals, q_range=q_range)
        
        if q_range is not None:
            q_min, q_max = q_range
            processed_q = results['processed_data']['q_values']
            assert np.all(processed_q >= q_min)
            assert np.all(processed_q <= q_max)
```

### Quality Gates

Before your PR can be merged, it must pass all quality gates:

✅ **Code Quality**
```bash
make lint               # Ruff linting
make format/check       # Code formatting
make lint/mypy          # Type checking
make security/bandit    # Security analysis
```

✅ **Testing**
```bash
make test               # Full test suite
make coverage           # ≥85% coverage required
make test/performance   # Performance benchmarks
```

✅ **Documentation**
```bash
make docs               # Documentation builds
# All public APIs must have docstrings
# Examples must execute correctly
```

## 🔬 Scientific Contribution Guidelines

### Algorithm Implementation
- **Cite relevant papers** in docstrings and comments
- **Validate against analytical solutions** when possible
- **Include numerical stability considerations**
- **Document parameter ranges and limitations**

### Data Format Support
- **Follow existing patterns** in `xpcs_toolkit/io/formats/`
- **Include format detection** in `ftype_utils.py`
- **Add comprehensive tests** with sample data files
- **Document the file format** in `FILE_FORMAT_GUIDE.md`

### Performance Optimization
- **Profile before optimizing** using `make profile`
- **Use NumPy vectorization** instead of Python loops
- **Consider memory usage** for large datasets
- **Benchmark improvements** with `make test/performance`

## 🎨 Documentation Guidelines

### User Documentation
- **Write for your audience** - assume scientific background but not XPCS expertise
- **Include working examples** that users can copy and run
- **Provide context** - explain when and why to use features
- **Link to scientific references** for theoretical background

### API Documentation
```python
def example_function(data: np.ndarray, threshold: float = 0.05) -> bool:
    """
    Brief one-line description of what the function does.
    
    Longer description explaining the function's purpose, scientific context,
    and any important implementation details. Include references to relevant
    papers or theoretical background.
    
    Parameters
    ----------
    data : np.ndarray
        Description of the data parameter, including expected shape,
        units, and any constraints or preprocessing requirements.
    threshold : float, default 0.05
        Description of threshold parameter with physical interpretation
        and guidance on appropriate values.
        
    Returns
    -------
    bool
        Description of return value and its interpretation.
        
    Raises
    ------
    ValueError
        When and why this exception is raised, with guidance on
        how to avoid or handle it.
        
    Notes
    -----
    Any important implementation details, algorithm references,
    or theoretical considerations that users should know about.
    
    References
    ----------
    .. [1] Author, A. et al. "Paper Title." Journal Name, vol, pages (year).
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.random((100, 10))
    >>> result = example_function(data, threshold=0.1)
    >>> print(f"Analysis result: {result}")
    Analysis result: True
    
    For more complex examples, show realistic use cases:
    
    >>> from xpcs_toolkit import XpcsDataFile
    >>> xpcs_data = XpcsDataFile('experiment.h5')
    >>> intensity_data = xpcs_data.saxs_2d
    >>> is_stable = example_function(intensity_data, threshold=0.02)
    >>> if is_stable:
    ...     print("Data is stable for correlation analysis")
    """
```

## 🎯 Issue and Pull Request Guidelines

### Creating Issues

#### Bug Report Template
```markdown
**Bug Description**
Clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Load data file '...'
2. Run command '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g. Ubuntu 22.04, macOS 13, Windows 11]
- Python version: [e.g. 3.12.1]
- XPCS Toolkit version: [e.g. 1.2.3]
- Relevant dependencies: [e.g. numpy 1.24.0, h5py 3.8.0]

**Additional Context**
- Sample data files (if possible)
- Full error traceback
- Screenshots if relevant
```

#### Feature Request Template
```markdown
**Scientific Use Case**
Describe the scientific problem you're trying to solve.

**Proposed Solution**
Describe what you'd like to happen. Include:
- Algorithm or method details
- Expected inputs and outputs
- Performance considerations

**Alternatives Considered**
Other approaches you've considered and why this solution is preferred.

**Additional Context**
- Literature references
- Similar implementations in other software
- Willingness to contribute implementation
```

### Pull Request Process

#### PR Description Template
```markdown
## Description
Brief description of changes and their purpose.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Performance optimization
- [ ] Refactoring (no functional changes)

## Scientific Context
Describe the scientific motivation and impact of these changes.

## Testing
- [ ] All existing tests pass locally
- [ ] New tests added for new functionality
- [ ] Performance benchmarks run (if applicable)
- [ ] Scientific accuracy validated against analytical solutions

## Quality Checklist
- [ ] Code follows project style guidelines (`make lint` passes)
- [ ] Self-review completed
- [ ] Documentation updated (docstrings, user guides)
- [ ] No breaking changes (or migration guide provided)
- [ ] Performance impact assessed

## Dependencies
List any new dependencies and justify their inclusion.

## Breaking Changes
If this is a breaking change, describe:
- What breaks
- How users can migrate
- Deprecation timeline
```

### Code Review Process

All contributions go through code review:

1. **Automated Checks** - GitHub Actions run quality gates
2. **Scientific Review** - Validate scientific correctness
3. **Code Review** - Review implementation and style
4. **Documentation Review** - Check docs and examples
5. **Final Approval** - Maintainer approval required

## 🏆 Recognition

### Contributors
All contributors are recognized in:
- **README.md** - Contributors section
- **CHANGELOG.md** - Release notes
- **GitHub Contributors** - Automatic recognition
- **Zenodo Citation** - Academic credit for releases

### Types of Contributions Recognized
- 💻 **Code contributions** - Bug fixes, features, optimizations
- 📖 **Documentation** - Tutorials, examples, improvements
- 🐛 **Bug reports** - High-quality bug reports with reproductions
- 💡 **Ideas & Discussion** - Feature suggestions and design input
- 🧪 **Testing** - Test contributions and validation
- 🌐 **Translations** - Documentation translations
- 📢 **Outreach** - Conference talks, blog posts, tutorials

## 📞 Getting Help

### Communication Channels
- **GitHub Discussions** - Questions and general discussion
- **GitHub Issues** - Bug reports and feature requests
- **Email** - weichen@anl.gov for direct contact
- **Matrix Chat** - [#xpcs-toolkit:matrix.org] (community chat)

### Development Support
- **Documentation** - See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- **API Reference** - [https://xpcs-toolkit.readthedocs.io/](https://xpcs-toolkit.readthedocs.io/)
- **Examples** - Browse the `examples/` directory
- **Tests** - Look at existing tests for patterns

### Beginner-Friendly Tasks
Look for issues labeled:
- 🟢 **good first issue** - Perfect for new contributors
- 📚 **documentation** - Documentation improvements
- 🧪 **testing** - Add tests for existing functionality
- 🐛 **bug** - Fix reported issues

## 📚 Resources

### Scientific Background
- **[SCIENTIFIC_BACKGROUND.md](SCIENTIFIC_BACKGROUND.md)** - XPCS theory and methods
- **XPCS Literature** - Key papers and reviews
- **Synchrotron Resources** - Beamline documentation

### Development Resources
- **[Python Packaging Guide](https://packaging.python.org/)**
- **[NumPy Documentation Style](https://numpydoc.readthedocs.io/)**
- **[Scientific Python Ecosystem](https://scientific-python.org/)**
- **[Sphinx Documentation](https://www.sphinx-doc.org/)**

### Similar Projects
- **PyFAI** - Azimuthal integration
- **SilX** - Synchrotron data analysis
- **scikit-beam** - X-ray scattering analysis
- **DPDAK** - Correlation spectroscopy

---

## 🙏 Thank You!

Thank you for contributing to XPCS Toolkit! Your contributions help advance X-ray science and support researchers worldwide. Every contribution, no matter how small, makes a difference.

**Questions?** Don't hesitate to ask in GitHub Discussions or reach out directly. We're here to help you succeed!

---

*This contributing guide is living document. If you have suggestions for improvements, please open an issue or submit a pull request.*