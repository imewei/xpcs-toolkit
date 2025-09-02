Contributing to XPCS Toolkit
============================

We welcome contributions from the scientific community! This guide will help you get started with contributing to XPCS Toolkit.

🎯 **Quick Start for Contributors**

1. Fork the repository on GitHub
2. Clone your fork: ``git clone https://github.com/YOUR-USERNAME/xpcs-toolkit.git``
3. Install development dependencies: ``pip install -e .[dev]``
4. Create a feature branch: ``git checkout -b feature/your-feature``
5. Make your changes with tests
6. Run quality checks: ``make lint`` and ``make test``
7. Submit a pull request

Types of Contributions
----------------------

We welcome several types of contributions:

**🐛 Bug Reports**
  Report issues, unexpected behavior, or errors you encounter

**💡 Feature Requests**
  Suggest new functionality or enhancements

**📖 Documentation**
  Improve documentation, add examples, or fix typos

**🔧 Code Contributions**
  Fix bugs, implement features, or improve performance

**🧪 Testing**
  Add test cases, improve test coverage, or test on new platforms

**🎨 User Experience**
  Improve CLI interface, error messages, or usability

Setting Up Development Environment
----------------------------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.12 or higher
- Git
- Text editor or IDE of your choice

Installation
~~~~~~~~~~~~

1. **Fork and clone the repository**:

.. code-block:: bash

   # Fork on GitHub first, then:
   git clone https://github.com/YOUR-USERNAME/xpcs-toolkit.git
   cd xpcs-toolkit

2. **Create and activate virtual environment**:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install in development mode**:

.. code-block:: bash

   pip install -e .[dev]

4. **Verify installation**:

.. code-block:: bash

   python -c "import xpcs_toolkit; print('✅ Development setup complete!')"

Development Workflow
--------------------

Creating a Feature Branch
~~~~~~~~~~~~~~~~~~~~~~~~~

Always create a new branch for your work:

.. code-block:: bash

   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number
   # or  
   git checkout -b docs/documentation-improvement

Making Changes
~~~~~~~~~~~~~~

1. **Write your code** following our style guidelines
2. **Add tests** for new functionality
3. **Update documentation** if needed
4. **Run quality checks** before committing

Quality Checks
~~~~~~~~~~~~~~

Run these checks before submitting:

.. code-block:: bash

   # Code formatting
   ruff format .
   
   # Linting
   ruff check .
   
   # Type checking
   mypy xpcs_toolkit
   
   # Run tests
   pytest
   
   # Run tests with coverage
   pytest --cov=xpcs_toolkit

Alternatively, use the Makefile shortcuts:

.. code-block:: bash

   make format    # Format code
   make lint      # Run all linting checks
   make test      # Run tests
   make coverage  # Run tests with coverage report

Committing Changes
~~~~~~~~~~~~~~~~~~

Follow conventional commit format:

.. code-block:: bash

   git commit -m "feat: add new correlation analysis function"
   git commit -m "fix: resolve memory leak in file loading"
   git commit -m "docs: improve SAXS analysis guide"

Commit message types:
- ``feat``: New feature
- ``fix``: Bug fix
- ``docs``: Documentation changes
- ``test``: Adding or fixing tests
- ``refactor``: Code refactoring
- ``perf``: Performance improvements
- ``style``: Code style changes

Submitting Pull Requests
------------------------

1. **Push your branch**:

.. code-block:: bash

   git push origin feature/your-feature-name

2. **Create pull request** on GitHub

3. **Fill out the PR template** with:
   - Clear description of changes
   - Link to related issues
   - Testing instructions
   - Breaking changes (if any)

4. **Address review feedback** promptly

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

**Good Pull Requests:**

- ✅ Single focus (one feature/fix per PR)
- ✅ Clear, descriptive title and description
- ✅ Include tests for new functionality
- ✅ Update documentation as needed
- ✅ Pass all quality checks
- ✅ Small, reviewable size (< 400 lines typically)

**What to Avoid:**

- ❌ Multiple unrelated changes in one PR
- ❌ Breaking changes without discussion
- ❌ Missing tests for new features
- ❌ Formatting changes mixed with functional changes
- ❌ Large PRs that are difficult to review

Code Standards
--------------

Style Guide
~~~~~~~~~~~

We follow Python standards with these tools:

- **Ruff**: For linting and formatting (replaces flake8, isort, black)
- **MyPy**: For type checking
- **Pre-commit**: For automated checks

Key principles:

- **PEP 8 compliant** with 88-character line limit
- **Type hints** for all public functions
- **Docstrings** for all public classes and functions
- **Clear, descriptive names** for variables and functions

Documentation Style
~~~~~~~~~~~~~~~~~~~

- **NumPy-style docstrings** for functions and classes
- **Sphinx-compatible** RST format for documentation
- **Examples** in docstrings where helpful
- **Clear parameter descriptions** with types and defaults

Example function docstring:

.. code-block:: python

   def analyze_correlation(
       tau: np.ndarray, 
       g2: np.ndarray, 
       fit_model: str = "exponential"
   ) -> Dict[str, float]:
       """
       Analyze correlation function data.
       
       Performs fitting and extracts characteristic parameters from
       intensity correlation functions g2(τ).
       
       Parameters
       ----------
       tau : np.ndarray
           Delay time values in seconds
       g2 : np.ndarray  
           Correlation function values
       fit_model : str, optional
           Fitting model to use, by default "exponential"
           
       Returns
       -------
       Dict[str, float]
           Dictionary containing fit parameters and quality metrics
           
       Examples
       --------
       >>> tau = np.logspace(-6, 0, 50)
       >>> g2 = 1.5 * np.exp(-2 * tau / 0.01) + 1.0
       >>> results = analyze_correlation(tau, g2)
       >>> print(f"Relaxation time: {results['tau_relax']:.3f} s")
       """

Testing Guidelines
------------------

Test Structure
~~~~~~~~~~~~~~

We use pytest with this structure:

.. code-block::

   xpcs_toolkit/tests/
   ├── unit/           # Unit tests for individual components
   ├── integration/    # Integration tests
   ├── performance/    # Performance tests
   └── fileio/         # File I/O tests

Writing Tests
~~~~~~~~~~~~~

1. **Test file naming**: ``test_*.py``
2. **Test function naming**: ``test_*`` 
3. **Use descriptive test names**
4. **Include docstrings for complex tests**
5. **Use fixtures for setup/teardown**

Example test:

.. code-block:: python

   def test_xpcs_file_loading():
       """Test that XpcsDataFile loads valid files correctly."""
       # Setup
       test_file = "path/to/test_file.hdf5"
       
       # Action
       data = XpcsDataFile(test_file)
       
       # Assertions
       assert data.analysis_type is not None
       assert hasattr(data, 'X_energy')
       assert data.X_energy > 0

Test Coverage
~~~~~~~~~~~~~

- **Aim for >85%** overall coverage
- **>95% coverage** for critical scientific functions
- **Test edge cases** and error conditions
- **Include integration tests** for workflows

Documentation Contributions
---------------------------

Types of Documentation
~~~~~~~~~~~~~~~~~~~~~~

- **API Documentation**: Automatically generated from docstrings
- **User Guides**: Step-by-step tutorials and how-tos  
- **Examples**: Code examples and notebooks
- **FAQ**: Common questions and solutions

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Build HTML documentation
   cd docs
   make html
   
   # Live reload during development
   make livehtml
   
   # Check links
   make linkcheck
   
   # Check coverage
   make coverage

Documentation Style
~~~~~~~~~~~~~~~~~~~

- **Clear, concise language**
- **Working code examples**
- **Screenshots where helpful** (especially for GUI features)
- **Cross-references** to related functions
- **Consistent formatting**

Scientific Contributions
------------------------

Domain Expertise
~~~~~~~~~~~~~~~~

We especially welcome contributions from scientists using XPCS:

- **Algorithm improvements**
- **New analysis methods**
- **Validation against experimental data**
- **Performance optimizations for scientific computing**

Scientific Standards
~~~~~~~~~~~~~~~~~~~~

- **Cite relevant literature** in docstrings/comments
- **Validate against known results** when possible
- **Include physical units** and meaningful defaults
- **Consider numerical precision** and stability

Example scientific function:

.. code-block:: python

   def calculate_structure_factor(q: np.ndarray, r: float) -> np.ndarray:
       """
       Calculate structure factor for spherical particles.
       
       Based on the form factor for homogeneous spheres [1]_.
       
       Parameters
       ----------
       q : np.ndarray
           Momentum transfer values in Å⁻¹
       r : float
           Sphere radius in Å
           
       Returns
       -------
       np.ndarray
           Structure factor values
           
       References
       ----------
       .. [1] Guinier, A. & Fournet, G. "Small-Angle Scattering of X-Rays"
              John Wiley & Sons, New York (1955).
       """

Review Process
--------------

What to Expect
~~~~~~~~~~~~~~

1. **Automated checks** run first (CI/CD)
2. **Maintainer review** for code quality and design
3. **Scientific review** for algorithm/analysis contributions
4. **Documentation review** for user-facing changes
5. **Final approval** and merge

Review Timeline
~~~~~~~~~~~~~~~

- **Simple fixes**: 1-3 days
- **New features**: 1-2 weeks  
- **Major changes**: 2-4 weeks

We aim to provide initial feedback within 48 hours.

Addressing Feedback
~~~~~~~~~~~~~~~~~~~

- **Respond promptly** to review comments
- **Ask questions** if feedback is unclear
- **Make requested changes** in additional commits
- **Mark conversations as resolved** when addressed

Getting Help
------------

Communication Channels
~~~~~~~~~~~~~~~~~~~~~~

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Email**: weichen@anl.gov for direct contact with maintainers
- **Code Review**: Comments on pull requests

Onboarding for New Contributors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New to open source? We're here to help!

- **Good first issues**: Look for "good first issue" labels
- **Mentorship available**: Ask for guidance in issues or discussions
- **Documentation contributions**: Great way to start contributing
- **Pair programming**: Available for complex contributions

Recognition
-----------

Contributors are recognized in:

- **CHANGELOG.md**: All contributors listed in release notes
- **GitHub contributors**: Automatic recognition on repository
- **Academic citations**: Significant scientific contributions acknowledged
- **Community recognition**: Highlighted in project communications

Thank You!
----------

Your contributions make XPCS Toolkit better for the entire scientific community. Whether you're fixing a typo, adding a feature, or improving documentation, every contribution matters.

**Questions?** Don't hesitate to reach out through GitHub Issues or Discussions. We're excited to work with you! 🚀

---

*This contributing guide is itself open to contributions! If you see ways to improve it, please submit a pull request.*