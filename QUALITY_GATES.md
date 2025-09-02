# Quality Gates for XPCS Toolkit

This document outlines the automated quality assurance system for the XPCS Toolkit project.

## Overview

The XPCS Toolkit maintains high code quality standards through automated quality gates that run on every commit and pull request. These gates ensure code reliability, performance, and scientific accuracy.

## Quality Metrics & Thresholds

### Code Coverage
- **Target**: ≥ 85% overall coverage
- **Critical Functions**: ≥ 95% for numerical/scientific functions
- **Tests**: 446 comprehensive tests covering unit, integration, and performance

### Code Quality
- **Formatting**: Ruff format compliance (PEP 8)
- **Linting**: Zero critical issues, minimal warnings
- **Type Coverage**: MyPy static type checking
- **Complexity**: Cyclomatic complexity ≤ 10

### Security
- **Vulnerability Scanning**: Bandit security analysis
- **Dependency Security**: Safety vulnerability database checks
- **No Critical Issues**: Zero high-severity security findings

### Performance
- **Import Time**: < 10 seconds for cold imports
- **Memory Usage**: Stable memory patterns
- **Scientific Accuracy**: Numerical precision within tolerance

## Automated Workflows

### 1. Pre-commit Hooks
Runs locally before each commit:
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**Checks Include:**
- Code formatting (Ruff)
- Type checking (MyPy)
- Security scanning (Bandit)  
- Numerical stability validation
- Import performance testing

### 2. CI/CD Pipeline
Runs on GitHub Actions for every PR and push:

#### Code Quality Job
- Multi-Python version testing (3.12, 3.13)
- Comprehensive linting and formatting
- Type checking with scientific library stubs
- Security vulnerability scanning
- Test suite execution with coverage

#### Scientific Validation Job
- Performance benchmarking
- Integration testing
- Scientific computing validation
- Memory usage analysis

#### Documentation Job
- Docstring coverage validation (≥80%)
- Documentation build verification

## Quality Gate Criteria

### Passing Requirements
All of the following must pass:

✅ **Code Coverage** ≥ 85%  
✅ **All Tests Passing** (446/446)  
✅ **Zero Critical Security Issues**  
✅ **Type Checking** with minimal errors  
✅ **Import Performance** < 10s  
✅ **Docstring Coverage** ≥ 80%  

### Blocking Conditions
Any of the following will block merging:

❌ Test failures  
❌ Coverage below 85%  
❌ Critical security vulnerabilities  
❌ Severe performance regressions  
❌ Import failures  

## Scientific Computing Standards

### Numerical Accuracy
- Floating-point precision validation
- Tolerance-based test comparisons
- Stable algorithm implementations

### Research Reproducibility  
- Consistent random seed handling
- Deterministic computation paths
- Version-controlled scientific parameters

### Performance Optimization
- Vectorized NumPy operations
- Efficient memory usage patterns
- Optimized HDF5 I/O operations

## Development Workflow

### Before Committing
1. Run pre-commit hooks: `pre-commit run --all-files`
2. Execute test suite: `pytest --cov=xpcs_toolkit`
3. Check type annotations: `mypy xpcs_toolkit`

### Pull Request Process
1. All quality gates must pass
2. Code review by maintainers
3. Scientific accuracy validation
4. Performance regression analysis

### Continuous Monitoring
- Daily dependency vulnerability scans
- Weekly performance benchmarking
- Monthly quality metrics review

## Quality Metrics Dashboard

The following metrics are tracked over time:

- **Test Coverage Trends**
- **Performance Baselines**  
- **Security Posture**
- **Code Quality Scores**
- **Documentation Coverage**

## Maintenance

### Quality Gate Updates
- Quality thresholds reviewed quarterly
- Tool versions updated monthly
- New checks added as needed

### Scientific Standards
- Numerical accuracy requirements
- Research reproducibility criteria
- Performance optimization targets

## Tools & Technologies

### Core Quality Tools
- **Ruff**: Fast Python linter and formatter
- **MyPy**: Static type checker
- **Pytest**: Comprehensive testing framework
- **Bandit**: Security vulnerability scanner
- **Safety**: Dependency vulnerability checker

### Scientific Computing Tools
- **NumPy**: Numerical precision validation
- **SciPy**: Scientific algorithm verification  
- **Coverage.py**: Test coverage analysis
- **Pre-commit**: Git hook automation

---

For questions about quality gates or to report issues, please open a GitHub issue or contact the development team.