# XPCS Toolkit Makefile Guide

## Overview

The XPCS Toolkit project includes a comprehensive Makefile that provides a powerful, user-friendly interface for all development, testing, building, and deployment tasks. This guide explains how to use the Makefile effectively for both development and production workflows.

## 🚀 Quick Start

```bash
# See all available commands
make help

# Check project status
make status

# Set up development environment
make dev

# Run tests
make test

# Clean everything
make clean
```

## 📋 Command Categories

The Makefile organizes commands into logical categories for easy navigation:

### Main Commands
- `help` - Show categorized help with all available commands
- `status` - Display comprehensive project and environment status
- `check-venv` - Verify virtual environment setup
- `venv` - Create a new virtual environment

### Development Workflow

#### Setup Commands
```bash
make venv                    # Create virtual environment
make install/dev            # Install with development dependencies
make dev                     # Quick development setup (combines above)
```

#### Testing Commands
```bash
make test                    # Run basic unit tests
make test/unit              # Run unit tests with detailed output
make test/integration       # Run integration tests
make test/cli              # Test CLI functionality
make test/performance      # Test performance and import times
make test/all              # Run all tests
make coverage              # Generate coverage report
make quick-test            # Fast test run for development
```

#### Code Quality Commands
```bash
make lint                   # Run all linting checks
make lint/ruff             # Check code style with ruff
make lint/mypy             # Run type checking with mypy
make format                # Format code with ruff
make format/check          # Check formatting without changes
make security/bandit       # Run security analysis
```

### Build & Distribution

#### Build Commands
```bash
make build                 # Build both wheel and source distribution
make build/check          # Verify build requirements
make build/wheel          # Build wheel package only
make build/sdist          # Build source distribution only
```

#### Installation Commands
```bash
make install              # Install in development mode
make install/dev          # Install with development dependencies
make install/prod         # Install for production use
```

### Release Management

#### Release Commands
```bash
make release/test         # Upload to test PyPI
make release/prod         # Upload to production PyPI
```

**Note**: Requires proper PyPI credentials and should be used carefully.

### Documentation

#### Documentation Commands
```bash
make docs                 # Generate Sphinx documentation
make docs/serve           # Generate docs and serve locally
make clean-docs           # Clean documentation artifacts
```

### Dependency Management

#### Dependency Commands
```bash
make deps/check           # Check dependency compatibility
make deps/update          # Update all dependencies
make deps/outdated        # Show outdated packages
make deps/tree            # Display dependency tree
```

### Maintenance

#### Cleaning Commands
```bash
make clean               # Remove build, test, and Python artifacts
make clean-all           # Deep clean including documentation and caches
make clean-build         # Remove only build artifacts
make clean-pyc           # Remove only Python bytecode files
make clean-test          # Remove only test artifacts
```

## 🎯 Development Workflows

### New Developer Setup
```bash
# 1. Clone the repository
git clone <repository-url>
cd xpcs_toolkit

# 2. Set up development environment
make venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
make dev

# 3. Verify installation
make status
make test/cli
```

### Daily Development Workflow
```bash
# 1. Check project status
make status

# 2. Run quick tests during development
make quick-test

# 3. Before committing, run full quality checks
make lint
make test
make coverage

# 4. Clean up when needed
make clean
```

### Pre-Release Workflow
```bash
# 1. Full quality assurance
make ci/check

# 2. Build distributions
make build

# 3. Test release (optional)
make release/test

# 4. Production release
make release/prod
```

## 🔧 Advanced Features

### Performance Profiling
```bash
make profile             # Profile package import performance
make test/performance    # Run performance benchmarks
make stats              # Show project statistics
```

### CI/CD Integration
```bash
make ci/check           # Complete CI pipeline
# Equivalent to: check-venv + lint + test + coverage
```

### Docker Support (if Dockerfile exists)
```bash
make docker/build       # Build Docker image
make docker/run         # Run container interactively
make docker/test        # Run tests in container
```

### Watch Mode for Development
```bash
make watch-test         # Continuously run tests (Ctrl+C to stop)
```

## 🎨 Output Features

### Colored Output
The Makefile uses ANSI color codes for better readability:
- 🔵 **Cyan**: Process descriptions
- 🟢 **Green**: Success messages
- 🟡 **Yellow**: Warnings and notes
- 🔴 **Red**: Errors
- 🟦 **Blue**: Section headers

### Progress Indicators
Commands show clear progress with:
- ✓ Success indicators
- ⚠ Warning indicators
- ✗ Error indicators
- Clear status messages

### Environment Detection
The Makefile automatically detects:
- Virtual environment status
- Python version and location
- Package installation status
- Dependency compatibility

## 📊 Status Information

### Project Status (`make status`)
Displays comprehensive information:
- Project and package names
- Python version and location
- Virtual environment status
- Package version (if installed)
- Dependency status

### Project Statistics (`make stats`)
Shows useful metrics:
- Lines of code count
- Test coverage percentage
- Package size on disk

## ⚙️ Configuration

### Environment Variables
The Makefile respects these environment variables:
- `PYTHON` - Python interpreter to use
- `PIP` - Pip command to use
- `PYTEST` - Pytest command to use

### Customization
Key variables can be customized at the top of the Makefile:
```makefile
PROJECT_NAME := xpcs-toolkit
PACKAGE_NAME := xpcs_toolkit
PYTHON := python
PIP := pip
```

## 🚨 Error Handling

### Common Issues and Solutions

#### Virtual Environment Issues
```bash
# Problem: No virtual environment detected
make check-venv          # Check status
make venv               # Create new environment
source .venv/bin/activate  # Activate it
```

#### Build Failures
```bash
# Problem: Build dependencies missing
make build/check        # Verify requirements
make deps/update        # Update dependencies
```

#### Test Failures
```bash
# Problem: Tests failing
make quick-test         # Fast feedback
make test/unit          # Detailed unit tests
make coverage          # Coverage analysis
```

#### Import Errors
```bash
# Problem: Package not installed
make install/dev        # Install in development mode
make status            # Verify installation
```

## 🔍 Troubleshooting

### Debug Mode
Add `-n` flag to see what commands would run without executing:
```bash
make -n test           # Show test commands without running
```

### Verbose Output
Some commands have verbose modes:
```bash
make test/unit         # Already includes -v flag
```

### Dependency Issues
```bash
make deps/check        # Check for conflicts
make deps/outdated     # Find outdated packages
make deps/tree         # Visualize dependency tree
```

## 🏆 Best Practices

### Development
1. **Always work in a virtual environment**
   ```bash
   make check-venv  # Verify before starting
   ```

2. **Run tests frequently**
   ```bash
   make quick-test  # During development
   make test       # Before commits
   ```

3. **Keep dependencies updated**
   ```bash
   make deps/outdated  # Check monthly
   make deps/update    # Update when needed
   ```

### Code Quality
1. **Format before committing**
   ```bash
   make format     # Auto-format code
   ```

2. **Check code quality**
   ```bash
   make lint      # Check style and types
   ```

3. **Monitor coverage**
   ```bash
   make coverage  # Generate coverage reports
   ```

### Releases
1. **Test thoroughly**
   ```bash
   make ci/check  # Full CI pipeline
   ```

2. **Build and verify**
   ```bash
   make build     # Create distributions
   ```

3. **Use test PyPI first**
   ```bash
   make release/test  # Test deployment
   ```

## 📚 Integration with IDEs

### VS Code
Add to `.vscode/tasks.json`:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Quick Test",
            "type": "shell",
            "command": "make quick-test",
            "group": "test"
        }
    ]
}
```

### PyCharm
Configure external tools to run Make commands through the IDE.

## 🤝 Contributing

When contributing to the project:
1. Use `make dev` for initial setup
2. Use `make quick-test` during development
3. Use `make ci/check` before submitting PRs
4. Use `make clean` to clean up artifacts

## 📞 Support

If you encounter issues with the Makefile:
1. Check `make help` for available commands
2. Use `make status` to check your environment
3. Refer to this guide for common solutions
4. Check the project's issue tracker for known problems

The Makefile is designed to be self-documenting and user-friendly. Most commands include helpful output and error messages to guide you through any issues.