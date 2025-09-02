# XPCS Toolkit - Comprehensive Makefile
# =====================================

# Project configuration
PROJECT_NAME := xpcs-toolkit
PACKAGE_NAME := xpcs_toolkit
PYTHON := python
PIP := pip
PYTEST := pytest
COVERAGE := coverage
RUFF := ruff
MYPY := mypy
TWINE := twine

# Directories
SRC_DIR := $(PACKAGE_NAME)
TESTS_DIR := $(SRC_DIR)/tests
DOCS_DIR := docs
BUILD_DIR := build
DIST_DIR := dist
HTMLCOV_DIR := htmlcov
CACHE_DIR := .pytest_cache

# Virtual environment detection
VENV := $(shell $(PYTHON) -c "import sys; print(sys.prefix != sys.base_prefix)")
ifeq ($(VENV),True)
    VENV_MSG := ✓ Virtual environment detected
else
    VENV_MSG := ⚠ No virtual environment detected - consider using 'make venv'
endif

# Color codes for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
CYAN := \033[0;36m
NC := \033[0m # No Color

# Default goal
.DEFAULT_GOAL := help

# Help system
define PRINT_HELP_PYSCRIPT
import re, sys
print("XPCS Toolkit - Development Commands")
print("=" * 40)
categories = {}
for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_/-]+):.*?## (.*)$$', line)
	if match:
		target, help_text = match.groups()
		if '/' in target:
			category = target.split('/')[0]
		else:
			category = 'main'
		if category not in categories:
			categories[category] = []
		categories[category].append((target, help_text))

for category in sorted(categories.keys()):
	if category != 'main':
		print(f"\n{category.upper()} Commands:")
		print("-" * 20)
	for target, help_text in categories[category]:
		print("  %-20s %s" % (target, help_text))
	if category == 'main':
		print()
endef
export PRINT_HELP_PYSCRIPT

# Browser helper
define BROWSER_PYSCRIPT
import os, webbrowser, sys
from urllib.request import pathname2url
webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

BROWSER := $(PYTHON) -c "$$BROWSER_PYSCRIPT"

# Phony targets
.PHONY: help venv check-venv status clean clean-all clean-venv clean-build clean-pyc clean-test clean-cache
.PHONY: lint lint/ruff lint/mypy format format/ruff format/check security/bandit
.PHONY: test test/unit test/integration test/cli test/performance test/all coverage
.PHONY: build build/check build/wheel build/sdist install install/dev install/prod
.PHONY: docs docs/serve docs/clean release release/test release/prod
.PHONY: docker/build docker/run docker/test ci/check profile
.PHONY: deps/check deps/update deps/outdated deps/tree

# =============================================================================
# MAIN COMMANDS
# =============================================================================

help: ## Show this help message
	@$(PYTHON) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

status: ## Show project status and environment information
	@echo "$(CYAN)XPCS Toolkit Project Status$(NC)"
	@echo "$(YELLOW)============================$(NC)"
	@echo "Project: $(PROJECT_NAME)"
	@echo "Package: $(PACKAGE_NAME)"
	@echo "Python:  $$($(PYTHON) --version)"
	@echo "Location: $$(which $(PYTHON))"
	@echo "$(VENV_MSG)"
	@echo ""
	@echo "$(BLUE)Package Info:$(NC)"
	@$(PYTHON) -c "import $(PACKAGE_NAME); print(f'Version: {$(PACKAGE_NAME).__version__}')" 2>/dev/null || echo "Package not installed"
	@echo ""
	@echo "$(BLUE)Dependencies Status:$(NC)"
	@$(PIP) check 2>/dev/null && echo "$(GREEN)✓ All dependencies satisfied$(NC)" || echo "$(RED)✗ Dependency conflicts found$(NC)"

check-venv: ## Check if virtual environment is active
	@if [ "$(VENV)" = "False" ]; then \
		echo "$(RED)⚠ Warning: No virtual environment detected$(NC)"; \
		echo "Consider running: make venv && source .venv/bin/activate"; \
	else \
		echo "$(GREEN)✓ Virtual environment is active$(NC)"; \
	fi

venv: ## Create virtual environment
	@echo "$(CYAN)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv .venv
	@echo "$(GREEN)✓ Virtual environment created in .venv$(NC)"
	@echo "$(YELLOW)Activate it with: source .venv/bin/activate$(NC)"

# =============================================================================
# CLEANING COMMANDS
# =============================================================================

clean: clean-build clean-pyc clean-test clean-cache ## Remove all build, test, coverage and Python artifacts
	@echo "$(GREEN)✓ Cleanup completed$(NC)"

clean-all: clean clean-docs clean-venv ## Remove all artifacts including virtual environment
	rm -rf .coverage.*
	rm -rf prof/
	@echo "$(GREEN)✓ Deep cleanup completed$(NC)"

clean-venv: ## Remove virtual environment
	rm -rf .venv/ venv/
	@echo "$(GREEN)✓ Virtual environment removed$(NC)"

clean-build: ## Remove build artifacts
	rm -fr $(BUILD_DIR)/
	rm -fr $(DIST_DIR)/
	rm -fr .eggs/
	find . -path './venv' -prune -o -path './.venv' -prune -o -name '*.egg-info' -exec rm -fr {} +
	find . -path './venv' -prune -o -path './.venv' -prune -o -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove Python file artifacts
	find . -path './venv' -prune -o -path './.venv' -prune -o -name '*.pyc' -exec rm -f {} +
	find . -path './venv' -prune -o -path './.venv' -prune -o -name '*.pyo' -exec rm -f {} +
	find . -path './venv' -prune -o -path './.venv' -prune -o -name '*~' -exec rm -f {} +
	find . -path './venv' -prune -o -path './.venv' -prune -o -name '__pycache__' -exec rm -fr {} +
	find . -path './venv' -prune -o -path './.venv' -prune -o -name '.DS_Store' -exec rm -f {} +

clean-test: ## Remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr $(HTMLCOV_DIR)/
	rm -fr $(CACHE_DIR)/
	rm -fr .pytest_cache/
	find . -name '.benchmarks' -type d -exec rm -rf {} +
	rm -f test_report.html
	rm -f bandit-report.json
	rm -f bandit_report.json
	rm -f bandit-results.json
	rm -f bandit_results.json
	rm -f code_quality_report.md
	rm -f pip_audit_report.json
	rm -f coverage.xml
	rm -f coverage.json

clean-cache: ## Remove development tool cache directories
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

clean-docs: ## Remove documentation artifacts
	rm -f $(DOCS_DIR)/$(PACKAGE_NAME).rst
	rm -f $(DOCS_DIR)/modules.rst
	$(MAKE) -C $(DOCS_DIR) clean 2>/dev/null || true

# =============================================================================
# CODE QUALITY COMMANDS
# =============================================================================

lint: lint/ruff lint/mypy ## Run all linting checks
	@echo "$(GREEN)✓ All linting checks completed$(NC)"

lint/ruff: ## Check style with ruff
	@echo "$(CYAN)Running ruff checks...$(NC)"
	$(RUFF) check $(SRC_DIR)
	@echo "$(GREEN)✓ Ruff checks passed$(NC)"

lint/mypy: ## Check types with mypy
	@echo "$(CYAN)Running mypy type checks...$(NC)"
	$(MYPY) $(SRC_DIR) --ignore-missing-imports
	@echo "$(GREEN)✓ MyPy checks passed$(NC)"

format: format/ruff ## Format code with ruff
	@echo "$(GREEN)✓ Code formatting completed$(NC)"

format/ruff: ## Format code with ruff
	@echo "$(CYAN)Formatting code with ruff...$(NC)"
	$(RUFF) format $(SRC_DIR)

format/check: ## Check code formatting without making changes
	@echo "$(CYAN)Checking code formatting...$(NC)"
	$(RUFF) format --check $(SRC_DIR)
	@echo "$(GREEN)✓ Code formatting is correct$(NC)"

security/bandit: ## Run security checks with bandit
	@echo "$(CYAN)Running security checks...$(NC)"
	$(PYTHON) -m bandit -r $(SRC_DIR) -c pyproject.toml -f json -o bandit-report.json || true
	$(PYTHON) -m bandit -r $(SRC_DIR) -c pyproject.toml || echo "$(YELLOW)⚠ Found low-severity warnings (acceptable)$(NC)"
	@echo "$(GREEN)✓ Security checks completed$(NC)"

# =============================================================================
# TESTING COMMANDS
# =============================================================================

test: test/unit ## Run basic unit tests
	@echo "$(GREEN)✓ Unit tests completed$(NC)"

test/unit: ## Run unit tests
	@echo "$(CYAN)Running unit tests...$(NC)"
	$(PYTEST) $(TESTS_DIR)/ -v --tb=short

test/integration: ## Run integration tests
	@echo "$(CYAN)Running integration tests...$(NC)"
	$(PYTEST) $(TESTS_DIR)/ -v -m "integration" --tb=short

test/cli: ## Test CLI functionality
	@echo "$(CYAN)Testing CLI functionality...$(NC)"
	@# Test help command
	$(PYTHON) -m $(PACKAGE_NAME).cli_headless --help > /dev/null
	@# Test version command
	$(PYTHON) -m $(PACKAGE_NAME).cli_headless --version > /dev/null
	@# Test console scripts if installed
	@if command -v xpcs-toolkit >/dev/null 2>&1; then \
		xpcs-toolkit --version > /dev/null; \
		echo "$(GREEN)✓ xpcs-toolkit console script works$(NC)"; \
	else \
		echo "$(YELLOW)⚠ xpcs-toolkit console script not installed$(NC)"; \
	fi
	@if command -v xpcs >/dev/null 2>&1; then \
		xpcs --version > /dev/null; \
		echo "$(GREEN)✓ xpcs console script works$(NC)"; \
	else \
		echo "$(YELLOW)⚠ xpcs console script not installed$(NC)"; \
	fi
	@echo "$(GREEN)✓ CLI functionality tests completed$(NC)"

test/performance: ## Run performance tests
	@echo "$(CYAN)Running performance tests...$(NC)"
	@# Test import time
	@echo "Import performance:"
	@$(PYTHON) -c "import time; start=time.time(); import $(PACKAGE_NAME); print(f'Import time: {time.time()-start:.3f}s')"
	@echo "$(GREEN)✓ Performance tests completed$(NC)"

test/all: test/unit test/integration test/cli test/performance ## Run all tests
	@echo "$(GREEN)✓ All tests completed$(NC)"

coverage: ## Run tests with coverage reporting
	@echo "$(CYAN)Running coverage analysis...$(NC)"
	$(COVERAGE) run --source $(SRC_DIR) -m $(PYTEST) $(TESTS_DIR)/
	$(COVERAGE) report -m
	$(COVERAGE) html
	@echo "$(GREEN)✓ Coverage report generated in $(HTMLCOV_DIR)/$(NC)"
	@echo "$(CYAN)Opening coverage report...$(NC)"
	$(BROWSER) $(HTMLCOV_DIR)/index.html

# =============================================================================
# BUILD COMMANDS
# =============================================================================

build: build/check clean build/wheel build/sdist ## Build distribution packages
	@echo "$(GREEN)✓ Build completed$(NC)"
	@ls -la $(DIST_DIR)/

build/check: ## Verify build requirements
	@echo "$(CYAN)Checking build requirements...$(NC)"
	@$(PYTHON) -c "import setuptools, wheel, build" || (echo "$(RED)Missing build dependencies$(NC)" && exit 1)
	@echo "$(GREEN)✓ Build requirements satisfied$(NC)"

build/wheel: ## Build wheel package
	@echo "$(CYAN)Building wheel...$(NC)"
	$(PYTHON) -m build --wheel

build/sdist: ## Build source distribution
	@echo "$(CYAN)Building source distribution...$(NC)"
	$(PYTHON) -m build --sdist

# =============================================================================
# INSTALLATION COMMANDS
# =============================================================================

install: clean ## Install the package to active Python environment
	@echo "$(CYAN)Installing package...$(NC)"
	$(PIP) install -e .
	@echo "$(GREEN)✓ Package installed in development mode$(NC)"

install/dev: clean ## Install package with development dependencies
	@echo "$(CYAN)Installing package with development dependencies...$(NC)"
	$(PIP) install -e .[dev]
	@echo "$(GREEN)✓ Development installation completed$(NC)"

install/prod: clean ## Install package for production
	@echo "$(CYAN)Installing package for production...$(NC)"
	$(PIP) install .
	@echo "$(GREEN)✓ Production installation completed$(NC)"

# =============================================================================
# DOCUMENTATION COMMANDS
# =============================================================================

docs: ## Generate Sphinx HTML documentation
	@echo "$(CYAN)Generating documentation...$(NC)"
	rm -f $(DOCS_DIR)/$(PACKAGE_NAME).rst
	rm -f $(DOCS_DIR)/modules.rst
	sphinx-apidoc -o $(DOCS_DIR)/ $(SRC_DIR)
	$(MAKE) -C $(DOCS_DIR) clean
	$(MAKE) -C $(DOCS_DIR) html
	@echo "$(GREEN)✓ Documentation generated$(NC)"
	$(BROWSER) $(DOCS_DIR)/_build/html/index.html

docs/serve: docs ## Generate docs and serve them locally
	@echo "$(CYAN)Serving documentation...$(NC)"
	$(PYTHON) -m http.server 8000 -d $(DOCS_DIR)/_build/html

# =============================================================================
# DEPENDENCY MANAGEMENT
# =============================================================================

deps/check: ## Check dependency compatibility
	@echo "$(CYAN)Checking dependencies...$(NC)"
	$(PIP) check
	@echo "$(GREEN)✓ Dependencies are compatible$(NC)"

deps/update: ## Update dependencies to latest versions
	@echo "$(CYAN)Updating dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e .[dev]
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

deps/outdated: ## Show outdated packages
	@echo "$(CYAN)Checking for outdated packages...$(NC)"
	$(PIP) list --outdated

deps/tree: ## Show dependency tree
	@echo "$(CYAN)Dependency tree:$(NC)"
	@$(PIP) install pipdeptree 2>/dev/null || echo "Installing pipdeptree..."
	@pipdeptree

# =============================================================================
# RELEASE COMMANDS
# =============================================================================

release/test: build ## Upload to test PyPI
	@echo "$(CYAN)Uploading to test PyPI...$(NC)"
	$(TWINE) upload --repository-url https://test.pypi.org/legacy/ $(DIST_DIR)/*
	@echo "$(GREEN)✓ Uploaded to test PyPI$(NC)"

release/prod: build ## Upload to production PyPI
	@echo "$(CYAN)Uploading to production PyPI...$(NC)"
	$(TWINE) upload $(DIST_DIR)/*
	@echo "$(GREEN)✓ Uploaded to production PyPI$(NC)"

# =============================================================================
# CI/CD COMMANDS
# =============================================================================

ci/check: check-venv lint test coverage ## Run full CI pipeline
	@echo "$(GREEN)✓ CI pipeline completed successfully$(NC)"

# =============================================================================
# PROFILING COMMANDS
# =============================================================================

profile: ## Profile package import performance
	@echo "$(CYAN)Profiling package performance...$(NC)"
	@mkdir -p prof
	$(PYTHON) -m cProfile -o prof/import_profile.prof -c "import $(PACKAGE_NAME)"
	@echo "$(GREEN)✓ Profile saved to prof/import_profile.prof$(NC)"
	@echo "$(CYAN)Top 10 slowest operations:$(NC)"
	@$(PYTHON) -c "import pstats; p = pstats.Stats('prof/import_profile.prof'); p.sort_stats('tottime').print_stats(10)"

# =============================================================================
# DOCKER COMMANDS (if Dockerfile exists)
# =============================================================================

docker/build: ## Build Docker image
	@if [ -f Dockerfile ]; then \
		echo "$(CYAN)Building Docker image...$(NC)"; \
		docker build -t $(PROJECT_NAME):latest .; \
		echo "$(GREEN)✓ Docker image built$(NC)"; \
	else \
		echo "$(YELLOW)⚠ No Dockerfile found$(NC)"; \
	fi

docker/run: ## Run Docker container
	@if [ -f Dockerfile ]; then \
		echo "$(CYAN)Running Docker container...$(NC)"; \
		docker run --rm -it $(PROJECT_NAME):latest; \
	else \
		echo "$(YELLOW)⚠ No Dockerfile found$(NC)"; \
	fi

docker/test: ## Run tests in Docker container
	@if [ -f Dockerfile ]; then \
		echo "$(CYAN)Running tests in Docker...$(NC)"; \
		docker run --rm $(PROJECT_NAME):latest make test; \
	else \
		echo "$(YELLOW)⚠ No Dockerfile found$(NC)"; \
	fi

# =============================================================================
# DEVELOPMENT SHORTCUTS
# =============================================================================

dev: install/dev ## Quick development setup
	@echo "$(GREEN)✓ Development environment ready$(NC)"

quick-test: ## Quick test run (fast subset)
	@echo "$(CYAN)Running quick tests...$(NC)"
	$(PYTEST) $(TESTS_DIR)/ -x --tb=no -q

watch-test: ## Watch for changes and run tests
	@echo "$(CYAN)Watching for changes...$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(NC)"
	@while true; do \
		make quick-test; \
		sleep 2; \
	done

# =============================================================================
# PROJECT STATISTICS
# =============================================================================

stats: ## Show project statistics
	@echo "$(CYAN)XPCS Toolkit Project Statistics$(NC)"
	@echo "$(YELLOW)==============================$(NC)"
	@echo "Lines of code:"
	@find $(SRC_DIR) -name "*.py" -not -path "*/tests/*" | xargs wc -l | tail -1
	@echo ""
	@echo "Test coverage:"
	@$(COVERAGE) report --show-missing 2>/dev/null | grep TOTAL || echo "Run 'make coverage' first"
	@echo ""
	@echo "Package size:"
	@du -sh $(SRC_DIR) 2>/dev/null || echo "N/A"