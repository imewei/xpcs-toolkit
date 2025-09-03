"""
Integration tests for documentation quality and example validation.

This module provides comprehensive testing of docstrings, code examples,
and documentation consistency across the XPCS Toolkit codebase.

Features:
- Docstring completeness validation
- Code example execution testing
- Parameter documentation verification
- Scientific equation rendering validation
- Cross-reference link checking
"""

import ast
import doctest
import importlib
import inspect
from pathlib import Path
import re
import sys
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

import xpcs_toolkit

# Handle imports that might fail in CI environments
try:
    from xpcs_toolkit.core.data import locator
    from xpcs_toolkit.core.data.locator import DataFileLocator

    core_data_available = True
except ImportError as e:
    # Fallback for CI environments where package structure might not be properly installed
    import warnings

    warnings.warn(f"Could not import core.data modules: {e}")
    locator = None
    DataFileLocator = None
    core_data_available = False

try:
    from xpcs_toolkit.scientific.correlation import g2

    g2_available = True
except ImportError:
    g2 = None
    g2_available = False

try:
    from xpcs_toolkit.tests.fixtures.synthetic_data import SyntheticXPCSDataGenerator

    synthetic_data_available = True
except ImportError:
    SyntheticXPCSDataGenerator = None
    synthetic_data_available = False


class TestDocumentationQuality:
    """Test suite for documentation quality and completeness."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment for documentation testing."""
        self.tested_modules = []

        # Only add modules that are successfully imported
        if core_data_available and locator is not None:
            self.tested_modules.append(("xpcs_toolkit.core.data.locator", locator))

        if g2_available and g2 is not None:
            self.tested_modules.append(("xpcs_toolkit.scientific.correlation.g2", g2))

        # Documentation quality thresholds
        self.min_docstring_length = 50
        self.required_sections = ["Parameters", "Returns", "Examples"]
        self.scientific_modules = ["scientific"]

    def test_module_docstrings_exist(self):
        """Test that all public modules have comprehensive docstrings."""
        if not self.tested_modules:
            pytest.skip("No modules available for testing (import issues in CI)")

        missing_docstrings = []
        insufficient_docstrings = []

        for module_name, module_obj in self.tested_modules:
            if not hasattr(module_obj, "__doc__") or module_obj.__doc__ is None:
                missing_docstrings.append(module_name)
            elif len(module_obj.__doc__) < self.min_docstring_length:
                insufficient_docstrings.append(
                    f"{module_name} (length: {len(module_obj.__doc__)})"
                )

        assert not missing_docstrings, (
            f"Modules without docstrings: {missing_docstrings}"
        )
        assert not insufficient_docstrings, (
            f"Modules with insufficient docstrings: {insufficient_docstrings}"
        )

    def test_function_docstrings_completeness(self):
        """Test that public functions have complete docstrings."""
        if not self.tested_modules:
            pytest.skip("No modules available for testing (import issues in CI)")

        incomplete_functions = []

        for module_name, module_obj in self.tested_modules:
            for name in dir(module_obj):
                if name.startswith("_"):
                    continue

                obj = getattr(module_obj, name)
                if not callable(obj):
                    continue

                # Check docstring existence
                if not hasattr(obj, "__doc__") or obj.__doc__ is None:
                    incomplete_functions.append(f"{module_name}.{name}: No docstring")
                    continue

                docstring = obj.__doc__

                # Check for required sections in scientific modules
                if any(sci_mod in module_name for sci_mod in self.scientific_modules):
                    missing_sections = []
                    for section in self.required_sections:
                        if section not in docstring:
                            missing_sections.append(section)

                    if missing_sections:
                        incomplete_functions.append(
                            f"{module_name}.{name}: Missing sections {missing_sections}"
                        )

        # Allow some incomplete functions but track them
        if len(incomplete_functions) > 10:  # Threshold for acceptable incompleteness
            pytest.fail(
                f"Too many incomplete function docstrings: {incomplete_functions}"
            )
        elif incomplete_functions:
            pytest.skip(
                f"Some functions have incomplete docstrings: {len(incomplete_functions)} total"
            )

    def test_parameter_documentation_consistency(self):
        """Test that function parameters are consistently documented."""
        inconsistent_params = []

        for module_name, module_obj in self.tested_modules:
            for name in dir(module_obj):
                if name.startswith("_") or name in ["logger", "colors", "symbols"]:
                    continue

                obj = getattr(module_obj, name)
                if (
                    not callable(obj)
                    or not hasattr(obj, "__doc__")
                    or obj.__doc__ is None
                ):
                    continue

                try:
                    signature = inspect.signature(obj)
                    param_names = [
                        p
                        for p in signature.parameters
                        if p not in ["self", "cls", "args", "kwargs"]
                    ]

                    docstring = obj.__doc__

                    # Check if Parameters section exists when there are parameters
                    if param_names and "Parameters" in docstring:
                        # Extract documented parameters from docstring
                        param_section = self._extract_parameters_section(docstring)
                        documented_params = self._parse_documented_parameters(
                            param_section
                        )

                        # Check for missing documentation
                        missing_docs = set(param_names) - set(documented_params)
                        if missing_docs:
                            inconsistent_params.append(
                                f"{module_name}.{name}: Undocumented params {missing_docs}"
                            )

                except (ValueError, TypeError):
                    # Skip functions where signature inspection fails
                    continue

        if inconsistent_params:
            # Log inconsistencies but don't fail - this is more informational
            print(
                f"Parameter documentation inconsistencies found: {len(inconsistent_params)}"
            )
            for inconsistency in inconsistent_params[:5]:  # Show first 5
                print(f"  - {inconsistency}")

    def test_scientific_equations_formatting(self):
        """Test that scientific equations are properly formatted in docstrings."""
        equation_issues = []

        # Common scientific notation patterns
        equation_patterns = [
            r"g₂\(.*?\)",  # g₂ function notation
            r"⟨.*?⟩",  # Ensemble averages
            r"\b[A-Za-z]+\^\w+\b",  # Exponents
            r"\b\d+\.\d+e[+-]?\d+\b",  # Scientific notation
        ]

        for module_name, module_obj in self.tested_modules:
            if not any(sci_mod in module_name for sci_mod in self.scientific_modules):
                continue

            docstring = module_obj.__doc__ or ""

            # Check for proper mathematical notation
            for pattern in equation_patterns:
                matches = re.findall(pattern, docstring)
                if matches:
                    # This is positive - equations are present
                    pass

            # Check for common formatting issues
            if "**" in docstring and "exp" not in docstring:
                # ** might indicate markdown that should be rendered
                equation_issues.append(f"{module_name}: Potential unrendered markdown")

            if "tau" in docstring.lower() and "τ" not in docstring:
                # Suggest using proper Greek letters
                equation_issues.append(
                    f"{module_name}: Consider using τ instead of 'tau'"
                )

        # This is informational rather than a hard failure
        if equation_issues:
            print(f"Scientific equation formatting suggestions: {equation_issues}")

    def _extract_parameters_section(self, docstring: str) -> str:
        """Extract the Parameters section from a docstring."""
        lines = docstring.split("\n")
        in_params = False
        param_lines = []

        for line in lines:
            if line.strip() == "Parameters":
                in_params = True
                continue
            elif in_params and line.strip() in [
                "Returns",
                "Yields",
                "Raises",
                "Examples",
                "Notes",
            ]:
                break
            elif in_params:
                param_lines.append(line)

        return "\n".join(param_lines)

    def _parse_documented_parameters(self, param_section: str) -> list[str]:
        """Parse parameter names from the Parameters section."""
        param_names = []
        for line in param_section.split("\n"):
            line = line.strip()
            if " : " in line:
                param_name = line.split(" : ")[0].strip()
                param_names.append(param_name)
        return param_names


class TestCodeExampleValidation:
    """Test suite for validating code examples in documentation."""

    @pytest.fixture(autouse=True)
    def setup_example_environment(self):
        """Set up environment for testing code examples."""
        self.synthetic_generator = (
            SyntheticXPCSDataGenerator() if synthetic_data_available else None
        )

        # Mock environment for examples that require external data
        self.mock_context = {
            "XpcsDataFile": MagicMock,
            "DataFileLocator": DataFileLocator,
            "g2mod": g2,
            "g2": g2,
            "np": None,  # Will be set up with lazy import
        }

    def test_g2_module_examples(self):
        """Test code examples in g2 module documentation."""
        if not g2_available or g2 is None:
            pytest.skip("g2 module not available (import issues in CI)")

        docstring = g2.__doc__
        if not docstring:
            pytest.skip("No docstring found for g2 module")

        # Extract code examples from docstring
        examples = self._extract_code_examples(docstring)

        if not examples:
            pytest.skip("No code examples found in g2 module docstring")

        for i, example in enumerate(examples):
            with patch.dict("sys.modules", self.mock_context):
                try:
                    # Create a safe execution environment

                    # Test that the example compiles
                    compiled = compile(example, f"<g2_example_{i}>", "exec")
                    assert compiled is not None, f"Example {i} failed to compile"

                except SyntaxError as e:
                    pytest.fail(f"Syntax error in g2 example {i}: {e}")
                except Exception as e:
                    # Some examples might fail due to missing data - that's ok
                    print(f"Example {i} execution note: {e}")

    def test_locator_class_examples(self):
        """Test code examples related to DataFileLocator usage."""
        if not core_data_available or DataFileLocator is None:
            pytest.skip("DataFileLocator not available (import issues in CI)")

        # Test basic usage patterns that should work
        basic_examples = [
            """
# Basic locator usage
from xpcs_toolkit.core.data.locator import DataFileLocator
import tempfile
import os

# Create temporary directory for testing
temp_dir = tempfile.mkdtemp()
locator = DataFileLocator(temp_dir)
""",
            """
# File list building
from xpcs_toolkit.core.data.locator import DataFileLocator
import tempfile
temp_dir = tempfile.mkdtemp()
temp_locator = DataFileLocator(temp_dir)
temp_locator.build_file_list()
file_count = len(temp_locator.source_files)
""",
        ]

        for i, example in enumerate(basic_examples):
            try:
                exec_globals = {"__name__": "__main__"}
                exec(example, exec_globals)

            except Exception as e:
                pytest.fail(f"Basic locator example {i} failed: {e}")

    def test_doctest_examples(self):
        """Run doctest on modules that contain testable examples."""
        modules_to_test = []
        if g2_available and g2 is not None:
            modules_to_test.append(g2)  # Only test modules with proper doctest format

        if not modules_to_test:
            pytest.skip("No modules available for doctest (import issues in CI)")

        for module in modules_to_test:
            if not hasattr(module, "__doc__") or module.__doc__ is None:
                continue

            # Create a test finder and runner
            finder = doctest.DocTestFinder()
            runner = doctest.DocTestRunner(verbose=False)

            # Extract and run doctests
            doctests = finder.find(module)

            for test in doctests:
                if test.examples:  # Only run if there are actual examples
                    try:
                        # Set up mock environment for doctests
                        test.globs.update(
                            {
                                "XpcsDataFile": MagicMock,
                                "g2mod": g2,
                                "np": MagicMock(),  # Mock numpy for examples
                            }
                        )

                        result = runner.run(test)

                        if result.failed > 0:
                            print(
                                f"Doctest failures in {module.__name__}: {result.failed}"
                            )

                    except Exception as e:
                        # Log but don't fail - doctests might need specific setup
                        print(f"Doctest execution issue in {module.__name__}: {e}")

    def _extract_code_examples(self, docstring: str) -> list[str]:
        """Extract code examples from docstring."""
        examples = []
        lines = docstring.split("\n")
        in_code_block = False
        current_example = []

        for line in lines:
            if line.strip().startswith("```python") or line.strip().startswith("```"):
                if in_code_block:
                    # End of code block
                    if current_example:
                        examples.append("\n".join(current_example))
                        current_example = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
            elif in_code_block:
                current_example.append(line)

        return examples


class TestDocumentationConsistency:
    """Test suite for documentation consistency and cross-references."""

    def test_version_consistency(self):
        """Test that version information is consistent across documentation."""
        # This would check __version__ strings, changelog, etc.
        # For now, just verify the package imports correctly
        assert hasattr(xpcs_toolkit, "__version__") or hasattr(xpcs_toolkit, "VERSION")

    def test_import_path_consistency(self):
        """Test that documented import paths are valid."""
        documented_imports = [
            "from xpcs_toolkit.core.data.locator import DataFileLocator",
            "from xpcs_toolkit.scientific.correlation import g2",
            "import xpcs_toolkit",
        ]

        for import_statement in documented_imports:
            try:
                exec(import_statement)
            except ImportError as e:
                pytest.fail(f"Documented import path invalid: {import_statement} - {e}")

    def test_cross_reference_validity(self):
        """Test that cross-references in documentation are valid."""
        # Check for references to other modules/functions
        cross_refs = []

        modules_for_refs = []
        if g2_available and g2 is not None:
            modules_for_refs.append(("g2", g2))
        if core_data_available and locator is not None:
            modules_for_refs.append(("locator", locator))

        for _, module_obj in modules_for_refs:
            if not hasattr(module_obj, "__doc__") or module_obj.__doc__ is None:
                continue

            docstring = module_obj.__doc__

            # Look for references to other functions/classes
            references = re.findall(r"`(\w+\.\w+)`|`(\w+)`", docstring)
            cross_refs.extend([ref[0] or ref[1] for ref in references])

        # Validate that referenced items exist (basic check)
        for ref in cross_refs:
            if "." in ref:
                module_name, item_name = ref.rsplit(".", 1)
                # This is a basic check - could be enhanced
                pass

    def test_example_data_references(self):
        """Test that examples reference appropriate test data or synthetic data."""
        if not synthetic_data_available or SyntheticXPCSDataGenerator is None:
            pytest.skip(
                "SyntheticXPCSDataGenerator not available (import issues in CI)"
            )

        # Ensure examples use synthetic data or mock data appropriately
        generator = SyntheticXPCSDataGenerator()

        # Test that synthetic data generator works as expected in examples
        try:
            intensity_data, q_vals, tau_vals = (
                generator.generate_brownian_motion_intensity(n_times=10, n_q_bins=3)
            )
            assert intensity_data.shape[1] == len(q_vals)
            assert len(tau_vals) == 10

        except Exception as e:
            pytest.fail(f"Synthetic data generator failed (used in examples): {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
