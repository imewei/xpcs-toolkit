"""
Configuration file for the Sphinx documentation builder.

This file contains the configuration needed for Sphinx documentation generation,
optimized for scientific Python packages and ReadTheDocs integration.
"""

from datetime import datetime
from pathlib import Path
import sys

# Add the project root to the Python path for autodoc
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import package info
try:
    import xpcs_toolkit

    release = xpcs_toolkit.__version__
except ImportError:
    release = "development"

# -- Project information -----------------------------------------------------

project = "XPCS Toolkit"
author = "Wei Chen"
copyright = f"{datetime.now().year}, Argonne National Laboratory"

# The short X.Y version
version = ".".join(release.split(".")[:2]) if "." in release else release
# The full version, including alpha/beta/rc tags
release = release

# -- General configuration ---------------------------------------------------

# Extensions to enable
extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
]

# Try to add optional extensions that are available
optional_required_extensions = [
    "myst_parser",  # Markdown support
    "sphinx_copybutton",  # Copy code button
    "sphinx_design",  # Modern UI components
]

for ext in optional_required_extensions:
    try:
        __import__(ext)
        extensions.append(ext)
        print(f"✅ Loaded extension: {ext}")
    except ImportError as e:
        print(f"⚠️  Extension {ext} not available: {e}")
        # Add fallback behavior if needed
        if ext == "sphinx_design":
            print("   Using basic styling without sphinx_design")

# Optional extensions (install if available)
optional_extensions = [
    "nbsphinx",  # Jupyter notebook support
    "sphinx_gallery.gen_gallery",  # Gallery generation
]

for ext in optional_extensions:
    try:
        __import__(ext)
        extensions.append(ext)
        print(f"✅ Loaded optional extension: {ext}")
    except (ImportError, OSError, PermissionError) as e:
        print(f"⚠️  Skipping optional extension {ext}: {type(e).__name__}: {e}")
        pass

# Add any paths that contain templates here, relative to this directory
templates_path = ["_templates"]

# Source file suffixes
source_suffix = {
    ".rst": None,
    ".md": "markdown",
}

# The root document
root_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = "furo"

# Theme options are theme-specific and customize the look and feel
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#2196F3",
        "color-brand-content": "#1976D2",
    },
    "dark_css_variables": {
        "color-brand-primary": "#42A5F5",
        "color-brand-content": "#64B5F6",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/imewei/xpcs-toolkit",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/xpcs-toolkit/",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 24 24">
                    <path d="M12 2A10 10 0 0 0 2 12a10 10 0 0 0 10 10 10 10 0 0 0 10-10A10 10 0 0 0 12 2zM12 20a8 8 0 0 1-8-8 8 8 0 0 1 8-8 8 8 0 0 1 8 8 8 8 0 0 1-8 8z"></path>
                    <path d="M12 6a6 6 0 0 0-6 6 6 6 0 0 0 6 6 6 6 0 0 0 6-6 6 6 0 0 0-6-6zm0 10a4 4 0 0 1-4-4 4 4 0 0 1 4-4 4 4 0 0 1 4 4 4 4 0 0 1-4 4z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Add any paths that contain custom static files (such as style sheets) here
html_static_path = ["_static"]

# Custom CSS files
html_css_files = []

# Custom JS files
html_js_files = []

# The name of the Pygments (syntax highlighting) style to use
pygments_style = "sphinx"
pygments_dark_style = "monokai"

# HTML title
html_title = f"{project} {version}"

# HTML short title
html_short_title = "XPCS Toolkit"

# Logo (commented out until available)
# html_logo = "_static/logo.png"

# Favicon (commented out until available)
# html_favicon = "_static/favicon.ico"

# -- Options for autodoc ----------------------------------------------------

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Mock modules that might not be available during doc build
autodoc_mock_imports = [
    "PyQt5",
    "pyqtgraph",
    "numba",
]

# Automatically extract typehints
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# -- Options for autosummary ------------------------------------------------

autosummary_generate = True
autosummary_imported_members = True

# -- Options for Napoleon (Google/NumPy docstrings) ------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for intersphinx ------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
}

# -- Options for copy button ------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

# -- Options for MyST parser ------------------------------------------------

myst_enable_extensions = [
    "deflist",
    "tasklist",
    "colon_fence",
    "fieldlist",
    "html_admonition",
    "html_image",
]

# -- Options for LaTeX output -----------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": r"""
\usepackage{charter}
\usepackage[defaultsans]{lato}
\usepackage{inconsolata}
""",
}

# -- Options for manual page output -----------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        root_doc,
        "xpcs-toolkit",
        "XPCS Toolkit Documentation",
        [author],
        1,
    )
]

# -- Options for Texinfo output ---------------------------------------------

# Grouping the document tree into Texinfo files
texinfo_documents = [
    (
        root_doc,
        "XPCSToolkit",
        "XPCS Toolkit Documentation",
        author,
        "XPCSToolkit",
        "Advanced X-ray Photon Correlation Spectroscopy analysis toolkit",
        "Miscellaneous",
    ),
]

# -- Options for epub output ------------------------------------------------

epub_title = project
epub_exclude_files = ["search.html"]

# -- Custom setup -----------------------------------------------------------


def setup(app):
    """Custom Sphinx setup function."""
    # Add custom CSS
    app.add_css_file("custom.css")

    # Add custom JavaScript
    # app.add_js_file("custom.js")

    # Custom build hooks
    app.connect("build-finished", on_build_finished)


def on_build_finished(app, exception):
    """Hook called when build is finished."""
    if exception is None:
        print("Documentation build completed successfully!")
    else:
        print(f"Documentation build failed: {exception}")
