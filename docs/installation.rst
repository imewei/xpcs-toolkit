Installation Guide
==================

This guide covers various installation methods for XPCS Toolkit, from quick pip installs to development setups.

Quick Installation
------------------

The fastest way to get started with XPCS Toolkit:

.. code-block:: bash

   pip install xpcs-toolkit

This installs the core functionality with minimal dependencies.

Requirements
------------

**Python Version**

XPCS Toolkit requires Python 3.12 or higher.

.. code-block:: bash

   python --version  # Should be 3.12+

**System Requirements**

- **Memory**: 4GB RAM minimum, 8GB+ recommended for large datasets
- **Storage**: 100MB for installation, additional space for data processing
- **OS**: Linux, macOS, or Windows (64-bit)

Installation Methods
--------------------

Standard Installation
~~~~~~~~~~~~~~~~~~~~~

Install XPCS Toolkit with core dependencies:

.. code-block:: bash

   pip install xpcs-toolkit

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

XPCS Toolkit offers several optional dependency groups for extended functionality:

.. code-block:: bash

   # Development tools (testing, linting, etc.)
   pip install xpcs-toolkit[dev]

   # Documentation tools
   pip install xpcs-toolkit[docs]

   # Performance optimizations (Numba, Cython)
   pip install xpcs-toolkit[performance]

   # Extended scientific computing (Dask, xarray)
   pip install xpcs-toolkit[extended]

   # All optional dependencies
   pip install xpcs-toolkit[all]

Conda Installation
~~~~~~~~~~~~~~~~~~

XPCS Toolkit will be available on conda-forge:

.. code-block:: bash

   # Coming soon
   conda install -c conda-forge xpcs-toolkit

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For contributing to XPCS Toolkit or running the latest development version:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/imewei/xpcs-toolkit.git
   cd xpcs-toolkit

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

   # Install in development mode
   pip install -e .[dev]

Minimal Installation
~~~~~~~~~~~~~~~~~~~~

For containerized deployments or embedded systems:

.. code-block:: bash

   pip install -r requirements-minimal.txt

This installs only the essential dependencies for core functionality.

Virtual Environments
--------------------

We strongly recommend using virtual environments to avoid dependency conflicts:

Using venv
~~~~~~~~~~

.. code-block:: bash

   # Create virtual environment
   python -m venv xpcs-env
   
   # Activate (Linux/macOS)
   source xpcs-env/bin/activate
   
   # Activate (Windows)
   xpcs-env\\Scripts\\activate
   
   # Install XPCS Toolkit
   pip install xpcs-toolkit
   
   # Deactivate when done
   deactivate

Using conda
~~~~~~~~~~~

.. code-block:: bash

   # Create conda environment
   conda create -n xpcs-env python=3.12
   conda activate xpcs-env
   
   # Install XPCS Toolkit
   pip install xpcs-toolkit
   
   # Deactivate when done
   conda deactivate

Verification
------------

Test your installation:

.. code-block:: bash

   # Check installation
   xpcs-toolkit --version
   
   # Run basic functionality test
   python -c "import xpcs_toolkit; print('✅ Installation successful!')"

Performance Validation
~~~~~~~~~~~~~~~~~~~~~~

Verify performance optimizations (if installed):

.. code-block:: python

   import xpcs_toolkit
   import time
   
   # Test import performance
   start = time.time()
   import xpcs_toolkit
   print(f"Import time: {time.time() - start:.2f}s")
   
   # Should be < 2 seconds with lazy loading

Common Installation Issues
--------------------------

HDF5 Dependencies
~~~~~~~~~~~~~~~~~

If you encounter HDF5-related issues:

.. code-block:: bash

   # Install HDF5 system libraries first
   # Ubuntu/Debian:
   sudo apt-get install libhdf5-dev
   
   # macOS:
   brew install hdf5
   
   # Then reinstall h5py
   pip install --no-cache-dir h5py

Scientific Libraries
~~~~~~~~~~~~~~~~~~~~

For issues with NumPy, SciPy, or other scientific libraries:

.. code-block:: bash

   # Use conda for scientific stack
   conda install numpy scipy matplotlib scikit-learn
   pip install xpcs-toolkit --no-deps

Windows Issues
~~~~~~~~~~~~~~

On Windows, you may need Microsoft Visual C++ Build Tools:

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install with "C++ build tools" workload
3. Restart your terminal and retry installation

Memory Issues
~~~~~~~~~~~~~

For large dataset processing, increase available memory:

.. code-block:: python

   # Monitor memory usage
   import psutil
   print(f"Available memory: {psutil.virtual_memory().available / 1e9:.1f} GB")

Docker Installation
-------------------

Use XPCS Toolkit in containerized environments:

.. code-block:: dockerfile

   FROM python:3.12-slim
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       libhdf5-dev \
       pkg-config \
       && rm -rf /var/lib/apt/lists/*
   
   # Install XPCS Toolkit
   RUN pip install xpcs-toolkit[performance]
   
   WORKDIR /workspace
   CMD ["xpcs-toolkit", "--help"]

Build and run:

.. code-block:: bash

   docker build -t xpcs-toolkit .
   docker run -v /path/to/data:/workspace/data xpcs-toolkit

Jupyter Environment
-------------------

For interactive analysis in Jupyter:

.. code-block:: bash

   pip install xpcs-toolkit jupyter
   jupyter lab

Then in a notebook:

.. code-block:: python

   import xpcs_toolkit
   
   # Enable inline plotting
   %matplotlib inline
   
   # Load and analyze data
   data = xpcs_toolkit.XpcsDataFile('experiment.hdf5')

Upgrading
---------

Upgrade to the latest version:

.. code-block:: bash

   pip install --upgrade xpcs-toolkit

Check for breaking changes in the :doc:`changelog` before upgrading.

Uninstallation
--------------

To completely remove XPCS Toolkit:

.. code-block:: bash

   pip uninstall xpcs-toolkit
   
   # Remove configuration files (optional)
   rm -rf ~/.config/xpcs-toolkit

Next Steps
----------

- Follow the :doc:`quickstart` tutorial
- Explore the API documentation
- Check out :doc:`guides/index` for detailed examples