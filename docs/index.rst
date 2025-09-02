XPCS Toolkit: Advanced X-ray Photon Correlation Spectroscopy Analysis
=====================================================================

.. image:: https://badge.fury.io/py/xpcs-toolkit.svg
   :target: https://badge.fury.io/py/xpcs-toolkit
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.12%2B-blue
   :target: https://python.org
   :alt: Python Version

.. image:: https://readthedocs.org/projects/xpcs-toolkit/badge/?version=latest
   :target: https://xpcs-toolkit.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/coverage-93%25-brightgreen
   :target: https://github.com/imewei/xpcs-toolkit
   :alt: Coverage Status

Welcome to the XPCS Toolkit documentation! This comprehensive toolkit provides advanced analysis capabilities for X-ray Photon Correlation Spectroscopy (XPCS) experiments, designed specifically for synchrotron beamline operations and research environments.

🚀 **Quick Start**
------------------

Get started with XPCS Toolkit in just a few minutes:

.. code-block:: bash

   pip install xpcs-toolkit
   xpcs-toolkit --help

.. code-block:: python

   import xpcs_toolkit
   
   # Load XPCS data file
   data = xpcs_toolkit.XpcsDataFile('experiment.hdf5')
   
   # Initialize analysis kernel
   kernel = xpcs_toolkit.AnalysisKernel('/path/to/data/')
   kernel.build_file_list()

🔬 **Scientific Features**
--------------------------

.. grid:: 2

    .. grid-item-card:: Multi-tau Correlation Analysis
        :img-top: _static/correlation.png
        
        Advanced g2(q,t) correlation function processing with comprehensive statistical analysis and fitting capabilities.

    .. grid-item-card:: Two-time Correlation
        :img-top: _static/twotime.png
        
        Time-resolved correlation analysis for studying dynamic processes and temporal evolution.

    .. grid-item-card:: SAXS Visualization
        :img-top: _static/saxs.png
        
        Small-Angle X-ray Scattering pattern analysis with publication-quality visualization tools.

    .. grid-item-card:: Stability Monitoring
        :img-top: _static/stability.png
        
        Real-time beam stability assessment and quality control for experimental validation.

📖 **Documentation Contents**
-----------------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   faq

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guides/index
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/index
   contributing
   changelog

🎯 **Use Cases**
---------------

**Synchrotron Beamlines**
   Real-time analysis during experiments, quality control, automated processing pipelines, and remote monitoring capabilities.

**Research Applications**
   Soft matter dynamics studies, materials characterization, time-resolved measurements, and comparative analysis workflows.

**Production Environments**
   Batch processing of experimental datasets, reproducible analysis with version control, and integration with existing pipelines.

🏆 **Performance & Quality**
----------------------------

XPCS Toolkit is designed for production use with enterprise-grade quality standards:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Metric
     - Value
     - Notes
   * - Test Coverage
     - 93%
     - Comprehensive test suite with 446 tests
   * - Import Time
     - ~1.5s
     - Optimized with lazy loading
   * - Memory Usage
     - 50-200MB
     - Depends on dataset size
   * - Quality Gates
     - ✅ All Passing
     - Automated CI/CD validation

🏢 **Institutional Support**
---------------------------

Developed at **Argonne National Laboratory** with support from the U.S. Department of Energy, Office of Science, Office of Basic Energy Sciences.

.. image:: _static/anl-logo.png
   :width: 200px
   :align: center
   :alt: Argonne National Laboratory

📞 **Getting Help**
------------------

- **Issues**: Report bugs and request features on `GitHub Issues <https://github.com/imewei/xpcs-toolkit/issues>`_
- **Discussions**: Join the community on `GitHub Discussions <https://github.com/imewei/xpcs-toolkit/discussions>`_
- **Email**: Contact the development team at weichen@anl.gov
- **Documentation**: This comprehensive documentation with examples and guides

📜 **Citation**
--------------

If you use XPCS Toolkit in your research, please cite:

.. code-block:: bibtex

   @software{xpcs_toolkit,
     title={XPCS Toolkit: Advanced X-ray Photon Correlation Spectroscopy Analysis},
     author={Chen, Wei},
     institution={Argonne National Laboratory},
     year={2024},
     url={https://github.com/imewei/xpcs-toolkit}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`