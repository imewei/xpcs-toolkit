Frequently Asked Questions
==========================

This page answers common questions about XPCS Toolkit installation, usage, and troubleshooting.

Installation and Setup
----------------------

**Q: What Python version is required?**

A: XPCS Toolkit requires Python 3.12 or higher. We recommend using the latest stable Python version for best performance and compatibility.

**Q: How do I install XPCS Toolkit?**

A: The easiest way is via pip:

.. code-block:: bash

   pip install xpcs-toolkit

For development or the latest features:

.. code-block:: bash

   git clone https://github.com/imewei/xpcs-toolkit.git
   cd xpcs-toolkit
   pip install -e .[dev]

**Q: I'm getting HDF5 errors during installation. What should I do?**

A: HDF5 issues are common. Try these solutions:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install libhdf5-dev pkg-config
   
   # macOS  
   brew install hdf5
   
   # Then reinstall h5py
   pip install --no-cache-dir h5py
   pip install xpcs-toolkit

**Q: Can I use XPCS Toolkit with conda?**

A: While XPCS Toolkit isn't yet on conda-forge, you can use conda for the scientific stack:

.. code-block:: bash

   conda create -n xpcs python=3.12
   conda activate xpcs
   conda install numpy scipy matplotlib scikit-learn h5py
   pip install xpcs-toolkit

File Formats and Data
---------------------

**Q: What file formats does XPCS Toolkit support?**

A: XPCS Toolkit supports:
- APS 8-ID-I NeXus format (modern HDF5-based format)
- Legacy XPCS HDF5 format 
- Automatic format detection

**Q: How do I know if my file is compatible?**

A: Try loading it:

.. code-block:: python

   import xpcs_toolkit
   
   try:
       data = xpcs_toolkit.XpcsDataFile('your_file.hdf5')
       print(f"✅ Compatible! Analysis type: {data.analysis_type}")
   except Exception as e:
       print(f"❌ Not compatible: {e}")

**Q: My file loads but I get AttributeError when accessing data. Why?**

A: Different analysis types contain different data. Check what's available:

.. code-block:: python

   data = xpcs_toolkit.XpcsDataFile('your_file.hdf5')
   
   # Check analysis type
   print(f"Analysis type: {data.analysis_type}")
   
   # List available attributes
   available = [attr for attr in dir(data) 
                if not attr.startswith('_') and hasattr(data, attr)]
   print("Available data:", available)

**Q: How do I convert legacy files to the new format?**

A: Currently, XPCS Toolkit reads legacy files directly. If you need format conversion, please open a GitHub issue with your specific requirements.

Analysis and Usage
------------------

**Q: How do I analyze correlation functions (g2)?**

A: For correlation analysis:

.. code-block:: bash

   # Command line
   xpcs-toolkit g2 /path/to/data/ --qmin 0.01 --qmax 0.1

.. code-block:: python

   # Python API
   data = xpcs_toolkit.XpcsDataFile('multitau_file.hdf5')
   if hasattr(data, 'g2'):
       g2_data = data.g2
       tau_data = data.tau
       # Your analysis here

**Q: How do I create SAXS plots?**

A: For SAXS visualization:

.. code-block:: bash

   # 2D patterns
   xpcs-toolkit saxs2d /path/to/data/ --scale log --colormap viridis
   
   # 1D profiles  
   xpcs-toolkit saxs1d /path/to/data/ --log-x --log-y

**Q: Can I batch process multiple files?**

A: Yes! Use the AnalysisKernel:

.. code-block:: python

   kernel = xpcs_toolkit.AnalysisKernel('/path/to/data/')
   kernel.build_file_list()
   files = kernel.get_selected_files()
   
   for file_path in files:
       data = xpcs_toolkit.XpcsDataFile(file_path)
       # Process each file

**Q: How do I export analysis results?**

A: Multiple options:

.. code-block:: python

   import numpy as np
   
   data = xpcs_toolkit.XpcsDataFile('file.hdf5')
   
   # Export correlation data
   if hasattr(data, 'g2'):
       np.savetxt('g2_data.txt', data.g2, header='G2 correlation functions')
   
   # Export SAXS profile
   if hasattr(data, 'saxs_1d'):
       q = data.saxs_1d['q']
       I = data.saxs_1d['Iq'][0]
       np.savetxt('saxs_profile.txt', np.column_stack([q, I]),
                  header='q(A^-1) I(q)')

Performance and Memory
----------------------

**Q: XPCS Toolkit is using too much memory. What can I do?**

A: Try these strategies:

1. **Process files individually**:

.. code-block:: python

   # Instead of loading all at once
   for file_path in file_list:
       data = xpcs_toolkit.XpcsDataFile(file_path)
       # Process and release
       del data

2. **Monitor memory usage**:

.. code-block:: python

   import psutil
   
   process = psutil.Process()
   memory_mb = process.memory_info().rss / 1024 / 1024
   print(f"Memory usage: {memory_mb:.1f} MB")

3. **Use lazy loading** (automatic in XPCS Toolkit)

**Q: Why is import slow?**

A: XPCS Toolkit uses lazy loading to minimize import time. First import may take 1-2 seconds, but subsequent imports are much faster. For even faster startup, use:

.. code-block:: python

   # Import only what you need
   from xpcs_toolkit import XpcsDataFile  # Faster than full import

**Q: How can I speed up analysis?**

A: Performance tips:

- Install performance dependencies: ``pip install xpcs-toolkit[performance]``
- Use NumPy vectorized operations
- Process in batches rather than individual files
- Consider using Numba-accelerated functions where available

Common Errors
-------------

**Q: I get "ImportError: No module named 'xpcs_toolkit'"**

A: Check your installation:

.. code-block:: bash

   pip list | grep xpcs
   
   # If not found, install:
   pip install xpcs-toolkit

**Q: I get "FileNotFoundError" even though the file exists**

A: Check:

1. File path is correct (use absolute paths)
2. File permissions are readable
3. File is not corrupted

.. code-block:: python

   from pathlib import Path
   
   file_path = Path('your_file.hdf5')
   print(f"Exists: {file_path.exists()}")
   print(f"Readable: {file_path.is_file()}")
   print(f"Size: {file_path.stat().st_size} bytes")

**Q: I get "ValueError: Unable to determine file format"**

A: The file format isn't recognized. Check:

1. Is it an HDF5 file? Try: ``h5dump -H your_file.hdf5``
2. Does it contain XPCS data structures?
3. Is the file complete (not truncated)?

**Q: Plots are not displaying in Jupyter notebooks**

A: Enable matplotlib backend:

.. code-block:: python

   %matplotlib inline
   import matplotlib.pyplot as plt
   
   # Or for interactive plots:
   %matplotlib widget

Development and Contributing
----------------------------

**Q: How do I contribute to XPCS Toolkit?**

A: We welcome contributions! See our :doc:`contributing` guide. Quick steps:

1. Fork the repository
2. Install development dependencies: ``pip install -e .[dev]``
3. Make your changes with tests
4. Run quality checks: ``make lint`` and ``make test``
5. Submit a pull request

**Q: How do I report bugs or request features?**

A: Please use GitHub Issues:

- **Bugs**: Include error messages, system info, and minimal reproduction example
- **Features**: Describe the use case and expected behavior
- **Questions**: Check this FAQ first, then open a discussion

**Q: How do I run the test suite?**

A: For development:

.. code-block:: bash

   # Install dev dependencies
   pip install -e .[dev]
   
   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=xpcs_toolkit
   
   # Run specific test
   pytest xpcs_toolkit/tests/test_analysis_kernel.py

**Q: Where can I find example data for testing?**

A: Currently, XPCS Toolkit is designed for APS 8-ID-I data formats. If you need example files for development or testing, please open a GitHub issue.

Integration and Advanced Usage
------------------------------

**Q: Can I use XPCS Toolkit with other scientific Python libraries?**

A: Absolutely! XPCS Toolkit integrates well with:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from scipy import optimize
   
   data = xpcs_toolkit.XpcsDataFile('file.hdf5')
   
   # Convert to pandas DataFrame
   if hasattr(data, 'g2'):
       df = pd.DataFrame(data.g2.T, columns=[f'q_{i}' for i in range(data.g2.shape[0])])
       df['tau'] = data.tau

**Q: How do I use XPCS Toolkit in a cluster environment?**

A: For HPC/cluster usage:

.. code-block:: python

   # Use absolute paths
   import os
   data_path = os.path.abspath('/path/to/data/')
   
   # Process in batches to manage memory
   # Consider using job arrays for parallel processing
   # Cache results to avoid recomputation

**Q: Can I extend XPCS Toolkit with custom analysis functions?**

A: Yes! The modular design allows extensions:

.. code-block:: python

   import xpcs_toolkit
   import numpy as np
   
   def custom_analysis(data):
       """Your custom analysis function."""
       if hasattr(data, 'g2'):
           # Your analysis here
           return results
   
   # Use with loaded data
   data = xpcs_toolkit.XpcsDataFile('file.hdf5')
   results = custom_analysis(data)

Getting Help
------------

**Q: Where can I get more help?**

A: Multiple resources available:

- **Documentation**: This comprehensive guide
- **GitHub Issues**: https://github.com/imewei/xpcs-toolkit/issues
- **GitHub Discussions**: https://github.com/imewei/xpcs-toolkit/discussions  
- **Email**: weichen@anl.gov for direct contact

**Q: How do I stay updated on new releases?**

A: - Watch the GitHub repository for notifications
- Check PyPI for new versions: https://pypi.org/project/xpcs-toolkit/
- Follow release announcements

**Q: Is commercial support available?**

A: XPCS Toolkit is developed at Argonne National Laboratory. For institutional partnerships or specialized support, contact weichen@anl.gov.

Still have questions?
-----------------------

If your question isn't answered here:

1. Check the API documentation for detailed function documentation
2. Browse the :doc:`guides/index` for in-depth examples  
3. Search existing `GitHub Issues <https://github.com/imewei/xpcs-toolkit/issues>`_
4. Open a new issue or discussion

We're here to help! 🚀