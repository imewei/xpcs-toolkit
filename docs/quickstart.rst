Quick Start Tutorial
===================

This tutorial will get you analyzing XPCS data in under 5 minutes. We'll cover the most common workflows and essential features.

Prerequisites
-------------

Make sure you have XPCS Toolkit installed:

.. code-block:: bash

   pip install xpcs-toolkit

Basic Usage
-----------

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

The fastest way to explore your data:

.. code-block:: bash

   # Display help and available commands
   xpcs-toolkit --help
   
   # List available data files in a directory
   xpcs-toolkit list /path/to/your/data/
   
   # Quick SAXS 2D visualization
   xpcs-toolkit saxs2d /path/to/your/data/ --output saxs_pattern.png

Python API
~~~~~~~~~~

For programmatic analysis and integration:

.. code-block:: python

   import xpcs_toolkit
   
   # Load an XPCS data file
   data = xpcs_toolkit.XpcsDataFile('your_experiment.hdf5')
   
   # Display basic file information
   print(f"File type: {data.analysis_type}")
   print(f"Beam energy: {data.X_energy} keV")

Working with Data Files
-----------------------

File Discovery
~~~~~~~~~~~~~~

Use the AnalysisKernel to work with directories of files:

.. code-block:: python

   # Initialize analysis kernel for a data directory
   kernel = xpcs_toolkit.AnalysisKernel('/path/to/your/data/')
   
   # Build list of available files
   kernel.build_file_list()
   
   # Get selected files for analysis
   files = kernel.get_selected_files()
   print(f"Found {len(files)} XPCS files")

File Information
~~~~~~~~~~~~~~~~

Extract metadata and experiment parameters:

.. code-block:: python

   # Load specific file
   data = xpcs_toolkit.XpcsDataFile('experiment.hdf5')
   
   # Access experiment metadata
   print(f"Detector distance: {data.det_dist} mm")
   print(f"Beam center: ({data.bcx}, {data.bcy})")
   print(f"Sample name: {data.sample_name}")

Common Analysis Tasks
---------------------

1. SAXS Pattern Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate 2D scattering patterns:

.. code-block:: bash

   # Command line - linear scale
   xpcs-toolkit saxs2d /path/to/data/ --scale linear --colormap viridis
   
   # Command line - log scale with custom range
   xpcs-toolkit saxs2d /path/to/data/ --scale log --vmin 1 --vmax 1000

.. code-block:: python

   # Python API
   import matplotlib.pyplot as plt
   
   data = xpcs_toolkit.XpcsDataFile('experiment.hdf5')
   
   # Access 2D SAXS data
   saxs_2d = data.saxs_2d
   
   # Simple visualization
   plt.figure(figsize=(8, 6))
   plt.imshow(saxs_2d, origin='lower', cmap='viridis')
   plt.colorbar(label='Intensity')
   plt.title('SAXS 2D Pattern')
   plt.show()

2. G2 Correlation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze temporal correlation functions:

.. code-block:: bash

   # Command line analysis
   xpcs-toolkit g2 /path/to/data/ --qmin 0.01 --qmax 0.1 --output g2_analysis.png

.. code-block:: python

   # Python API
   data = xpcs_toolkit.XpcsDataFile('multitau_file.hdf5')
   
   # Access correlation data
   g2_data = data.g2
   tau_data = data.tau
   
   # Plot correlation functions
   plt.figure(figsize=(10, 6))
   for i in range(min(5, g2_data.shape[0])):  # Plot first 5 q-points
       plt.semilogx(tau_data, g2_data[i, :], label=f'q-bin {i}')
   
   plt.xlabel('Delay Time τ (s)')
   plt.ylabel('g₂(τ)')
   plt.legend()
   plt.title('Intensity Correlation Functions')
   plt.show()

3. SAXS 1D Profiles
~~~~~~~~~~~~~~~~~~~

Create radial intensity profiles:

.. code-block:: bash

   # Command line with log scales
   xpcs-toolkit saxs1d /path/to/data/ --log-x --log-y --output profile.png

.. code-block:: python

   # Python API
   data = xpcs_toolkit.XpcsDataFile('experiment.hdf5')
   
   # Access 1D SAXS profile
   saxs_1d = data.saxs_1d
   I_q = saxs_1d['Iq'][0]  # First phi bin
   q_values = saxs_1d['q']
   
   # Plot I(q) profile
   plt.figure(figsize=(8, 6))
   plt.loglog(q_values, I_q, 'b-', linewidth=2)
   plt.xlabel('q (Å⁻¹)')
   plt.ylabel('I(q)')
   plt.title('Radial SAXS Profile')
   plt.grid(True, alpha=0.3)
   plt.show()

4. Stability Monitoring
~~~~~~~~~~~~~~~~~~~~~~~

Monitor beam and sample stability:

.. code-block:: bash

   # Command line stability analysis
   xpcs-toolkit stability /path/to/data/ --output stability_report.png

.. code-block:: python

   # Python API for stability analysis
   data = xpcs_toolkit.XpcsDataFile('experiment.hdf5')
   
   if hasattr(data, 'Int_t'):
       # Plot intensity vs time
       plt.figure(figsize=(12, 4))
       plt.plot(data.Int_t, alpha=0.7)
       plt.xlabel('Frame Number')
       plt.ylabel('Integrated Intensity')
       plt.title('Beam Stability Monitor')
       plt.show()

Advanced Examples
-----------------

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple files efficiently:

.. code-block:: python

   import os
   from pathlib import Path
   
   # Process all files in a directory
   data_dir = Path('/path/to/your/data/')
   
   for hdf_file in data_dir.glob('*.hdf5'):
       try:
           data = xpcs_toolkit.XpcsDataFile(str(hdf_file))
           print(f"Processing {hdf_file.name}:")
           print(f"  - Analysis type: {data.analysis_type}")
           print(f"  - Beam energy: {data.X_energy} keV")
           
           # Perform your analysis here
           
       except Exception as e:
           print(f"  - Error: {e}")

Data Export
~~~~~~~~~~~

Export analysis results:

.. code-block:: python

   import numpy as np
   
   data = xpcs_toolkit.XpcsDataFile('experiment.hdf5')
   
   # Export correlation data
   if hasattr(data, 'g2'):
       np.savetxt('g2_data.txt', data.g2, 
                  header='G2 correlation functions')
   
   # Export SAXS profile
   if hasattr(data, 'saxs_1d'):
       saxs_1d = data.saxs_1d
       export_data = np.column_stack([saxs_1d['q'], saxs_1d['Iq'][0]])
       np.savetxt('saxs_profile.txt', export_data, 
                  header='q(A^-1) I(q)')

Integration with Scientific Stack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine with other scientific Python tools:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from scipy import optimize
   
   data = xpcs_toolkit.XpcsDataFile('experiment.hdf5')
   
   # Convert to pandas for analysis
   if hasattr(data, 'g2'):
       g2_df = pd.DataFrame(data.g2.T, columns=[f'q_{i}' for i in range(data.g2.shape[0])])
       g2_df['tau'] = data.tau
       
       # Statistical analysis
       print("G2 correlation statistics:")
       print(g2_df.describe())

Error Handling
--------------

Robust error handling for production use:

.. code-block:: python

   def safe_load_xpcs_file(filepath):
       """Safely load XPCS file with error handling."""
       try:
           data = xpcs_toolkit.XpcsDataFile(filepath)
           return data
       except FileNotFoundError:
           print(f"File not found: {filepath}")
       except ValueError as e:
           print(f"Invalid file format: {e}")
       except Exception as e:
           print(f"Unexpected error: {e}")
       return None
   
   # Usage
   data = safe_load_xpcs_file('experiment.hdf5')
   if data is not None:
       # Proceed with analysis
       pass

Performance Tips
----------------

1. **Use lazy loading**: XPCS Toolkit loads data on-demand to minimize memory usage
2. **Process in batches**: For large datasets, process files in smaller batches
3. **Monitor memory**: Use system monitoring to track memory usage during processing
4. **Cache results**: Save intermediate results to avoid recomputation

.. code-block:: python

   # Monitor memory usage
   import psutil
   
   process = psutil.Process()
   memory_mb = process.memory_info().rss / 1024 / 1024
   print(f"Current memory usage: {memory_mb:.1f} MB")

Next Steps
----------

Now that you've completed the quick start:

1. **Explore the API**: Check out the complete :doc:`api/index`
2. **Read user guides**: Dive deeper with :doc:`guides/index`
3. **Try tutorials**: Work through :doc:`tutorials/index`
4. **Get help**: Visit our :doc:`faq` or GitHub issues

Common File Formats
--------------------

XPCS Toolkit supports several file formats:

- **APS 8-ID-I NeXus format**: Modern HDF5-based format with full metadata
- **Legacy HDF5 format**: Backward compatibility with older XPCS files
- **Automatic detection**: The toolkit automatically identifies file format

Need Help?
----------

- **Documentation**: You're reading it! 📚
- **Examples**: Check the ``examples/`` directory in the repository
- **Issues**: Report problems on `GitHub <https://github.com/imewei/xpcs-toolkit/issues>`_
- **Discussions**: Join the community on `GitHub Discussions <https://github.com/imewei/xpcs-toolkit/discussions>`_

Happy analyzing! 🔬✨