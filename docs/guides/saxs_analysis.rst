SAXS Analysis Guide
==================

This guide covers Small-Angle X-ray Scattering (SAXS) analysis using XPCS Toolkit, from basic visualization to advanced quantitative analysis.

Introduction to SAXS
--------------------

Small-Angle X-ray Scattering provides structural information about materials at the nanoscale. XPCS Toolkit supports both 2D pattern analysis and 1D radial profiling for comprehensive SAXS characterization.

**Key Concepts:**

- **q-vector**: Momentum transfer, q = 4π sin(θ)/λ
- **Scattering intensity**: I(q) related to structure factor
- **Beam center**: Critical for accurate q calibration
- **Radial averaging**: Converting 2D patterns to 1D profiles

2D SAXS Pattern Analysis
------------------------

Visualizing 2D Scattering Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Basic 2D visualization:

.. code-block:: python

   import xpcs_toolkit
   import matplotlib.pyplot as plt
   
   # Load XPCS file with SAXS data
   data = xpcs_toolkit.XpcsDataFile('saxs_experiment.hdf5')
   
   # Access 2D SAXS pattern
   saxs_2d = data.saxs_2d
   
   # Basic visualization
   plt.figure(figsize=(8, 8))
   plt.imshow(saxs_2d, origin='lower', cmap='viridis', 
              norm=plt.LogNorm(vmin=1, vmax=saxs_2d.max()))
   plt.colorbar(label='Intensity')
   plt.title('2D SAXS Pattern')
   plt.xlabel('Pixel X')
   plt.ylabel('Pixel Y')
   plt.show()

Command line visualization:

.. code-block:: bash

   # Generate 2D SAXS plot with log scale
   xpcs-toolkit saxs2d /path/to/data/ --scale log --colormap plasma
   
   # Linear scale with custom intensity range
   xpcs-toolkit saxs2d /path/to/data/ --scale linear --vmin 10 --vmax 1000

Advanced 2D Analysis
~~~~~~~~~~~~~~~~~~~~

Beam center determination and q-calibration:

.. code-block:: python

   # Access beam center information
   bcx, bcy = data.bcx, data.bcy
   print(f"Beam center: ({bcx:.1f}, {bcy:.1f})")
   
   # Get detector parameters for q-calibration
   det_dist = data.det_dist  # mm
   pixel_size = data.pix_dim_x  # mm
   energy = data.X_energy  # keV
   wavelength = 12.398 / energy  # Angstroms
   
   print(f"Detector distance: {det_dist:.1f} mm")
   print(f"Pixel size: {pixel_size:.4f} mm")
   print(f"Wavelength: {wavelength:.4f} Å")

Q-space mapping:

.. code-block:: python

   import numpy as np
   
   # Create q-space map
   if hasattr(data, 'qmap'):
       qmap = data.qmap
       
       # Visualize q-space
       plt.figure(figsize=(10, 4))
       
       plt.subplot(1, 2, 1)
       plt.imshow(saxs_2d, origin='lower', cmap='viridis', norm=plt.LogNorm())
       plt.title('Real Space (detector)')
       plt.colorbar()
       
       plt.subplot(1, 2, 2)
       plt.imshow(qmap, origin='lower', cmap='viridis')
       plt.title('Q-space mapping')
       plt.colorbar(label='q (Å⁻¹)')
       
       plt.tight_layout()
       plt.show()

1D Radial Profile Analysis
--------------------------

Generating Radial Profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract 1D intensity profiles from 2D patterns:

.. code-block:: bash

   # Command line radial profiling
   xpcs-toolkit saxs1d /path/to/data/ --log-x --log-y --output profile.png

.. code-block:: python

   # Python API for radial profiling
   if hasattr(data, 'saxs_1d'):
       saxs_1d = data.saxs_1d
       
       # Extract q and I(q) data
       q_values = saxs_1d['q']
       I_q = saxs_1d['Iq'][0]  # First phi bin
       
       # Plot radial profile
       plt.figure(figsize=(8, 6))
       plt.loglog(q_values, I_q, 'b-', linewidth=2)
       plt.xlabel('q (Å⁻¹)')
       plt.ylabel('I(q)')
       plt.title('Radial SAXS Profile')
       plt.grid(True, alpha=0.3)
       plt.show()

Multi-Phi Analysis
~~~~~~~~~~~~~~~~~~~

Analyze angular-dependent scattering:

.. code-block:: python

   if hasattr(data, 'saxs_1d'):
       saxs_1d = data.saxs_1d
       q_values = saxs_1d['q']
       I_q_phi = saxs_1d['Iq']  # Shape: (n_phi, n_q)
       
       # Plot multiple phi sectors
       plt.figure(figsize=(10, 6))
       
       n_phi = min(8, I_q_phi.shape[0])
       colors = plt.cm.viridis(np.linspace(0, 1, n_phi))
       
       for i in range(n_phi):
           phi_angle = i * 360 / I_q_phi.shape[0]
           plt.loglog(q_values, I_q_phi[i], 
                     color=colors[i], linewidth=1.5,
                     label=f'φ = {phi_angle:.0f}°')
       
       plt.xlabel('q (Å⁻¹)')
       plt.ylabel('I(q)')
       plt.title('Angular-Dependent SAXS Profiles')
       plt.legend()
       plt.grid(True, alpha=0.3)
       plt.show()

Quantitative Analysis
---------------------

Guinier Analysis
~~~~~~~~~~~~~~~~

Extract radius of gyration using Guinier approximation:

.. code-block:: python

   def guinier_analysis(q, I, q_max_factor=1.3):
       """
       Perform Guinier analysis to extract radius of gyration.
       
       Guinier equation: ln(I) = ln(I₀) - (R_g²/3) * q²
       """
       from scipy import stats
       import numpy as np
       
       # Find Guinier region (q * R_g < 1.3)
       # Initial estimate from data range
       q_max_guinier = q_max_factor / np.sqrt(np.var(q))
       mask = q <= q_max_guinier
       
       if np.sum(mask) < 5:
           return None, None, "Insufficient points for Guinier analysis"
       
       q_guinier = q[mask]
       ln_I = np.log(I[mask])
       
       # Linear regression on q²
       slope, intercept, r_value, p_value, std_err = stats.linregress(
           q_guinier**2, ln_I
       )
       
       # Extract radius of gyration
       R_g = np.sqrt(-3 * slope)
       I_0 = np.exp(intercept)
       
       return R_g, I_0, f"R² = {r_value**2:.3f}"

   # Apply Guinier analysis
   if hasattr(data, 'saxs_1d'):
       q_values = data.saxs_1d['q']
       I_q = data.saxs_1d['Iq'][0]
       
       R_g, I_0, fit_info = guinier_analysis(q_values, I_q)
       
       if R_g is not None:
           print(f"Radius of gyration: {R_g:.2f} Å")
           print(f"Forward scattering: {I_0:.2e}")
           print(f"Fit quality: {fit_info}")

Porod Analysis
~~~~~~~~~~~~~~

High-q analysis for surface area information:

.. code-block:: python

   def porod_analysis(q, I, q_min_factor=3.0):
       """
       Perform Porod analysis to extract surface area information.
       
       Porod law: I(q) = I_p * q⁻⁴ for q >> R_g⁻¹
       """
       from scipy import stats
       import numpy as np
       
       # Find Porod region (high q)
       q_min_porod = q_min_factor * np.mean(q)
       mask = q >= q_min_porod
       
       if np.sum(mask) < 5:
           return None, None, "Insufficient points for Porod analysis"
       
       q_porod = q[mask]
       I_porod = I[mask]
       
       # Linear regression in log-log space
       log_q = np.log(q_porod)
       log_I = np.log(I_porod)
       
       slope, intercept, r_value, p_value, std_err = stats.linregress(
           log_q, log_I
       )
       
       # Extract Porod parameters
       porod_exponent = -slope
       porod_constant = np.exp(intercept)
       
       return porod_exponent, porod_constant, f"R² = {r_value**2:.3f}"

   # Apply Porod analysis
   if hasattr(data, 'saxs_1d'):
       q_values = data.saxs_1d['q']
       I_q = data.saxs_1d['Iq'][0]
       
       exponent, constant, fit_info = porod_analysis(q_values, I_q)
       
       if exponent is not None:
           print(f"Porod exponent: {exponent:.2f}")
           print(f"Porod constant: {constant:.2e}")
           print(f"Fit quality: {fit_info}")

Power Law Fitting
~~~~~~~~~~~~~~~~~

General power law analysis:

.. code-block:: python

   from scipy.optimize import curve_fit
   
   def power_law(q, A, alpha):
       """Power law: I(q) = A * q^(-alpha)"""
       return A * np.power(q, -alpha)
   
   # Fit power law to data
   if hasattr(data, 'saxs_1d'):
       q_values = data.saxs_1d['q']
       I_q = data.saxs_1d['Iq'][0]
       
       # Fit in appropriate q range
       q_min, q_max = np.percentile(q_values, [10, 90])
       mask = (q_values >= q_min) & (q_values <= q_max)
       
       try:
           popt, pcov = curve_fit(power_law, 
                                 q_values[mask], I_q[mask],
                                 p0=[I_q[mask][0], 2.0])
           
           A_fit, alpha_fit = popt
           alpha_err = np.sqrt(pcov[1, 1])
           
           print(f"Power law exponent: {alpha_fit:.2f} ± {alpha_err:.2f}")
           
           # Plot fit
           plt.figure(figsize=(8, 6))
           plt.loglog(q_values, I_q, 'bo', alpha=0.6, label='Data')
           plt.loglog(q_values[mask], power_law(q_values[mask], *popt), 
                     'r-', linewidth=2, 
                     label=f'Power law fit: α = {alpha_fit:.2f}')
           plt.xlabel('q (Å⁻¹)')
           plt.ylabel('I(q)')
           plt.legend()
           plt.grid(True, alpha=0.3)
           plt.show()
           
       except Exception as e:
           print(f"Power law fit failed: {e}")

Data Export and Processing
--------------------------

Export SAXS Data
~~~~~~~~~~~~~~~~

Save analysis results for further processing:

.. code-block:: python

   import numpy as np
   import pandas as pd
   
   # Export radial profile
   if hasattr(data, 'saxs_1d'):
       saxs_1d = data.saxs_1d
       q_values = saxs_1d['q']
       I_q = saxs_1d['Iq']
       
       # Create comprehensive export
       export_dict = {'q_A-1': q_values}
       
       # Add all phi sectors
       for i in range(I_q.shape[0]):
           phi_angle = i * 360 / I_q.shape[0]
           export_dict[f'I_phi_{phi_angle:.0f}deg'] = I_q[i]
       
       # Save to files
       df = pd.DataFrame(export_dict)
       df.to_csv('saxs_profiles.csv', index=False)
       
       # Simple numpy export
       np.savetxt('saxs_radial.txt', 
                  np.column_stack([q_values, I_q[0]]),
                  header='q(A^-1) I(q)', 
                  fmt='%.6e')

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple SAXS files:

.. code-block:: python

   from pathlib import Path
   import matplotlib.pyplot as plt
   
   def batch_saxs_analysis(data_dir, output_dir):
       """Process all SAXS files in directory."""
       data_path = Path(data_dir)
       output_path = Path(output_dir)
       output_path.mkdir(exist_ok=True)
       
       results = []
       
       for hdf_file in data_path.glob('*.hdf5'):
           try:
               data = xpcs_toolkit.XpcsDataFile(str(hdf_file))
               
               if hasattr(data, 'saxs_1d'):
                   # Extract profile
                   q_values = data.saxs_1d['q']
                   I_q = data.saxs_1d['Iq'][0]
                   
                   # Simple analysis
                   q_max = q_values[np.argmax(I_q)]
                   I_max = np.max(I_q)
                   
                   results.append({
                       'file': hdf_file.name,
                       'q_peak': q_max,
                       'I_peak': I_max,
                       'integrated_intensity': np.trapz(I_q, q_values)
                   })
                   
                   # Save individual profile
                   output_file = output_path / f"{hdf_file.stem}_profile.txt"
                   np.savetxt(output_file, 
                             np.column_stack([q_values, I_q]),
                             header='q(A^-1) I(q)')
                   
               print(f"Processed: {hdf_file.name}")
               
           except Exception as e:
               print(f"Error processing {hdf_file.name}: {e}")
       
       # Save summary
       if results:
           summary_df = pd.DataFrame(results)
           summary_df.to_csv(output_path / 'saxs_summary.csv', index=False)
           print(f"Processed {len(results)} files successfully")
       
       return results
   
   # Usage
   results = batch_saxs_analysis('/path/to/data/', './saxs_results/')

Best Practices
--------------

**Data Quality Checks**

1. Verify beam center accuracy
2. Check for detector artifacts
3. Validate q-calibration
4. Monitor background subtraction

**Analysis Workflow**

1. Start with 2D pattern inspection
2. Generate and validate 1D profiles  
3. Perform appropriate model fitting
4. Export results with metadata

**Performance Tips**

- Use lazy loading for large datasets
- Process in batches for many files
- Cache intermediate results
- Monitor memory usage

Troubleshooting
~~~~~~~~~~~~~~~

**Common Issues:**

- **Missing SAXS data**: Verify file contains SAXS analysis results
- **Incorrect beam center**: Check experimental parameters
- **Poor fitting results**: Adjust fitting ranges and models
- **Memory issues**: Process files individually or in smaller batches

**Quality Assessment:**

- Check residuals after fitting
- Validate against known standards
- Compare results across similar samples
- Monitor statistical uncertainties