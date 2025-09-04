"""
XPCS Toolkit - Data Averaging and Quality Control Toolbox (average_toolbox)

This module provides comprehensive tools for averaging multiple XPCS/SAXS datasets
with quality control, statistical validation, and batch processing capabilities.
Data averaging is essential for improving signal-to-noise ratios, removing outliers,
and combining measurements from equivalent samples or conditions.

## Scientific Background

Data averaging in X-ray scattering experiments serves multiple purposes:

### Statistical Enhancement
- **Signal-to-noise improvement**: √N enhancement from N averaged datasets
- **Statistical reliability**: Reduced impact of measurement fluctuations
- **Outlier mitigation**: Robust averaging excludes corrupted or anomalous data
- **Reproducibility assessment**: Quantify measurement consistency across samples

### Quality Control Metrics
- **Baseline validation**: Ensure proper correlation function normalization
- **Data consistency**: Check for systematic variations or drift
- **Temporal stability**: Monitor sample stability during measurements
- **Instrumental reliability**: Detect systematic errors or hardware issues

### Physical Interpretation
- **Equilibrium properties**: Time-averaged behavior of stable systems
- **Ensemble averaging**: Statistical mechanics quantities from multiple realizations
- **Sample heterogeneity**: Assess variability across different sample preparations
- **Measurement precision**: Quantify experimental uncertainties and reproducibility

## Analysis Capabilities

### Automatic Quality Assessment
- **G₂ baseline validation**: Check correlation function normalization (g₂→1 at long times)
- **Statistical clustering**: Identify and group similar measurements using k-means
- **Outlier detection**: Automatically exclude anomalous datasets
- **Intensity consistency**: Monitor scattered intensity levels across measurements

### Flexible Averaging Options
- **Multi-field support**: Average 2D images, 1D profiles, correlation functions simultaneously
- **Weighted averaging**: Account for different measurement times or statistics
- **Error propagation**: Proper statistical uncertainty calculation for averaged quantities
- **Selective inclusion**: Manual or automatic selection of datasets to include

### Batch Processing
- **Large dataset handling**: Efficient processing of hundreds of files
- **Memory optimization**: Chunk-based processing for large datasets
- **Progress monitoring**: Real-time status updates and completion estimates
- **Robust error handling**: Continue processing despite individual file failures

### Real-time Monitoring
- **Live quality plots**: Monitor baseline values during processing
- **Progress visualization**: Track completion status and estimated time
- **Interactive controls**: Pause, resume, or cancel operations
- **Result preview**: Inspect averaged data before final save

## Typical Analysis Workflow

1. **File Selection**: Choose datasets for averaging based on experimental conditions
2. **Quality Assessment**: Set baseline validation criteria for correlation functions
3. **Clustering Analysis**: Group similar measurements and identify outliers
4. **Parameter Configuration**: Set averaging windows, field selection, and output options
5. **Batch Processing**: Execute averaging with real-time monitoring
6. **Result Validation**: Inspect averaged data quality and statistics
7. **Data Export**: Save averaged results with proper metadata preservation

## Applications

### Experimental Design
- **Sample optimization**: Determine minimum measurement time per dataset
- **Statistical planning**: Calculate required number of measurements for target precision
- **Quality thresholds**: Establish acceptance criteria for data inclusion
- **Protocol validation**: Verify measurement reproducibility across conditions

### Data Analysis
- **Enhanced correlation functions**: Improve g₂(q,τ) signal quality for fitting
- **Structure determination**: Average SAXS patterns for reliable size/shape analysis
- **Dynamics characterization**: Reduce noise in dynamic light scattering measurements
- **Comparative studies**: Create high-quality reference datasets for comparisons

### Quality Control
- **Instrument monitoring**: Track detector stability and beam consistency
- **Sample screening**: Identify samples with anomalous behavior
- **Method validation**: Assess measurement precision and accuracy
- **Publication standards**: Generate high-quality averaged data for publication

### High-throughput Analysis
- **Automated processing**: Process large datasets with minimal supervision
- **Screening studies**: Rapidly assess many samples or conditions
- **Time-series analysis**: Average over temporal windows for evolution studies
- **Multi-sample comparisons**: Generate consistent datasets for statistical analysis

## Module Components

### AverageToolbox Class
Main class providing comprehensive averaging functionality with:
- File management and validation
- Quality control and clustering
- Real-time progress monitoring
- Interactive visualization

### Standalone Functions
- `do_average()`: Simple batch averaging function
- `average_plot_cluster()`: K-means clustering for quality assessment

## Usage Examples

```python
# Basic averaging workflow
from xpcs_toolkit.module.average_toolbox import AverageToolbox

# Initialize averaging job
avg_job = AverageToolbox(
    work_dir='/path/to/data/',
    flist=['file001.h5', 'file002.h5', 'file003.h5']
)

# Configure averaging parameters
avg_job.setup(
    chunk_size=256,           # Process in chunks for memory efficiency
    avg_window=3,             # Baseline validation window (last 3 points)
    avg_qindex=0,             # Q-bin for baseline check
    avg_blmin=0.95,           # Minimum acceptable baseline
    avg_blmax=1.05,           # Maximum acceptable baseline
    fields=['saxs_2d', 'g2']  # Data fields to average
)

# Execute averaging
results = avg_job.do_average(save_path='averaged_data.h5')

# Simple standalone averaging
from xpcs_toolkit.module.average_toolbox import do_average

baseline_values = do_average(
    flist=['file001.h5', 'file002.h5'],
    work_dir='/data/',
    save_path='average.h5',
    avg_blmin=0.98,
    avg_blmax=1.02
)
```

## Quality Metrics

### G₂ Baseline Validation
The correlation function baseline g₂(τ→∞) should approach 1.0 for properly normalized data:
- **Acceptable range**: Typically 0.95 ≤ g₂(∞) ≤ 1.05
- **Physical meaning**: Values outside this range may indicate systematic errors
- **Validation window**: Average over last few lag time points for stability

### Statistical Clustering
K-means clustering on intensity statistics helps identify:
- **Consistent measurements**: Main cluster represents typical behavior
- **Outliers**: Points outside main cluster may have experimental issues
- **Systematic variations**: Separate clusters may indicate different conditions

## References

- Schätzel, "Suppression of multiple scattering by photon cross-correlation techniques" (1993)
- Pusey & van Megen, "Dynamic light scattering by non-ergodic media" (1989)
- Cipelletti & Weitz, "Ultralow-angle dynamic light scattering with a charge coupled device camera based multispeckle, multitau correlator" (1999)
- Fluerasu et al., "Slow dynamics and aging in colloidal gels studied by x-ray photon correlation spectroscopy" (2007)

## Author

XPCS Toolkit Development Team
Advanced Photon Source, Argonne National Laboratory
"""

import logging
import os
from shutil import copyfile
import time
import uuid

from xpcs_toolkit.mpl_compat import MockSignal

# Use lazy imports for heavy dependencies
from .._lazy_imports import lazy_import

np = lazy_import("numpy")
trange = lazy_import("tqdm", "trange")

from xpcs_toolkit.fileIO.hdf_reader import put
from xpcs_toolkit.helper.listmodel import ListDataModel
from xpcs_toolkit.xpcs_file import XpcsDataFile as XF

# Optional imports
try:
    from sklearn.cluster import KMeans as sk_kmeans
except ImportError:
    sk_kmeans = None

# PyQtGraph import removed for headless operation
pg = None


logger = logging.getLogger(__name__)


def average_plot_cluster(self, hdl1, num_clusters=2):
    """Plot clustering results - disabled in headless mode"""
    raise NotImplementedError(
        "GUI plotting functionality has been disabled in headless mode."
    )


class WorkerSignal:
    """Mock worker signal for headless operation"""

    def __init__(self):
        self.progress = MockSignal()
        self.values = MockSignal()
        self.status = MockSignal()


class AverageToolbox:
    def __init__(self, work_dir=None, flist=None, jid=None) -> None:
        if flist is None:
            flist = ["hello"]
        self.file_list = flist.copy()
        self.model = ListDataModel(self.file_list)

        self.work_dir = work_dir
        self.signals = WorkerSignal()
        self.kwargs = {}
        if jid is None:
            self.jid = uuid.uuid4()
        else:
            self.jid = jid
        self.submit_time = time.strftime("%H:%M:%S")
        self.stime = self.submit_time
        self.etime = "--:--:--"
        self.status = "wait"
        self.baseline = np.zeros(max(len(self.model), 10), dtype=np.float32)
        self.ptr = 0
        self.short_name = self.generate_avg_fname()
        self.eta = "..."
        self.size = len(self.model)
        self._progress = "0%"
        # axis to show the baseline;
        self.ax = None
        # use one file as template
        # Ensure work_dir and model[0] are strings for path joining
        if self.work_dir is not None and len(self.model) > 0:
            self.origin_path = os.path.join(str(self.work_dir), str(self.model[0]))
        else:
            self.origin_path = None

        self.is_killed = False

    def kill(self):
        self.is_killed = True

    def __str__(self) -> str:
        return str(self.jid)

    def generate_avg_fname(self):
        if len(self.model) == 0:
            return
        fname = self.model[0]
        end = fname.rfind("_")
        if end == -1:
            end = len(fname)
        new_fname = "Avg" + fname[slice(0, end)]
        # if new_fname[-3:] not in ['.h5', 'hdf']:
        #     new_fname += '.hdf'
        return new_fname

    def run(self):
        self.do_average(*self.args, **self.kwargs)

    def setup(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def do_average(
        self,
        chunk_size=256,
        save_path=None,
        avg_window=3,
        avg_qindex=0,
        avg_blmin=0.95,
        avg_blmax=1.05,
        fields=None,
    ):
        if fields is None:
            fields = ["saxs_2d"]
        self.stime = time.strftime("%H:%M:%S")
        self.status = "running"
        logger.info("average job %d starts", self.jid)
        tot_num = len(self.model)
        steps = (tot_num + chunk_size - 1) // chunk_size
        mask = np.zeros(tot_num, dtype=np.int64)
        prev_percentage = 0

        def validate_g2_baseline(g2_data, q_idx):
            # Use min to avoid bounds checking
            idx = min(q_idx, g2_data.shape[1] - 1) if g2_data.shape[1] > 0 else 0
            if idx != q_idx:
                logger.info("q_index is out of range; using %d instead", idx)

            # More efficient slicing and calculation
            g2_baseline = g2_data[-avg_window:, idx].mean()
            return avg_blmin <= g2_baseline <= avg_blmax, g2_baseline

        result = {}
        for key in fields:
            result[key] = None

        t0 = time.perf_counter()
        for n in range(steps):
            beg = chunk_size * (n + 0)
            end = chunk_size * (n + 1)
            end = min(tot_num, end)
            for m in range(beg, end):
                # time.sleep(0.5)
                if self.is_killed:
                    logger.info("the averaging instance has been killed.")
                    self._progress = "killed"
                    self.status = "killed"
                    return

                curr_percentage = int((m + 1) * 100 / tot_num)
                if curr_percentage >= prev_percentage:
                    prev_percentage = curr_percentage
                    dt = (time.perf_counter() - t0) / (m + 1)
                    eta = dt * (tot_num - m - 1)
                    self.eta = eta
                    self._progress = "%d%%" % (curr_percentage)
                    # self.signals.progress.emit((self.jid, curr_percentage))

                fname = self.model[m]
                xf = None  # Initialize to avoid unbound variable
                try:
                    # Ensure work_dir and fname are strings
                    if self.work_dir is not None:
                        xf = XF(
                            os.path.join(str(self.work_dir), str(fname)), fields=fields
                        )
                    else:
                        xf = XF(str(fname), fields=fields)
                    flag, val = validate_g2_baseline(xf.g2, avg_qindex)
                    self.baseline[self.ptr] = val
                    self.ptr += 1
                except (OSError, ValueError) as e:
                    # Specific exception handling for file and data errors
                    flag, val = False, 0
                    logger.error("Failed to process file %s: %s", fname, str(e))
                except Exception as e:
                    # Catch-all for unexpected errors with more detail
                    flag, val = False, 0
                    logger.error(
                        "Unexpected error processing file %s: %s", fname, str(e)
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Full traceback:", exc_info=True)

                if flag and xf is not None:
                    for key in fields:
                        # More efficient attribute access
                        if key == "saxs_1d":
                            data = getattr(xf, "saxs_1d", {}).get("data_raw")
                            if data is None:
                                continue
                        else:
                            data = getattr(xf, key, None)
                            if data is None:
                                continue

                        if result[key] is None:
                            # Use copy to avoid memory aliasing issues
                            result[key] = data.copy() if hasattr(data, "copy") else data
                            mask[m] = 1
                        elif result[key].shape == data.shape:
                            # In-place addition for memory efficiency
                            result[key] += data
                            mask[m] = 1
                        else:
                            logger.info(
                                "data shape does not match for key %s, %s", key, fname
                            )

                self.signals.values.emit((self.jid, val))

        if np.sum(mask) == 0:
            logger.info("no dataset is valid; check the baseline criteria.")
            return
        else:
            for key in fields:
                if key == "saxs_1d":
                    # only keep the Iq component, put method doesn't accept dict
                    result["saxs_1d"] = result["saxs_1d"] / np.sum(mask)
                else:
                    result[key] /= np.sum(mask)
                if key == "g2_err":
                    result[key] /= np.sqrt(np.sum(mask))
                if key == "saxs_2d":
                    # saxs_2d needs to be (1, height, width)
                    saxs_2d = result[key]
                    if saxs_2d.ndim == 2:
                        saxs_2d = np.expand_dims(saxs_2d, axis=0)
                    result[key] = saxs_2d

            logger.info("the valid dataset number is %d / %d" % (np.sum(mask), tot_num))

        logger.info(f"create file: {save_path}")
        # Ensure origin_path and save_path are not None before copying
        if self.origin_path is not None and save_path is not None:
            copyfile(self.origin_path, str(save_path))
        else:
            logger.warning("No origin path available for copying")

        # Ensure save_path is a string
        if save_path is not None:
            put(str(save_path), result, file_type="nexus", mode="alias")
        else:
            logger.warning("No save path provided for putting results")

        self.status = "finished"
        self.signals.status.emit((self.jid, self.status))
        self.etime = time.strftime("%H:%M:%S")
        self.model.layoutChanged.emit()
        self.signals.progress.emit((self.jid, 100))
        logger.info("average job %d finished", self.jid)
        return result

    def initialize_plot(self, hdl):
        """Initialize plot - disabled in headless mode"""
        raise NotImplementedError(
            "GUI plotting functionality has been disabled in headless mode."
        )

    def update_plot(self):
        """Update plot - disabled in headless mode"""
        pass  # No-op in headless mode

    def get_pg_tree(self):
        """Get PyQtGraph tree widget - disabled in headless mode"""
        return None  # Always return None in headless mode


def do_average(
    flist,
    work_dir=None,
    save_path=None,
    avg_window=3,
    avg_qindex=0,
    avg_blmin=0.95,
    avg_blmax=1.05,
    fields=None,
):
    if fields is None:
        fields = ["saxs_2d", "saxs_1d", "g2", "g2_err"]
    if work_dir is None:
        work_dir = "./"

    tot_num = len(flist)
    abs_cs_scale_tot = 0.0
    baseline = np.zeros(tot_num, dtype=np.float32)
    mask = np.zeros(tot_num, dtype=np.int64)

    def validate_g2_baseline(g2_data, q_idx):
        # Use min to avoid bounds checking (consistent with class method)
        idx = min(q_idx, g2_data.shape[1] - 1) if g2_data.shape[1] > 0 else 0
        if idx != q_idx:
            logger.info("q_index is out of range; using %d instead", idx)

        # More efficient slicing and calculation
        g2_baseline = g2_data[-avg_window:, idx].mean()
        return avg_blmin <= g2_baseline <= avg_blmax, g2_baseline

    result = {}
    for key in fields:
        result[key] = None

    for m in trange(tot_num):
        fname = flist[m]
        xf = None  # Initialize to avoid unbound variable
        try:
            # Ensure work_dir and fname are strings
            if work_dir is not None:
                xf = XF(os.path.join(str(work_dir), str(fname)), fields=fields)
            else:
                xf = XF(str(fname), fields=fields)
            flag, val = validate_g2_baseline(xf.g2, avg_qindex)
            baseline[m] = val
        except (OSError, ValueError) as e:
            # Specific exception handling for file and data errors
            flag, val = False, 0
            logger.error("Failed to process file %s: %s", fname, str(e))
        except Exception as e:
            # Catch-all for unexpected errors with more detail
            flag, val = False, 0
            logger.error("Unexpected error processing file %s: %s", fname, str(e))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Full traceback:", exc_info=True)

        if flag and xf is not None:
            for key in fields:
                scale = 1.0  # Default scale value
                if key != "saxs_1d":
                    data = xf.at(key) if hasattr(xf, "at") else None  # type: ignore[union-attr]
                else:
                    if hasattr(xf, "at"):
                        saxs_1d_data = xf.at("saxs_1d")  # type: ignore[union-attr]
                        if saxs_1d_data is not None:
                            data = saxs_1d_data["data_raw"]
                            scale = xf.abs_cross_section_scale
                            if scale is None:
                                scale = 1.0
                            data *= scale
                        else:
                            data = None
                            scale = 1.0  # Keep default scale
                    else:
                        data = None
                        scale = 1.0
                    abs_cs_scale_tot += scale

                if result[key] is None:
                    result[key] = data
                    mask[m] = 1
                elif data is not None and result[key].shape == data.shape:
                    result[key] += data
                    mask[m] = 1
                elif data is not None:
                    logger.info(f"data shape does not match for key {key}, {fname}")

    if np.sum(mask) == 0:
        logger.info("no dataset is valid; check the baseline criteria.")
        return
    else:
        for key in fields:
            if key == "saxs_1d":
                result["saxs_1d"] /= abs_cs_scale_tot
            else:
                result[key] /= np.sum(mask)
            if key == "g2_err":
                result[key] /= np.sqrt(np.sum(mask))

        logger.info("the valid dataset number is %d / %d" % (np.sum(mask), tot_num))

    original_file = os.path.join(work_dir, flist[0])
    if save_path is None:
        save_path = "AVG" + os.path.basename(flist[0])
    logger.info(f"create file: {save_path}")
    # Ensure original_file is not None before copying
    if original_file is not None:
        copyfile(original_file, save_path)
    else:
        logger.warning("No original file available for copying")

    # Ensure save_path is a string
    put(
        str(save_path) if save_path is not None else "",
        result,
        file_type="nexus",
        mode="alias",
    )

    return baseline
