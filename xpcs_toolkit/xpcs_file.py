import os
import re
import warnings
from typing import Any, Optional, Tuple, List, Dict, cast

# Use lazy import for numpy to improve startup time
from ._lazy_imports import lazy_import
np = lazy_import('numpy')

from .fileIO.hdf_reader import get, get_analysis_type, read_metadata_to_dict
from .helper.fitting import fit_with_fixed
from .fileIO.qmap_utils import get_qmap
from .module.twotime_utils import get_c2_stream, get_single_c2_from_hdf
import logging

logger = logging.getLogger(__name__)


def single_exp_all(x, a, b, c, d):
    """
    Single exponential fitting for XPCS-multitau analysis.

    Parameters
    ----------
    x : float or ndarray
        Delay in seconds.
    a : float
        Contrast.
    b : float
        Characteristic time (tau).
    c : float
        Restriction factor.
    d : float
        Baseline offset.

    Returns
    -------
    float or ndarray
        Computed value of the single exponential model.
    """
    return a * np.exp(-2 * (x / b) ** c) + d


def double_exp_all(x, a, b1, c1, d, b2, c2, f):
    """
    Double exponential fitting for XPCS-multitau analysis.

    Parameters
    ----------
    x : float or ndarray
        Delay in seconds.
    a : float
        Contrast.
    b1 : float
        Characteristic time (tau) of the first exponential component.
    c1 : float
        Restriction factor for the first component.
    d : float
        Baseline offset.
    b2 : float
        Characteristic time (tau) of the second exponential component.
    c2 : float
        Restriction factor for the second component.
    f : float
        Fractional contribution of the first exponential component (0 ≤ f ≤ 1).

    Returns
    -------
    float or ndarray
        Computed value of the double exponential model.
    """
    t1 = np.exp(-1 * (x / b1) ** c1) * f
    t2 = np.exp(-1 * (x / b2) ** c2) * (1 - f)
    return a * (t1 + t2) ** 2 + d


def power_law(x, a, b):
    """
    Power-law fitting for diffusion behavior.

    Parameters
    ----------
    x : float or ndarray
        Independent variable, typically time delay (tau).
    a : float
        Scaling factor.
    b : float
        Power exponent.

    Returns
    -------
    float or ndarray
        Computed value based on the power-law model.
    """
    return a * x**b


def create_identifier(filename: str, label_style: str | None = None, simplify_flag: bool = True) -> str:
    """
    Generate a simplified or customized ID string from a filename.

    Parameters
    ----------
    filename : str
        Input file name, possibly with path and extension.
    label_style : str or None, optional
        Comma-separated string of indices to extract specific components from the file name.
    simplify_flag : bool, optional
        Whether to simplify the file name by removing leading zeros and stripping suffixes.

    Returns
    -------
    str
        A simplified or customized ID string derived from the input filename.
    """
    filename = os.path.basename(filename)

    if simplify_flag:
        # Remove leading zeros from structured parts like '_t0600' → '_t600'
        filename = re.sub(r"_(\w)0+(\d+)", r"_\1\2", filename)
        # Remove trailing _results and file extension
        filename = re.sub(r"(_results)?\.hdf$", "", filename, flags=re.IGNORECASE)

    if len(filename) < 10 or not label_style:
        return filename

    try:
        selection = [int(x.strip()) for x in label_style.split(",")]
        if not selection:
            warnings.warn("Empty label_style selection. Returning simplified filename.")
            return filename
    except ValueError:
        warnings.warn("Invalid label_style format. Must be comma-separated integers.")
        return filename

    segments = filename.split("_")
    selected_segments = []

    for i in selection:
        if i < len(segments):
            selected_segments.append(segments[i])
        else:
            warnings.warn(f"Index {i} out of range for segments {segments}")

    if not selected_segments:
        return filename  # fallback if nothing valid was selected

    return "_".join(selected_segments)


class XpcsDataFile:
    """
    XpcsDataFile - Comprehensive XPCS Dataset Handler

    This class provides complete functionality for loading, analyzing, and manipulating
    X-ray Photon Correlation Spectroscopy (XPCS) datasets from the customized NeXus
    file format developed at Argonne National Laboratory's Advanced Photon Source
    beamline 8-ID-I.

    ## Supported Analysis Types

    ### Multi-tau Correlation Analysis
    - Complete g2(q,t) correlation function datasets
    - Time-delay correlation analysis from microseconds to hours
    - Statistical error propagation and uncertainty quantification
    - Advanced fitting capabilities for extracting relaxation dynamics
    - Support for both equilibrium and non-equilibrium systems

    ### Two-time Correlation Analysis
    - Two-dimensional C(q,t1,t2) correlation function visualization
    - Time-resolved dynamics for aging and non-stationary systems
    - Interactive exploration of speckle pattern evolution
    - Support for studying glass transitions and phase changes
    - Age-dependent correlation analysis

    ## File Format Support

    ### APS 8-ID-I NeXus Format
    - Native support for the customized NeXus format
    - Complete experimental metadata preservation
    - Optimized data structures for large-scale experiments
    - Multi-dimensional dataset handling
    - Comprehensive q-space mapping information

    ### Data Components
    - **Scattering Patterns**: 2D detector images with calibration
    - **Correlation Functions**: Multi-tau or two-time correlation data
    - **Q-space Maps**: Momentum transfer calibration and binning
    - **Detector Geometry**: Pixel positions and solid angles
    - **Experimental Parameters**: Beam energy, detector distance, etc.
    - **Analysis Metadata**: Processing parameters and timestamps

    ## Key Features

    ### Intelligent Data Loading
    - Lazy loading of large datasets to minimize memory usage
    - Automatic detection of analysis type (multi-tau vs two-time)
    - Efficient handling of multi-gigabyte datasets
    - Selective field loading for specific analysis needs

    ### Advanced Q-space Handling
    - Automatic q-vector calculation from detector geometry
    - Support for both isotropic and anisotropic samples
    - Dynamic and static q-binning for different analysis types
    - Phi-angle sectoring for anisotropy studies

    ### Correlation Analysis Tools
    - G2 function extraction with customizable q-ranges
    - Time-range selection for specific dynamics
    - Statistical error propagation from raw photon statistics
    - Baseline correction and normalization

    ### Visualization Support
    - Direct integration with matplotlib for publication-quality plots
    - Interactive visualization tools for data exploration
    - Export capabilities for external analysis software
    - Support for both linear and logarithmic scaling

    ## Typical Usage

    ```python
    # Load XPCS dataset
    xpcs_data = XpcsDataFile('experiment_data.hdf')

    # Check available analysis types
    print(f"Analysis types: {xpcs_data.analysis_type}")

    # Extract correlation data
    q_vals, t_vals, g2, g2_err, labels = xpcs_data.get_g2_data(q_range=(0.01, 0.1))

    # Access scattering pattern
    saxs_pattern = xpcs_data.saxs_2d

    # Get experimental metadata
    metadata = xpcs_data.get_hdf_metadata()
    ```

    Parameters
    ----------
    filename : str
        Path to the HDF file containing XPCS analysis data.
        Must be in APS 8-ID-I NeXus format or compatible legacy format.
    fields : list of str, optional
        Additional data fields to load beyond the standard set.
        Useful for accessing specialized analysis results or raw data.
    label_style : str, optional
        Comma-separated string of indices to customize the file label creation.
        Used for organizing multiple datasets in batch analysis.
    qmap_manager : QMapManager, optional
        Q-map manager object for handling q-space mapping and caching.
        Improves performance when analyzing multiple related files.

    Attributes
    ----------
    filename : str
        Path to the loaded data file
    analysis_type : tuple
        Types of analysis available ('Multitau', 'Twotime', or both)
    q_space_map : QMap
        Q-space mapping and calibration information
    label : str
        Human-readable identifier for the dataset
    g2 : ndarray, optional
        Multi-tau correlation function data (if available)
    saxs_2d : ndarray
        2D scattering pattern from the detector
    time_elapsed : ndarray, optional
        Time delay values for correlation analysis (if available)

    """

    def __init__(self, filename: str, fields: list[str] | None = None,
                 label_style: str | None = None, qmap_manager=None):
        self.filename = filename
        if qmap_manager is None:
            self.q_space_map = get_qmap(self.filename)
        else:
            self.q_space_map = qmap_manager.get_qmap(self.filename)
        self.analysis_type = get_analysis_type(self.filename)
        self.label = self.update_label(label_style)
        payload_dictionary = self.load_dataset(fields)
        self.__dict__.update(payload_dictionary)
        self.hdf_metadata = None
        self.fit_summary = None
        self.c2_all_data = None
        self.c2_kwargs = None
        # label is a short string to describe the file/filename
        # place holder for self.saxs_2d;
        self.saxs_2d_data = None
        self.saxs_2d_log_data = None

    def update_label(self, label_style: str | None) -> str:
        """Update the label for this data file.

        Parameters
        ----------
        label_style : str, optional
            Style specification for label creation.

        Returns
        -------
        str
            Updated label string.
        """
        self.label = create_identifier(self.filename, label_style=label_style)
        return self.label

    def __str__(self) -> str:
        ans = ["File:" + str(self.filename)]
        for key, val in self.__dict__.items():
            # omit those to avoid lengthy output
            if key == "hdf_metadata":
                continue
            elif isinstance(val, np.ndarray) and val.size > 1:
                val = str(val.shape)
            else:
                val = str(val)
            ans.append(f"   {key.ljust(12)}: {val.ljust(30)}")
        return "\n".join(ans)

    def __repr__(self):
        ans = str(type(self))
        ans = "\n".join([ans, self.__str__()])
        return ans

    def get_hdf_metadata(self, filter_strings: list[str] | None = None) -> dict[str, Any]:
        """
        Get a text representation of the XPCS file metadata.

        The entries are organized in a tree structure for easy navigation.

        Parameters
        ----------
        filter_strings : list of str, optional
            List of filter strings to apply to the metadata tree.

        Returns
        -------
        dict
            Dictionary containing the HDF metadata tree structure.
        """
        # cache the data because it may take long time to generate the str
        if self.hdf_metadata is None:
            self.hdf_metadata = read_metadata_to_dict(self.filename)
        return self.hdf_metadata

    def load_dataset(self, extra_fields: list[str] | None = None) -> dict[str, Any]:
        """
        Load dataset from the HDF file.

        Parameters
        ----------
        extra_fields : list of str, optional
            Additional data fields to load from the file.

        Returns
        -------
        dict
            Dictionary containing the loaded dataset.
        """
        # default common fields for both twotime and multitau analysis
        fields = ["saxs_1d", "Iqp", "Int_t", "t0", "t1", "start_time"]

        if "Multitau" in self.analysis_type:
            fields = fields + ["tau", "g2", "g2_err", "stride_frame", "avg_frame"]
        if "Twotime" in self.analysis_type:
            fields = fields + [
                "c2_g2",
                "c2_g2_segments",
                "c2_processed_bins",
                "c2_stride_frame",
                "c2_avg_frame",
            ]

        # append other extra fields, eg "G2", "IP", "IF"
        if isinstance(extra_fields, list):
            fields += extra_fields

        # avoid duplicated keys
        fields = list(set(fields))

        result_data = get(self.filename, fields, "alias", file_type="nexus")

        if result_data is None:
            # Return empty dict if get() returns None
            return {}

        # Type assertion to help the type checker understand this is a dictionary
        result_data = cast(Dict[str, Any], result_data)

        if "Twotime" in self.analysis_type:
            stride_frame = result_data.pop("c2_stride_frame", 1)  # Provide default value
            avg_frame = result_data.pop("c2_avg_frame", 1)  # Provide default value
            if "c2_t0" in result_data and "t0" in result_data:
                result_data["c2_t0"] = result_data["t0"] * stride_frame * avg_frame

        if "Multitau" in self.analysis_type:
            # correct g2_err to avoid fitting divergence
            if "g2_err" in result_data:
                result_data["g2_err"] = self.correct_g2_err(result_data["g2_err"])
            stride_frame = result_data.pop("stride_frame", 1)  # Provide default value
            avg_frame = result_data.pop("avg_frame", 1)  # Provide default value
            if "t0" in result_data:
                result_data["t0"] = result_data["t0"] * stride_frame * avg_frame
                if "tau" in result_data:
                    result_data["time_elapsed"] = result_data["tau"] * result_data["t0"]
                result_data["g2_t0"] = result_data["t0"]

        if "saxs_1d" in result_data:
            result_data["saxs_1d"] = self.q_space_map.reshape_phi_analysis(
                result_data["saxs_1d"], self.label, mode="saxs_1d"
            )
        if "Iqp" in result_data:
            result_data["Iqp"] = self.q_space_map.reshape_phi_analysis(
                result_data["Iqp"], self.label, mode="stability"
            )

        result_data["abs_cross_section_scale"] = 1.0
        return result_data

    def __getattr__(self, key):
        # keys from q_space_map
        if key in [
            "sqlist",
            "dqlist",
            "dqmap",
            "sqmap",
            "mask",
            "bcx",
            "bcy",
            "det_dist",
            "pixel_size",
            "X_energy",
            "splist",
            "dplist",
            "static_num_pts",
            "dynamic_num_pts",
            "map_names",
            "map_units",
            "get_qbin_label",
        ]:
            return self.q_space_map.__dict__[key]
        # delayed loading of saxs_2d due to its large size
        elif key == "saxs_2d":
            if self.saxs_2d_data is None:
                result_data = get(self.filename, ["saxs_2d"], "alias", file_type="nexus")
                if result_data is not None and isinstance(result_data, dict) and "saxs_2d" in result_data:
                    self.saxs_2d_data = result_data["saxs_2d"]
            return self.saxs_2d_data
        elif key == "saxs_2d_log":
            if self.saxs_2d_log_data is None:
                saxs_2d = self.saxs_2d
                if saxs_2d is not None:
                    saxs = saxs_2d.copy()  # More idiomatic numpy copy
                    # Use np.any for faster check
                    if not np.any(saxs > 0):
                        self.saxs_2d_log_data = np.zeros_like(saxs, dtype=np.float32)
                    else:
                        # More efficient masking and log computation
                        min_val = saxs[saxs > 0].min()
                        saxs = np.where(saxs > 0, saxs, min_val)
                        self.saxs_2d_log_data = np.log10(saxs, dtype=np.float32)
                else:
                    # Return None if saxs_2d data is not available
                    self.saxs_2d_log_data = None
            return self.saxs_2d_log_data
        elif key == "Int_t_fft":
            Int_t = getattr(self, 'Int_t', None)
            if Int_t is None:
                return None
            # Use rfft for real input data (more efficient)
            y = np.abs(np.fft.rfft(Int_t[1]))
            t0 = getattr(self, 't0', 1)
            # More efficient array generation
            n_half = y.size
            x = np.arange(n_half, dtype=np.float32) / (2 * (n_half - 1) * t0)
            y = y.astype(np.float32)
            y[0] = 0  # Set DC component to zero
            return np.vstack((x, y))
        elif key in self.__dict__:
            return self.__dict__[key]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def get_info_at_position(self, x: int, y: int) -> Optional[str]:
        """Get information at a specific detector position.

        Parameters
        ----------
        x, y : int
            Pixel coordinates on the detector.

        Returns
        -------
        str or None
            Formatted string with intensity and q-map information, or None if out of bounds.
        """
        x, y = int(x), int(y)
        saxs_2d = self.saxs_2d
        if saxs_2d is None:
            return None
        shape = saxs_2d.shape
        if x < 0 or x >= shape[1] or y < 0 or y >= shape[0]:
            return None
        else:
            scat_intensity = saxs_2d[y, x]
            qmap_info = self.q_space_map.get_qmap_at_pos(x, y)
            return f"I={scat_intensity:.4e} {qmap_info}"

    def get_detector_extent(self) -> Tuple[float, float, float, float]:
        """Get the detector extent for plotting.

        Returns
        -------
        tuple
            (left, right, bottom, top) extent values.
        """
        return self.q_space_map.extent

    def get_qbin_label(self, qbin: int, append_qbin: bool = False) -> str:
        """Get label for a specific q-bin.

        Parameters
        ----------
        qbin : int
            Q-bin index.
        append_qbin : bool, optional
            Whether to append the q-bin number to the label.

        Returns
        -------
        str
            Formatted q-bin label.
        """
        return self.q_space_map.get_qbin_label(qbin, append_qbin=append_qbin)

    def get_qbinlist_at_qindex(self, qindex: int, zero_based: bool = True) -> List[int]:
        """Get list of q-bins at a specific q-index.

        Parameters
        ----------
        qindex : int
            Q-index value.
        zero_based : bool, optional
            Whether to use zero-based indexing.

        Returns
        -------
        list of int
            List of q-bin indices.
        """
        return self.q_space_map.get_qbinlist_at_qindex(qindex, zero_based=zero_based)

    def get_g2_data(self, q_range: Optional[Tuple[float, float]] = None,
                    time_range: Optional[Tuple[float, float]] = None) -> Tuple[Any, Any, Any, Any, List[str]]:
        """Get G2 correlation data within specified q and time ranges.

        Parameters
        ----------
        q_range : tuple of float, optional
            (q_min, q_max) range for q-values.
        time_range : tuple of float, optional
            (t_min, t_max) range for time values.

        Returns
        -------
        tuple
            (q_values, time_elapsed, g2, g2_err, labels)
        """
        assert "Multitau" in self.analysis_type, "only multitau is supported"
        # q_range can be None
        qindex_selected, qvalues = self.q_space_map.get_qbin_in_qrange(q_range, zero_based=True)
        g2_data = getattr(self, 'g2', None)
        g2_err_data = getattr(self, 'g2_err', None)
        if g2_data is None or g2_err_data is None:
            raise ValueError("G2 data is not available")
        g2 = g2_data[:, qindex_selected]
        g2_err = g2_err_data[:, qindex_selected]
        labels = [self.q_space_map.get_qbin_label(qbin + 1) for qbin in qindex_selected]

        time_elapsed_data = getattr(self, 'time_elapsed', None)
        if time_elapsed_data is None:
            raise ValueError("time_elapsed data is not available")

        if time_range is not None:
            t_roi = (time_elapsed_data >= time_range[0]) * (time_elapsed_data <= time_range[1])
            g2 = g2[t_roi]
            g2_err = g2_err[t_roi]
            time_elapsed = time_elapsed_data[t_roi]
        else:
            time_elapsed = time_elapsed_data

        return qvalues, time_elapsed, g2, g2_err, labels

    def get_saxs1d_data(
        self,
        bkg_xf=None,
        bkg_weight=1.0,
        q_range=None,
        sampling=1,
        use_absolute_crosssection=False,
        norm_method=None,
        target="saxs1d",
        qrange=None,  # Deprecated: use q_range instead
    ):
        # Handle backward compatibility for qrange parameter
        if qrange is not None and q_range is None:
            import warnings
            warnings.warn(
                "Parameter 'qrange' is deprecated, use 'q_range' instead",
                DeprecationWarning,
                stacklevel=2
            )
            q_range = qrange
            
        assert target in ["saxs1d", "saxs1d_partial"]
        saxs_1d = getattr(self, 'saxs_1d', None)
        if saxs_1d is None:
            raise ValueError("saxs_1d data is not available")
        if target == "saxs1d":
            q, Iq = saxs_1d["q"], saxs_1d["Iq"]
        else:
            Iqp_data = getattr(self, 'Iqp', None)
            if Iqp_data is None:
                raise ValueError("Iqp data is not available")
            q, Iq = saxs_1d["q"], Iqp_data
        if bkg_xf is not None:
            bkg_saxs_1d = getattr(bkg_xf, 'saxs_1d', None)
            if bkg_saxs_1d is not None and np.allclose(q, bkg_saxs_1d["q"]):
                Iq = Iq - bkg_weight * bkg_saxs_1d["Iq"]
                Iq[Iq < 0] = np.nan
            else:
                logger.warning(
                    "background subtraction is not applied because q is not matched"
                )
        if q_range is not None:
            q_roi = (q >= q_range[0]) * (q <= q_range[1])
            if q_roi.sum() > 0:
                q = q[q_roi]
                Iq = Iq[:, q_roi]
            else:
                logger.warning("q_range is not applied because it is out of range")
        if use_absolute_crosssection and self.abs_cross_section_scale is not None:
            Iq *= self.abs_cross_section_scale

        # apply sampling
        if sampling > 1:
            q, Iq = q[::sampling], Iq[:, ::sampling] if Iq.ndim > 1 else Iq[::sampling]
        # apply normalization
        q, Iq, xlabel, ylabel = self.norm_saxs_data(q, Iq, norm_method=norm_method)
        return q, Iq, xlabel, ylabel

    def norm_saxs_data(self, q, Iq, norm_method=None):
        assert norm_method in (None, "q2", "q4", "I0")
        if norm_method is None:
            return q, Iq, "q (Å⁻¹)", "Intensity"
        ylabel = "Intensity"
        if norm_method == "q2":
            Iq = Iq * np.square(q)
            ylabel = ylabel + " * q^2"
        elif norm_method == "q4":
            Iq = Iq * np.square(np.square(q))
            ylabel = ylabel + " * q^4"
        elif norm_method == "I0":
            baseline = Iq[0]
            Iq = Iq / baseline
            ylabel = ylabel + " / I_0"
        xlabel = "q (Å⁻¹)"
        return q, Iq, xlabel, ylabel

    def get_twotime_qbin_labels(self):
        qbin_labels = []
        c2_processed_bins = getattr(self, 'c2_processed_bins', None)
        if c2_processed_bins is None:
            return []
        for qbin in c2_processed_bins.tolist():
            qbin_labels.append(self.get_qbin_label(qbin, append_qbin=True))
        return qbin_labels

    def get_twotime_maps(
        self, scale="log", auto_crop=True, highlight_xy=None, selection=None
    ):
        # emphasize the beamstop region which has qindex = 0;
        dqmap = np.copy(self.dqmap)
        if scale == "log":
            saxs = self.saxs_2d_log
        else:
            saxs = self.saxs_2d

        if auto_crop:
            idx = np.nonzero(dqmap >= 1)
            if len(idx[0]) > 0 and len(idx[1]) > 0:
                sl_v = slice(np.min(idx[0]), np.max(idx[0]) + 1)
                sl_h = slice(np.min(idx[1]), np.max(idx[1]) + 1)
            else:
                sl_v = slice(None)
                sl_h = slice(None)
            dqmap = dqmap[sl_v, sl_h]
            if saxs is not None:
                saxs = saxs[sl_v, sl_h]

        qindex_max = np.max(dqmap)
        dqlist = np.unique(dqmap)[1:]
        dqmap = dqmap.astype(np.float32)
        dqmap[dqmap == 0] = np.nan

        dqmap_disp = np.flipud(np.copy(dqmap))

        dq_bin = None
        if highlight_xy is not None:
            x, y = highlight_xy
            if x >= 0 and y >= 0 and x < dqmap.shape[1] and y < dqmap.shape[0]:
                dq_bin = dqmap_disp[y, x]
        elif selection is not None:
            dq_bin = dqlist[selection]

        if dq_bin is not None and dq_bin != np.nan and dq_bin > 0:
            # highlight the selected qbin if it's valid
            dqmap_disp[dqmap_disp == dq_bin] = qindex_max + 1
            selection = np.where(dqlist == dq_bin)[0][0]
        else:
            selection = None
        return dqmap_disp, saxs, selection

    def get_twotime_c2(self, selection=0, correct_diag=True, max_size=32678):
        c2_processed_bins = getattr(self, 'c2_processed_bins', None)
        if c2_processed_bins is None:
            raise ValueError("c2_processed_bins is not available")
        dq_processed = tuple(c2_processed_bins.tolist())
        assert selection >= 0 and selection < len(
            dq_processed
        ), f"selection {selection} out of range {dq_processed}"  # noqa: E501
        config = (selection, correct_diag, max_size)
        if self.c2_kwargs == config:
            return self.c2_all_data
        else:
            fname = getattr(self, 'fname', self.filename)
            c2_result = get_single_c2_from_hdf(
                fname,
                selection=selection,
                max_size=max_size,
                t0=getattr(self, 't0', None),
                correct_diag=correct_diag,
            )
            self.c2_all_data = c2_result
            self.c2_kwargs = config
        return c2_result

    def get_twotime_stream(self, **kwargs):
        fname = getattr(self, 'fname', self.filename)
        return get_c2_stream(fname, **kwargs)

    # def get_g2_fitting_line(self, q, tor=1e-6):
    #     """
    #     get the fitting line for q, within tor
    #     """
    #     if self.fit_summary is None:
    #         return None, None
    #     idx = np.argmin(np.abs(self.fit_summary["q_val"] - q))
    #     if abs(self.fit_summary["q_val"][idx] - q) > tor:
    #         return None, None

    #     fit_x = self.fit_summary["fit_line"][idx]["fit_x"]
    #     fit_y = self.fit_summary["fit_line"][idx]["fit_y"]
    #     return fit_x, fit_y

    def get_fitting_info(self, mode="g2_fitting"):
        if self.fit_summary is None:
            return "fitting is not ready for %s" % self.label

        if mode == "g2_fitting":
            result = self.fit_summary.copy()
            # fit_line is not useful to display
            result.pop("fit_line", None)
            val = result.pop("fit_val", None)
            if result["fit_func"] == "single":
                prefix = ["a", "b", "c", "d"]
            else:
                prefix = ["a", "b", "c", "d", "b2", "c2", "f"]

            msg = []
            for n in range(val.shape[0]):
                temp = []
                for m in range(len(prefix)):
                    temp.append(
                        "%s = %f ± %f" % (prefix[m], val[n, 0, m], val[n, 1, m])
                    )
                msg.append(", ".join(temp))
            result["fit_val"] = np.array(msg)

        elif mode == "tauq_fitting":
            if "tauq_fit_val" not in self.fit_summary:
                result = "tauq fitting is not available"
            else:
                v = self.fit_summary["tauq_fit_val"]
                result = "a = %e ± %e; b = %f ± %f" % (
                    v[0, 0],
                    v[1, 0],
                    v[0, 1],
                    v[1, 1],
                )
        else:
            raise ValueError("mode not supported.")

        return result

    def fit_g2(
        self, q_range=None, t_range=None, bounds=None, fit_flag=None, fit_func="single"
    ):
        """
        fit the g2 values using single exponential decay function
        :param q_range: a tuple of q lower bound and upper bound
        :param t_range: a tuple of t lower bound and upper bound
        :param bounds: bounds for fitting;
        :param fit_flag: tuple of bools; True to fit and False to float
        :param fit_func: ["single" | "double"]: to fit with single exponential
            or double exponential function
        :return: dictionary with the fitting result;
        """
        if bounds is None:
            raise ValueError("bounds cannot be None")
        assert len(bounds) == 2
        if fit_func == "single":
            assert (
                len(bounds[0]) == 4
            ), "for single exp, the shape of bounds must be (2, 4)"
            if fit_flag is None:
                fit_flag = [True for _ in range(4)]
            func = single_exp_all
        else:
            assert (
                len(bounds[0]) == 7
            ), "for single exp, the shape of bounds must be (2, 4)"
            if fit_flag is None:
                fit_flag = [True for _ in range(7)]
            func = double_exp_all

        q_val, t_el, g2, sigma, label = self.get_g2_data(q_range=q_range, time_range=t_range)

        # set the initial guess
        p0 = np.array(bounds).mean(axis=0)
        # tau"s bounds are in log scale, set as the geometric average
        p0[1] = np.sqrt(bounds[0][1] * bounds[1][1])
        if fit_func == "double":
            p0[4] = np.sqrt(bounds[0][4] * bounds[1][4])

        fit_x = np.logspace(
            np.log10(np.min(t_el)) - 0.5, np.log10(np.max(t_el)) + 0.5, 128
        )

        fit_line, fit_val = fit_with_fixed(
            func, t_el, g2, sigma, bounds, fit_flag, fit_x, p0=p0
        )

        self.fit_summary = {
            "fit_func": fit_func,
            "fit_val": fit_val,
            "t_el": t_el,
            "q_val": q_val,
            "q_range": str(q_range),
            "t_range": str(t_range),
            "bounds": bounds,
            "fit_flag": str(fit_flag),
            "fit_line": fit_line,
            "label": label,
        }

        return self.fit_summary

    @staticmethod
    def correct_g2_err(g2_err=None, threshold=1e-6):
        # correct the err for some data points with really small error, which
        # may cause the fitting to blowup

        if g2_err is None:
            return None
        g2_err_mod = np.copy(g2_err)
        for n in range(g2_err.shape[1]):
            data = g2_err[:, n]
            idx = data > threshold
            # avoid averaging of empty slice
            if np.sum(idx) > 0:
                avg = np.mean(data[idx])
            else:
                avg = threshold
            g2_err_mod[np.logical_not(idx), n] = avg

        return g2_err_mod

    def fit_tauq(self, q_range, bounds, fit_flag):
        if self.fit_summary is None:
            return

        x = self.fit_summary["q_val"]
        q_slice = (x >= q_range[0]) * (x <= q_range[1])
        x = x[q_slice]

        y = self.fit_summary["fit_val"][q_slice, 0, 1]
        sigma = self.fit_summary["fit_val"][q_slice, 1, 1]

        # filter out those invalid fittings; failed g2 fitting has -1 err
        valid_idx = sigma > 0

        if np.sum(valid_idx) == 0:
            self.fit_summary["tauq_success"] = False
            return

        x = x[valid_idx]
        y = y[valid_idx]
        sigma = sigma[valid_idx]

        # reshape to two-dimension so the fit_with_fixed function works
        y = y.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # the initial value for typical gel systems
        p0 = [1.0e-7, -2.0]
        fit_x = np.logspace(np.log10(np.min(x) / 1.1), np.log10(np.max(x) * 1.1), 128)

        fit_line, fit_val = fit_with_fixed(
            power_law, x, y, sigma, bounds, fit_flag, fit_x, p0=p0
        )

        # fit_line and fit_val are lists with just one element;
        self.fit_summary["tauq_success"] = fit_line[0]["success"]
        self.fit_summary["tauq_q"] = x
        self.fit_summary["tauq_tau"] = np.squeeze(y)
        self.fit_summary["tauq_tau_err"] = np.squeeze(sigma)
        self.fit_summary["tauq_fit_line"] = fit_line[0]
        self.fit_summary["tauq_fit_val"] = fit_val[0]

        return self.fit_summary

    def compute_qmap(self):
        """Compute qmap from q_space_map if available."""
        if hasattr(self.q_space_map, 'compute_qmap'):
            return self.q_space_map.compute_qmap()
        return None

    def get_roi_data(self, roi_parameter, phi_num=180):
        qmap_all = self.compute_qmap()
        if qmap_all is None:
            raise ValueError("Cannot compute qmap")
        qmap = qmap_all["q"] if isinstance(qmap_all, dict) else None
        pmap = qmap_all["phi"] if isinstance(qmap_all, dict) else None
        rmap = qmap_all["r_pixel"] if isinstance(qmap_all, dict) else None

        if qmap is None or pmap is None or rmap is None:
            raise ValueError("Required qmap components are not available")

        if roi_parameter["sl_type"] == "Pie":
            pmin, pmax = roi_parameter["angle_range"]
            if pmax < pmin:
                pmax += 360.0
                pmap[pmap < pmin] += 360.0
            proi = np.logical_and(pmap >= pmin, pmap < pmax)
            mask = getattr(self, 'mask', None)
            if mask is None:
                raise ValueError("Mask is not available")
            proi = np.logical_and(proi, (mask > 0))
            qmap_idx = np.zeros_like(qmap, dtype=np.uint32)

            index = 1
            sqspan = getattr(self, 'sqspan', None)
            if sqspan is None:
                raise ValueError("sqspan is not available")
            qsize = len(sqspan) - 1
            for n in range(qsize):
                q0, q1 = sqspan[n : n + 2]
                select = (qmap >= q0) * (qmap < q1)
                qmap_idx[select] = index
                index += 1
            qmap_idx = (qmap_idx * proi).ravel()

            saxs_2d = self.saxs_2d
            if saxs_2d is None:
                raise ValueError("saxs_2d is not available")
            saxs_roi = np.bincount(qmap_idx, saxs_2d.ravel(), minlength=qsize + 1)
            saxs_nor = np.bincount(qmap_idx, minlength=qsize + 1)
            saxs_nor[saxs_nor == 0] = 1.0
            saxs_roi = saxs_roi * 1.0 / saxs_nor

            # remove the 0th term
            saxs_roi = saxs_roi[1:]

            # set the qmax cutoff
            dist = roi_parameter["dist"]
            # qmax = qmap[int(self.bcy), int(self.bcx + dist)]
            X_energy = getattr(self, 'X_energy', None)
            pix_dim_x = getattr(self, 'pix_dim_x', None)
            det_dist = getattr(self, 'det_dist', None)
            if X_energy is None or pix_dim_x is None or det_dist is None:
                raise ValueError("Required attributes for qmax calculation are not available")
            wlength = 12.398 / X_energy
            qmax = dist * pix_dim_x / det_dist * 2 * np.pi / wlength
            saxs_roi[self.sqlist >= qmax] = 0
            saxs_roi[saxs_roi <= 0] = np.nan
            return self.sqlist, saxs_roi

        elif roi_parameter["sl_type"] == "Ring":
            rmin, rmax = roi_parameter["radius"]
            if rmin > rmax:
                rmin, rmax = rmax, rmin
            rroi = np.logical_and(rmap >= rmin, rmap < rmax)
            mask = getattr(self, 'mask', None)
            if mask is None:
                raise ValueError("Mask is not available")
            rroi = np.logical_and(rroi, (mask > 0))

            phi_min, phi_max = np.min(pmap[rroi]), np.max(pmap[rroi])
            x = np.linspace(phi_min, phi_max, phi_num)
            delta = (phi_max - phi_min) / phi_num
            index = ((pmap - phi_min) / delta).astype(np.int64)
            index[index == phi_num] = phi_num - 1
            index += 1
            # q_avg = qmap[rroi].mean()
            index = (index * rroi).ravel()

            saxs_2d = self.saxs_2d
            if saxs_2d is None:
                raise ValueError("saxs_2d is not available")
            saxs_roi = np.bincount(index, saxs_2d.ravel(), minlength=phi_num + 1)
            saxs_nor = np.bincount(index, minlength=phi_num + 1)
            saxs_nor[saxs_nor == 0] = 1.0
            saxs_roi = saxs_roi * 1.0 / saxs_nor

            # remove the 0th term
            saxs_roi = saxs_roi[1:]
            return x, saxs_roi

    def export_saxs1d(self, roi_list, folder):
        # export ROI
        idx = 0
        if roi_list is None:
            return
        for roi in roi_list:
            fname = os.path.join(
                folder, self.label + "_" + roi["sl_type"] + f"_{idx:03d}.txt"
            )
            idx += 1
            roi_data = self.get_roi_data(roi)
            if roi_data is None:
                continue
            x, y = roi_data
            if roi["sl_type"] == "Ring":
                header = "phi(degree) Intensity"
            else:
                header = "q(1/Angstron) Intensity"
            np.savetxt(fname, np.vstack([x, y]).T, header=header)

        # export all saxs1d
        fname = os.path.join(folder, self.label + "_" + "saxs1d.txt")
        saxs_1d = getattr(self, 'saxs_1d', None)
        if saxs_1d is None:
            raise ValueError("saxs_1d data is not available")
        Iq, q = saxs_1d["Iq"], saxs_1d["q"]
        header = "q(1/Angstron) Intensity"
        for n in range(Iq.shape[0] - 1):
            header += f" Intensity_phi{n + 1 :03d}"
        np.savetxt(fname, np.vstack([q, Iq]).T, header=header)

    def get_pg_tree(self):
        """Return data tree in dictionary format for headless usage"""
        from .mpl_compat import DataTreeWidget

        data = self.load_dataset()
        if data is None:
            data = {}
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                if val.size > 4096:
                    data[key] = "data size is too large"
                # squeeze one-element array
                if val.size == 1:
                    data[key] = float(val)
        data["analysis_type"] = getattr(self, 'atype', self.analysis_type)
        data["label"] = self.label

        # Return mock tree widget for compatibility
        tree = DataTreeWidget(data=data)
        tree.setWindowTitle(getattr(self, 'fname', self.filename))
        tree.resize(600, 800)
        return tree


# Backward compatibility layer
class XpcsFile(XpcsDataFile):
    """Deprecated: Use XpcsDataFile instead.

    This class is provided for backward compatibility only.
    New code should use XpcsDataFile directly.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "XpcsFile is deprecated, use XpcsDataFile instead",
            DeprecationWarning,
            stacklevel=2
        )
        # Handle the old parameter name 'fname'
        if 'fname' in kwargs:
            kwargs['filename'] = kwargs.pop('fname')
        super().__init__(*args, **kwargs)

        # Add backward compatibility attributes
        self.fname = self.filename
        self.qmap = self.q_space_map
        self.atype = self.analysis_type
        self.hdf_info = self.hdf_metadata

    # Deprecated method aliases
    def get_hdf_info(self, fstr=None):
        """Deprecated: Use get_hdf_metadata() instead."""
        warnings.warn(
            "get_hdf_info is deprecated, use get_hdf_metadata instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_hdf_metadata(filter_strings=fstr)

    def load_data(self, extra_fields=None):
        """Deprecated: Use load_dataset() instead."""
        warnings.warn(
            "load_data is deprecated, use load_dataset instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.load_dataset(extra_fields=extra_fields)


# Deprecated function alias
def create_id(*args, **kwargs):
    """Deprecated: Use create_identifier() instead."""
    warnings.warn(
        "create_id is deprecated, use create_identifier instead",
        DeprecationWarning,
        stacklevel=2
    )
    return create_identifier(*args, **kwargs)


def test1():
    # Example usage (commented out for testing)
    # cwd = "../../../xpcs_data"
    # af = XpcsFile(filename="N077_D100_att02_0128_0001-100000.hdf")
    # af.plot_saxs2d()  # Method doesn't exist in current implementation
    pass


if __name__ == "__main__":
    test1()
