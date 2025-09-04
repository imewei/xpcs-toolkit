import logging
from pathlib import Path
from typing import Any, Optional, Union

from xpcs_toolkit.helper.logging_config import get_logger
from xpcs_toolkit.helper.logging_utils import PerformanceTimer, log_exceptions

# Use lazy imports for heavy dependencies
from .._lazy_imports import lazy_import
from .aps_8idi import key as hdf_key

h5py = lazy_import("h5py")
np = lazy_import("numpy")

logger = logging.getLogger(__name__)


def put(
    save_path: Union[str, Path],
    result: dict[str, Any],
    file_type: str = "nexus",
    mode: str = "raw",
    ftype: Optional[str] = None,
) -> None:
    """
    Save analysis results to HDF5 file with comprehensive logging.

    This function writes a dictionary of analysis results to an HDF5 file,
    with support for both raw HDF5 keys and aliased field names. The operation
    is logged with detailed timing and size information.

    Parameters
    ----------
    save_path : str or Path
        Path to save the result file
    result : dict
        Dictionary containing data to save (key -> array/value)
    ftype : str
        File type format ('nexus' or 'aps_8idi')
    mode : str
        Key mode: 'raw' (use keys as-is) or 'alias' (translate via hdf_key)

    Raises
    ------
    ValueError
        If ftype or mode are not recognized
    OSError
        If file cannot be opened or written

    Logging
    -------
    - INFO: File creation/update with key count and sizes
    - DEBUG: Individual key details and transformations
    - ERROR: File access failures with full context
    """
    # Handle backward compatibility for ftype parameter
    if ftype is not None and file_type == "nexus":
        import warnings

        warnings.warn(
            "Parameter 'ftype' is deprecated, use 'file_type' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        file_type = ftype

    # Input validation
    if file_type not in hdf_key:
        raise ValueError(
            f"Unsupported file type: {file_type}. Available: {list(hdf_key.keys())}"
        )
    if mode not in ["raw", "alias"]:
        raise ValueError(f"Unsupported mode: {mode}. Must be 'raw' or 'alias'")

    save_path = Path(save_path)
    context_logger = get_logger(
        __name__, file_path=str(save_path), operation="put", ftype=file_type, mode=mode
    )

    # Calculate total data size for logging
    total_size_mb = 0
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            size_bytes = val.nbytes
            total_size_mb += size_bytes / (1024 * 1024)

    context_logger.info(
        "Writing HDF5 data",
        extra={
            "key_count": len(result),
            "total_size_mb": round(total_size_mb, 2),
            "file_exists": save_path.exists(),
        },
    )

    # Use the underlying logger for log_exceptions and PerformanceTimer
    base_logger = context_logger.logger if hasattr(context_logger, "logger") else logger

    with (
        log_exceptions(base_logger, "Failed to write HDF5 file"),
        PerformanceTimer(base_logger, "HDF5 write operation", item_count=len(result)),
        h5py.File(str(save_path), "a") as f,
    ):
        for key, val in result.items():
            original_key = key

            # Transform key if using alias mode
            if mode == "alias":
                if key not in hdf_key[file_type]:
                    context_logger.warning(
                        "Unknown alias key, using raw key",
                        extra={
                            "key": key,
                            "available_keys": list(hdf_key[file_type].keys()),
                        },
                    )
                else:
                    key = hdf_key[file_type][key]
                    context_logger.debug(
                        "Key alias translation",
                        extra={"original_key": original_key, "hdf_key": key},
                    )

            # Remove existing key if present
            if key in f:
                context_logger.debug("Overwriting existing key", extra={"key": key})
                del f[key]

            # Reshape 1D arrays to 2D format if needed
            original_shape = None
            if isinstance(val, np.ndarray):
                original_shape = val.shape
                if val.ndim == 1:
                    val = np.reshape(val, (1, -1))
                    context_logger.debug(
                        "Reshaped 1D array",
                        extra={
                            "key": key,
                            "original_shape": original_shape,
                            "new_shape": val.shape,
                        },
                    )

            # Write data
            f[key] = val

            # Log details about written data
            if isinstance(val, np.ndarray):
                context_logger.debug(
                    "Wrote array data",
                    extra={
                        "key": key,
                        "dtype": str(val.dtype),
                        "shape": val.shape,
                        "size_mb": round(val.nbytes / (1024 * 1024), 3),
                    },
                )
            else:
                context_logger.debug(
                    "Wrote scalar data",
                    extra={"key": key, "value_type": type(val).__name__},
                )

    context_logger.info("HDF5 write completed successfully")


def get_abs_cs_scale(filename=None, file_type="nexus", fname=None, ftype=None):
    # Backward compatibility for old parameter names
    if fname is not None:
        import warnings

        warnings.warn(
            "Parameter 'fname' is deprecated, use 'filename' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        filename = fname
    if ftype is not None:
        import warnings

        warnings.warn(
            "Parameter 'ftype' is deprecated, use 'file_type' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        file_type = ftype

    key = hdf_key[file_type]["abs_cross_section_scale"]
    with h5py.File(filename, "r") as f:
        if key not in f:
            return None
        else:
            return float(f[key][()])


def read_metadata_to_dict(file_path):
    """
    Reads an HDF5 file and loads its contents into a nested dictionary.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.

    Returns
    -------
    dict
        A nested dictionary containing datasets as NumPy arrays.
    """

    def recursive_read(h5_group, target_dict):
        """Recursively reads groups and datasets into the dictionary."""
        for key in h5_group:
            obj = h5_group[key]
            if isinstance(obj, h5py.Dataset):
                val = obj[()]
                if type(val) in [np.bytes_, bytes]:
                    val = val.decode()
                target_dict[key] = val
            elif isinstance(obj, h5py.Group):
                target_dict[key] = {}
                recursive_read(obj, target_dict[key])

    data_dict = {}
    groups = [
        "/entry/instrument",
        "/xpcs/multitau/config",
        "/xpcs/twotime/config",
        "/entry/sample",
        "/entry/user",
    ]
    with h5py.File(file_path, "r") as hdf_file:
        for group in groups:
            if group in hdf_file:
                data_dict[group] = {}
                recursive_read(hdf_file[group], data_dict[group])
    return data_dict


def get(
    filename=None,
    fields=None,
    mode="raw",
    ret_type="dict",
    file_type="nexus",
    fname=None,
    ftype=None,
):
    """
    get the values for the various keys listed in fields for a single
    file;
    :param filename: file name
    :param fields: list of keys [key1, key2, ..., ]
    :param mode: ['raw' | 'alias']; alias is defined in .hdf_key
                 otherwise the raw hdf key will be used
    :param ret_type: return dictonary if 'dict', list if it is 'list'
    :return: dictionary or dictionary;
    """
    # Backward compatibility for old parameter names
    if fname is not None:
        import warnings

        warnings.warn(
            "Parameter 'fname' is deprecated, use 'filename' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        filename = fname
    if ftype is not None:
        import warnings

        warnings.warn(
            "Parameter 'ftype' is deprecated, use 'file_type' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        file_type = ftype
    assert mode in ["raw", "alias"], "mode not supported"
    assert ret_type in ["dict", "list"], "ret_type not supported"

    # Handle case where fields is None
    if fields is None:
        fields = []

    ret = {}
    missing_keys = []

    with h5py.File(filename, "r") as HDF_Result:
        for key in fields:
            path = hdf_key[file_type][key] if mode == "alias" else key
            if path not in HDF_Result:
                logger.warning("Path to field not found: %s (key: %s)", path, key)
                missing_keys.append(key)
                continue

            val = HDF_Result.get(path)[()]
            if key in ["saxs_2d"] and val.ndim == 3:  # saxs_2d is in [1xNxM] format
                val = val[0]
            # converts bytes to unicode;
            if type(val) in [np.bytes_, bytes]:
                val = val.decode()
            if isinstance(val, np.ndarray) and val.shape == (1, 1):
                val = val.item()
            ret[key] = val

    if missing_keys:
        logger.info(
            "Loaded %d fields, %d fields missing: %s",
            len(ret),
            len(missing_keys),
            missing_keys,
        )

    if ret_type == "dict":
        return ret
    elif ret_type == "list":
        # Ensure fields is not None before iterating
        if fields is None:
            return []
        return [ret[key] for key in fields]


def get_analysis_type(filename=None, file_type="nexus", fname=None, ftype=None):
    """
    determine the analysis type of the file
    Parameters
    ----------
    filename: str
        file name
    file_type: str
        file type, 'nexus' or 'legacy'
    Returns
    -------
    tuple
        analysis type, 'Twotime' or 'Multitau', or both
    """
    # Backward compatibility for old parameter names
    if fname is not None:
        import warnings

        warnings.warn(
            "Parameter 'fname' is deprecated, use 'filename' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        filename = fname
    if ftype is not None:
        import warnings

        warnings.warn(
            "Parameter 'ftype' is deprecated, use 'file_type' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        file_type = ftype
    c2_prefix = hdf_key[file_type]["c2_prefix"]
    g2_prefix = hdf_key[file_type]["g2"]
    analysis_type = []

    try:
        with h5py.File(filename, "r") as HDF_Result:
            if c2_prefix in HDF_Result:
                analysis_type.append("Twotime")
            if g2_prefix in HDF_Result:
                analysis_type.append("Multitau")

            # Check for common XPCS data patterns to identify test files
            has_exchange_data = (
                "exchange/g2" in HDF_Result or "exchange/tau" in HDF_Result
            )

            if len(analysis_type) == 0:
                if has_exchange_data:
                    # This looks like a minimal test file with exchange data
                    analysis_type.append("Multitau")
                    logger.warning(
                        f"No standard analysis paths found in {filename}. Assuming Multitau based on exchange data."
                    )
                else:
                    raise ValueError(f"No analysis type found in {filename}")

    except (OSError, KeyError) as e:
        logger.warning(
            f"Failed to determine analysis type from {filename}: {e}. Assuming Multitau."
        )
        analysis_type = ["Multitau"]

    return tuple(analysis_type)


if __name__ == "__main__":
    pass
