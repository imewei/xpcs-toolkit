"""
File type detection utilities for XPCS data files.

This module provides functions to identify and classify different
types of XPCS data files (NeXus, legacy formats).
"""

import os
from typing import Union

import h5py


def isNeXusFile(filename: str) -> bool:
    """
    Check if a file follows NeXus format standards.

    Parameters
    ----------
    filename : str
        Path to the file to check

    Returns
    -------
    bool
        True if file is NeXus format, False otherwise
    """
    try:
        with h5py.File(filename, "r") as f:
            if "/entry/instrument/bluesky/metadata/" in f:
                return True
    except (OSError, FileNotFoundError, PermissionError):
        return False
    return False


def isLegacyFile(filename: str) -> bool:
    """
    Check if a file uses legacy XPCS format.

    Parameters
    ----------
    filename : str
        Path to the file to check

    Returns
    -------
    bool
        True if file is legacy XPCS format, False otherwise
    """
    try:
        with h5py.File(filename, "r") as f:
            if "/xpcs/Version" in f:
                return True
    except (OSError, FileNotFoundError, PermissionError):
        return False
    return False


def get_ftype(filename: str) -> Union[str, bool]:
    """
    Determine the file type of an XPCS data file.

    Parameters
    ----------
    filename : str
        Path to the file to analyze

    Returns
    -------
    Union[str, bool]
        "nexus" for NeXus format, "legacy" for legacy format,
        False if file doesn't exist or format is unrecognized
    """
    if not os.path.isfile(filename):
        return False

    if isLegacyFile(filename):
        return "legacy"
    elif isNeXusFile(filename):
        return "nexus"
    else:
        return False
