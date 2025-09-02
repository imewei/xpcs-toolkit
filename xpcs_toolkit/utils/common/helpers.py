"""
Common utility functions for data processing and visualization.

This module provides shared utility functions used throughout the XPCS toolkit
for data manipulation, normalization, and range calculations.
"""

import numpy as np


def get_min_max(data, min_percent=0, max_percent=100, **kwargs):
    """
    Calculate minimum and maximum values for data visualization.

    Parameters
    ----------
    data : array_like
        Input data array
    min_percent : float, default 0
        Minimum percentile for range calculation
    max_percent : float, default 100
        Maximum percentile for range calculation
    **kwargs
        Additional plotting parameters including plot_norm and plot_type

    Returns
    -------
    tuple
        (vmin, vmax) values for visualization range
    """
    vmin = np.percentile(data.ravel(), min_percent)
    vmax = np.percentile(data.ravel(), max_percent)
    if "plot_norm" in kwargs and "plot_type" in kwargs and kwargs["plot_norm"] == 3:
        if kwargs["plot_type"] == "log":
            t = max(abs(vmin), abs(vmax))
            vmin, vmax = -t, t
        else:
            t = max(abs(1 - vmin), abs(vmax - 1))
            vmin, vmax = 1 - t, 1 + t

    return vmin, vmax


def norm_saxs_data(Iq, q, plot_norm=0):
    """
    Normalize SAXS intensity data for different plotting modes.

    Parameters
    ----------
    Iq : array_like
        SAXS intensity values
    q : array_like
        Scattering vector magnitudes
    plot_norm : int, default 0
        Normalization mode (0=none, 1=q^2, 2=q^4, 3=I/I_0)

    Returns
    -------
    tuple
        (normalized_Iq, xlabel, ylabel) for plotting
    """
    ylabel = "Intensity"
    if plot_norm == 1:
        Iq = Iq * np.square(q)
        ylabel = ylabel + " * q^2"
    elif plot_norm == 2:
        Iq = Iq * np.square(np.square(q))
        ylabel = ylabel + " * q^4"
    elif plot_norm == 3:
        baseline = Iq[0]
        Iq = Iq / baseline
        ylabel = ylabel + " / I_0"

    xlabel = "$q (\\AA^{-1})$"
    return Iq, xlabel, ylabel


def create_slice(arr, x_range):
    """
    Create a slice object for array range selection.

    Parameters
    ----------
    arr : array_like
        Input array to slice
    x_range : tuple
        (start, end) range values

    Returns
    -------
    slice
        Slice object for array indexing
    """
    start, end = 0, arr.size - 1
    while arr[start] < x_range[0]:
        start += 1
        if start == arr.size:
            break

    while arr[end] >= x_range[1]:
        end -= 1
        if end == 0:
            break

    return slice(start, end + 1)
