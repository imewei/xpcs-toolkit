"""
Intensity dynamics analysis for XPCS experiments.

This module provides functions to analyze and visualize intensity time series
data from XPCS measurements, with support for data smoothing and sampling.
"""

import numpy as np

# PyQtGraph import removed for headless operation
pg = None

colors = [
    (192, 0, 0),
    (0, 176, 80),
    (0, 32, 96),
    (255, 0, 0),
    (0, 176, 240),
    (0, 32, 96),
    (255, 164, 0),
    (146, 208, 80),
    (0, 112, 192),
    (112, 48, 160),
    (54, 96, 146),
    (150, 54, 52),
    (118, 147, 60),
    (96, 73, 122),
    (49, 134, 155),
    (226, 107, 10),
]


def smooth_data(fc, window=1, sampling=1):
    """
    Smooth and sample intensity dynamics data.

    Parameters
    ----------
    fc : object
        XPCS file object containing Int_t intensity data
    window : int, default 1
        Moving average window size for smoothing
    sampling : int, default 1
        Downsampling factor for data reduction

    Returns
    -------
    tuple
        (x, y) arrays of frame indices and smoothed intensities
    """
    # some bad frames have both x and y = 0;
    # x, y = fc.Int_t[0], fc.Int_t[1]
    y = fc.Int_t[1]
    x = np.arange(y.shape[0])

    if window > 1:
        y = np.cumsum(y, dtype=float, axis=0)
        y = (y[window:] - y[:-window]) / window
        x = x[window:]
    if sampling >= 2:
        y = y[::sampling]
        x = x[::sampling]

    return x, y


def plot(xf_list, pg_hdl, enable_zoom=True, xlabel="Frame Index", **kwargs):
    """
    Plot intensity vs time - disabled in headless mode.

    This function has been disabled as it requires PyQtGraph GUI functionality.
    Use the matplotlib-based CLI interface for visualization instead.

    :param xf_list: list of xf objects
    :param pg_hdl: pyqtgraph handler to plot (ignored in headless mode)
    :param enable_zoom: bool, if to plot the zoom view or not (ignored)
    :param xlabel: x-axis label (ignored)
    :param kwargs: used to define how to average/sample the data (ignored)
    :return: None
    """
    raise NotImplementedError(
        "GUI plotting functionality has been disabled in headless mode. "
        "Use the matplotlib-based CLI interface for visualization instead."
    )
