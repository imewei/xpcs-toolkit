"""
Matplotlib compatibility layer to replace PyQtGraph functionality.

This module provides matplotlib-based replacements for PyQtGraph functions
used throughout the XPCS analysis modules. It maintains API compatibility
while using matplotlib as the backend.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import warnings
from typing import Optional, Union, Tuple, Any, Dict


class MockDataTreeWidget:
    """Mock replacement for PyQtGraph's DataTreeWidget"""
    def __init__(self, data=None):
        self.data = data
        self._window_title = "Data Tree"
        self._size = (800, 600)
    
    def setWindowTitle(self, title: str):
        self._window_title = title
    
    def resize(self, width: int, height: int):
        self._size = (width, height)
    
    def show(self):
        print(f"DataTree: {self._window_title}")
        if self.data:
            print(self._format_dict(self.data))
    
    def _format_dict(self, d, indent=0):
        """Format dictionary for console display"""
        lines = []
        for key, value in d.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_dict(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)


class CompatPlotWidget:
    """Compatibility wrapper that returns matplotlib Figure/Axes"""
    def __init__(self, figsize=(8, 6)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
    def setImage(self, image, levels=None, **kwargs):
        """Display 2D image data"""
        if levels is not None:
            vmin, vmax = levels
            kwargs.update({'vmin': vmin, 'vmax': vmax})
        
        im = self.ax.imshow(image, **kwargs)
        self.fig.colorbar(im, ax=self.ax)
        return im
    
    def plot(self, x, y=None, pen=None, **kwargs):
        """Plot line data"""
        if y is None:
            y = x
            x = np.arange(len(y))
        
        # Convert PyQtGraph pen notation to matplotlib
        if pen is not None:
            if isinstance(pen, str):
                kwargs['color'] = pen
            elif isinstance(pen, (tuple, list)) and len(pen) >= 3:
                kwargs['color'] = pen[:3] if len(pen) == 3 else pen[:4]
        
        line = self.ax.plot(x, y, **kwargs)
        return line[0]
    
    def clear(self):
        """Clear the plot"""
        self.ax.clear()
    
    def setLabel(self, axis: str, text: str, units: Optional[str] = None):
        """Set axis labels"""
        label = f"{text} ({units})" if units else text
        if axis.lower() == 'left':
            self.ax.set_ylabel(label)
        elif axis.lower() == 'bottom':
            self.ax.set_xlabel(label)


def mkPen(color=None, width=1, style='-'):
    """Create matplotlib-compatible line style dictionary"""
    style_map = {
        '-': '-',
        '--': '--', 
        ':': ':',
        '-.': '-.'
    }
    
    pen_dict = {
        'linewidth': width,
        'linestyle': style_map.get(style, '-')
    }
    
    if color is not None:
        pen_dict['color'] = color
    
    return pen_dict


def mkBrush(color=None, alpha=1.0):
    """Create matplotlib-compatible brush (fill) dictionary"""
    brush_dict = {}
    
    if color is not None:
        brush_dict['facecolor'] = color
    
    if alpha != 1.0:
        brush_dict['alpha'] = alpha
    
    return brush_dict


def ErrorBarItem(x, y, top=None, bottom=None, **kwargs):
    """Create matplotlib errorbar plot"""
    yerr = None
    if top is not None and bottom is not None:
        yerr = [bottom, top]
    elif top is not None:
        yerr = top
    elif bottom is not None:
        yerr = bottom
    
    fig, ax = plt.subplots()
    line = ax.errorbar(x, y, yerr=yerr, **kwargs)
    return fig, ax, line


def PlotWidget(title: str = "", labels: Optional[Dict[str, str]] = None):
    """Create a matplotlib-based plot widget"""
    widget = CompatPlotWidget()
    
    if title:
        widget.ax.set_title(title)
    
    if labels:
        if 'left' in labels:
            widget.ax.set_ylabel(labels['left'])
        if 'bottom' in labels:
            widget.ax.set_xlabel(labels['bottom'])
    
    return widget


def ImageView():
    """Create matplotlib image viewer"""
    return CompatPlotWidget()


def DataTreeWidget(data=None):
    """Create mock data tree widget"""
    return MockDataTreeWidget(data)


# Colormap utilities
def colormap_to_matplotlib(name: str):
    """Convert PyQtGraph colormap names to matplotlib"""
    colormap_mapping = {
        'thermal': 'hot',
        'flame': 'afmhot',
        'yellowy': 'YlOrRd',
        'bipolar': 'RdBu',
        'spectrum': 'nipy_spectral',
        'cyclic': 'hsv',
        'greyclip': 'gray',
        'grey': 'gray'
    }
    return colormap_mapping.get(name, name)


# Signal/slot mock for compatibility
class MockSignal:
    """Mock Qt signal for compatibility"""
    def __init__(self):
        self._callbacks = []
    
    def connect(self, callback):
        self._callbacks.append(callback)
    
    def disconnect(self, callback=None):
        if callback:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass
        else:
            self._callbacks.clear()
    
    def emit(self, *args, **kwargs):
        for callback in self._callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                warnings.warn(f"Signal callback failed: {e}")


# PyQtGraph compatibility functions
def plot(x, y=None, pen=None, symbol=None, **kwargs):
    """Direct plotting function compatible with PyQtGraph"""
    fig, ax = plt.subplots()
    
    if y is None:
        y = x
        x = np.arange(len(y))
    
    # Handle pen (color/style)
    plot_kwargs = {}
    if pen is not None:
        if isinstance(pen, str):
            plot_kwargs['color'] = pen
        elif isinstance(pen, dict):
            plot_kwargs.update(pen)
    
    # Handle symbols (markers)
    if symbol is not None:
        symbol_map = {
            'o': 'o', 's': 's', 't': '^', 'd': 'D', 
            '+': '+', 'x': 'x', '*': '*'
        }
        plot_kwargs['marker'] = symbol_map.get(symbol, symbol)
        plot_kwargs['linestyle'] = 'None'
    
    plot_kwargs.update(kwargs)
    
    line = ax.plot(x, y, **plot_kwargs)
    return fig, ax, line[0]


def image(img, **kwargs):
    """Display image with matplotlib"""
    fig, ax = plt.subplots()
    im = ax.imshow(img, **kwargs)
    fig.colorbar(im, ax=ax)
    return fig, ax, im


# Layout and configuration
setConfigOptions = lambda **kwargs: None  # No-op for matplotlib
setConfigOption = lambda key, value: None  # No-op for matplotlib
