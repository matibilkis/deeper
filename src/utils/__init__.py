"""
Utility functions.

Contains plotting utilities and miscellaneous helper functions.
"""

from src.utils.misc import record

# Plotting functions require matplotlib
try:
    from src.utils.plots import plot_learning_curves, plot_beta_histogram
    __all__ = ['plot_learning_curves', 'plot_beta_histogram', 'record']
except ImportError:
    # Matplotlib not available
    __all__ = ['record']

