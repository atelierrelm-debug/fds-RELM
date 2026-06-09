"""pyrelm — a Pythonic interface for FDS (Fire Dynamics Simulator).

Build fire models in Python, run them on the real FDS solver, and read
the results back as numpy/pandas objects.
"""

from .model import Model, Mesh
from .namelist import Namelist
from .runner import find_fds, run
from .results import Results, Slice
from . import plot

__version__ = "0.1.0"

__all__ = ["Model", "Mesh", "Namelist", "find_fds", "run", "Results",
           "Slice", "plot"]
