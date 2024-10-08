from importlib.metadata import version

from . import dt, io, pl, pp, tl
from .core import EHRData

__all__ = ["EHRData", "dt", "io", "pl", "pp", "tl"]

__version__ = version("ehrdata")
