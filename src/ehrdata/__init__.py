from importlib.metadata import version

from . import dt, io, pl, tl
from .core import EHRData

__all__ = ["EHRData", "dt", "io", "pl", "tl"]

__version__ = version("ehrdata")
