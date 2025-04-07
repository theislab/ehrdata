from importlib.metadata import version

from . import dt, io, pl, tl
from .core import EHRData

__all__ = ["EHRData", "dt", "io", "tl", "pl"]

__version__ = version("ehrdata")
