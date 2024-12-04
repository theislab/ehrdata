from importlib.metadata import version

from . import dt, io, pl
from .core import EHRData

__all__ = ["EHRData", "dt", "io", "pl"]

__version__ = version("ehrdata")
