from importlib.metadata import version

from core import EHRData

from . import dt, io, pl, pp, tl

__all__ = ["EHRData", "dt", "io", "pl", "pp", "tl"]

__version__ = version("ehrdata")
