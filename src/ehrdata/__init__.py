from importlib.metadata import version

from . import dt, io, pl, pp, tl

__all__ = ["dt", "io", "pl", "pp", "tl"]

__version__ = version("ehrdata")
