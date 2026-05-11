"""Source-specific adapter modules.

Each adapter translates one data source's raw table shapes into canonical
DataFrames validated against :mod:`ehrdata.io.source.schema`.
"""

from . import cprd, lced, marketscan

__all__ = ["cprd", "lced", "marketscan"]
