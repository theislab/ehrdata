from importlib.util import find_spec

__all__ = ["BasicClass", "basic_plot", "vitessce"]

from .basic import BasicClass, basic_plot

if find_spec("vitessce"):
    from . import vitessce
