from importlib.metadata import version

from . import dt, integrations, io
from ._codes import aggregate_codes, annotate_codes
from ._feature_types import feature_type_overview, harmonize_missing_values, infer_feature_types, replace_feature_types
from ._move_data import move_to_obs, move_to_x
from ._rebin import rebin
from .core import EHRData

__all__ = [
    "EHRData",
    "aggregate_codes",
    "annotate_codes",
    "dt",
    "feature_type_overview",
    "harmonize_missing_values",
    "infer_feature_types",
    "integrations",
    "io",
    "move_to_obs",
    "move_to_x",
    "rebin",
    "replace_feature_types",
]

__version__ = version("ehrdata")
