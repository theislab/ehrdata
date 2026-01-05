from importlib.metadata import version

from . import dt, integrations, io
from ._feature_types import feature_type_overview, harmonize_missing_values, infer_feature_types, replace_feature_types
from ._move_data import move_to_obs, move_to_x
from .core import EHRData

__all__ = [
    "EHRData",
    "dt",
    "feature_type_overview",
    "harmonize_missing_values",
    "infer_feature_types",
    "integrations",
    "io",
    "move_to_X",
    "move_to_obs",
    "replace_feature_types",
]

__version__ = version("ehrdata")
