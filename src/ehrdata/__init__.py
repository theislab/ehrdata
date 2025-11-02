from importlib.metadata import version

from . import dt, integrations, io
from ._feature_types import feature_type_overview, harmonize_missing_values, infer_feature_types, replace_feature_types
from .core import EHRData

__all__ = [
    "EHRData",
    "dt",
    "feature_type_overview",
    "harmonize_missing_values",
    "infer_feature_types",
    "integrations",
    "io",
    "replace_feature_types",
]

__version__ = version("ehrdata")

import anndata as ad

# Opt to use this newer feature of anndata https://github.com/scverse/anndata/blob/6a6bde151eeb231eebac20b66e6002b88052e8db/src/anndata/_io/specs/methods.py#L1151
ad.settings.allow_write_nullable_strings = True
