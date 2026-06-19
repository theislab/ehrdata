from pathlib import Path

# Typing Column
# -----------------------
# The column name and used values in adata.var for column types.

FEATURE_TYPE_KEY = "feature_type"
NUMERIC_TAG = "numeric"
CATEGORICAL_TAG = "categorical"
DATE_TAG = "date"

DEFAULT_TEM_LAYER_NAME = "tem_data"

# On-disk format versions for ehrdata
# -----------------------------------
# EHRData stores time-series as 3D arrays (#observations x #variables x #timesteps) in `X`/`layers`.
# AnnData only guarantees 2D arrays in `X`/`layers`, so we version how 3D data is laid out on disk, see https://github.com/scverse/anndata/issues/2430
# v1: 3D arrays written directly into `X`/`layers`. This is the legacy layout (no on-disk marker); it
#     only works on anndata versions that still permit 3D arrays there, so it is no longer written.
# v2: 3D arrays relocated into `.obsm` on write and restored to `X`/`layers` on read. This is the only
#     layout we write, since it works regardless of whether anndata permits 3D arrays in `X`/`layers`.
# v3: placeholder for the future, once anndata relaxes its spec to allow 3D arrays in `X`/`layers`
#     again; it would write them there directly. Not implemented yet.
EHRDATA_FORMAT_V1 = 1
EHRDATA_FORMAT_V2 = 2
EHRDATA_FORMAT_V3 = 3
EHRDATA_DEFAULT_FORMAT_VERSION = EHRDATA_FORMAT_V2

# The encoding-type/-version stamped onto the store (zarr) or root group (h5ad) so an ehrdata file is
# identifiable and its format version discoverable. Reading does not rely on the stamped version: the
# v2 layout is self-describing through the reserved `.obsm` keys below, so v1 (no reserved keys) and v2
# (reserved keys present) are distinguished by inspecting `.obsm` alone.
EHRDATA_ENCODING_TYPE = "ehrdata"

# Reserved `.obsm` keys used by format v2 to relocate 3D arrays out of `X`/`layers` and back on read.
# `X` is stored under `_ed_ondisk_X`; a layer named `<name>` under `_ed_ondisk_layers_<name>`.
EHRDATA_OBSM_3D_X_KEY = "_ed_ondisk_X"
EHRDATA_OBSM_3D_LAYER_PREFIX = "_ed_ondisk_layers_"

# Recommended on-disk file extensions, mirroring anndata's `.h5ad`/`.zarr`.
EHRDATA_H5_SUFFIX = ".h5ed"
EHRDATA_ZARR_SUFFIX = ".ehrdata.zarr"

# Missing values
# --------------
# These values if encountered as strings are considered to represent missing values in the data

MISSING_VALUES = (
    "nan",  # this is the result value of str(np.nan)
    "np.nan",  # this is very explicit about what the value should mean
    "<NA>",  # this is the result value of str(pd.NA)
    "pd.NA",  # this is very explicit about what the value should mean
)


# Pandas Format
# --------------
# The format of the pandas dataframe to be used in the to_pandas and from_pandas functions

PANDAS_FORMATS = ["flat", "wide", "long"]


# Default data path
# ------------------
# The default path to store and load data

DEFAULT_DATA_PATH = Path("ehrapy_data")
