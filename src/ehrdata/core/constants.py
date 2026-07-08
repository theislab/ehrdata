from pathlib import Path

# Typing Column
# -----------------------
# The column name and used values in adata.var for column types.

FEATURE_TYPE_KEY = "feature_type"
NUMERIC_TAG = "numeric"
CATEGORICAL_TAG = "categorical"
DATE_TAG = "date"

DEFAULT_TEM_LAYER_NAME = "tem_data"

# On-disk format (see :mod:`ehrdata.io._ondisk` for the layout).
# EHRData stores time-series as 3D arrays in `X`/`layers`, but anndata only guarantees 2D arrays there (https://github.com/scverse/anndata/issues/2430) and enforces this since release 0.13.0
# Since storage version 0.2.0 (ehrdata release 0.3.0), 3D arrays are relocated into the reserved `.obsm` keys below on write and restored on read.
# Bump only when the on-disk layout changes.
EHRDATA_ENCODING_TYPE_KEY = "ehrdata-encoding-type"
EHRDATA_ENCODING_TYPE = "ehrdata"
EHRDATA_ONDISK_VERSION_KEY = "ehrdata-encoding-version"
EHRDATA_ONDISK_VERSION = "0.2.0"

# Reserved `.obsm` keys holding a 3D `X` / a 3D layer `<name>` on disk.
EHRDATA_OBSM_3D_X_KEY = "_ed_ondisk_X"
EHRDATA_OBSM_3D_LAYER_PREFIX = "_ed_ondisk_layers_"

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
