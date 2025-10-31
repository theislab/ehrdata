from pathlib import Path

# Typing Column
# -----------------------
# The column name and used values in adata.var for column types.

FEATURE_TYPE_KEY = "feature_type"
NUMERIC_TAG = "numeric"
CATEGORICAL_TAG = "categorical"
DATE_TAG = "date"

DEFAULT_TEM_LAYER_NAME = "tem_data"

EHRDATA_ZARR_ENCODING_VERSION = "0.0.1"

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
