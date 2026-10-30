from __future__ import annotations

import contextlib
import datetime as dt
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import sparse

if TYPE_CHECKING:
    from ehrdata import EHRData
from ehrdata._logger import logger


def _is_numeric_or_bool(array: Any) -> bool:
    return np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)


def _cast_columns_to_float(array: Any) -> Any:
    """Cast the values of each variable of a non-numeric array to float, leaving the variables where this fails unaltered."""
    if not isinstance(array, np.ndarray):
        # scipy sparse arrays are always numeric, and a non-numeric sparse.COO can only be cast as a whole
        with contextlib.suppress(TypeError, ValueError):
            return array.astype(np.float64)
        return array

    out = array.astype(
        object
    ).copy()  # A fresh object array is built so columns can independently hold floats: numpy 2 / pandas 3 read string data back as a homogeneous ``StringDType`` array, into which an in-place ``astype(float64)`` assignment would merely re-stringify the values.

    for column in range(out.shape[1]):
        with contextlib.suppress(TypeError, ValueError):
            out[:, column] = out[:, column].astype(np.float64)

    return out


def _cast_variables_to_float(edata: EHRData) -> None:
    """Cast the dtype of variables to float, and overwrite the values of the original arrays with the casted columns."""
    if edata.isbacked:
        msg = "Cannot cast variables to float when EHRData is backed."
        raise ValueError(msg)

    if edata.X is not None and not _is_numeric_or_bool(edata.X):
        edata.X = _cast_columns_to_float(edata.X)

    for key, array in list(edata.layers.items()):
        if key is None:  # anndata 0.13: the unified `.X` slot, already handled above
            continue
        if array is not None and not _is_numeric_or_bool(array):
            edata.layers[key] = _cast_columns_to_float(array)


def _cast_object_array_to_float_or_str(array: Any, name: str) -> Any:
    """Cast an `object` array held in memory to float, and if this fails, to str; return any other array as it is."""
    if not isinstance(array, np.ndarray | sparse.COO) or array.dtype != np.object_:
        # a backed or dask array is written as it is
        return array
    if isinstance(array, np.ndarray):
        # missing values become np.nan, which a str array holds as "nan", a value harmonize_missing_values recognizes
        array = np.where(pd.isna(array), np.nan, array)
    try:
        return array.astype(np.float64)
    except (TypeError, ValueError):
        logger.warning(f"edata.{name} is of dtype object and not numeric: it is cast to dtype 'str' for writing.")
        return array.astype(str)


def _to_string(value: Any) -> str:
    if isinstance(value, dt.timedelta | np.timedelta64):
        return pd.Timedelta(value).isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, dt.date | dt.time):
        return value.isoformat()
    return str(value)


def _to_string_array(column: pd.Series) -> pd.api.extensions.ExtensionArray:
    """Write the values of a column as strings, datetimes and timedeltas in ISO 8601, keeping missing values missing."""
    is_observed = column.notna().to_numpy()
    values = column.to_numpy(dtype=object)
    return pd.array(
        [_to_string(value) if observed else None for value, observed in zip(values, is_observed, strict=True)],
        dtype="string",
    )


def _to_numeric(column: pd.Series) -> pd.Series:
    """Cast a column to a numeric dtype, keeping integers and booleans exact with a nullable dtype if values are missing."""
    numeric = pd.to_numeric(column, dtype_backend="numpy_nullable")
    observed_values = column[column.notna()]
    if numeric.dtype == object or (
        pd.api.types.is_float_dtype(numeric.dtype)
        and observed_values.map(lambda value: isinstance(value, int | np.integer)).all()
    ):
        # integers beyond what an int64 or uint64 holds
        msg = f"Column {column.name!r} holds integers that no numeric dtype holds exactly."
        raise ValueError(msg)
    if pd.api.types.is_float_dtype(numeric.dtype) or not numeric.isna().any():
        return _to_numpy_dtype(numeric)
    return numeric


def _to_numpy_dtype(column: pd.Series) -> pd.Series:
    # anndata has no writer for nullable floats, and np.nan marks their missing values just as well
    return pd.Series(
        column.to_numpy(dtype=column.dtype.numpy_dtype, na_value=np.nan), index=column.index, name=column.name
    )


def _cast_dataframe_columns_to_writable_dtype(df: pd.DataFrame, slot: str) -> pd.DataFrame:
    """Return a copy of a dataframe of `.obs`, `.var` or `.tem` with the columns anndata cannot write cast.

    anndata has no writer for datetime and timedelta columns, and h5py rejects an `object` column holding anything
    but strings, which is what a database read leaves behind for a column it found empty or with missing values.
    Datetimes and timedeltas are written as ISO 8601 strings, such as `2100-01-01T08:30:00` and `P0DT1H0M0S`.
    `object` columns are cast following the rule that `X` and the layers follow: to a numeric dtype, and to a
    string dtype if that fails or if they hold strings next to other values. Integers and booleans stay exact,
    in a nullable dtype if values are missing. Nullable float columns, which anndata cannot write either, are cast to
    float. Missing values stay missing, and columns anndata can write are left as they are.
    """
    df = df.copy()
    cast_to_iso_8601 = []
    cast_to_string = []

    # by position, as a dataframe with repeated column names is left for anndata to reject
    for position, column_name in enumerate(df.columns):
        column = df.iloc[:, position]

        if pd.api.types.is_datetime64_any_dtype(column) or pd.api.types.is_timedelta64_dtype(column):
            df.isetitem(position, _to_string_array(column))
            cast_to_iso_8601.append(column_name)

        elif isinstance(column.dtype, pd.Float32Dtype | pd.Float64Dtype):
            df.isetitem(position, _to_numpy_dtype(column))

        elif column.dtype == object:
            observed_values = column[column.notna()]
            is_string = observed_values.map(lambda value: isinstance(value, str))
            if observed_values.empty:
                df.isetitem(position, np.full(len(column), np.nan))
            elif is_string.all():
                if len(observed_values) < len(column):
                    # anndata writes an all-string column itself, but not the missing values among them
                    df.isetitem(position, column.astype("string"))
            else:
                try:
                    if is_string.any():
                        # strings next to other values stay strings, so that codes like "007" keep their leading zeros
                        msg = f"Column {column_name!r} holds strings next to other values."
                        raise ValueError(msg)
                    df.isetitem(position, _to_numeric(column))
                except (TypeError, ValueError):
                    df.isetitem(position, _to_string_array(column))
                    cast_to_string.append(column_name)

    if cast_to_iso_8601:
        logger.info(f"Columns {cast_to_iso_8601} of .{slot} are datetimes or timedeltas, written as ISO 8601 strings.")
    if cast_to_string:
        logger.warning(
            f"Columns {cast_to_string} of .{slot} hold values that cannot be written as numbers, and are written as strings."
        )

    return df
