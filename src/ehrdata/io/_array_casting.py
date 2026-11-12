from __future__ import annotations

import contextlib
import datetime as dt
import decimal
import numbers
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import sparse

if TYPE_CHECKING:
    from ehrdata import EHRData
from ehrdata._logger import logger
from ehrdata.core.constants import VALUE_KIND_MISSING, VALUE_KIND_NUMBER, VALUE_KIND_TEXT


def _is_numeric_or_bool(array: Any) -> bool:
    return np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)


def _cast_columns_to_float(array: Any) -> Any:
    """Cast each variable of a non-numeric array to float where possible."""
    if not isinstance(array, np.ndarray):
        # scipy sparse is always numeric; sparse.COO can only be cast as a whole
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


# `infer_dtype` of object arrays that are cast to float as a whole
_FLOAT_CASTABLE = {"empty", "floating", "integer", "mixed-integer-float", "boolean", "decimal"}


def _is_exact_integer(value: Any) -> bool:
    try:
        return float(int(value)) == int(value)
    except OverflowError:
        return False


def _number_to_string(value: Any) -> str:
    # as written if that reads back as the same float, e.g. "3" for 3
    text = str(value)
    with contextlib.suppress(ValueError):
        if float(text) == float(value):
            return text
    return repr(float(value))


def _encode_object_array(array: Any, name: str) -> tuple[Any, np.ndarray | None]:
    """Encode an in-memory `object` array for writing; return other arrays unchanged.

    If every value is a real number (also a bool or `Decimal`), and an integer only if float64 holds it exactly, or missing,
    the values are returned as float, with `None`.
    Otherwise they are returned as strings, with the kind of each value: text, number or missing.
    Strings are never parsed into numbers.
    """
    if not isinstance(array, np.ndarray | sparse.COO) or array.dtype != np.object_:
        # backed and dask arrays are written as they are
        return array, None
    if isinstance(array, sparse.COO):
        try:
            return array.astype(np.float64), None
        except (TypeError, ValueError):
            logger.warning(f"edata.{name} is of dtype object and not numeric: it is cast to dtype 'str' for writing.")
            return array.astype(str), None

    is_missing = np.asarray(
        pd.isna(array), dtype=bool
    )  # TODO: find out if scipy sparse can ever enter, as this would be a densification.
    if pd.api.types.infer_dtype(array.ravel(), skipna=True) in _FLOAT_CASTABLE:
        with contextlib.suppress(OverflowError):
            values = np.where(is_missing, np.nan, array).astype(np.float64)
            # integers from 2**53 on may not be exact
            if not (np.abs(values[~is_missing]) >= 2**53).any():
                return values, None

    flat, flat_missing = array.ravel(), is_missing.ravel()
    kinds = np.full(flat.shape, VALUE_KIND_MISSING, dtype=np.uint8)
    strings = np.full(flat.shape, "nan", dtype=object)  # an object array of str, not fixed width
    type_ids = np.fromiter(map(id, map(type, flat)), dtype=np.uintp, count=flat.size)
    inexact_integers = False
    # by type, to vectorize
    for value_type in set(map(type, flat[~flat_missing])):
        mask = (type_ids == id(value_type)) & ~flat_missing
        values = flat[mask]
        if issubclass(value_type, str):
            kinds[mask], strings[mask] = VALUE_KIND_TEXT, values
        elif issubclass(value_type, int | np.integer) and not issubclass(value_type, bool):
            try:
                exact = np.abs(values.astype(np.float64)) < 2**53
            except OverflowError:
                exact = np.zeros(values.shape, dtype=bool)
            exact[~exact] = np.frompyfunc(_is_exact_integer, 1, 1)(values[~exact]).astype(bool)
            inexact_integers |= not exact.all()
            kinds[mask] = np.where(exact, VALUE_KIND_NUMBER, VALUE_KIND_TEXT)
            strings[mask] = np.fromiter(map(str, values), dtype=object, count=values.size)
        elif issubclass(value_type, numbers.Real | decimal.Decimal | np.bool_):
            to_string = str if value_type in {float, np.float64} else _number_to_string
            kinds[mask] = VALUE_KIND_NUMBER
            strings[mask] = np.fromiter(map(to_string, values), dtype=object, count=values.size)
        else:
            kinds[mask] = VALUE_KIND_TEXT
            strings[mask] = np.fromiter(map(_to_string, values), dtype=object, count=values.size)

    if not (kinds == VALUE_KIND_TEXT).any():
        return np.where(is_missing, np.nan, array).astype(np.float64), None
    if inexact_integers:
        logger.warning(f"edata.{name} holds integers that float64 does not hold exactly: they are written as text.")
    return strings.reshape(array.shape), kinds.reshape(array.shape)


def _decode_object_array(strings: Any, kinds: np.ndarray) -> np.ndarray:
    """Restore the values of an array encoded by :func:`_encode_object_array`."""
    strings = np.asarray(strings)
    values = strings.astype(object)
    is_number = kinds == VALUE_KIND_NUMBER
    values[is_number] = strings[is_number].astype(np.float64)
    values[kinds == VALUE_KIND_MISSING] = np.nan
    return values


def _to_string(value: Any) -> str:
    if isinstance(value, dt.timedelta | np.timedelta64):
        return pd.Timedelta(value).isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, dt.date | dt.time):
        return value.isoformat()
    return str(value)


def _to_string_array(column: pd.Series) -> pd.api.extensions.ExtensionArray:
    """Column values as strings, datetimes and timedeltas in ISO 8601; missing values stay missing."""
    is_observed = column.notna().to_numpy()
    values = column.to_numpy(dtype=object)
    return pd.array(
        [_to_string(value) if observed else None for value, observed in zip(values, is_observed, strict=True)],
        dtype="string",
    )


def _to_numeric(column: pd.Series) -> pd.Series:
    """Cast a column to numeric; integers and booleans stay exact, nullable if values are missing."""
    numeric = pd.to_numeric(column, dtype_backend="numpy_nullable")
    observed_values = column[column.notna()]
    if numeric.dtype == object or (
        pd.api.types.is_float_dtype(numeric.dtype)
        and observed_values.map(lambda value: isinstance(value, int | np.integer)).all()
    ):
        # integers beyond int64/uint64
        msg = f"Column {column.name!r} holds integers that no numeric dtype holds exactly."
        raise ValueError(msg)
    if pd.api.types.is_float_dtype(numeric.dtype) or not numeric.isna().any():
        return _to_numpy_dtype(numeric)
    return numeric


def _to_numpy_dtype(column: pd.Series) -> pd.Series:
    # anndata cannot write nullable floats
    return pd.Series(
        column.to_numpy(dtype=column.dtype.numpy_dtype, na_value=np.nan), index=column.index, name=column.name
    )


def _cast_dataframe_columns_to_writable_dtype(df: pd.DataFrame, slot: str) -> pd.DataFrame:
    """Return a copy of `.obs`, `.var` or `.tem` with the columns anndata cannot write cast.

    Datetimes and timedeltas become ISO 8601 strings. `object` columns become numeric where possible (integers and
    booleans exact, nullable if values are missing), and strings otherwise or if they mix strings with other values.
    Nullable floats become float. Missing values stay missing; all other columns are left as they are.
    """
    df = df.copy()
    cast_to_iso_8601 = []
    cast_to_string = []

    # by position, so that anndata itself rejects repeated column names
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
                    # anndata cannot write missing values in an `object` string column
                    df.isetitem(position, column.astype("string"))
            else:
                try:
                    if is_string.any():
                        # keeps codes like "007" intact
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
