from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ehrdata import EHRData
from ehrdata._logger import logger


def _cast_variables_to_float(edata: EHRData) -> None:
    """Cast the dtype of variables to float, and overwrite the values of the original arrays with the casted columns."""
    if edata.isbacked:
        msg = "Cannot cast variables to float when EHRData is backed."
        raise ValueError(msg)

    if edata.X is not None and not (np.issubdtype(edata.X.dtype, np.number) or np.issubdtype(edata.X.dtype, np.bool_)):
        # note that every scipy sparse array is of a numeric dtype and will enter this if block
        # further sparse.COO, while being allowed in theory to be str dtype, is only allowed numeric dtypes under binsparse specification which we follow closely
        out = edata.X.astype(
            object
        ).copy()  # A fresh object array is built so columns can independently hold floats: numpy 2 / pandas 3 read string data back as a homogeneous ``StringDType`` array, into which an in-place ``astype(float64)`` assignment would merely re-stringify the values.

        for column in range(out.shape[1]):
            with contextlib.suppress(ValueError):
                out[:, column] = out[:, column].astype(np.float64)

        edata.X = out

    for key in edata.layers:
        if edata.layers[key] is not None and not (
            np.issubdtype(edata.layers[key].dtype, np.number) or np.issubdtype(edata.layers[key].dtype, np.bool_)
        ):
            # note that every scipy sparse array is of a numeric dtype and will enter this if block
            # further sparse.COO, while being allowed in theory to be str dtype, is only allowed numeric dtypes under binsparse specification which we follow closely
            out = edata.X.astype(object).copy()

            for column in range(out.shape[1]):
                with contextlib.suppress(ValueError):
                    out[:, column] = out[:, column].astype(np.float64)

            edata.layers[key] = out


def _cast_arrays_dtype_to_float_or_str_if_nonnumeric_object(edata: EHRData) -> EHRData:
    """Cast the dtype of object arrays to float, and if this fails, to str."""
    if edata.X is not None and edata.X.dtype == np.object_:
        edata = edata.copy()
        try:
            edata.X = edata.X.astype(np.float64)
        except ValueError:
            edata.X = edata.X.astype(str)
        for layer, array in edata.layers.items():
            if array.dtype == np.object_:
                try:
                    edata.layers[layer] = array.astype(np.float64)
                except ValueError:
                    logger.warning(
                        f"edata.layers[{layer}] is of dtype {edata.layers[layer].dtype}: this is casted to dtype 'str' for saving to zarr."
                    )
                    edata.layers[layer] = array.astype(str)

    return edata


def _cast_dataframe_columns_to_writable_dtype(df: pd.DataFrame, slot: str) -> pd.DataFrame:
    """Return a copy of a dataframe of `.obs`, `.var` or `.tem` with the columns anndata cannot write cast.

    anndata has no writer for datetime columns, and h5py rejects an `object` column holding anything
    but strings, which is what a database read leaves behind for a column it found empty.
    Such columns are cast following the rule that `X` and the layers follow: to a numeric dtype, and
    to a string dtype if that fails. Datetimes are written as ISO 8601 strings, from which
    :func:`~ehrdata.infer_feature_types` recognizes the column as a date again. Missing values stay missing.
    """
    df = df.copy()
    cast_to_string = []

    for column_name in df.columns:
        column = df[column_name]

        if pd.api.types.is_datetime64_any_dtype(column):
            # element-wise, since a mapped column of ISO 8601 strings is inferred back to datetimes by pandas
            iso_8601 = np.array([None if pd.isna(value) else value.isoformat() for value in column], dtype=object)
            df[column_name] = pd.Categorical(iso_8601)
            cast_to_string.append(column_name)

        elif column.dtype == object:
            observed_values = column[column.notna()]
            if observed_values.empty:
                df[column_name] = np.full(len(column), np.nan)
            elif observed_values.map(lambda value: isinstance(value, str)).all():
                # anndata writes an all-string column itself, but not the missing values among them
                df[column_name] = pd.Categorical(column)
            else:
                try:
                    df[column_name] = pd.to_numeric(column)
                except (TypeError, ValueError):
                    df[column_name] = pd.Categorical(column.astype(str).where(column.notna()))
                    cast_to_string.append(column_name)

    if cast_to_string:
        logger.warning(
            f"Columns {cast_to_string} of .{slot} have a dtype that cannot be written, and are written as strings."
        )

    return df
