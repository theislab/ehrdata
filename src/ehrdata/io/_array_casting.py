from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ehrdata._feature_types import _detect_feature_type
from ehrdata._logger import logger
from ehrdata.core.constants import NUMERIC_TAG

if TYPE_CHECKING:
    from ehrdata import EHRData


def _recover_numeric_columns(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` as an object array with each variable column ehrdata infers as numeric cast to float.

    Which columns are numeric is decided by ehrdata's feature-type inference
    (:func:`~ehrdata._feature_types._detect_feature_type`) rather than an ad-hoc float cast, so dates
    and non-numeric categoricals are left untouched. A fresh object array is built so columns can
    independently hold floats: numpy 2 / pandas 3 read string data back as a homogeneous
    ``StringDType`` array, into which an in-place ``astype(float64)`` assignment would merely
    re-stringify the values.
    """
    out = np.asarray(arr, dtype=object).copy()
    for column in range(out.shape[1]):
        col = pd.Series(np.asarray(out[:, column]).reshape(-1))
        try:
            feature_type, _ = _detect_feature_type(col, binary_as="numeric")
        except ValueError:
            continue  # e.g. an all-missing column: nothing to recover
        if feature_type == NUMERIC_TAG:
            out[:, column] = out[:, column].astype(np.float64)
    return out


def _cast_variables_to_float(edata: EHRData) -> None:
    """Cast the dtype of variables to float, and overwrite the values of the original arrays with the casted columns."""
    if edata.isbacked:
        msg = "Cannot cast variables to float when EHRData is backed."
        raise ValueError(msg)

    if edata.X is not None and not np.issubdtype(edata.X.dtype, np.number):
        edata.X = _recover_numeric_columns(edata.X)

    for key in edata.layers:
        if edata.layers[key] is not None and not np.issubdtype(edata.layers[key].dtype, np.number):
            edata.layers[key] = _recover_numeric_columns(edata.layers[key])


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
