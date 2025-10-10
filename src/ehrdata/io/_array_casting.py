from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ehrdata import EHRData
from lamin_utils import logger


def _cast_variables_to_float(edata: EHRData) -> None:
    """Cast the dtype of variables to float, and overwrite the values of the original arrays with the casted columns."""
    if edata.isbacked:
        msg = "Cannot cast variables to float when EHRData is backed."
        raise ValueError(msg)

    if edata.X is not None and not np.issubdtype(edata.X.dtype, np.number):
        for column in range(edata.shape[1]):
            with contextlib.suppress(ValueError):
                edata.X[:, column] = edata.X[:, column].astype(np.float64)

    for key in edata.layers:
        if edata.layers[key] is not None and not np.issubdtype(edata.layers[key].dtype, np.number):
            for column in range(edata.shape[1]):
                with contextlib.suppress(ValueError):
                    edata.layers[key][:, column] = edata.layers[key][:, column].astype(np.float64)


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
