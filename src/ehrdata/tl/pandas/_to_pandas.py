from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
import xarray as xr
from scipy.sparse import issparse

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ehrdata import EHRData


def to_pandas(
    edata: EHRData,
    layer: str | None = None,
    obs_cols: Iterable[str] | str | None = None,
    var_col: str | None = None,
    format: Literal["wide", "long"] = "wide",
) -> pd.DataFrame:
    """Transform an EHRData object to a Pandas DataFrame.

    Args:
        edata: The EHRData object to be transformed into a pandas DataFrame
        layer: The layer to access the values of. If not specified, it uses the `X` matrix.
        obs_cols: The columns of `obs` to add to the DataFrame.
        var_col: The column of `var` to create the column names from in the created DataFrame. If not specified, the `var_names` will be used.
        format: The format of the output DataFrame. This is relevant for longitudinal data. If "wide", the output dataframe will write a column for each (variable, time) tuple, naming the colun as <variable_name>_t_<tem.index value>. If "long", the output dataframe will be in long format, with columns "observation_id", "variable", "time", and "value".

    Returns:
        The data object as a pandas DataFrame.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> df = ep.tl.to_pandas(edata)
    """
    X = edata.layers[layer] if layer is not None else edata.X

    if var_col is not None and var_col not in edata.var.columns:
        err_msg = f"Variable column {var_col} not found in edata.var"
        raise ValueError(err_msg)

    var_names = edata.var_names if var_col is None else edata.var[var_col]

    if issparse(X):  # pragma: no cover
        X = X.toarray()

    if format == "wide":
        if len(X.shape) == 2:
            df = pd.DataFrame(X, columns=var_names)
        elif len(X.shape) == 3:
            X_wide = X.reshape(X.shape[0], -1)
            column_names = [
                f"{edata.var_names[i]}_t_{edata.tem.index[j]}" for i in range(X.shape[1]) for j in range(X.shape[2])
            ]
            df = pd.DataFrame(X_wide, columns=column_names)

        if obs_cols:
            if len(edata.obs.columns) == 0:
                msg = "Cannot slice columns from empty obs!"
                raise ValueError(msg)
            if isinstance(obs_cols, str):
                obs_cols = list(obs_cols)
            if isinstance(obs_cols, list):  # pragma: no cover
                obs_slice = edata.obs[obs_cols]
            # reset index needed since we slice all or at least some columns from obs DataFrame
            obs_slice = obs_slice.reset_index(drop=True)
            df = pd.concat([df, obs_slice], axis=1)
        df.index = edata.obs_names
        return df

    elif format == "long":
        if obs_cols:
            err_msg = "Long format does not support obs_cols"
            raise NotImplementedError(err_msg)

        if len(X.shape) == 2:
            df = pd.DataFrame(X, columns=var_names)
            df = df.melt(id_vars=edata.obs_names, var_name="variable", value_name="value")
            # to long
        elif len(X.shape) == 3:
            data_array = xr.DataArray(
                X,
                dims=["observation_id", "variable", "time"],
                coords={
                    "observation_id": edata.obs_names,
                    "variable": edata.var_names,
                    "time": edata.tem.index,
                },
            )
            data_array.name = "value"
            df = data_array.to_dataframe().reset_index()

        return df

    else:
        err_msg = f"Invalid format: {format}"
        raise ValueError(err_msg)
