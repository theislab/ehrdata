from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from scipy.sparse import issparse

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ehrdata import EHRData


def to_dataframe(
    edata: EHRData,
    layer: str | None = None,
    obs_cols: Iterable[str] | str | None = None,
    var_cols: Iterable[str] | str | None = None,
) -> pd.DataFrame:
    """Transform an EHRData object to a Pandas DataFrame.

    Args:
        edata: The EHRData object to be transformed into a pandas DataFrame
        layer: The layer to access the values of. If not specified, it uses the `X` matrix.
        obs_cols: The columns of `obs` to add to the DataFrame.
        var_cols: The columns of `var` to fetch values from.

    Returns:
        The AnnData object as a pandas DataFrame

    Examples:
        >>> import ehrapy as ep
        >>> edata = ep.dt.mimic_2(encoded=True)
        >>> df = ep.ad.anndata_to_df(edata)
    """
    X = edata.layers[layer] if layer is not None else edata.X

    if issparse(X):  # pragma: no cover
        X = X.toarray()

    df = pd.DataFrame(X, columns=list(edata.var_names))
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
    if var_cols:
        if len(edata.var.columns) == 0:
            msg = "Cannot slice columns from empty var!"
            raise ValueError(msg)
        if isinstance(var_cols, str):
            var_cols = list(var_cols)
        if isinstance(var_cols, list):
            var_slice = edata.var[var_cols]
        # reset index needed since we slice all or at least some columns from var DataFrame
        var_slice = var_slice.reset_index(drop=True)
        df = pd.concat([df, var_slice], axis=1)

    return df
