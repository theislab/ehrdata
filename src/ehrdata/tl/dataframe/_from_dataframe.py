from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ehrdata import EHRData


def from_dataframe(
    df: pd.DataFrame,
    columns_obs_only: list[str] | None = None,
    index_column: str | int | None = None,
) -> EHRData:
    """Transform a given Pandas DataFrame into an EHRData object.

    Note that columns containing boolean values (either 0/1 or T(t)rue/F(f)alse)
    will be stored as boolean columns whereas the other non-numerical columns will be stored as categorical values.

    Args:
        df: The pandas dataframe to be transformed.
        columns_obs_only: An optional list of column names that should belong to obs only and not X.
        index_column: The index column of obs. This can be either a column name (or its numerical index in the DataFrame) or the index of the dataframe.

    Returns:
        An EHRData object created from the given DataFrame.

    Examples:
        >>> import ehrdata as ed
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "patient_id": ["0", "1", "2", "3", "4"],
        ...         "age": [65, 72, 58, 78, 82],
        ...         "sex": ["M", "F", "F", "M", "F"],
        ...     }
        ... )
        >>> edata = ed.io.from_dataframe(df, index_column="patient_id")
    """
    from ehrdata import EHRData

    df = df.copy()
    # Check and handle the overlap of index_column in columns_obs_only
    if index_column is not None:
        if isinstance(index_column, int):
            if index_column >= len(df.columns):
                err_msg = f"index_column integer index is out of bounds. DataFrame has {len(df.columns)} columns."
                raise IndexError(err_msg)
            index_column = df.columns[index_column]
        if not df.index.name or df.index.name != index_column:
            if index_column in df.columns:
                df.set_index(index_column, inplace=True)
            else:
                err_msg = f"Column {index_column} not found in DataFrame."
                raise ValueError(err_msg)

    # Now handle columns_obs_only with consideration of the new index
    if columns_obs_only:
        if index_column in columns_obs_only:
            columns_obs_only.remove(index_column)
        missing_cols = [col for col in columns_obs_only if col not in df.columns]
        if missing_cols:
            err_msg = f"Columns {missing_cols} specified in columns_obs_only are not in the DataFrame."
            raise ValueError(err_msg)
        obs = df.loc[:, columns_obs_only].copy()
        df.drop(columns=columns_obs_only, inplace=True, errors="ignore")
    else:
        obs = pd.DataFrame(index=df.index)

    for col in obs.columns:
        if obs[col].dtype == "bool":
            obs[col] = obs[col].astype(bool)
        elif obs[col].dtype == "object":
            obs[col] = obs[col].astype("category")

    # Prepare the AnnData object
    X = df.to_numpy(copy=True)
    obs.index = obs.index.astype(str)
    var = pd.DataFrame(index=df.columns)
    var.index = var.index.astype(str)

    # Handle dtype of X based on presence of numerical columns only
    all_numeric = df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]
    X = X.astype(np.float64 if all_numeric else object)

    edata = EHRData(X=X, obs=obs, var=var)
    edata.obs_names = edata.obs_names.astype(str)
    edata.var_names = edata.var_names.astype(str)

    return edata
