from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ehrdata import EHRData

PANDAS_FORMATS = ["flat", "wide", "long"]


def from_pandas(
    df: pd.DataFrame,
    columns_obs_only: list[str] | None = None,
    index_column: str | int | None = None,
    format: Literal["flat", "wide", "long"] = "flat",
    wide_format_time_suffix: str | None = None,
    long_format_keys: dict[Literal["observation_column", "variable_column", "time_column", "value_column"], str]
    | None = None,
) -> EHRData:
    """Transform a given Pandas DataFrame into an EHRData object.

    Note that columns containing boolean values (either 0/1 or T(t)rue/F(f)alse)
    will be stored as boolean columns whereas the other non-numerical columns will be stored as categorical values.

    Args:
        df: The pandas dataframe to be transformed.
        columns_obs_only: An optional list of column names that should belong to obs only and not X.
        index_column: The index column of obs. This can be either a column name (or its numerical index in the DataFrame) or the index of the dataframe.
        format: The format of the input dataframe. If the data is not longitudinal, choose "flat". If the data is longitudinal in the long format, choose "long". If the data is longitudinal in a wide format, choose, "wide".
        wide_format_time_suffix: Use only if format="wide". Suffixes int the in the variable columns that indicate the time.
        long_format_keys: Use only if format="long". The keys of the dataframe in the long format. The dictionary should have the following structure: {"observation_column": "<the column name of the observation ids>", "variable_column": "<the column name of the variable ids>", "time_column": "<the column name of the time>", "value_column": "<the column name of the values>"}.

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
        >>> edata = ed.tl.from_pandas(df, index_column="patient_id")
    """
    from ehrdata import EHRData

    df = df.copy()

    if format not in PANDAS_FORMATS:
        err_msg = f"Format {format} is not supported. Please choose from {PANDAS_FORMATS}."
        raise ValueError(err_msg)

    if format != "long" and long_format_keys is not None:
        err_msg = "long_format_keys should only be used if format is 'long'."
        raise ValueError(err_msg)

    elif format == "long" and long_format_keys is None:
        long_format_keys = {
            "observation_column": "observation_id",
            "variable_column": "variable",
            "time_column": "time",
            "value_column": "value",
        }
    elif format == "long" and long_format_keys is not None:
        valid_long_format_keys = ["observation_column", "variable_column", "time_column", "value_column"]
        invalid_keys = [key for key in long_format_keys if key not in valid_long_format_keys]
        if invalid_keys:
            err_msg = f"Invalid keys: {invalid_keys}. Please use only the following keys: {valid_long_format_keys}."
            raise ValueError(err_msg)

    if format != "wide" and wide_format_time_suffix is not None:
        err_msg = "wide_format_time_suffix should only be used if format is 'wide'."
        raise ValueError(err_msg)
    elif format == "wide" and wide_format_time_suffix is None:
        wide_format_time_suffix = "_t_"

    # index column:
    if format in ["flat", "wide"]:
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

    elif format == "long" and index_column is not None:
        msg = "Long format with index column is not implemented yet."
        raise NotImplementedError(msg)
    elif format == "long" and columns_obs_only is not None:
        msg = "Long format with columns_obs_only is not implemented yet."
        raise NotImplementedError(msg)

    # Prepare the AnnData object
    if format == "flat":
        X = df.to_numpy(copy=True)
        obs.index = obs.index.astype(str)
        var = pd.DataFrame(index=df.columns)
        var.index = var.index.astype(str)

        # Handle dtype of X based on presence of numerical columns only
        all_numeric = df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]
        X = X.astype(np.float64 if all_numeric else object)

        edata = EHRData(X=X, obs=obs, var=var)

    elif format == "wide":
        # Define the variable name without suffix for each column.
        variables, timepoints = zip(*[col.split("_t_") for col in df.columns], strict=False)
        unique_variables = np.array(sorted(set(variables)))
        unique_timepoints = np.array(sorted(set(timepoints)))
        variables = np.array(variables)
        timepoints = np.array(timepoints)

        R = np.full((len(df), len(unique_variables), len(unique_timepoints)), np.nan)
        for i, timepoint in enumerate(unique_timepoints):
            for j, variable in enumerate(unique_variables):
                if variable + wide_format_time_suffix + timepoint in df.columns:
                    R[:, j, i] = (
                        df.loc[:, (variables == variable) & (timepoints == timepoint)].to_numpy(copy=True).flatten()
                    )

        obs.index = obs.index.astype(str)
        var = pd.DataFrame(index=unique_variables)
        var.index = var.index.astype(str)
        tem = pd.DataFrame(index=unique_timepoints)
        tem.index = tem.index.astype(str)

        edata = EHRData(R=R, obs=obs, var=var, tem=tem)

    elif format == "long":
        df_pivot = df.pivot_table(
            index=[
                long_format_keys["observation_column"],
                long_format_keys["variable_column"],
                long_format_keys["time_column"],
            ],
            values=long_format_keys["value_column"],
        )
        xr_dataset = df_pivot.to_xarray()
        R = xr_dataset[long_format_keys["value_column"]].values

        obs = pd.DataFrame(index=xr_dataset[long_format_keys["observation_column"]].values)
        obs.index = obs.index.astype(str)
        var = pd.DataFrame(index=xr_dataset[long_format_keys["variable_column"]].values)
        var.index = var.index.astype(str)
        tem = pd.DataFrame(index=xr_dataset[long_format_keys["time_column"]].values)
        tem.index = tem.index.astype(str)

        edata = EHRData(R=R, obs=obs, var=var, tem=tem)

    edata.obs_names = edata.obs_names.astype(str)
    edata.var_names = edata.var_names.astype(str)

    return edata
