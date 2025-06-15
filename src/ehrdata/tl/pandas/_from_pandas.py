from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ehrdata import EHRData
PANDAS_FORMATS = ["flat", "wide", "long"]


def from_pandas(
    df: pd.DataFrame,
    columns_obs_only: Iterable[str] | None = None,
    index_column: str | int | None = None,
    format: Literal["flat", "wide", "long"] = "flat",
    wide_format_time_suffix: str | None = None,
    long_format_keys: dict[Literal["observation_column", "variable_column", "time_column", "value_column"], str]
    | None = None,
) -> EHRData:
    """Transform a given :class:`~pandas.DataFrame` into an :class:`~ehrdata.EHRData` object.

    Note that columns containing boolean values (either 0/1 or T(t)rue/F(f)alse)
    will be stored as boolean columns whereas the other non-numerical columns will be stored as categorical values.

    Args:
        df: The dataframe to be transformed.
        columns_obs_only: An optional list of column names that should belong to obs only and not X.
        index_column: The index column of obs. This can be either a column name (or its numerical index in the DataFrame) or the index of the dataframe.
        format: The format of the input dataframe. If the data is not longitudinal, choose `format="flat"`. If the data is longitudinal in the long format, choose `format="long"`. If the data is longitudinal in a wide format, choose `format="wide"`.
        wide_format_time_suffix: Use only if `format="wide"`. Suffices in the variable columns that indicate the time of the observation. The collected suffices will be sorted lexicographically, and the variables ordered accordingly along the 3rd axis of the :class:`~ehrdata.EHRData` object.
        long_format_keys: Use only if `format="long"`. The keys of the dataframe in the long format. The dictionary should have the following structure: {"observation_column": "<the column name of the observation ids>", "variable_column": "<the column name of the variable ids>", "time_column": "<the column name of the time>", "value_column": "<the column name of the values>"}.

    Returns:
        The :class:`~ehrdata.EHRData` object created from the given :class:`~pandas.DataFrame`.

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
        >>> edata

        >>> EHRData object with n_obs × n_vars = 5 × 2
        >>> shape of .X: (5, 2)

        >>> df_wide = pd.DataFrame(
        ...     {
        ...         "patient_id": ["0", "1"],
        ...         "sex": ["F", "M"],
        ...         "systolic_blood_pressure_t_0": [120, 130],
        ...         "systolic_blood_pressure_t_2": [125, np.nan],  # the suffix strings are sorted lexicographically
        ...         "systolic_blood_pressure_t_1": [np.nan, 135],
        ...     }
        ... )
        >>> edata = ed.tl.from_pandas(df_wide, format="wide", columns_obs_only=["patient_id", "sex"])
        >>> edata

        >>> EHRData object with n_obs × n_vars × n_t = 2 × 1 × 3
        >>> obs: 'patient_id', 'sex'
        >>> shape of .X: (2, 1)
        >>> shape of .R: (2, 1, 3)


        >>> df_long = pd.DataFrame(
        ...     {
        ...         "observation_id": ["0", "0", "0", "1", "1", "1"],
        ...         "variable": [
        ...             "sex",
        ...             "systolic_blood_pressure",
        ...             "systolic_blood_pressure",
        ...             "sex",
        ...             "systolic_blood_pressure",
        ...             "systolic_blood_pressure",
        ...         ],
        ...         "time": ["t_0", "t_0", "t_1", "t_0", "t_0", "t_2"],
        ...         "value": ["F", 120, 125, "M", 130, 135],
        ...     }
        ... )
        >>> edata = ed.tl.from_pandas(df_long, format="long", columns_obs_only=["sex"])
        >>> edata

        >>> EHRData object with n_obs × n_vars × n_t = 2 × 1 × 3
        >>> obs: "sex"
        >>> shape of .X: (2, 1)
        >>> shape of .R: (2, 1, 3)

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
            columns_obs_only = list(columns_obs_only)
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
        try:
            variables, timepoints = zip(*[col.split(wide_format_time_suffix) for col in df.columns], strict=False)
        except ValueError:
            err_msg = f"The wide format time suffix {wide_format_time_suffix} is not found in the column names. For variables not containing the time suffix, please put them as static variables into `obs` using the `columns_obs_only` parameter. Please specify the time suffix using the wide_format_time_suffix parameter."
            raise ValueError(err_msg) from None

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
        if index_column is not None:
            err_msg = (
                "For long format, use the `observation_column` parameter to specify the index of the observations."
            )
            raise ValueError(err_msg)

        unique_observations = df[long_format_keys["observation_column"]].unique()
        obs = pd.DataFrame(index=unique_observations)

        if columns_obs_only is not None:
            df_obs_only = df[df[long_format_keys["variable_column"]].isin(columns_obs_only)].drop(
                columns=long_format_keys["time_column"]
            )

            df_obs_pivoted = pd.pivot(
                df_obs_only, index=long_format_keys["observation_column"], columns=long_format_keys["variable_column"]
            )
            df_obs_pivoted.columns = df_obs_pivoted.columns.droplevel(0)

            obs = obs.join(df_obs_pivoted, how="left")

        if columns_obs_only is not None:
            df = df[~df[long_format_keys["variable_column"]].isin(columns_obs_only)]
        xr_dataarray = df.set_index(
            [
                long_format_keys["observation_column"],
                long_format_keys["variable_column"],
                long_format_keys["time_column"],
            ]
        ).to_xarray()

        R = xr_dataarray[long_format_keys["value_column"]].values

        var = pd.DataFrame(index=xr_dataarray[long_format_keys["variable_column"]].values)
        var.index = var.index.astype(str)
        tem = pd.DataFrame(index=xr_dataarray[long_format_keys["time_column"]].values)
        tem.index = tem.index.astype(str)

        edata = EHRData(R=R, obs=obs, var=var, tem=tem)

    edata.obs_names = edata.obs_names.astype(str)
    edata.var_names = edata.var_names.astype(str)

    return edata
