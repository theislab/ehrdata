from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from ehrdata import EHRData


def read_csv(
    filename: Path | str,
    *,
    sep: str = ",",
    index_column: str | None = None,
    columns_obs_only: Iterable[str] | None = None,
    format: Literal["flat", "wide", "long"] = "flat",
    wide_format_time_suffix: str | None = None,
    long_format_keys: dict[Literal["observation_column", "variable_column", "time_column", "value_column"], str]
    | None = None,
    **kwargs,
) -> EHRData:
    """Read a comma-separated values (csv) file into an :class:`~ehrdata.EHRData` object.

    It first reads the csv file using :func:`pandas.read_csv`, and then passes the resulting :class:`~pandas.DataFrame` to :func:`ehrdata.io.from_pandas`.
    See the documentation of :func:`ehrdata.io.from_pandas` for more details of table layouts.

    Args:
        filename: Path to the file or directory to read. Delegates to :func:`pandas.read_csv`.
        sep: Separator in the file. Delegates to :func:`pandas.read_csv`.
        index_column: If specified, this column of the csv file will be used for the `.obs` dataframe.
            Delegates to :func:`~ehrdata.io.from_pandas`.
        columns_obs_only: These columns will be added to the `.obs` dataframe only.
            Delegates to :func:`~ehrdata.io.from_pandas`.
        format: The format of the input dataframe. If the data is not longitudinal, choose `format="flat"`.
            If the data is longitudinal in the long format, choose `format="long"`.
            If the data is longitudinal in a wide format, choose `format="wide"`.
            Delegates to :func:`~ehrdata.io.from_pandas`.
        wide_format_time_suffix: Use only if `format="wide"`.
            Suffices in the variable columns that indicate the time of the observation.
            The collected suffices will be sorted lexicographically, and the variables ordered accordingly along the 3rd axis of the :class:`~ehrdata.EHRData` object.
            Delegates to :func:`~ehrdata.io.from_pandas`.
        long_format_keys: Use only if `format="long"`.
            The keys of the dataframe in the long format.
            The dictionary should have the following structure:
            {"observation_column": "<the column name of the observation ids>",
            "variable_column": "<the column name of the variable ids>",
            "time_column": "<the column name of the time>",
            "value_column": "<the column name of the values>"}.
            Delegates to :func:`~ehrdata.io.from_pandas`.
        **kwargs: Passed to :func:`pandas.read_csv`.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.io.read_csv("myfile.csv")
    """
    from ehrdata.io import from_pandas

    df = pd.read_csv(filename, sep=sep, index_col=index_column, **kwargs)
    edata = from_pandas(
        df,
        columns_obs_only=columns_obs_only,
        format=format,
        wide_format_time_suffix=wide_format_time_suffix,
        long_format_keys=long_format_keys,
    )

    return edata
