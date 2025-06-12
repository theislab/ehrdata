from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from ehrdata import EHRData


def read_csv(
    filename: Path | str,
    *,
    sep: str = ",",
    index_column: str | None = None,
    columns_obs_only: list[str] | None = None,
    **kwargs,
) -> EHRData:
    """Reads a csv file.

    This function reads a csv file, and creates an :class:`ehrdata.EHRData` object.
    It first reads the csv file using :func:`pandas.read_csv`, and then passes the resulting DataFrame to :func:`ehrdata.tl.from_pandas`.

    Args:
        filename: Path to the file or directory to read.
        sep: Separator in the file. Delegates to pandas.read_csv().
        index_column: If specified, this column of the csv file will be used for the `.obs` DataFrame.
        columns_obs_only: These columns will be added to the `.obs` DataFrame only.
        **kwargs: Passed to :func:`pandas.read_csv`

    Returns:
        The dataset in the form of an :class:`ehrdata.EHRData` object.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.io.read_csv("myfile.csv")
    """
    from ehrdata.tl import from_pandas

    df = pd.read_csv(filename, sep=sep, index_col=index_column, **kwargs)
    edata = from_pandas(df, columns_obs_only=columns_obs_only)

    return edata
