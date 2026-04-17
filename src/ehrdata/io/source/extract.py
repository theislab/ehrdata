"""Generic extraction helpers for the non-OMOP source ingestion layer.

Provides DataFrame-level utilities that correspond to the repeating patterns
found across the original MarketScan, LCED, and CPRD R scripts:

- ``union_tables`` mirrors dplyr ``union_all() + distinct_all()``
- ``unnest_codes`` mirrors SQL ``unnest(array[dx1, dx2, ...])``
- ``read_zipped_tsv`` / ``read_zipped_tsvs`` mirror
  ``fread(cmd=paste("unzip -p", file))``
- ``read_csv_with_duckdb`` mirrors ``data.table::fread`` for larger files
"""

from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


def union_tables(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate DataFrames and remove duplicate rows.

    Mirrors the ``union_all() |> distinct_all()`` idiom used in the LCED and
    MarketScan R ETL scripts when assembling diagnosis/procedure tables from
    multiple source views.

    Args:
        dfs: Non-empty list of DataFrames with compatible columns.

    Returns:
        Single deduplicated DataFrame; index reset to 0-based integers.

    Raises:
        ValueError: If *dfs* is empty.
    """
    if not dfs:
        msg = "dfs must contain at least one DataFrame"
        raise ValueError(msg)
    return pd.concat(dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)


def unnest_codes(
    df: pd.DataFrame,
    id_cols: list[str],
    code_cols: list[str],
    *,
    value_name: str = "code",
) -> pd.DataFrame:
    """Pivot wide-format code columns to long format, dropping null values.

    Mirrors ``SELECT unnest(array[dx1, dx2, dx3]) AS dx FROM ...`` used in
    the LCED and MarketScan ETL to expand multi-code rows into one row per
    code.

    Example::

        # Wide: (patient_id=1, eventdate=…, dx1="E11", dx2="I10", dx3=None)
        # Long: (patient_id=1, eventdate=…, dx="E11")
        #       (patient_id=1, eventdate=…, dx="I10")

    Args:
        df: Wide-format DataFrame.
        id_cols: Columns to keep as identifiers (e.g. ``["patient_id", "eventdate"]``).
        code_cols: Wide code columns to unnest (e.g. ``["dx1", "dx2", "dx3"]``).
        value_name: Name for the resulting long-format code column.

    Returns:
        Long-format DataFrame with columns ``id_cols + [value_name]``, null
        codes removed and duplicate rows dropped; index reset.
    """
    long = df[id_cols + code_cols].melt(
        id_vars=id_cols,
        value_vars=code_cols,
        value_name=value_name,
    )
    return long.dropna(subset=[value_name]).drop(columns="variable").drop_duplicates().reset_index(drop=True)


def read_zipped_tsv(
    zip_path: Path | str,
    member: str,
    *,
    usecols: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    r"""Read a single TSV member from a zip archive.

    Mirrors ``fread(cmd=paste("unzip -p", file), sep="\\t")`` used in the
    CPRD R ETL to stream individual extract files without unzipping to disk.

    Args:
        zip_path: Path to the ``.zip`` file.
        member: Filename of the TSV inside the archive.
        usecols: Columns to select; all columns if ``None``.
        **kwargs: Forwarded to :func:`pandas.read_csv`.

    Returns:
        DataFrame with the member's contents.
    """
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as fh:
        return pd.read_csv(fh, sep="\t", usecols=usecols, **kwargs)


def read_zipped_tsvs(
    zip_path: Path | str,
    *,
    pattern: str | None = None,
    usecols: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Read and concatenate all matching TSV members from a zip archive.

    Mirrors the ``foreach``-based parallel unzip-and-bind pattern in the
    CPRD ETL that combined Controls and Main extract files.

    Args:
        zip_path: Path to the ``.zip`` file.
        pattern: Optional substring to filter member names by; all ``.txt``,
            ``.tsv``, and ``.csv`` members are included if ``None``.
        usecols: Columns to select from each member.
        **kwargs: Forwarded to :func:`pandas.read_csv`.

    Returns:
        Concatenated DataFrame from all matched members; duplicates preserved
        (call :func:`union_tables` afterwards to deduplicate).
    """
    _READABLE = (".txt", ".tsv", ".csv")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(_READABLE)]
        if pattern is not None:
            members = [m for m in members if pattern in m]
        dfs = []
        for member in members:
            with zf.open(member) as fh:
                dfs.append(pd.read_csv(fh, sep="\t", usecols=usecols, **kwargs))
    if not dfs:
        return pd.DataFrame(columns=usecols or [])
    return pd.concat(dfs, ignore_index=True)


def read_csv_with_duckdb(
    path: Path | str,
    *,
    columns: list[str] | None = None,
    where: str | None = None,
) -> pd.DataFrame:
    """Read a CSV file (or glob pattern) via DuckDB for efficient loading.

    Use this instead of :func:`pandas.read_csv` when the source file is large
    or when a ``WHERE`` clause can substantially reduce the loaded rows.

    Args:
        path: Path to a CSV file or a glob pattern (e.g. ``"data/*.csv"``).
            DuckDB resolves globs and unions matching files automatically.
        columns: Columns to project; all columns if ``None``.
        where: Optional SQL ``WHERE`` clause applied inside DuckDB before
            materialising results (e.g. ``"patient_id IN (1, 2, 3)"``).

    Returns:
        DataFrame with the query results.
    """
    import duckdb

    col_list = ", ".join(columns) if columns else "*"
    sql = f"SELECT {col_list} FROM read_csv_auto('{path}')"
    if where:
        sql += f" WHERE {where}"
    con = duckdb.connect(database=":memory:")
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()
