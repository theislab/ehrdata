"""LOINC (Logical Observation Identifiers Names and Codes) vocabulary loader.

Loads a minimal LOINC reference table and provides a helper to join
human-readable component names onto a lab-test DataFrame.

The full LOINC corpus (~750 MB) is not vendored in this package.  Only a
small fixture-based subset is used in tests.  Download the full table from
https://loinc.org/downloads/ and pass its path to :func:`load_loinc_map`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_loinc_map(path: Path | str) -> pd.DataFrame:
    """Load a LOINC reference CSV into a lookup DataFrame.

    The file must be comma-separated and contain at minimum the columns
    ``loinc`` (or ``loinc_num``) and ``component``.  Long-common-name and
    other columns are preserved if present.

    Args:
        path: Path to a LOINC CSV file.

    Returns:
        DataFrame with ``loinc`` as the first column, followed by any
        additional columns present in the file.
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.lower().str.strip()
    # LOINC distributes the code field as "loinc_num" in the full table;
    # normalise to "loinc" for consistency with our canonical schema.
    if "loinc_num" in df.columns and "loinc" not in df.columns:
        df = df.rename(columns={"loinc_num": "loinc"})
    df["loinc"] = df["loinc"].str.strip()
    df["component"] = df["component"].str.strip()
    return df


def join_component_by_loinc(
    df: pd.DataFrame,
    loinc_map: pd.DataFrame,
    *,
    loinc_col: str = "loinc",
) -> pd.DataFrame:
    """Add a ``component`` column to *df* via a left join on LOINC code.

    Args:
        df: DataFrame containing a LOINC code column.
        loinc_map: Lookup DataFrame as returned by :func:`load_loinc_map`.
        loinc_col: Name of the LOINC column in *df*.

    Returns:
        Copy of *df* with a ``component`` column appended or replaced.
    """
    df = df.copy()
    if "component" in df.columns:
        df = df.drop(columns="component")
    lookup = loinc_map[["loinc", "component"]].drop_duplicates(subset="loinc")
    return df.merge(
        lookup.rename(columns={"loinc": loinc_col}),
        on=loinc_col,
        how="left",
    )
