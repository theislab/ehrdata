"""CPRD Read code vocabulary helpers.

Loads the CPRD ``medical.txt`` lookup (medcode → Read code) and provides a
helper to left-join Read codes onto a DataFrame by medcode.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_medical_map(path: Path | str) -> pd.DataFrame:
    """Load CPRD ``medical.txt`` → (medcode, readcode) mapping.

    ``medical.txt`` is tab-delimited with at least the columns ``medcode``,
    ``readcode``, and ``desc``.  Column names are lower-cased and stripped
    before processing; only ``medcode`` and ``readcode`` are returned.

    Args:
        path: Path to ``medical.txt`` (or a compatible tab-delimited file).

    Returns:
        DataFrame with columns ``medcode`` (str) and ``readcode`` (str),
        deduplicated on ``medcode``.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.lower().strip() for c in df.columns]
    df["medcode"] = df["medcode"].str.strip()
    df["readcode"] = df["readcode"].str.strip()
    return df[["medcode", "readcode"]].drop_duplicates(subset=["medcode"]).reset_index(drop=True)


def join_readcode_by_medcode(
    df: pd.DataFrame,
    medical_map: pd.DataFrame,
    *,
    medcode_col: str = "medcode",
) -> pd.DataFrame:
    """Left-join ``readcode`` onto *df* using *medcode_col* as the key.

    Rows in *df* whose medcode is absent from the map receive ``NaN`` for
    ``readcode``.  *df* is not mutated.

    Args:
        df: Source DataFrame containing a medcode column.
        medical_map: Mapping DataFrame as returned by :func:`load_medical_map`.
        medcode_col: Column in *df* that holds medcode values.

    Returns:
        Copy of *df* with a ``readcode`` column appended; index reset.
    """
    map_ = medical_map[["medcode", "readcode"]].drop_duplicates(subset=["medcode"])
    result = df.copy().merge(map_, left_on=medcode_col, right_on="medcode", how="left")
    if medcode_col != "medcode" and "medcode" in result.columns:
        result = result.drop(columns=["medcode"])
    return result.reset_index(drop=True)
