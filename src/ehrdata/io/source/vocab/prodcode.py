"""CPRD product code vocabulary helpers.

Loads the CPRD ``product.txt`` / ``product.csv`` lookup (prodcode → drug
substance) and provides a helper to left-join drug substance onto a DataFrame
by prodcode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


def load_product_map(path: Path | str) -> pd.DataFrame:
    """Load CPRD ``product.txt`` or ``product.csv`` → (prodcode, drugsubstance).

    Both the raw ``product.txt`` (tab-delimited) and the derived ``product.csv``
    (comma-delimited, may contain a ``drugsubstance.updated`` column) are
    supported.  When ``drugsubstance.updated`` is present it is preferred over
    the raw ``drugsubstance`` column so that the normalised substance names
    produced by the Preparation step are used.

    Args:
        path: Path to ``product.txt`` or ``product.csv``.

    Returns:
        DataFrame with columns ``prodcode`` (str) and ``drugsubstance`` (str),
        deduplicated on ``prodcode``.
    """
    sep = "," if str(path).endswith(".csv") else "\t"
    df = pd.read_csv(path, sep=sep, dtype=str)
    df.columns = [c.lower().strip() for c in df.columns]
    ingredient_col = "drugsubstance.updated" if "drugsubstance.updated" in df.columns else "drugsubstance"
    df = df[["prodcode", ingredient_col]].rename(columns={ingredient_col: "drugsubstance"})
    df["prodcode"] = df["prodcode"].str.strip()
    df["drugsubstance"] = df["drugsubstance"].str.strip()
    return df.drop_duplicates(subset=["prodcode"]).reset_index(drop=True)


def join_drugsubstance_by_prodcode(
    df: pd.DataFrame,
    product_map: pd.DataFrame,
    *,
    prodcode_col: str = "prodcode",
) -> pd.DataFrame:
    """Left-join ``drugsubstance`` (ingredient) onto *df* via *prodcode_col*.

    Rows whose prodcode is absent from the map receive ``NaN`` for
    ``drugsubstance``.  *df* is not mutated.

    Args:
        df: Source DataFrame containing a prodcode column.
        product_map: Mapping DataFrame as returned by :func:`load_product_map`.
        prodcode_col: Column in *df* that holds prodcode values.

    Returns:
        Copy of *df* with a ``drugsubstance`` column appended; index reset.
    """
    map_ = product_map[["prodcode", "drugsubstance"]].drop_duplicates(subset=["prodcode"])
    result = df.copy().merge(map_, left_on=prodcode_col, right_on="prodcode", how="left")
    if prodcode_col != "prodcode" and "prodcode" in result.columns:
        result = result.drop(columns=["prodcode"])
    return result.reset_index(drop=True)
