"""RxCUI-to-ingredient vocabulary loader.

The mapping file (``rxcui ingredient map.txt``) shipped with the IBM LCED
Coding System has 24,659 rows with columns ``V1`` (row index), ``rxcui``, and
``ingredient``.  This module loads that file and provides a helper to join
ingredient information onto a therapy DataFrame by RxNorm concept identifier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


def load_rxcui_ingredient_map(path: Path | str) -> pd.DataFrame:
    """Load the RxCUI-to-ingredient mapping file.

    Args:
        path: Path to ``rxcui ingredient map.txt`` (comma-separated, with
            header ``V1,rxcui,ingredient`` where ``V1`` is a row index).

    Returns:
        DataFrame with columns ``rxcui`` (string) and ``ingredient`` (string).
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.lower().str.strip()
    df["rxcui"] = df["rxcui"].str.strip()
    df["ingredient"] = df["ingredient"].str.strip()
    return df[["rxcui", "ingredient"]]


def join_ingredient_by_rxcui(
    df: pd.DataFrame,
    rxcui_map: pd.DataFrame,
    *,
    rxcui_col: str = "rxcui",
) -> pd.DataFrame:
    """Add an ``ingredient`` column to *df* via a left join on RxCUI.

    If *df* already contains an ``ingredient`` column it is overwritten with
    the joined values.

    Args:
        df: DataFrame containing an RxCUI column.
        rxcui_map: Mapping DataFrame as returned by
            :func:`load_rxcui_ingredient_map`.
        rxcui_col: Name of the RxCUI column in *df*.

    Returns:
        Copy of *df* with an ``ingredient`` column appended or replaced.
    """
    df = df.copy()
    if "ingredient" in df.columns:
        df = df.drop(columns="ingredient")
    rxcui_lookup = rxcui_map[["rxcui", "ingredient"]].drop_duplicates(subset="rxcui")
    merged = df.merge(
        rxcui_lookup.rename(columns={"rxcui": rxcui_col}),
        on=rxcui_col,
        how="left",
    )
    return merged
