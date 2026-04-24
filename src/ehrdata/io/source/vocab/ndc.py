"""NDC-to-ingredient vocabulary loader.

The mapping file (``ndc ingredient map.txt``) shipped with the IBM LCED
Coding System has 52,859 rows with columns ``ndc11``, ``rxcui``, and
``ingredient``.  This module loads that file and provides a helper to join
ingredient information onto a therapy DataFrame by NDC-11 code.

NDC-11 codes are zero-padded to exactly 11 characters on load so that
matching is robust to leading-zero loss during CSV export.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

_NDC11_LEN = 11


def load_ndc_ingredient_map(path: Path | str) -> pd.DataFrame:
    """Load the NDC-to-ingredient mapping file.

    Args:
        path: Path to ``ndc ingredient map.txt`` (comma-separated, with
            header ``ndc11,rxcui,ingredient``).

    Returns:
        DataFrame with columns ``ndc11`` (zero-padded 11-char string),
        ``rxcui`` (string), and ``ingredient`` (string).
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.lower().str.strip()
    df["ndc11"] = df["ndc11"].str.strip().str.zfill(_NDC11_LEN)
    df["rxcui"] = df["rxcui"].str.strip()
    df["ingredient"] = df["ingredient"].str.strip()
    return df[["ndc11", "rxcui", "ingredient"]]


def join_ingredient_by_ndc(
    df: pd.DataFrame,
    ndc_map: pd.DataFrame,
    *,
    ndc_col: str = "ndc11",
) -> pd.DataFrame:
    """Add an ``ingredient`` column to *df* via a left join on NDC code.

    If *df* already contains an ``ingredient`` column it is overwritten with
    the joined values (preserving ``NaN`` where no match is found).

    Args:
        df: DataFrame containing an NDC column.
        ndc_map: Mapping DataFrame as returned by :func:`load_ndc_ingredient_map`.
        ndc_col: Name of the NDC column in *df*.

    Returns:
        Copy of *df* with an ``ingredient`` column appended or replaced.
    """
    df = df.copy()
    if "ingredient" in df.columns:
        df = df.drop(columns="ingredient")
    ndc_lookup = ndc_map[["ndc11", "ingredient"]].drop_duplicates(subset="ndc11")
    merged = df.merge(
        ndc_lookup.rename(columns={"ndc11": ndc_col}),
        on=ndc_col,
        how="left",
    )
    return merged
