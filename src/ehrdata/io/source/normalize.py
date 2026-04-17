"""Normalization helpers for the non-OMOP source ingestion layer.

Provides atomic transformations (ID coercion, date parsing, ICD version
inference, deduplication, sorting) and composite ``normalize_*`` pipelines
that apply them in the correct order for each canonical table type.

All functions return new DataFrames and do not mutate their inputs.
"""

from __future__ import annotations

import pandas as pd


# ICD-9 uses E-codes (external causes) and V-codes (supplemental factors).
# When ``dxver`` is missing in a claims record, a leading E or V reliably
# identifies ICD-9 coding — heuristic taken from the original LCED ETL.
_ICD9_LEADING_CHARS = frozenset("EV")


def coerce_patient_id(series: pd.Series) -> pd.Series:
    """Cast patient identifiers to strings and strip surrounding whitespace.

    Handles integer primary keys (``enrolid``, ``patid``) as well as string
    keys with padding artefacts from CSV exports.

    Args:
        series: Raw patient ID column of any dtype.

    Returns:
        String series with whitespace stripped and original index preserved.
    """
    return series.astype(str).str.strip()


def coerce_date(series: pd.Series, formats: list[str] | None = None) -> pd.Series:
    """Parse a date series to ``datetime64[ns]``, trying multiple formats.

    Unparseable values become ``NaT`` rather than raising.

    Args:
        series: Raw date column (strings, objects, or already datetime).
        formats: Ordered list of ``strptime`` format strings to attempt before
            falling back to pandas auto-inference.  Useful when the source has
            a known non-ISO format (e.g. ``"%d/%m/%Y"`` for CPRD extracts).

    Returns:
        Series of ``datetime64[ns]`` with the original index preserved.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if formats:
        for fmt in formats:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
            if parsed.notna().any():
                return parsed
    return pd.to_datetime(series, errors="coerce")


def infer_icd_version(
    df: pd.DataFrame,
    *,
    dx_col: str = "dx",
    dxver_col: str = "dxver",
) -> pd.DataFrame:
    """Fill missing ICD version flags using code-prefix heuristics.

    When ``dxver`` is ``None`` / ``NaN`` and the code starts with ``E`` or
    ``V`` the version is set to ``"9"`` (ICD-9).  All other missing values are
    left as ``None``.  Existing non-null values are never overwritten.

    This reproduces the LCED ETL logic::

        dxver = if_else(is.na(dxver) & substr(dx,1,1) %in% c("E","V"), "9", dxver)

    Args:
        df: DataFrame containing diagnosis codes and a version column.
        dx_col: Column holding the raw ICD code string.
        dxver_col: Column holding the version flag
            (``"0"`` = ICD-10, ``"9"`` = ICD-9).

    Returns:
        Copy of *df* with ``dxver`` filled where inferrable.
    """
    df = df.copy()
    # Cast to object so string assignment into a float NaN column doesn't warn.
    df[dxver_col] = df[dxver_col].astype(object)
    missing_ver = df[dxver_col].isna()
    starts_icd9 = df[dx_col].str[:1].isin(_ICD9_LEADING_CHARS)
    df.loc[missing_ver & starts_icd9, dxver_col] = "9"
    return df


def deduplicate(df: pd.DataFrame, *, subset: list[str] | None = None) -> pd.DataFrame:
    """Remove duplicate rows, optionally restricted to a column subset.

    Args:
        df: Input DataFrame.
        subset: Columns to consider for duplication detection; all columns if
            ``None``.

    Returns:
        DataFrame with duplicates removed and index reset to 0-based integers.
    """
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def sort_events(
    df: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    date_col: str = "eventdate",
) -> pd.DataFrame:
    """Sort a clinical event table by patient then event date.

    Rows with missing dates are placed last, matching the ``arrange()``
    behaviour in the original R ETL scripts.

    Args:
        df: Clinical event DataFrame containing patient and date columns.
        patient_col: Name of the patient identifier column.
        date_col: Name of the event date column.

    Returns:
        Sorted DataFrame with index reset.
    """
    return df.sort_values([patient_col, date_col], na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Composite normalisation pipelines
# ---------------------------------------------------------------------------


def normalize_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw diagnosis DataFrame to the canonical schema.

    Applies (in order): patient ID coercion, date parsing, ICD version
    inference, deduplication, and chronological sort.

    Args:
        df: Raw diagnosis DataFrame.  Must contain ``patient_id``,
            ``eventdate``, and ``dx``.  ``dxver`` is optional; if absent it is
            added as a column of ``None`` values before inference.

    Returns:
        Normalised diagnosis DataFrame conforming to
        :data:`~ehrdata.io.source.schema.DIAGNOSIS`.
    """
    df = df.copy()
    df["patient_id"] = coerce_patient_id(df["patient_id"])
    df["eventdate"] = coerce_date(df["eventdate"])
    if "dxver" not in df.columns:
        df["dxver"] = None
    df = infer_icd_version(df)
    df = deduplicate(df)
    df = sort_events(df)
    return df


def normalize_therapy(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw therapy DataFrame to the canonical schema.

    Coerces patient IDs and all date columns, then deduplicates.

    Args:
        df: Raw therapy DataFrame.  Must contain ``patient_id``.  Date columns
            ``prescription_date``, ``start_date``, ``fill_date``, and
            ``end_date`` are parsed when present.

    Returns:
        Normalised therapy DataFrame conforming to
        :data:`~ehrdata.io.source.schema.THERAPY`.
    """
    df = df.copy()
    df["patient_id"] = coerce_patient_id(df["patient_id"])
    for col in ("prescription_date", "start_date", "fill_date", "end_date"):
        if col in df.columns:
            df[col] = coerce_date(df[col])
        else:
            df[col] = pd.NaT
    df = deduplicate(df)
    return df


def normalize_labtest(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw lab test DataFrame to the canonical schema.

    Args:
        df: Raw labtest DataFrame.  Must contain ``patient_id`` and
            ``eventdate``.

    Returns:
        Normalised labtest DataFrame conforming to
        :data:`~ehrdata.io.source.schema.LABTEST`.
    """
    df = df.copy()
    df["patient_id"] = coerce_patient_id(df["patient_id"])
    df["eventdate"] = coerce_date(df["eventdate"])
    df = deduplicate(df)
    df = sort_events(df)
    return df


def normalize_procedure(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw procedure DataFrame to the canonical schema.

    Args:
        df: Raw procedure DataFrame.  Must contain ``patient_id``,
            ``eventdate``, and ``proc``.

    Returns:
        Normalised procedure DataFrame conforming to
        :data:`~ehrdata.io.source.schema.PROCEDURE`.
    """
    df = df.copy()
    df["patient_id"] = coerce_patient_id(df["patient_id"])
    df["eventdate"] = coerce_date(df["eventdate"])
    df = deduplicate(df)
    df = sort_events(df)
    return df
