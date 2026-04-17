"""ICD coding system utilities.

Provides the E/V prefix heuristic used to infer ICD-9 version flags from raw
claims codes, plus stubs for GEM (General Equivalence Mapping) and CCS
(Clinical Classifications Software) look-ups that require external files.
"""

from __future__ import annotations

import pandas as pd

# ICD-9 code series that begin with E (external causes) or V (supplemental
# factors) are unambiguously ICD-9 when the version flag is absent in claims
# data.  Used by the LCED ETL and replicated in normalize.infer_icd_version.
ICD9_PREFIXES: frozenset[str] = frozenset("EV")


def classify_icd_version(series: pd.Series) -> pd.Series:
    """Infer ICD version for each code based on first-character heuristics.

    Returns ``"9"`` for codes starting with ``E`` or ``V`` (ICD-9 E/V codes),
    ``None`` for all other codes.  This is a heuristic — codes that start with
    ``E`` or ``V`` exist in ICD-10 as well, so the result is reliable only
    when applied to codes whose version is already unknown and the context
    suggests they originate from legacy claims data.

    Args:
        series: Series of ICD code strings.

    Returns:
        Series of ``"9"`` or ``None`` with the same index.
    """
    return series.str[:1].map(lambda c: "9" if c in ICD9_PREFIXES else None)


def normalize_icd_code(code: str) -> str:
    """Strip whitespace and upper-case an ICD code string.

    Args:
        code: Raw ICD code string.

    Returns:
        Cleaned code.
    """
    return code.strip().upper()


# ---------------------------------------------------------------------------
# Stubs — require external files not vendored in this package
# ---------------------------------------------------------------------------


def load_gem_map(path: str) -> pd.DataFrame:  # pragma: no cover
    """Load an ICD-9 / ICD-10 General Equivalence Mapping (GEM) file.

    GEM files are published by CMS and map ICD-9-CM ↔ ICD-10-CM.  They are
    not vendored in this package due to their size; download from
    https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-10-cm-and-gems.

    Args:
        path: Path to a GEM flat file (pipe-delimited, as released by CMS).

    Returns:
        DataFrame with columns ``source_code``, ``target_code``, and
        ``flags``.

    Raises:
        NotImplementedError: Always — implementation pending external data.
    """
    raise NotImplementedError("GEM loader requires external CMS GEM file; not yet implemented.")


def load_ccs_map(path: str) -> pd.DataFrame:  # pragma: no cover
    """Load an AHRQ Clinical Classifications Software (CCS) mapping file.

    Args:
        path: Path to the CCS CSV file (AHRQ format).

    Returns:
        DataFrame with columns ``icd_code``, ``ccs_category``, and
        ``ccs_description``.

    Raises:
        NotImplementedError: Always — implementation pending external data.
    """
    raise NotImplementedError("CCS loader requires external AHRQ CCS file; not yet implemented.")
