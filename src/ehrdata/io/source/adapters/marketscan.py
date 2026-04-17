"""IBM MarketScan claims adapter.

Translates raw IBM MarketScan Commercial Claims source tables (supplied as
:class:`~pandas.DataFrame` objects) into canonical DataFrames conforming to
the schemas defined in :mod:`ehrdata.io.source.schema`.

Each ``build_*`` function accepts the specific source tables it needs, applies
column selection / renaming, unnests wide code arrays into long format, unions
multiple sources, and delegates final normalization to
:mod:`ehrdata.io.source.normalize`.

Source column reference (original MarketScan PostgreSQL view names):

- ``facility_header``: facility/outpatient encounter header
- ``inpatient_admissions``: inpatient admission records
- ``inpatient_services``: inpatient line-item services
- ``outpatient_services``: outpatient line-item services
- ``outpatient_prescription_drugs``: pharmacy claims
- ``enrollment_annual_summary``: annual enrollment snapshot
- ``enrollment_detail``: enrollment coverage detail with plan info
"""

from __future__ import annotations

import pandas as pd

from ehrdata.io.source.extract import union_tables, unnest_codes
from ehrdata.io.source.normalize import (
    coerce_date,
    coerce_patient_id,
    deduplicate,
    infer_icd_version,
    sort_events,
)

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

# Canonical patient ID (MarketScan calls it enrolid)
_PID = "patient_id"
_ENROLID = "enrolid"

# Patinfo columns that exist in MarketScan beyond the canonical three
_PATINFO_EXTRA_COLS = [
    "efamid", "year", "region", "msa", "wgtkey",
    "eeclass", "eestatu", "egeoloc", "emprel", "indstry",
]

# All patinfo columns as they appear in MarketScan source tables
_PATINFO_SRC_COLS = [_ENROLID, "dobyr", "sex"] + _PATINFO_EXTRA_COLS


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


def build_diagnosis(
    facility_header: pd.DataFrame,
    inpatient_admissions: pd.DataFrame,
    inpatient_services: pd.DataFrame,
    outpatient_services: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical diagnosis table from four MarketScan sources.

    Mirrors the ``unnest(array[dx1..dx9])`` + ``union_all`` + ``distinct``
    pattern from ``MarketScan Data Cleaning.R`` (lines 44–56).

    Args:
        facility_header: ``commercial.facility_header`` view data.
            Must contain ``enrolid``, ``dxver``, ``svcdate``, ``dx1``–``dx9``.
        inpatient_admissions: ``commercial.inpatient_admissions`` view data.
            Must contain ``enrolid``, ``dxver``, ``admdate``, ``pdx``,
            ``dx1``–``dx15``.
        inpatient_services: ``commercial.inpatient_services`` view data.
            Must contain ``enrolid``, ``dxver``, ``svcdate``, ``pdx``,
            ``dx1``–``dx4``.
        outpatient_services: ``commercial.outpatient_services`` view data.
            Must contain ``enrolid``, ``dxver``, ``svcdate``, ``dx1``–``dx4``.

    Returns:
        Canonical diagnosis DataFrame conforming to
        :data:`~ehrdata.io.source.schema.DIAGNOSIS`.
    """
    parts = [
        _unnest_dx(facility_header, date_col="svcdate",
                   dx_cols=[f"dx{i}" for i in range(1, 10)]),
        _unnest_dx(inpatient_admissions, date_col="admdate",
                   dx_cols=["pdx"] + [f"dx{i}" for i in range(1, 16)]),
        _unnest_dx(inpatient_services, date_col="svcdate",
                   dx_cols=["pdx"] + [f"dx{i}" for i in range(1, 5)]),
        _unnest_dx(outpatient_services, date_col="svcdate",
                   dx_cols=[f"dx{i}" for i in range(1, 5)]),
    ]
    df = union_tables(parts)
    df[_PID] = coerce_patient_id(df[_PID])
    df["eventdate"] = coerce_date(df["eventdate"])
    df = infer_icd_version(df)
    df = deduplicate(df)
    df = sort_events(df)
    return df


def _unnest_dx(src: pd.DataFrame, *, date_col: str, dx_cols: list[str]) -> pd.DataFrame:
    """Select, rename, and unnest diagnosis codes from one source table."""
    present_dx = [c for c in dx_cols if c in src.columns]
    dxver_col = ["dxver"] if "dxver" in src.columns else []
    tmp = src[[_ENROLID, date_col] + dxver_col + present_dx].copy()
    tmp = tmp.rename(columns={_ENROLID: _PID, date_col: "eventdate"})
    if "dxver" not in tmp.columns:
        tmp["dxver"] = None
    return unnest_codes(tmp, id_cols=[_PID, "eventdate", "dxver"], code_cols=present_dx, value_name="dx")


# ---------------------------------------------------------------------------
# Therapy
# ---------------------------------------------------------------------------


def build_therapy(
    outpatient_prescription_drugs: pd.DataFrame,
    *,
    ndc_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the canonical therapy table from MarketScan pharmacy claims.

    Mirrors ``MarketScan Data Cleaning.R`` lines 82–98:
    - Maps ``svcdate`` → ``fill_date``
    - Calculates ``end_date = fill_date + daysupp`` days
    - Renames ``ndcnum`` → ``ndc11``
    - Optionally joins ingredient from NDC map

    Args:
        outpatient_prescription_drugs: ``commercial.outpatient_prescription_drugs``
            view data.  Must contain ``enrolid``, ``svcdate``, ``daysupp``,
            ``refill``, and ``ndcnum``.
        ndc_map: Optional NDC→ingredient map as returned by
            :func:`~ehrdata.io.source.vocab.ndc.load_ndc_ingredient_map`.
            When ``None`` the ``ingredient`` column is set to ``None``.

    Returns:
        Canonical therapy DataFrame conforming to
        :data:`~ehrdata.io.source.schema.THERAPY`.
    """
    src = outpatient_prescription_drugs
    df = pd.DataFrame()
    df[_PID] = coerce_patient_id(src[_ENROLID])
    df["prescription_date"] = pd.NaT
    df["start_date"] = pd.NaT
    df["fill_date"] = coerce_date(src["svcdate"])
    df["refill"] = pd.to_numeric(src["refill"], errors="coerce").astype("Int64")
    df["rxcui"] = None
    df["ndc11"] = src["ndcnum"].astype(str).str.strip().str.zfill(11)

    # Calculate end_date = fill_date + daysupp days
    daysupp = pd.to_numeric(src["daysupp"], errors="coerce")
    df["end_date"] = df["fill_date"] + pd.to_timedelta(daysupp, unit="D")

    if ndc_map is not None:
        from ehrdata.io.source.vocab.ndc import join_ingredient_by_ndc
        df = join_ingredient_by_ndc(df, ndc_map)
    else:
        df["ingredient"] = None

    df = deduplicate(df)
    return df[["patient_id", "prescription_date", "start_date", "fill_date",
               "end_date", "refill", "rxcui", "ndc11", "ingredient"]]


# ---------------------------------------------------------------------------
# Procedure
# ---------------------------------------------------------------------------


def build_procedure(
    facility_header: pd.DataFrame,
    inpatient_admissions: pd.DataFrame,
    inpatient_services: pd.DataFrame,
    outpatient_services: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical procedure table from four MarketScan sources.

    Mirrors ``MarketScan Data Cleaning.R`` lines 121–140.

    Note: ``inpatient_services`` unnests ``array[pdx, proc1]`` — this
    replicates the original ETL which included the primary diagnosis code
    alongside the procedure code in the inpatient services procedure array.

    Args:
        facility_header: Must contain ``enrolid``, ``svcdate``,
            ``proc1``–``proc6``.
        inpatient_admissions: Must contain ``enrolid``, ``admdate``,
            ``pproc``, ``proc1``–``proc15``.
        inpatient_services: Must contain ``enrolid``, ``svcdate``,
            ``proctyp``, ``pdx``, ``proc1``.
        outpatient_services: Must contain ``enrolid``, ``svcdate``,
            ``proctyp``, ``proc1``.

    Returns:
        Canonical procedure DataFrame conforming to
        :data:`~ehrdata.io.source.schema.PROCEDURE`.
    """
    parts = [
        _unnest_proc(facility_header, date_col="svcdate",
                     proc_cols=[f"proc{i}" for i in range(1, 7)], proctype_col=None),
        _unnest_proc(inpatient_admissions, date_col="admdate",
                     proc_cols=["pproc"] + [f"proc{i}" for i in range(1, 16)], proctype_col=None),
        _unnest_proc(inpatient_services, date_col="svcdate",
                     proc_cols=["pdx", "proc1"], proctype_col="proctyp"),
        _unnest_proc(outpatient_services, date_col="svcdate",
                     proc_cols=["proc1"], proctype_col="proctyp"),
    ]
    df = union_tables(parts)
    df[_PID] = coerce_patient_id(df[_PID])
    df["eventdate"] = coerce_date(df["eventdate"])
    df = deduplicate(df)
    df = sort_events(df)
    return df


def _unnest_proc(
    src: pd.DataFrame,
    *,
    date_col: str,
    proc_cols: list[str],
    proctype_col: str | None,
) -> pd.DataFrame:
    """Select, rename, and unnest procedure codes from one source table."""
    present_proc = [c for c in proc_cols if c in src.columns]
    tmp = src[[_ENROLID, date_col] + present_proc].copy()
    tmp = tmp.rename(columns={_ENROLID: _PID, date_col: "eventdate"})
    if proctype_col and proctype_col in src.columns:
        tmp["proctype"] = src[proctype_col].values
    else:
        tmp["proctype"] = None
    long = unnest_codes(tmp, id_cols=[_PID, "eventdate", "proctype"], code_cols=present_proc, value_name="proc")
    return long


# ---------------------------------------------------------------------------
# Patinfo
# ---------------------------------------------------------------------------


def build_patinfo(*source_tables: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical patient-info table by unioning all MarketScan sources.

    Accepts any number of source DataFrames that contain ``enrolid``,
    ``dobyr``, and ``sex`` columns.  Extra MarketScan columns (``efamid``,
    ``year``, ``region``, ``msa``, etc.) are carried through when present in
    all inputs.

    Mirrors the seven-source INSERT + ``distinct_all()`` pattern in
    ``MarketScan Data Cleaning.R`` lines 159–210.

    Args:
        *source_tables: One or more source DataFrames.  Each must contain at
            least ``enrolid``, ``dobyr``, and ``sex``.

    Returns:
        Canonical patinfo DataFrame.  Always contains ``patient_id``,
        ``dobyr``, ``sex``.  Extra MarketScan columns are included when
        present.
    """
    _REQUIRED = [_ENROLID, "dobyr", "sex"]
    available_extra = [
        c for c in _PATINFO_EXTRA_COLS
        if all(c in t.columns for t in source_tables)
    ]
    keep_cols = _REQUIRED + available_extra

    parts = [t[[c for c in keep_cols if c in t.columns]].copy() for t in source_tables]
    df = union_tables(parts)
    df = df.rename(columns={_ENROLID: _PID})
    df[_PID] = coerce_patient_id(df[_PID])
    df["dobyr"] = pd.to_numeric(df["dobyr"], errors="coerce").astype("Int64")
    df = deduplicate(df)
    return df


# ---------------------------------------------------------------------------
# Insurance
# ---------------------------------------------------------------------------


def build_insurance(
    facility_header: pd.DataFrame,
    inpatient_services: pd.DataFrame,
    outpatient_prescription_drugs: pd.DataFrame,
    outpatient_services: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical insurance table from four MarketScan sources.

    Mirrors ``MarketScan Data Cleaning.R`` lines 221–250.

    Args:
        facility_header: Must contain ``enrolid``, ``svcdate``, ``cob``,
            ``coins``, ``copay``.
        inpatient_services: Same columns.
        outpatient_prescription_drugs: Same columns.
        outpatient_services: Same columns.

    Returns:
        Canonical insurance DataFrame conforming to
        :data:`~ehrdata.io.source.schema.INSURANCE`.
    """
    _INS_COLS = [_ENROLID, "svcdate", "cob", "coins", "copay"]
    parts = []
    for src in (facility_header, inpatient_services, outpatient_prescription_drugs, outpatient_services):
        present = [c for c in _INS_COLS if c in src.columns]
        parts.append(src[present].copy())

    df = union_tables(parts)
    df = df.rename(columns={_ENROLID: _PID})
    df[_PID] = coerce_patient_id(df[_PID])
    df["svcdate"] = coerce_date(df["svcdate"])
    for col in ("cob", "coins", "copay"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = deduplicate(df)
    return df


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


def build_provider(enrollment_detail: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical provider table from enrollment detail.

    Mirrors ``MarketScan Data Cleaning.R`` lines 256–265.

    Args:
        enrollment_detail: ``commercial.enrollment_detail`` view data.
            Must contain ``enrolid``.  Optional columns: ``dtstart``,
            ``dtend``, ``plantyp``, ``rx``, ``hlthplan``.

    Returns:
        Canonical provider DataFrame conforming to
        :data:`~ehrdata.io.source.schema.PROVIDER`.
    """
    _PROV_COLS = [_ENROLID, "dtstart", "dtend", "plantyp", "rx", "hlthplan"]
    present = [c for c in _PROV_COLS if c in enrollment_detail.columns]
    df = enrollment_detail[present].copy()
    df = df.rename(columns={_ENROLID: _PID})
    df[_PID] = coerce_patient_id(df[_PID])
    for col in ("dtstart", "dtend"):
        if col in df.columns:
            df[col] = coerce_date(df[col])
    df = deduplicate(df)
    return df
