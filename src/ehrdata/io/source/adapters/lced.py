"""IBM LCED (Limited Claims-EMR Data) adapter.

Translates raw IBM LCED source tables (supplied as :class:`~pandas.DataFrame`
objects) into canonical DataFrames conforming to the schemas in
:mod:`ehrdata.io.source.schema`.

LCED differs from MarketScan in three important ways:

1. The patient identifier is already named ``patient_id`` in every source view
   (no ``enrolid`` rename needed).
2. Therapy comes from **two** independent sources that must be joined separately
   before unioning — ``v_drug`` (RxCUI-based, prescription records) and
   ``v_outpatient_drug_claims`` (NDC-based, pharmacy claims).
3. Lab tests are present in two sources — ``v_observation`` (structured EMR
   observations) and ``v_lab_results`` (claims-based lab results).
4. An extra ``habit`` table captures lifestyle/survey data (smoking, BMI, etc.)
   obtained by joining ``v_habit`` with ``v_encounter``.

Source view reference (original LCED PostgreSQL schema ``lced.*``):

- ``v_facility_header``: facility encounters
- ``v_inpatient_admissions``: inpatient admissions
- ``v_inpatient_services``: inpatient service lines
- ``v_lab_results``: lab result claims
- ``v_outpatient_services``: outpatient service lines
- ``v_drug``: EMR prescription records (RxCUI)
- ``v_outpatient_drug_claims``: pharmacy claims (NDC)
- ``v_observation``: structured EMR lab observations (LOINC)
- ``v_habit``: patient lifestyle/survey responses
- ``v_encounter``: encounter dates for habit join
- ``v_annual_summary_enrollment``: annual enrollment snapshots
- ``v_detail_enrollment``: enrollment coverage detail with plan info
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

_PID = "patient_id"


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


def build_diagnosis(
    facility_header: pd.DataFrame,
    inpatient_admissions: pd.DataFrame,
    inpatient_services: pd.DataFrame,
    lab_results: pd.DataFrame,
    outpatient_services: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical diagnosis table from five LCED sources.

    Mirrors ``LCED Data Cleaning.R`` lines 30–54.  Identical in structure to
    the MarketScan adapter except that ``patient_id`` is already the correct
    column name and a fifth source (``v_lab_results``) contributes ``dx1``.

    Args:
        facility_header: Must contain ``patient_id``, ``dxver``, ``svcdate``,
            ``dx1``–``dx9``.
        inpatient_admissions: Must contain ``patient_id``, ``dxver``,
            ``admdate``, ``pdx``, ``dx1``–``dx15``.
        inpatient_services: Must contain ``patient_id``, ``dxver``,
            ``svcdate``, ``pdx``, ``dx1``–``dx4``.
        lab_results: Must contain ``patient_id``, ``dxver``, ``svcdate``,
            ``dx1``.
        outpatient_services: Must contain ``patient_id``, ``dxver``,
            ``svcdate``, ``dx1``–``dx4``.

    Returns:
        Canonical diagnosis DataFrame conforming to
        :data:`~ehrdata.io.source.schema.DIAGNOSIS`.
    """
    parts = [
        _unnest_dx(facility_header,   date_col="svcdate",
                   dx_cols=[f"dx{i}" for i in range(1, 10)]),
        _unnest_dx(inpatient_admissions, date_col="admdate",
                   dx_cols=["pdx"] + [f"dx{i}" for i in range(1, 16)]),
        _unnest_dx(inpatient_services, date_col="svcdate",
                   dx_cols=["pdx"] + [f"dx{i}" for i in range(1, 5)]),
        _unnest_dx(lab_results,       date_col="svcdate", dx_cols=["dx1"]),
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
    """Select and unnest diagnosis codes from one LCED source table."""
    present_dx = [c for c in dx_cols if c in src.columns]
    dxver_cols = ["dxver"] if "dxver" in src.columns else []
    tmp = src[[_PID, date_col] + dxver_cols + present_dx].copy()
    tmp = tmp.rename(columns={date_col: "eventdate"})
    if "dxver" not in tmp.columns:
        tmp["dxver"] = None
    return unnest_codes(tmp, id_cols=[_PID, "eventdate", "dxver"], code_cols=present_dx, value_name="dx")


# ---------------------------------------------------------------------------
# Therapy
# ---------------------------------------------------------------------------


def build_therapy(
    v_drug: pd.DataFrame,
    v_outpatient_drug_claims: pd.DataFrame,
    *,
    ndc_map: pd.DataFrame | None = None,
    rxcui_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the canonical therapy table from two LCED drug sources.

    Mirrors ``LCED Data Cleaning.R`` lines 68–88.

    ``v_drug`` holds EMR prescriptions keyed by RxCUI; ``v_outpatient_drug_claims``
    holds pharmacy claims keyed by NDC.  Each source is ingredient-joined with
    its respective vocabulary before the two are unioned.

    Args:
        v_drug: ``lced.v_drug`` view data.  Must contain ``patient_id``,
            ``prescription_date``, ``start_date``, ``end_date``, ``rx_cui``.
        v_outpatient_drug_claims: ``lced.v_outpatient_drug_claims`` view data.
            Must contain ``patient_id``, ``svcdate``, ``daysupp``, ``refill``,
            ``ndcnum``.
        ndc_map: Optional NDC→ingredient map (from
            :func:`~ehrdata.io.source.vocab.ndc.load_ndc_ingredient_map`).
        rxcui_map: Optional RxCUI→ingredient map (from
            :func:`~ehrdata.io.source.vocab.rxnorm.load_rxcui_ingredient_map`).

    Returns:
        Canonical therapy DataFrame conforming to
        :data:`~ehrdata.io.source.schema.THERAPY`.
    """
    drug_part = _build_drug_part(v_drug, rxcui_map=rxcui_map)
    claims_part = _build_claims_part(v_outpatient_drug_claims, ndc_map=ndc_map)
    df = union_tables([drug_part, claims_part])
    df = deduplicate(df)
    return df[["patient_id", "prescription_date", "start_date", "fill_date",
               "end_date", "refill", "rxcui", "ndc11", "ingredient"]]


def _build_drug_part(src: pd.DataFrame, *, rxcui_map: pd.DataFrame | None) -> pd.DataFrame:
    """Build the RxCUI-based (v_drug) contribution to therapy."""
    df = pd.DataFrame()
    df[_PID] = coerce_patient_id(src[_PID])
    df["prescription_date"] = coerce_date(src.get("prescription_date"))
    df["start_date"] = coerce_date(src.get("start_date"))
    df["fill_date"] = pd.NaT
    df["end_date"] = coerce_date(src.get("end_date"))
    df["refill"] = pd.array([pd.NA] * len(src), dtype="Int64")
    df["rxcui"] = src["rx_cui"].astype(str).str.strip() if "rx_cui" in src.columns else None
    df["ndc11"] = None
    if rxcui_map is not None:
        from ehrdata.io.source.vocab.rxnorm import join_ingredient_by_rxcui
        df = join_ingredient_by_rxcui(df, rxcui_map)
    else:
        df["ingredient"] = None
    return df


def _build_claims_part(src: pd.DataFrame, *, ndc_map: pd.DataFrame | None) -> pd.DataFrame:
    """Build the NDC-based (v_outpatient_drug_claims) contribution to therapy."""
    df = pd.DataFrame()
    df[_PID] = coerce_patient_id(src[_PID])
    df["prescription_date"] = pd.NaT
    df["start_date"] = pd.NaT
    df["fill_date"] = coerce_date(src["svcdate"])
    daysupp = pd.to_numeric(src["daysupp"], errors="coerce")
    df["end_date"] = df["fill_date"] + pd.to_timedelta(daysupp, unit="D")
    df["refill"] = pd.to_numeric(src["refill"], errors="coerce").astype("Int64")
    df["rxcui"] = None
    df["ndc11"] = src["ndcnum"].astype(str).str.strip().str.zfill(11)
    if ndc_map is not None:
        from ehrdata.io.source.vocab.ndc import join_ingredient_by_ndc
        df = join_ingredient_by_ndc(df, ndc_map)
    else:
        df["ingredient"] = None
    return df


# ---------------------------------------------------------------------------
# Lab tests
# ---------------------------------------------------------------------------


def build_labtest(
    v_observation: pd.DataFrame,
    v_lab_results: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical lab-test table from two LCED sources.

    Mirrors ``LCED Data Cleaning.R`` lines 93–98.

    ``v_observation`` provides structured EMR observations (``std_value`` /
    ``std_uom`` / ``loinc_test_id``); ``v_lab_results`` provides claims-based
    results (``result`` / ``resltcat`` / ``resunit`` / ``loinccd``).

    Args:
        v_observation: ``lced.v_observation`` view data.  Must contain
            ``patient_id`` and ``observation_date``.  Optional:
            ``std_value``, ``std_uom``, ``loinc_test_id``.
        v_lab_results: ``lced.v_lab_results`` view data.  Must contain
            ``patient_id`` and ``svcdate``.  Optional: ``result``,
            ``resltcat``, ``resunit``, ``loinccd``.

    Returns:
        Canonical labtest DataFrame conforming to
        :data:`~ehrdata.io.source.schema.LABTEST`.
    """
    obs_part = _build_observation_part(v_observation)
    lab_part = _build_lab_results_part(v_lab_results)
    df = union_tables([obs_part, lab_part])
    df[_PID] = coerce_patient_id(df[_PID])
    df["eventdate"] = coerce_date(df["eventdate"])
    df = deduplicate(df)
    df = sort_events(df)
    return df


def _build_observation_part(src: pd.DataFrame) -> pd.DataFrame:
    """Map v_observation columns to canonical labtest schema."""
    df = pd.DataFrame()
    df[_PID] = src[_PID]
    df["eventdate"] = src.get("observation_date")
    df["value"] = src.get("std_value", pd.Series(dtype=object, index=src.index))
    df["valuecat"] = None
    df["unit"] = src.get("std_uom", pd.Series(dtype=object, index=src.index))
    df["loinc"] = src.get("loinc_test_id", pd.Series(dtype=object, index=src.index))
    return df


def _build_lab_results_part(src: pd.DataFrame) -> pd.DataFrame:
    """Map v_lab_results columns to canonical labtest schema."""
    df = pd.DataFrame()
    df[_PID] = src[_PID]
    df["eventdate"] = src.get("svcdate")
    df["value"] = src.get("result", pd.Series(dtype=object, index=src.index)).astype(str).where(
        src.get("result", pd.Series(dtype=object, index=src.index)).notna(), None
    )
    df["valuecat"] = src.get("resltcat", pd.Series(dtype=object, index=src.index))
    df["unit"] = src.get("resunit", pd.Series(dtype=object, index=src.index))
    df["loinc"] = src.get("loinccd", pd.Series(dtype=object, index=src.index))
    return df


# ---------------------------------------------------------------------------
# Procedure
# ---------------------------------------------------------------------------


def build_procedure(
    facility_header: pd.DataFrame,
    inpatient_admissions: pd.DataFrame,
    inpatient_services: pd.DataFrame,
    lab_results: pd.DataFrame,
    outpatient_services: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical procedure table from five LCED sources.

    Mirrors ``LCED Data Cleaning.R`` lines 110–133.  Identical in structure to
    MarketScan except ``patient_id`` is already correct and ``v_lab_results``
    contributes an additional procedure source.

    Note: ``inpatient_services`` unnests ``array[pdx, proc1]``, replicating
    the original ETL which included the primary diagnosis alongside the
    procedure code.

    Args:
        facility_header: Must contain ``patient_id``, ``svcdate``,
            ``proc1``–``proc6``.
        inpatient_admissions: Must contain ``patient_id``, ``admdate``,
            ``pproc``, ``proc1``–``proc15``.
        inpatient_services: Must contain ``patient_id``, ``svcdate``,
            ``proctyp``, ``pdx``, ``proc1``.
        lab_results: Must contain ``patient_id``, ``svcdate``, ``proctyp``,
            ``proc1``.
        outpatient_services: Must contain ``patient_id``, ``svcdate``,
            ``proctyp``, ``proc1``.

    Returns:
        Canonical procedure DataFrame conforming to
        :data:`~ehrdata.io.source.schema.PROCEDURE`.
    """
    parts = [
        _unnest_proc(facility_header,    date_col="svcdate",
                     proc_cols=[f"proc{i}" for i in range(1, 7)], proctype_col=None),
        _unnest_proc(inpatient_admissions, date_col="admdate",
                     proc_cols=["pproc"] + [f"proc{i}" for i in range(1, 16)], proctype_col=None),
        _unnest_proc(inpatient_services,  date_col="svcdate",
                     proc_cols=["pdx", "proc1"], proctype_col="proctyp"),
        _unnest_proc(lab_results,         date_col="svcdate",
                     proc_cols=["proc1"],        proctype_col="proctyp"),
        _unnest_proc(outpatient_services, date_col="svcdate",
                     proc_cols=["proc1"],        proctype_col="proctyp"),
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
    """Select and unnest procedure codes from one LCED source table."""
    present_proc = [c for c in proc_cols if c in src.columns]
    tmp = src[[_PID, date_col] + present_proc].copy()
    tmp = tmp.rename(columns={date_col: "eventdate"})
    if proctype_col and proctype_col in src.columns:
        tmp["proctype"] = src[proctype_col].values
    else:
        tmp["proctype"] = None
    return unnest_codes(tmp, id_cols=[_PID, "eventdate", "proctype"], code_cols=present_proc, value_name="proc")


# ---------------------------------------------------------------------------
# Habit
# ---------------------------------------------------------------------------


def build_habit(
    v_habit: pd.DataFrame,
    v_encounter: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical habit table from LCED survey/lifestyle data.

    Mirrors ``LCED Data Cleaning.R`` lines 148–155.

    Joins ``v_habit`` (patient lifestyle responses) with ``v_encounter``
    (encounter dates) on ``encounter_join_id``, then drops rows where
    ``mapped_question_answer`` is null and removes the join key column.

    Args:
        v_habit: ``lced.v_habit`` view data.  Must contain ``patient_id``,
            ``mapped_question_answer``, and ``encounter_join_id``.
        v_encounter: ``lced.v_encounter`` view data.  Must contain
            ``encounter_date`` and ``encounter_join_id``.

    Returns:
        Canonical habit DataFrame conforming to
        :data:`~ehrdata.io.source.schema.HABIT`.
    """
    df = v_habit.merge(
        v_encounter[["encounter_join_id", "encounter_date"]],
        on="encounter_join_id",
        how="left",
    )
    df = df[df["mapped_question_answer"].notna()].copy()
    df[_PID] = coerce_patient_id(df[_PID])
    df["encounter_date"] = coerce_date(df["encounter_date"])
    df = df.drop(columns="encounter_join_id")
    df = deduplicate(df)
    df = df.sort_values([_PID, "encounter_date", "mapped_question_answer"],
                        na_position="last").reset_index(drop=True)
    return df[[_PID, "encounter_date", "mapped_question_answer"]]


# ---------------------------------------------------------------------------
# Patinfo
# ---------------------------------------------------------------------------


def build_patinfo(*source_tables: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical patient-info table by unioning LCED sources.

    LCED patinfo only extracts the three canonical columns
    (``patient_id``, ``dobyr``, ``sex``) — unlike MarketScan, no extra
    regional/plan columns are available at this level.

    Mirrors the seven-source union + ``distinct_all()`` in
    ``LCED Data Cleaning.R`` lines 166–182.

    Args:
        *source_tables: One or more DataFrames each containing at least
            ``patient_id``, ``dobyr``, and ``sex``.

    Returns:
        Canonical patinfo DataFrame with columns ``patient_id``, ``dobyr``,
        ``sex``.
    """
    parts = [t[[c for c in ["patient_id", "dobyr", "sex"] if c in t.columns]].copy()
             for t in source_tables]
    df = union_tables(parts)
    df[_PID] = coerce_patient_id(df[_PID])
    df["dobyr"] = pd.to_numeric(df["dobyr"], errors="coerce").astype("Int64")
    return deduplicate(df)


# ---------------------------------------------------------------------------
# Insurance
# ---------------------------------------------------------------------------


def build_insurance(
    facility_header: pd.DataFrame,
    inpatient_services: pd.DataFrame,
    outpatient_drug_claims: pd.DataFrame,
    outpatient_services: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical insurance table from four LCED sources.

    Mirrors ``LCED Data Cleaning.R`` lines 193–202.

    Args:
        facility_header: Must contain ``patient_id``, ``svcdate``, ``cob``,
            ``coins``, ``copay``.
        inpatient_services: Same columns.
        outpatient_drug_claims: Same columns.
        outpatient_services: Same columns.

    Returns:
        Canonical insurance DataFrame conforming to
        :data:`~ehrdata.io.source.schema.INSURANCE`.
    """
    _COLS = ["patient_id", "svcdate", "cob", "coins", "copay"]
    parts = [src[[c for c in _COLS if c in src.columns]].copy()
             for src in (facility_header, inpatient_services, outpatient_drug_claims, outpatient_services)]
    df = union_tables(parts)
    df[_PID] = coerce_patient_id(df[_PID])
    df["svcdate"] = coerce_date(df["svcdate"])
    for col in ("cob", "coins", "copay"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return deduplicate(df)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


def build_provider(v_detail_enrollment: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical provider table from LCED enrollment detail.

    Mirrors ``LCED Data Cleaning.R`` lines 214–221.

    Args:
        v_detail_enrollment: ``lced.v_detail_enrollment`` view data.  Must
            contain ``patient_id``.  Optional: ``dtstart``, ``dtend``,
            ``plantyp``, ``rx``, ``hlthplan``.

    Returns:
        Canonical provider DataFrame conforming to
        :data:`~ehrdata.io.source.schema.PROVIDER`.
    """
    _COLS = ["patient_id", "dtstart", "dtend", "plantyp", "rx", "hlthplan"]
    present = [c for c in _COLS if c in v_detail_enrollment.columns]
    df = v_detail_enrollment[present].copy()
    df[_PID] = coerce_patient_id(df[_PID])
    for col in ("dtstart", "dtend"):
        if col in df.columns:
            df[col] = coerce_date(df[col])
    return deduplicate(df)
