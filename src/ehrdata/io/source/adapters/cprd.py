"""CPRD (Clinical Practice Research Datalink) adapter.

Translates raw CPRD source tables (supplied as :class:`~pandas.DataFrame`
objects) into canonical DataFrames conforming to the schemas in
:mod:`ehrdata.io.source.schema`.

CPRD differs from MarketScan and LCED in several important ways:

1. Diagnoses are coded with **Read codes** (a UK-specific hierarchy), not ICD.
   The ``dxver`` field is therefore always ``None`` for CPRD outputs.
2. Drugs are identified by **prodcode** (CPRD product dictionary) rather than
   NDC or RxCUI.  Drug substance names come from ``product.txt`` / ``product.csv``.
3. Lab tests are stored across two complementary file types:

   - **Clinical** + **Additional** files joined on ``(patid, adid, enttype)`` —
     the Additional file holds the actual numeric test values.
   - **Test** files that include both ``eventdate`` and data columns directly.

4. CPRD dates are formatted ``DD/MM/YYYY``, not ``YYYY-MM-DD``.
5. There is no insurance or provider table in the original CPRD ETL.

Source file reference (original CPRD GOLD extract layout):

- ``Clinical``: patid, eventdate, medcode, enttype, adid, constype, consid
- ``Referral``: patid, eventdate, medcode, constype, consid, ...
- ``Test``: patid, eventdate, medcode, enttype, data1-data7, constype, consid
- ``Additional``: patid, enttype, adid, data1-data7
- ``Therapy``: patid, eventdate, prodcode, bnfcode, qty, issueseq
- ``Patient``: patid, dobyr, sex, pracid
- ``Practice``: pracid, region, lcd, uts
"""

from __future__ import annotations

import pandas as pd

from ehrdata.io.source.extract import union_tables
from ehrdata.io.source.normalize import (
    coerce_date,
    coerce_patient_id,
    deduplicate,
    sort_events,
)

_PID = "patient_id"
_CPRD_DATE_FMT = "%d/%m/%Y"


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


def build_diagnosis(
    clinical: pd.DataFrame,
    referral: pd.DataFrame,
    test_data: pd.DataFrame,
    *,
    medical_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build canonical DIAGNOSIS from CPRD clinical, referral, and test files.

    Each source contributes ``(patid, eventdate, medcode)``.  When
    *medical_map* is provided the medcode is translated to its Read code and
    stored in ``dx``; otherwise the raw medcode string is used.

    CPRD does not use ICD coding so ``dxver`` is always ``None``.

    Args:
        clinical: Clinical file rows (must include patid, eventdate, medcode).
        referral: Referral file rows (must include patid, eventdate, medcode).
        test_data: Test file rows (must include patid, eventdate, medcode).
        medical_map: Optional medcode→readcode DataFrame from
            :func:`~ehrdata.io.source.vocab.readcode.load_medical_map`.

    Returns:
        Canonical DIAGNOSIS DataFrame sorted by patient then eventdate.
    """
    parts = []
    for src in (clinical, referral, test_data):
        p = src[["patid", "eventdate", "medcode"]].copy()
        p["patid"] = coerce_patient_id(p["patid"])
        p["eventdate"] = coerce_date(p["eventdate"], formats=[_CPRD_DATE_FMT])
        if medical_map is not None:
            map_ = medical_map[["medcode", "readcode"]].drop_duplicates(subset=["medcode"])
            p["medcode"] = p["medcode"].astype(str)
            p = p.merge(map_, on="medcode", how="left")
            p["dx"] = p["readcode"].astype(object)
            p = p.drop(columns=["readcode", "medcode"])
        else:
            p["dx"] = p["medcode"].astype(str)
            p = p.drop(columns=["medcode"])
        p = p.rename(columns={"patid": _PID})
        p = p.dropna(subset=["dx"])
        parts.append(p[[_PID, "eventdate", "dx"]])

    out = union_tables(parts)
    out["dxver"] = pd.array([None] * len(out), dtype=object)
    out = out[[_PID, "dxver", "eventdate", "dx"]]
    return deduplicate(sort_events(out, patient_col=_PID, date_col="eventdate"))


# ---------------------------------------------------------------------------
# Therapy
# ---------------------------------------------------------------------------


def build_therapy(
    therapy_data: pd.DataFrame,
    *,
    product_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build canonical THERAPY from CPRD therapy files.

    ``eventdate`` is mapped to ``fill_date`` (the prescription/dispensing date).
    ``prescription_date``, ``start_date``, and ``end_date`` are not available in
    CPRD and are set to ``NaT``.  ``ndc11`` and ``rxcui`` are ``None``.

    When *product_map* is provided, prodcode is joined to obtain the drug
    substance name stored in ``ingredient``.

    Args:
        therapy_data: Therapy file rows (patid, eventdate, prodcode, ...).
        product_map: Optional prodcode→drugsubstance DataFrame from
            :func:`~ehrdata.io.source.vocab.prodcode.load_product_map`.

    Returns:
        Canonical THERAPY DataFrame.
    """
    out = therapy_data[["patid", "eventdate", "prodcode"]].copy()
    out["patid"] = coerce_patient_id(out["patid"])
    out["fill_date"] = coerce_date(out["eventdate"], formats=[_CPRD_DATE_FMT])
    out = out.drop(columns=["eventdate"])

    if product_map is not None:
        map_ = product_map[["prodcode", "drugsubstance"]].drop_duplicates(subset=["prodcode"])
        out = out.merge(map_, on="prodcode", how="left")
        out["ingredient"] = out["drugsubstance"].astype(object)
        out = out.drop(columns=["drugsubstance"])
    else:
        out["ingredient"] = pd.array([None] * len(out), dtype=object)

    out = out.drop(columns=["prodcode"])
    out = out.rename(columns={"patid": _PID})

    for col in ("prescription_date", "start_date", "end_date"):
        out[col] = pd.NaT
    out["refill"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["rxcui"] = pd.array([None] * len(out), dtype=object)
    out["ndc11"] = pd.array([None] * len(out), dtype=object)

    out = out[
        [_PID, "prescription_date", "start_date", "fill_date", "end_date", "refill", "rxcui", "ndc11", "ingredient"]
    ]
    return deduplicate(out)


# ---------------------------------------------------------------------------
# Lab test
# ---------------------------------------------------------------------------


def build_labtest(
    clinical: pd.DataFrame,
    additional: pd.DataFrame,
    test_data: pd.DataFrame,
    *,
    entity_enttypes: set[int] | None = None,
) -> pd.DataFrame:
    """Build canonical LABTEST from CPRD clinical+additional and test files.

    Two complementary sources are combined:

    1. **Clinical** (patid, eventdate, enttype, adid) inner-joined with
       **Additional** (patid, enttype, adid, data2, data3, data4) on the
       composite key ``(patid, enttype, adid)`` — replicates the
       ``merge(..., by=c("patid","adid","enttype"))`` in the original R ETL.
    2. **Test** files (patid, eventdate, enttype, data2, data3, data4) that
       already include both date and data columns.

    Column mapping: ``data2`` → ``value``, ``data3`` → ``unit``,
    ``data4`` → ``valuecat``.  CPRD does not provide LOINC codes so ``loinc``
    is always ``None``.

    Args:
        clinical: Clinical file rows (patid, eventdate, enttype, adid).
        additional: Additional file rows (patid, enttype, adid, data2, data3, data4).
        test_data: Test file rows (patid, eventdate, enttype, data2, data3, data4).
        entity_enttypes: Optional set of ``enttype`` integer codes to retain;
            all entity types are included when ``None``.

    Returns:
        Canonical LABTEST DataFrame sorted by patient then eventdate.
    """
    # Part 1: clinical + additional inner-join to recover eventdate for the data rows
    clin_cols = [c for c in ("patid", "eventdate", "enttype", "adid") if c in clinical.columns]
    addi_cols = [c for c in ("patid", "enttype", "adid", "data2", "data3", "data4") if c in additional.columns]
    clin_addi = (
        clinical[clin_cols]
        .merge(additional[addi_cols], on=["patid", "enttype", "adid"], how="inner")
        .drop(columns=["adid"])
    )

    # Part 2: test file already has all needed columns
    test_cols = [c for c in ("patid", "eventdate", "enttype", "data2", "data3", "data4") if c in test_data.columns]
    test_part = test_data[test_cols].copy()

    out = pd.concat([clin_addi, test_part], ignore_index=True)

    if entity_enttypes is not None:
        out = out[out["enttype"].isin(entity_enttypes)]

    out = out.rename(
        columns={
            "patid": _PID,
            "data2": "value",
            "data3": "unit",
            "data4": "valuecat",
        }
    )
    out[_PID] = coerce_patient_id(out[_PID])
    out["eventdate"] = coerce_date(out["eventdate"], formats=[_CPRD_DATE_FMT])
    out["loinc"] = pd.array([None] * len(out), dtype=object)

    out = out[[_PID, "eventdate", "value", "valuecat", "unit", "loinc"]]
    return deduplicate(sort_events(out, patient_col=_PID, date_col="eventdate"))


# ---------------------------------------------------------------------------
# Patient information
# ---------------------------------------------------------------------------


def build_patinfo(*patient_tables: pd.DataFrame) -> pd.DataFrame:
    """Build canonical PATINFO from one or more CPRD patient extract files.

    Each table must contain at least ``patid``, ``dobyr``, and ``sex``.
    Rows are unioned across all tables and duplicates removed.

    Args:
        *patient_tables: One or more patient DataFrames.

    Returns:
        Canonical PATINFO DataFrame (patient_id, dobyr, sex).
    """
    parts = []
    for pt in patient_tables:
        dob_col = "dobyr" if "dobyr" in pt.columns else "yob"
        p = pt[["patid", dob_col, "sex"]].copy()
        p = p.rename(columns={"patid": _PID, dob_col: "dobyr"})
        p[_PID] = coerce_patient_id(p[_PID])
        parts.append(p)

    out = union_tables(parts)
    out["dobyr"] = pd.array(pd.to_numeric(out["dobyr"], errors="coerce"), dtype="Int64")
    out["sex"] = out["sex"].astype(object)
    return deduplicate(out[[_PID, "dobyr", "sex"]])
