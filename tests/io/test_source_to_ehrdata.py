"""Tests for src/ehrdata/io/source/to_ehrdata.py.

Verifies that the source-layer canonical DataFrames are correctly assembled
into EHRData objects: obs, var, X shape and contents, uns provenance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ehrdata.io.source.to_ehrdata import to_ehrdata


# ---------------------------------------------------------------------------
# Minimal fixture helpers
# ---------------------------------------------------------------------------


def _patinfo(*patient_ids: str) -> pd.DataFrame:
    return pd.DataFrame({
        "patient_id": list(patient_ids),
        "dobyr": [1960 + i for i in range(len(patient_ids))],
        "sex": ["F" if i % 2 == 0 else "M" for i in range(len(patient_ids))],
    })


def _diagnosis(rows: list[tuple[str, str]]) -> pd.DataFrame:
    pids, dxs = zip(*rows) if rows else ([], [])
    return pd.DataFrame({
        "patient_id": list(pids),
        "dxver": [None] * len(pids),
        "eventdate": pd.NaT,
        "dx": list(dxs),
    })


def _therapy(rows: list[tuple[str, str | None]]) -> pd.DataFrame:
    pids, ings = zip(*rows) if rows else ([], [])
    return pd.DataFrame({
        "patient_id": list(pids),
        "prescription_date": pd.NaT,
        "start_date": pd.NaT,
        "fill_date": pd.NaT,
        "end_date": pd.NaT,
        "refill": pd.array([pd.NA] * len(pids), dtype="Int64"),
        "rxcui": None,
        "ndc11": None,
        "ingredient": list(ings),
    })


def _labtest(rows: list[tuple[str, str | None]]) -> pd.DataFrame:
    pids, loincs = zip(*rows) if rows else ([], [])
    return pd.DataFrame({
        "patient_id": list(pids),
        "eventdate": pd.NaT,
        "value": None,
        "valuecat": None,
        "unit": None,
        "loinc": list(loincs),
    })


def _procedure(rows: list[tuple[str, str]]) -> pd.DataFrame:
    pids, procs = zip(*rows) if rows else ([], [])
    return pd.DataFrame({
        "patient_id": list(pids),
        "proctype": None,
        "eventdate": pd.NaT,
        "proc": list(procs),
    })


# ---------------------------------------------------------------------------
# TestObsPopulation
# ---------------------------------------------------------------------------


class TestObsPopulation:
    def test_obs_index_is_patient_id(self):
        edata = to_ehrdata(_patinfo("P1", "P2", "P3"))
        assert set(edata.obs_names) == {"P1", "P2", "P3"}

    def test_obs_index_dtype_is_string(self):
        edata = to_ehrdata(_patinfo("P1", "P2"))
        assert edata.obs.index.dtype == object

    def test_obs_contains_dobyr_and_sex(self):
        edata = to_ehrdata(_patinfo("P1"))
        assert "dobyr" in edata.obs.columns
        assert "sex" in edata.obs.columns

    def test_obs_row_count(self):
        edata = to_ehrdata(_patinfo("P1", "P2", "P3"))
        assert edata.n_obs == 3

    def test_duplicate_patients_in_patinfo_deduplicated(self):
        patinfo = pd.concat([_patinfo("P1"), _patinfo("P1")], ignore_index=True)
        edata = to_ehrdata(patinfo)
        assert edata.n_obs == 1

    def test_patients_not_in_patinfo_excluded_from_X(self):
        patinfo = _patinfo("P1")
        dx = _diagnosis([("P1", "E11.9"), ("GHOST", "I10")])
        edata = to_ehrdata(patinfo, diagnosis=dx)
        assert "P1" in edata.obs_names
        assert "GHOST" not in edata.obs_names


# ---------------------------------------------------------------------------
# TestVarPopulation
# ---------------------------------------------------------------------------


class TestVarPopulation:
    def test_empty_tables_gives_empty_var(self):
        edata = to_ehrdata(_patinfo("P1"))
        assert edata.n_vars == 0

    def test_diagnosis_concepts_prefixed(self):
        dx = _diagnosis([("P1", "E11.9"), ("P1", "I10")])
        edata = to_ehrdata(_patinfo("P1"), diagnosis=dx)
        assert "diagnosis:E11.9" in edata.var_names
        assert "diagnosis:I10" in edata.var_names

    def test_therapy_concepts_prefixed(self):
        th = _therapy([("P1", "metformin")])
        edata = to_ehrdata(_patinfo("P1"), therapy=th)
        assert "therapy:metformin" in edata.var_names

    def test_labtest_concepts_prefixed(self):
        lb = _labtest([("P1", "14749-6")])
        edata = to_ehrdata(_patinfo("P1"), labtest=lb)
        assert "labtest:14749-6" in edata.var_names

    def test_procedure_concepts_prefixed(self):
        pr = _procedure([("P1", "99213")])
        edata = to_ehrdata(_patinfo("P1"), procedure=pr)
        assert "procedure:99213" in edata.var_names

    def test_var_has_concept_source_column(self):
        dx = _diagnosis([("P1", "E11.9")])
        edata = to_ehrdata(_patinfo("P1"), diagnosis=dx)
        assert "concept_source" in edata.var.columns

    def test_var_has_concept_code_column(self):
        dx = _diagnosis([("P1", "E11.9")])
        edata = to_ehrdata(_patinfo("P1"), diagnosis=dx)
        assert "concept_code" in edata.var.columns

    def test_concept_source_value(self):
        dx = _diagnosis([("P1", "E11.9")])
        edata = to_ehrdata(_patinfo("P1"), diagnosis=dx)
        row = edata.var.loc["diagnosis:E11.9"]
        assert row["concept_source"] == "diagnosis"
        assert row["concept_code"] == "E11.9"

    def test_null_loinc_excluded_from_var(self):
        lb = _labtest([("P1", None), ("P1", "14749-6")])
        edata = to_ehrdata(_patinfo("P1"), labtest=lb)
        assert edata.n_vars == 1
        assert "labtest:14749-6" in edata.var_names

    def test_null_ingredient_excluded_from_var(self):
        th = _therapy([("P1", None), ("P1", "aspirin")])
        edata = to_ehrdata(_patinfo("P1"), therapy=th)
        assert edata.n_vars == 1
        assert "therapy:aspirin" in edata.var_names

    def test_concepts_from_multiple_tables_unioned(self):
        dx = _diagnosis([("P1", "E11.9")])
        th = _therapy([("P1", "metformin")])
        edata = to_ehrdata(_patinfo("P1"), diagnosis=dx, therapy=th)
        assert "diagnosis:E11.9" in edata.var_names
        assert "therapy:metformin" in edata.var_names

    def test_var_index_is_sorted(self):
        dx = _diagnosis([("P1", "Z99"), ("P1", "A01")])
        edata = to_ehrdata(_patinfo("P1"), diagnosis=dx)
        assert list(edata.var_names) == sorted(edata.var_names.tolist())


# ---------------------------------------------------------------------------
# TestXMatrix
# ---------------------------------------------------------------------------


class TestXMatrix:
    def test_x_shape(self):
        dx = _diagnosis([("P1", "E11.9"), ("P2", "I10")])
        edata = to_ehrdata(_patinfo("P1", "P2"), diagnosis=dx)
        assert edata.X.shape == (2, 2)

    def test_x_dtype_is_float64(self):
        dx = _diagnosis([("P1", "E11.9")])
        edata = to_ehrdata(_patinfo("P1"), diagnosis=dx)
        assert edata.X.dtype == np.float64

    def test_x_binary_values_only(self):
        dx = _diagnosis([("P1", "E11.9"), ("P1", "I10"), ("P2", "E11.9")])
        edata = to_ehrdata(_patinfo("P1", "P2"), diagnosis=dx)
        unique_vals = np.unique(edata.X)
        assert set(unique_vals.tolist()).issubset({0.0, 1.0})

    def test_x_is_one_when_patient_has_concept(self):
        dx = _diagnosis([("P1", "E11.9")])
        edata = to_ehrdata(_patinfo("P1", "P2"), diagnosis=dx)
        p1_idx = edata.obs_names.tolist().index("P1")
        c_idx = edata.var_names.tolist().index("diagnosis:E11.9")
        assert edata.X[p1_idx, c_idx] == 1.0

    def test_x_is_zero_when_patient_lacks_concept(self):
        dx = _diagnosis([("P1", "E11.9")])
        edata = to_ehrdata(_patinfo("P1", "P2"), diagnosis=dx)
        p2_idx = edata.obs_names.tolist().index("P2")
        c_idx = edata.var_names.tolist().index("diagnosis:E11.9")
        assert edata.X[p2_idx, c_idx] == 0.0

    def test_duplicate_events_do_not_inflate_x(self):
        # P1 has E11.9 three times — X should still be 1.0
        dx = _diagnosis([("P1", "E11.9"), ("P1", "E11.9"), ("P1", "E11.9")])
        edata = to_ehrdata(_patinfo("P1"), diagnosis=dx)
        assert edata.X[0, 0] == 1.0

    def test_no_event_tables_gives_zero_column_x(self):
        edata = to_ehrdata(_patinfo("P1", "P2"))
        assert edata.X.shape == (2, 0)

    def test_x_empty_when_no_patients(self):
        patinfo = pd.DataFrame({"patient_id": [], "dobyr": [], "sex": []})
        dx = _diagnosis([("P1", "E11.9")])
        edata = to_ehrdata(patinfo, diagnosis=dx)
        assert edata.n_obs == 0

    def test_x_rows_align_with_obs(self):
        dx = _diagnosis([("P2", "E11.9")])
        edata = to_ehrdata(_patinfo("P1", "P2"), diagnosis=dx)
        p1_idx = edata.obs_names.tolist().index("P1")
        p2_idx = edata.obs_names.tolist().index("P2")
        c_idx = edata.var_names.tolist().index("diagnosis:E11.9")
        assert edata.X[p1_idx, c_idx] == 0.0
        assert edata.X[p2_idx, c_idx] == 1.0


# ---------------------------------------------------------------------------
# TestUns
# ---------------------------------------------------------------------------


class TestUns:
    def test_source_stored_in_uns(self):
        edata = to_ehrdata(_patinfo("P1"), source="marketscan")
        assert edata.uns["source_io_source"] == "marketscan"

    def test_source_none_when_not_provided(self):
        edata = to_ehrdata(_patinfo("P1"))
        assert edata.uns["source_io_source"] is None

    def test_tables_used_diagnosis(self):
        dx = _diagnosis([("P1", "E11.9")])
        edata = to_ehrdata(_patinfo("P1"), diagnosis=dx)
        assert "diagnosis" in edata.uns["source_io_tables"]

    def test_tables_used_therapy(self):
        th = _therapy([("P1", "metformin")])
        edata = to_ehrdata(_patinfo("P1"), therapy=th)
        assert "therapy" in edata.uns["source_io_tables"]

    def test_tables_used_lists_all_provided(self):
        dx = _diagnosis([("P1", "E11.9")])
        th = _therapy([("P1", "metformin")])
        lb = _labtest([("P1", "14749-6")])
        edata = to_ehrdata(_patinfo("P1"), diagnosis=dx, therapy=th, labtest=lb)
        assert set(edata.uns["source_io_tables"]) == {"diagnosis", "therapy", "labtest"}

    def test_tables_used_empty_when_no_event_tables(self):
        edata = to_ehrdata(_patinfo("P1"))
        assert edata.uns["source_io_tables"] == []


# ---------------------------------------------------------------------------
# TestReturnType
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_ehrdata(self):
        from ehrdata import EHRData

        edata = to_ehrdata(_patinfo("P1"))
        assert isinstance(edata, EHRData)

    def test_full_pipeline_with_all_tables(self):
        from ehrdata import EHRData

        patinfo = _patinfo("P1", "P2", "P3")
        dx = _diagnosis([("P1", "E11.9"), ("P2", "I10"), ("P3", "E11.9")])
        th = _therapy([("P1", "metformin"), ("P2", None)])
        lb = _labtest([("P3", "14749-6"), ("P1", None)])
        pr = _procedure([("P2", "99213")])

        edata = to_ehrdata(patinfo, diagnosis=dx, therapy=th, labtest=lb, procedure=pr, source="test")

        assert isinstance(edata, EHRData)
        assert edata.n_obs == 3
        # concepts: diagnosis:E11.9, diagnosis:I10, therapy:metformin, labtest:14749-6, procedure:99213
        assert edata.n_vars == 5
        assert edata.uns["source_io_source"] == "test"
        assert set(edata.uns["source_io_tables"]) == {"diagnosis", "therapy", "labtest", "procedure"}
