"""Tests for the IBM MarketScan adapter.

All source DataFrames are constructed inline to mirror the minimum columns
present in the actual MarketScan PostgreSQL views.  Only a handful of rows
are needed per table to exercise the ETL logic.
"""

from pathlib import Path

import pandas as pd
import pytest

from ehrdata.io.source.adapters.marketscan import (
    build_diagnosis,
    build_insurance,
    build_patinfo,
    build_procedure,
    build_provider,
    build_therapy,
)
from ehrdata.io.source.schema import DIAGNOSIS, INSURANCE, PATINFO, PROCEDURE, PROVIDER, THERAPY
from ehrdata.io.source.vocab.ndc import load_ndc_ingredient_map

VOCAB_DIR = Path("tests/data/source_vocab")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ndc_map():
    return load_ndc_ingredient_map(VOCAB_DIR / "ndc_ingredient_map.txt")


@pytest.fixture
def facility_header():
    return pd.DataFrame(
        {
            "enrolid": [1001, 1002, 1001],
            "dxver": ["0", "9", "0"],
            "svcdate": ["2020-01-15", "2020-02-01", "2020-03-10"],
            "dx1": ["E11.9", "I10", "E11.9"],
            "dx2": ["I10", None, None],
            "dx3": [None, None, None],
            "proc1": ["99213", None, "99213"],
            "proc2": [None, "93000", None],
            "proc3": [None, None, None],
            "cob": [0.0, 10.0, 0.0],
            "coins": [20.0, 0.0, 20.0],
            "copay": [30.0, 15.0, 30.0],
            "dobyr": [1960, 1970, 1960],
            "sex": ["M", "F", "M"],
            "efamid": [5001, 5002, 5001],
            "year": [2020, 2020, 2020],
            "region": ["NE", "SE", "NE"],
            "msa": [10, 20, 10],
            "wgtkey": [1, 2, 1],
            "eeclass": ["A", "B", "A"],
            "eestatu": ["1", "2", "1"],
            "egeoloc": ["11", "22", "11"],
            "emprel": ["1", "2", "1"],
            "indstry": ["001", "002", "001"],
        }
    )


@pytest.fixture
def inpatient_admissions():
    return pd.DataFrame(
        {
            "enrolid": [1001, 1003],
            "dxver": ["0", None],
            "admdate": ["2020-04-01", "2020-05-15"],
            "pdx": ["J18.9", "E10.9"],
            "dx1": ["I10", None],
            "dx2": [None, None],
            "pproc": ["27447", None],
            "proc1": [None, "99232"],
            "dobyr": [1960, 1980],
            "sex": ["M", "M"],
            "efamid": [5001, 5003],
            "year": [2020, 2020],
            "region": ["NE", "MW"],
            "msa": [10, 30],
            "wgtkey": [1, 3],
            "eeclass": ["A", "C"],
            "eestatu": ["1", "3"],
            "egeoloc": ["11", "33"],
            "emprel": ["1", "3"],
            "indstry": ["001", "003"],
        }
    )


@pytest.fixture
def inpatient_services():
    return pd.DataFrame(
        {
            "enrolid": [1001, 1002],
            "dxver": ["0", "0"],
            "svcdate": ["2020-04-02", "2020-04-10"],
            "pdx": ["J18.9", "K92.1"],
            "dx1": ["I10", None],
            "dx2": [None, None],
            "proctyp": ["ICD", "CPT"],
            "proc1": ["99232", "44950"],
            "cob": [5.0, 0.0],
            "coins": [10.0, 25.0],
            "copay": [20.0, 40.0],
            "dobyr": [1960, 1970],
            "sex": ["M", "F"],
            "efamid": [5001, 5002],
            "year": [2020, 2020],
            "region": ["NE", "SE"],
            "msa": [10, 20],
            "wgtkey": [1, 2],
            "eeclass": ["A", "B"],
            "eestatu": ["1", "2"],
            "egeoloc": ["11", "22"],
            "emprel": ["1", "2"],
            "indstry": ["001", "002"],
        }
    )


@pytest.fixture
def outpatient_services():
    return pd.DataFrame(
        {
            "enrolid": [1002, 1003],
            "dxver": [None, "0"],
            "svcdate": ["2020-06-01", "2020-07-01"],
            "dx1": ["E10.9", "I10"],
            "dx2": [None, None],
            "proctyp": ["CPT", "CPT"],
            "proc1": ["99213", "99214"],
            "cob": [0.0, 0.0],
            "coins": [15.0, 20.0],
            "copay": [25.0, 35.0],
            "dobyr": [1970, 1980],
            "sex": ["F", "M"],
            "efamid": [5002, 5003],
            "year": [2020, 2020],
            "region": ["SE", "MW"],
            "msa": [20, 30],
            "wgtkey": [2, 3],
            "eeclass": ["B", "C"],
            "eestatu": ["2", "3"],
            "egeoloc": ["22", "33"],
            "emprel": ["2", "3"],
            "indstry": ["002", "003"],
        }
    )


@pytest.fixture
def outpatient_prescription_drugs():
    return pd.DataFrame(
        {
            "enrolid": [1001, 1002, 1001],
            "svcdate": ["2020-01-20", "2020-02-05", "2020-01-20"],
            "daysupp": [30, 90, 30],
            "refill": [0, 1, 0],
            "ndcnum": ["00071015523", "00310751030", "00071015523"],  # metformin, insulin glargine
            "cob": [0.0, 5.0, 0.0],
            "coins": [10.0, 0.0, 10.0],
            "copay": [5.0, 10.0, 5.0],
            "dobyr": [1960, 1970, 1960],
            "sex": ["M", "F", "M"],
            "efamid": [5001, 5002, 5001],
            "year": [2020, 2020, 2020],
            "region": ["NE", "SE", "NE"],
            "msa": [10, 20, 10],
            "wgtkey": [1, 2, 1],
            "eeclass": ["A", "B", "A"],
            "eestatu": ["1", "2", "1"],
            "egeoloc": ["11", "22", "11"],
            "emprel": ["1", "2", "1"],
            "indstry": ["001", "002", "001"],
        }
    )


@pytest.fixture
def enrollment_annual_summary():
    return pd.DataFrame(
        {
            "enrolid": [1001, 1002, 1003],
            "dobyr": [1960, 1970, 1980],
            "sex": ["M", "F", "M"],
            "efamid": [5001, 5002, 5003],
            "year": [2020, 2020, 2020],
            "region": ["NE", "SE", "MW"],
            "msa": [10, 20, 30],
            "wgtkey": [1, 2, 3],
            "eeclass": ["A", "B", "C"],
            "eestatu": ["1", "2", "3"],
            "egeoloc": ["11", "22", "33"],
            "emprel": ["1", "2", "3"],
            "indstry": ["001", "002", "003"],
        }
    )


@pytest.fixture
def enrollment_detail():
    return pd.DataFrame(
        {
            "enrolid": [1001, 1001, 1002],
            "dtstart": ["2020-01-01", "2021-01-01", "2020-01-01"],
            "dtend": ["2020-12-31", "2021-12-31", "2020-12-31"],
            "plantyp": [10, 10, 20],
            "rx": ["Y", "Y", "N"],
            "hlthplan": ["BlueCross", "BlueCross", "Aetna"],
            "dobyr": [1960, 1960, 1970],
            "sex": ["M", "M", "F"],
            "efamid": [5001, 5001, 5002],
            "year": [2020, 2021, 2020],
            "region": ["NE", "NE", "SE"],
            "msa": [10, 10, 20],
            "wgtkey": [1, 1, 2],
            "eeclass": ["A", "A", "B"],
            "eestatu": ["1", "1", "2"],
            "egeoloc": ["11", "11", "22"],
            "emprel": ["1", "1", "2"],
            "indstry": ["001", "001", "002"],
        }
    )


# ---------------------------------------------------------------------------
# build_diagnosis
# ---------------------------------------------------------------------------


class TestBuildDiagnosis:
    def test_schema_valid(self, facility_header, inpatient_admissions, inpatient_services, outpatient_services):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        assert DIAGNOSIS.validate(result) == []

    def test_patient_id_is_string(self, facility_header, inpatient_admissions, inpatient_services, outpatient_services):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        assert result["patient_id"].dtype == object

    def test_eventdate_is_datetime(
        self, facility_header, inpatient_admissions, inpatient_services, outpatient_services
    ):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        assert pd.api.types.is_datetime64_any_dtype(result["eventdate"])

    def test_no_null_dx_codes(self, facility_header, inpatient_admissions, inpatient_services, outpatient_services):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        assert result["dx"].notna().all()

    def test_enrolid_renamed_to_patient_id(
        self, facility_header, inpatient_admissions, inpatient_services, outpatient_services
    ):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        assert "patient_id" in result.columns
        assert "enrolid" not in result.columns

    def test_no_duplicate_rows(self, facility_header, inpatient_admissions, inpatient_services, outpatient_services):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        assert not result.duplicated().any()

    def test_sorted_by_patient_then_date(
        self, facility_header, inpatient_admissions, inpatient_services, outpatient_services
    ):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        patients = list(result["patient_id"])
        assert patients == sorted(patients)

    def test_icd_version_inferred_for_missing(
        self, facility_header, inpatient_admissions, inpatient_services, outpatient_services
    ):
        # outpatient_services has dxver=None for enrolid=1002 with dx=E10.9
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        e10_rows = result[(result["patient_id"] == "1002") & (result["dx"] == "E10.9")]
        # E prefix + null dxver → inferred as "9"
        assert (e10_rows["dxver"] == "9").all()

    def test_wide_dx_unnested(self, facility_header, inpatient_admissions, inpatient_services, outpatient_services):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        # facility_header row 0 has dx1=E11.9 and dx2=I10 → both should appear
        patient_1001_dx = set(result[result["patient_id"] == "1001"]["dx"].tolist())
        assert "E11.9" in patient_1001_dx
        assert "I10" in patient_1001_dx

    def test_all_four_sources_contribute(
        self, facility_header, inpatient_admissions, inpatient_services, outpatient_services
    ):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        # Patient 1003 only appears in inpatient_admissions and outpatient_services
        assert "1003" in result["patient_id"].values


# ---------------------------------------------------------------------------
# build_therapy
# ---------------------------------------------------------------------------


class TestBuildTherapy:
    def test_schema_valid(self, outpatient_prescription_drugs, ndc_map):
        result = build_therapy(outpatient_prescription_drugs, ndc_map=ndc_map)
        assert THERAPY.validate(result) == []

    def test_fill_date_is_svcdate(self, outpatient_prescription_drugs):
        result = build_therapy(outpatient_prescription_drugs)
        assert pd.api.types.is_datetime64_any_dtype(result["fill_date"])
        assert result["fill_date"].iloc[0] == pd.Timestamp("2020-01-20")

    def test_end_date_calculated(self, outpatient_prescription_drugs):
        result = build_therapy(outpatient_prescription_drugs)
        # fill_date 2020-01-20 + 30 days = 2020-02-19
        row = result[result["patient_id"] == "1001"].iloc[0]
        assert row["end_date"] == pd.Timestamp("2020-02-19")

    def test_prescription_date_is_nat(self, outpatient_prescription_drugs):
        result = build_therapy(outpatient_prescription_drugs)
        assert result["prescription_date"].isna().all()

    def test_start_date_is_nat(self, outpatient_prescription_drugs):
        result = build_therapy(outpatient_prescription_drugs)
        assert result["start_date"].isna().all()

    def test_rxcui_is_none(self, outpatient_prescription_drugs):
        result = build_therapy(outpatient_prescription_drugs)
        assert result["rxcui"].isna().all()

    def test_ndc11_zero_padded(self, outpatient_prescription_drugs):
        result = build_therapy(outpatient_prescription_drugs)
        assert result["ndc11"].str.len().eq(11).all()

    def test_ingredient_joined_from_ndc_map(self, outpatient_prescription_drugs, ndc_map):
        result = build_therapy(outpatient_prescription_drugs, ndc_map=ndc_map)
        metformin_rows = result[result["ndc11"] == "00071015523"]
        assert (metformin_rows["ingredient"] == "metformin").all()

    def test_ingredient_none_without_ndc_map(self, outpatient_prescription_drugs):
        result = build_therapy(outpatient_prescription_drugs)
        assert result["ingredient"].isna().all()

    def test_duplicate_removed(self, outpatient_prescription_drugs, ndc_map):
        # Source has enrolid=1001 twice with same svcdate/ndcnum/daysupp/refill
        result = build_therapy(outpatient_prescription_drugs, ndc_map=ndc_map)
        assert len(result[result["patient_id"] == "1001"]) == 1

    def test_enrolid_renamed_to_patient_id(self, outpatient_prescription_drugs):
        result = build_therapy(outpatient_prescription_drugs)
        assert "patient_id" in result.columns
        assert "enrolid" not in result.columns


# ---------------------------------------------------------------------------
# build_procedure
# ---------------------------------------------------------------------------


class TestBuildProcedure:
    def test_schema_valid(self, facility_header, inpatient_admissions, inpatient_services, outpatient_services):
        result = build_procedure(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        assert PROCEDURE.validate(result) == []

    def test_no_null_proc_codes(self, facility_header, inpatient_admissions, inpatient_services, outpatient_services):
        result = build_procedure(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        assert result["proc"].notna().all()

    def test_enrolid_renamed(self, facility_header, inpatient_admissions, inpatient_services, outpatient_services):
        result = build_procedure(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        assert "patient_id" in result.columns
        assert "enrolid" not in result.columns

    def test_eventdate_is_datetime(
        self, facility_header, inpatient_admissions, inpatient_services, outpatient_services
    ):
        result = build_procedure(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        assert pd.api.types.is_datetime64_any_dtype(result["eventdate"])

    def test_proctype_null_for_facility_header(
        self, facility_header, inpatient_admissions, inpatient_services, outpatient_services
    ):
        result = build_procedure(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        # facility_header contributions should have null proctype
        fh_date = pd.Timestamp("2020-01-15")
        fh_rows = result[(result["patient_id"] == "1001") & (result["eventdate"] == fh_date)]
        assert fh_rows["proctype"].isna().all()

    def test_proctype_set_for_outpatient_services(
        self, facility_header, inpatient_admissions, inpatient_services, outpatient_services
    ):
        result = build_procedure(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        op_rows = result[result["patient_id"] == "1002"]
        cpt_rows = op_rows[op_rows["proctype"] == "CPT"]
        assert len(cpt_rows) > 0

    def test_no_duplicate_rows(self, facility_header, inpatient_admissions, inpatient_services, outpatient_services):
        result = build_procedure(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        assert not result.duplicated().any()

    def test_sorted_by_patient_then_date(
        self, facility_header, inpatient_admissions, inpatient_services, outpatient_services
    ):
        result = build_procedure(facility_header, inpatient_admissions, inpatient_services, outpatient_services)
        patients = list(result["patient_id"])
        assert patients == sorted(patients)


# ---------------------------------------------------------------------------
# build_patinfo
# ---------------------------------------------------------------------------


class TestBuildPatinfo:
    def test_schema_valid(self, enrollment_annual_summary, enrollment_detail):
        result = build_patinfo(enrollment_annual_summary, enrollment_detail)
        assert PATINFO.validate(result) == []

    def test_enrolid_renamed(self, enrollment_annual_summary):
        result = build_patinfo(enrollment_annual_summary)
        assert "patient_id" in result.columns
        assert "enrolid" not in result.columns

    def test_deduplicates_across_sources(self, enrollment_annual_summary, facility_header):
        # enrollment_annual_summary and facility_header both contain enrolid=1001
        # with identical (dobyr, sex, efamid, year, ...) → full-row dedup collapses them
        result = build_patinfo(enrollment_annual_summary, facility_header)
        assert not result.duplicated().any()

    def test_extra_marketscan_columns_present(self, enrollment_annual_summary):
        result = build_patinfo(enrollment_annual_summary)
        assert "region" in result.columns
        assert "year" in result.columns

    def test_dobyr_nullable_int(self, enrollment_annual_summary):
        result = build_patinfo(enrollment_annual_summary)
        assert result["dobyr"].dtype.name == "Int64"

    def test_single_source(self, enrollment_annual_summary):
        result = build_patinfo(enrollment_annual_summary)
        assert len(result) == 3  # 3 unique patients in fixture

    def test_all_patients_present(self, enrollment_annual_summary, facility_header):
        result = build_patinfo(enrollment_annual_summary, facility_header)
        pids = set(result["patient_id"].tolist())
        assert "1001" in pids
        assert "1002" in pids


# ---------------------------------------------------------------------------
# build_insurance
# ---------------------------------------------------------------------------


class TestBuildInsurance:
    def test_schema_valid(
        self, facility_header, inpatient_services, outpatient_prescription_drugs, outpatient_services
    ):
        result = build_insurance(
            facility_header, inpatient_services, outpatient_prescription_drugs, outpatient_services
        )
        assert INSURANCE.validate(result) == []

    def test_enrolid_renamed(
        self, facility_header, inpatient_services, outpatient_prescription_drugs, outpatient_services
    ):
        result = build_insurance(
            facility_header, inpatient_services, outpatient_prescription_drugs, outpatient_services
        )
        assert "patient_id" in result.columns
        assert "enrolid" not in result.columns

    def test_svcdate_is_datetime(
        self, facility_header, inpatient_services, outpatient_prescription_drugs, outpatient_services
    ):
        result = build_insurance(
            facility_header, inpatient_services, outpatient_prescription_drugs, outpatient_services
        )
        assert pd.api.types.is_datetime64_any_dtype(result["svcdate"])

    def test_cob_coins_copay_numeric(
        self, facility_header, inpatient_services, outpatient_prescription_drugs, outpatient_services
    ):
        result = build_insurance(
            facility_header, inpatient_services, outpatient_prescription_drugs, outpatient_services
        )
        for col in ("cob", "coins", "copay"):
            assert pd.api.types.is_float_dtype(result[col])

    def test_no_duplicates(
        self, facility_header, inpatient_services, outpatient_prescription_drugs, outpatient_services
    ):
        result = build_insurance(
            facility_header, inpatient_services, outpatient_prescription_drugs, outpatient_services
        )
        assert not result.duplicated().any()


# ---------------------------------------------------------------------------
# build_provider
# ---------------------------------------------------------------------------


class TestBuildProvider:
    def test_schema_valid(self, enrollment_detail):
        result = build_provider(enrollment_detail)
        assert PROVIDER.validate(result) == []

    def test_enrolid_renamed(self, enrollment_detail):
        result = build_provider(enrollment_detail)
        assert "patient_id" in result.columns
        assert "enrolid" not in result.columns

    def test_date_columns_parsed(self, enrollment_detail):
        result = build_provider(enrollment_detail)
        assert pd.api.types.is_datetime64_any_dtype(result["dtstart"])
        assert pd.api.types.is_datetime64_any_dtype(result["dtend"])

    def test_correct_row_count(self, enrollment_detail):
        # 3 rows in fixture, all distinct
        result = build_provider(enrollment_detail)
        assert len(result) == 3

    def test_hlthplan_preserved(self, enrollment_detail):
        result = build_provider(enrollment_detail)
        assert "BlueCross" in result["hlthplan"].values

    def test_no_duplicates(self, enrollment_detail):
        result = build_provider(enrollment_detail)
        assert not result.duplicated().any()

    def test_minimal_input_no_optional_cols(self):
        # Adapter must not crash if optional columns are absent
        df = pd.DataFrame({"enrolid": [1001, 1002]})
        result = build_provider(df)
        assert "patient_id" in result.columns
        assert len(result) == 2
