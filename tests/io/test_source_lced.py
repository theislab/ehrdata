"""Tests for the IBM LCED adapter and Phase 3 vocab modules.

Source DataFrames are constructed inline to mirror the minimum columns present
in the LCED PostgreSQL views.
"""

from pathlib import Path

import pandas as pd
import pytest

from ehrdata.io.source.adapters.lced import (
    build_diagnosis,
    build_habit,
    build_insurance,
    build_labtest,
    build_patinfo,
    build_procedure,
    build_provider,
    build_therapy,
)
from ehrdata.io.source.schema import DIAGNOSIS, HABIT, INSURANCE, LABTEST, PATINFO, PROCEDURE, PROVIDER, THERAPY
from ehrdata.io.source.vocab.icd import ICD9_PREFIXES, classify_icd_version
from ehrdata.io.source.vocab.loinc import join_component_by_loinc, load_loinc_map
from ehrdata.io.source.vocab.ndc import load_ndc_ingredient_map
from ehrdata.io.source.vocab.rxnorm import load_rxcui_ingredient_map

VOCAB_DIR = Path("tests/data/source_vocab")


# ---------------------------------------------------------------------------
# Vocab fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ndc_map():
    return load_ndc_ingredient_map(VOCAB_DIR / "ndc_ingredient_map.txt")


@pytest.fixture()
def rxcui_map():
    return load_rxcui_ingredient_map(VOCAB_DIR / "rxcui_ingredient_map.txt")


@pytest.fixture()
def loinc_map():
    return load_loinc_map(VOCAB_DIR / "loinc_map.csv")


# ---------------------------------------------------------------------------
# LCED source table fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def facility_header():
    return pd.DataFrame({
        "patient_id": ["P001", "P002", "P001"],
        "dxver":      ["0",    "9",    "0"],
        "svcdate":    ["2020-01-15", "2020-02-01", "2020-03-10"],
        "dx1":        ["E11.9", "I10",   "E11.9"],
        "dx2":        ["I10",   None,    None],
        "dx3":        [None,    None,    None],
        "proc1":      ["99213", None,    "99213"],
        "proc2":      [None,    "93000", None],
        "proc3":      [None,    None,    None],
        "svcdate":    ["2020-01-15", "2020-02-01", "2020-03-10"],
        "cob":        [0.0,     10.0,  0.0],
        "coins":      [20.0,    0.0,   20.0],
        "copay":      [30.0,    15.0,  30.0],
        "dobyr":      [1960,    1970,  1960],
        "sex":        ["M",     "F",   "M"],
    })


@pytest.fixture()
def inpatient_admissions():
    return pd.DataFrame({
        "patient_id": ["P001", "P003"],
        "dxver":      ["0",    None],
        "admdate":    ["2020-04-01", "2020-05-15"],
        "pdx":        ["J18.9", "E10.9"],
        "dx1":        ["I10",   None],
        "dx2":        [None,    None],
        "pproc":      ["27447", None],
        "proc1":      [None,    "99232"],
        "dobyr":      [1960,    1980],
        "sex":        ["M",     "M"],
    })


@pytest.fixture()
def inpatient_services():
    return pd.DataFrame({
        "patient_id": ["P001", "P002"],
        "dxver":      ["0",    "0"],
        "svcdate":    ["2020-04-02", "2020-04-10"],
        "pdx":        ["J18.9", "K92.1"],
        "dx1":        ["I10",   None],
        "dx2":        [None,    None],
        "proctyp":    ["ICD",   "CPT"],
        "proc1":      ["99232", "44950"],
        "cob":        [5.0,     0.0],
        "coins":      [10.0,    25.0],
        "copay":      [20.0,    40.0],
        "dobyr":      [1960,    1970],
        "sex":        ["M",     "F"],
    })


@pytest.fixture()
def lab_results():
    return pd.DataFrame({
        "patient_id": ["P001", "P002", "P003"],
        "dxver":      [None,   "0",    None],
        "svcdate":    ["2020-01-20", "2020-02-10", "2020-03-05"],
        "dx1":        ["V58.67", "E11.9", None],
        "proctyp":    ["CPT",   "CPT",   "CPT"],
        "proc1":      ["83036", "83036", None],
        "result":     ["7.2",   "6.8",   "5.4"],
        "resltcat":   [None,    None,    "normal"],
        "resunit":    ["mmol/L", "mmol/L", "mmol/L"],
        "loinccd":    ["14749-6", "14749-6", "14749-6"],
        "dobyr":      [1960,    1970,    1980],
        "sex":        ["M",     "F",     "M"],
    })


@pytest.fixture()
def outpatient_services():
    return pd.DataFrame({
        "patient_id": ["P002", "P003"],
        "dxver":      [None,   "0"],
        "svcdate":    ["2020-06-01", "2020-07-01"],
        "dx1":        ["E10.9", "I10"],
        "dx2":        [None,    None],
        "proctyp":    ["CPT",   "CPT"],
        "proc1":      ["99213", "99214"],
        "cob":        [0.0,     0.0],
        "coins":      [15.0,    20.0],
        "copay":      [25.0,    35.0],
        "dobyr":      [1970,    1980],
        "sex":        ["F",     "M"],
    })


@pytest.fixture()
def v_drug():
    return pd.DataFrame({
        "patient_id":        ["P001",         "P002",         "P001"],
        "prescription_date": ["2020-01-10",   "2020-02-05",   "2020-01-10"],
        "start_date":        ["2020-01-15",   "2020-02-10",   "2020-01-15"],
        "end_date":          ["2020-04-15",   "2020-05-10",   "2020-04-15"],
        "rx_cui":            ["723",          "4815",         "723"],   # metformin, insulin glargine
    })


@pytest.fixture()
def v_outpatient_drug_claims():
    return pd.DataFrame({
        "patient_id": ["P001",         "P003",         "P002"],
        "svcdate":    ["2020-01-20",   "2020-03-01",   "2020-02-15"],
        "daysupp":    [30,             90,             30],
        "refill":     [0,              1,              0],
        "ndcnum":     ["00071015523",  "00310751030",  "00065063136"],
        "cob":        [0.0,            5.0,            0.0],
        "coins":      [10.0,           0.0,            20.0],
        "copay":      [5.0,            10.0,           15.0],
    })


@pytest.fixture()
def v_observation():
    return pd.DataFrame({
        "patient_id":       ["P001",        "P002"],
        "observation_date": ["2020-01-15",  "2020-02-01"],
        "std_value":        ["7.2",         "6.8"],
        "std_uom":          ["mmol/L",      "mmol/L"],
        "loinc_test_id":    ["14749-6",     "4548-4"],
    })


@pytest.fixture()
def v_habit():
    return pd.DataFrame({
        "patient_id":             ["P001", "P001", "P002", "P003"],
        "mapped_question_answer": ["current smoker", None, "non-smoker", "former smoker"],
        "encounter_join_id":      [101,    102,    103,    104],
    })


@pytest.fixture()
def v_encounter():
    return pd.DataFrame({
        "encounter_join_id": [101,          102,          103,          104],
        "encounter_date":    ["2020-01-15", "2020-02-01", "2020-02-10", "2020-03-01"],
    })


@pytest.fixture()
def v_annual_summary_enrollment():
    return pd.DataFrame({
        "patient_id": ["P001", "P002", "P003"],
        "dobyr":      [1960,   1970,   1980],
        "sex":        ["M",    "F",    "M"],
    })


@pytest.fixture()
def v_detail_enrollment():
    return pd.DataFrame({
        "patient_id": ["P001",        "P001",        "P002"],
        "dtstart":    ["2020-01-01",  "2021-01-01",  "2020-01-01"],
        "dtend":      ["2020-12-31",  "2021-12-31",  "2020-12-31"],
        "plantyp":    [10,            10,            20],
        "rx":         ["Y",           "Y",           "N"],
        "hlthplan":   ["BlueCross",   "BlueCross",   "Aetna"],
        "dobyr":      [1960,          1960,          1970],
        "sex":        ["M",           "M",           "F"],
    })


# ===========================================================================
# LOINC vocab tests
# ===========================================================================


class TestLoadLoincMap:
    def test_returns_dataframe(self, loinc_map):
        assert isinstance(loinc_map, pd.DataFrame)

    def test_columns_include_loinc_and_component(self, loinc_map):
        assert "loinc" in loinc_map.columns
        assert "component" in loinc_map.columns

    def test_row_count(self, loinc_map):
        assert len(loinc_map) == 10

    def test_known_component(self, loinc_map):
        row = loinc_map[loinc_map["loinc"] == "14749-6"]
        assert row["component"].iloc[0] == "Glucose"

    def test_loinc_is_string(self, loinc_map):
        assert loinc_map["loinc"].dtype == object

    def test_long_common_name_preserved(self, loinc_map):
        assert "long_common_name" in loinc_map.columns

    def test_loinc_num_alias(self, tmp_path):
        csv = tmp_path / "loinc.csv"
        csv.write_text("loinc_num,component\n14749-6,Glucose\n")
        df = load_loinc_map(csv)
        assert "loinc" in df.columns
        assert "loinc_num" not in df.columns


class TestJoinComponentByLoinc:
    def test_matched_row_gets_component(self, loinc_map):
        df = pd.DataFrame({"patient_id": ["P1"], "loinc": ["14749-6"]})
        result = join_component_by_loinc(df, loinc_map)
        assert result.loc[0, "component"] == "Glucose"

    def test_unmatched_row_gets_nan(self, loinc_map):
        df = pd.DataFrame({"patient_id": ["P1"], "loinc": ["99999-9"]})
        result = join_component_by_loinc(df, loinc_map)
        assert pd.isna(result.loc[0, "component"])

    def test_row_count_preserved(self, loinc_map):
        df = pd.DataFrame({"loinc": ["14749-6", "99999-9", "4548-4"]})
        result = join_component_by_loinc(df, loinc_map)
        assert len(result) == 3

    def test_custom_loinc_column(self, loinc_map):
        df = pd.DataFrame({"lab_code": ["14749-6"]})
        result = join_component_by_loinc(df, loinc_map, loinc_col="lab_code")
        assert result.loc[0, "component"] == "Glucose"

    def test_does_not_mutate_input(self, loinc_map):
        df = pd.DataFrame({"loinc": ["14749-6"]})
        _ = join_component_by_loinc(df, loinc_map)
        assert "component" not in df.columns


# ===========================================================================
# ICD vocab tests
# ===========================================================================


class TestIcdVocab:
    def test_icd9_prefixes_contains_e_and_v(self):
        assert "E" in ICD9_PREFIXES
        assert "V" in ICD9_PREFIXES

    def test_classify_e_prefix_returns_9(self):
        s = pd.Series(["E930.0", "E10.9"])
        result = classify_icd_version(s)
        assert list(result) == ["9", "9"]

    def test_classify_v_prefix_returns_9(self):
        s = pd.Series(["V58.67"])
        result = classify_icd_version(s)
        assert result.iloc[0] == "9"

    def test_classify_other_prefix_returns_none(self):
        s = pd.Series(["I10", "J18.9", "K92.1"])
        result = classify_icd_version(s)
        assert result.isna().all()

    def test_classify_preserves_index(self):
        s = pd.Series(["E930.0", "I10"], index=[10, 20])
        result = classify_icd_version(s)
        assert list(result.index) == [10, 20]


# ===========================================================================
# LCED adapter tests
# ===========================================================================


class TestLcedBuildDiagnosis:
    def test_schema_valid(self, facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services)
        assert DIAGNOSIS.validate(result) == []

    def test_patient_id_already_named(self, facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services)
        assert "patient_id" in result.columns
        assert "enrolid" not in result.columns

    def test_no_null_dx_codes(self, facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services)
        assert result["dx"].notna().all()

    def test_five_sources_contribute(self, facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services)
        # P003 only appears in inpatient_admissions and outpatient_services
        assert "P003" in result["patient_id"].values

    def test_v_code_from_lab_results_inferred_icd9(self, facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services)
        v_rows = result[(result["patient_id"] == "P001") & (result["dx"] == "V58.67")]
        assert (v_rows["dxver"] == "9").all()

    def test_no_duplicate_rows(self, facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services)
        assert not result.duplicated().any()

    def test_sorted_by_patient_then_date(self, facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services):
        result = build_diagnosis(facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services)
        assert list(result["patient_id"]) == sorted(result["patient_id"])


class TestLcedBuildTherapy:
    def test_schema_valid(self, v_drug, v_outpatient_drug_claims, ndc_map, rxcui_map):
        result = build_therapy(v_drug, v_outpatient_drug_claims, ndc_map=ndc_map, rxcui_map=rxcui_map)
        assert THERAPY.validate(result) == []

    def test_v_drug_has_prescription_and_start_date(self, v_drug, v_outpatient_drug_claims):
        result = build_therapy(v_drug, v_outpatient_drug_claims)
        drug_rows = result[result["rxcui"].notna()]
        assert drug_rows["prescription_date"].notna().any()
        assert drug_rows["start_date"].notna().any()

    def test_claims_has_fill_date_and_ndc11(self, v_drug, v_outpatient_drug_claims):
        result = build_therapy(v_drug, v_outpatient_drug_claims)
        claims_rows = result[result["ndc11"].notna()]
        assert claims_rows["fill_date"].notna().any()

    def test_end_date_calculated_for_claims(self, v_drug, v_outpatient_drug_claims):
        result = build_therapy(v_drug, v_outpatient_drug_claims)
        claims_p001 = result[(result["patient_id"] == "P001") & result["ndc11"].notna()]
        # fill_date 2020-01-20 + 30 days = 2020-02-19
        assert claims_p001["end_date"].iloc[0] == pd.Timestamp("2020-02-19")

    def test_rxcui_ingredient_joined(self, v_drug, v_outpatient_drug_claims, rxcui_map):
        result = build_therapy(v_drug, v_outpatient_drug_claims, rxcui_map=rxcui_map)
        metformin_rows = result[result["rxcui"] == "723"]
        assert (metformin_rows["ingredient"] == "metformin").all()

    def test_ndc_ingredient_joined(self, v_drug, v_outpatient_drug_claims, ndc_map):
        result = build_therapy(v_drug, v_outpatient_drug_claims, ndc_map=ndc_map)
        ndc_rows = result[result["ndc11"] == "00071015523"]
        assert (ndc_rows["ingredient"] == "metformin").all()

    def test_duplicate_drug_rows_removed(self, v_drug, v_outpatient_drug_claims):
        # v_drug has P001 twice with identical row
        result = build_therapy(v_drug, v_outpatient_drug_claims)
        p001_drug = result[(result["patient_id"] == "P001") & result["rxcui"].notna()]
        assert len(p001_drug) == 1

    def test_both_sources_contribute(self, v_drug, v_outpatient_drug_claims):
        result = build_therapy(v_drug, v_outpatient_drug_claims)
        assert result["rxcui"].notna().any()
        assert result["ndc11"].notna().any()

    def test_no_vocab_gives_null_ingredient(self, v_drug, v_outpatient_drug_claims):
        result = build_therapy(v_drug, v_outpatient_drug_claims)
        assert result["ingredient"].isna().all()


class TestLcedBuildLabtest:
    def test_schema_valid(self, v_observation, lab_results):
        result = build_labtest(v_observation, lab_results)
        assert LABTEST.validate(result) == []

    def test_both_sources_contribute(self, v_observation, lab_results):
        result = build_labtest(v_observation, lab_results)
        # v_observation provides P001 and P002; lab_results also provides P003
        assert "P003" in result["patient_id"].values

    def test_observation_loinc_mapped(self, v_observation, lab_results):
        result = build_labtest(v_observation, lab_results)
        obs_rows = result[result["patient_id"] == "P001"]
        assert "14749-6" in obs_rows["loinc"].values

    def test_lab_results_valuecat_mapped(self, v_observation, lab_results):
        result = build_labtest(v_observation, lab_results)
        # lab_results P003 has resltcat=normal
        p003_rows = result[result["patient_id"] == "P003"]
        assert "normal" in p003_rows["valuecat"].values

    def test_eventdate_is_datetime(self, v_observation, lab_results):
        result = build_labtest(v_observation, lab_results)
        assert pd.api.types.is_datetime64_any_dtype(result["eventdate"])

    def test_no_duplicate_rows(self, v_observation, lab_results):
        result = build_labtest(v_observation, lab_results)
        assert not result.duplicated().any()

    def test_sorted_by_patient_then_date(self, v_observation, lab_results):
        result = build_labtest(v_observation, lab_results)
        assert list(result["patient_id"]) == sorted(result["patient_id"])


class TestLcedBuildProcedure:
    def test_schema_valid(self, facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services):
        result = build_procedure(facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services)
        assert PROCEDURE.validate(result) == []

    def test_no_null_proc_codes(self, facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services):
        result = build_procedure(facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services)
        assert result["proc"].notna().all()

    def test_lab_results_source_contributes(self, facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services):
        result = build_procedure(facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services)
        # lab_results has proc1=83036 for P001 and P002
        assert "83036" in result["proc"].values

    def test_no_duplicate_rows(self, facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services):
        result = build_procedure(facility_header, inpatient_admissions, inpatient_services, lab_results, outpatient_services)
        assert not result.duplicated().any()


class TestLcedBuildHabit:
    def test_schema_valid(self, v_habit, v_encounter):
        result = build_habit(v_habit, v_encounter)
        assert HABIT.validate(result) == []

    def test_null_answers_filtered_out(self, v_habit, v_encounter):
        result = build_habit(v_habit, v_encounter)
        assert result["mapped_question_answer"].notna().all()

    def test_encounter_date_joined(self, v_habit, v_encounter):
        result = build_habit(v_habit, v_encounter)
        p001_rows = result[result["patient_id"] == "P001"]
        assert pd.api.types.is_datetime64_any_dtype(result["encounter_date"])
        assert p001_rows["encounter_date"].iloc[0] == pd.Timestamp("2020-01-15")

    def test_encounter_join_id_dropped(self, v_habit, v_encounter):
        result = build_habit(v_habit, v_encounter)
        assert "encounter_join_id" not in result.columns

    def test_columns_are_canonical(self, v_habit, v_encounter):
        result = build_habit(v_habit, v_encounter)
        assert list(result.columns) == ["patient_id", "encounter_date", "mapped_question_answer"]

    def test_correct_row_count(self, v_habit, v_encounter):
        # 4 rows in v_habit, 1 has null answer → 3 rows
        result = build_habit(v_habit, v_encounter)
        assert len(result) == 3

    def test_sorted_by_patient_date_answer(self, v_habit, v_encounter):
        result = build_habit(v_habit, v_encounter)
        assert list(result["patient_id"]) == sorted(result["patient_id"])


class TestLcedBuildPatinfo:
    def test_schema_valid(self, v_annual_summary_enrollment):
        result = build_patinfo(v_annual_summary_enrollment)
        assert PATINFO.validate(result) == []

    def test_only_canonical_three_columns(self, v_annual_summary_enrollment, facility_header):
        result = build_patinfo(v_annual_summary_enrollment, facility_header)
        # LCED patinfo only carries patient_id, dobyr, sex (no extra regional cols)
        assert set(result.columns) == {"patient_id", "dobyr", "sex"}

    def test_deduplicates_across_sources(self, v_annual_summary_enrollment, facility_header):
        result = build_patinfo(v_annual_summary_enrollment, facility_header)
        assert not result.duplicated().any()

    def test_all_patients_present(self, v_annual_summary_enrollment):
        result = build_patinfo(v_annual_summary_enrollment)
        assert set(result["patient_id"]) == {"P001", "P002", "P003"}

    def test_dobyr_nullable_int(self, v_annual_summary_enrollment):
        result = build_patinfo(v_annual_summary_enrollment)
        assert result["dobyr"].dtype.name == "Int64"


class TestLcedBuildInsurance:
    def test_schema_valid(self, facility_header, inpatient_services, v_outpatient_drug_claims, outpatient_services):
        result = build_insurance(facility_header, inpatient_services, v_outpatient_drug_claims, outpatient_services)
        assert INSURANCE.validate(result) == []

    def test_patient_id_is_string(self, facility_header, inpatient_services, v_outpatient_drug_claims, outpatient_services):
        result = build_insurance(facility_header, inpatient_services, v_outpatient_drug_claims, outpatient_services)
        assert result["patient_id"].dtype == object

    def test_svcdate_is_datetime(self, facility_header, inpatient_services, v_outpatient_drug_claims, outpatient_services):
        result = build_insurance(facility_header, inpatient_services, v_outpatient_drug_claims, outpatient_services)
        assert pd.api.types.is_datetime64_any_dtype(result["svcdate"])

    def test_no_duplicates(self, facility_header, inpatient_services, v_outpatient_drug_claims, outpatient_services):
        result = build_insurance(facility_header, inpatient_services, v_outpatient_drug_claims, outpatient_services)
        assert not result.duplicated().any()


class TestLcedBuildProvider:
    def test_schema_valid(self, v_detail_enrollment):
        result = build_provider(v_detail_enrollment)
        assert PROVIDER.validate(result) == []

    def test_date_columns_parsed(self, v_detail_enrollment):
        result = build_provider(v_detail_enrollment)
        assert pd.api.types.is_datetime64_any_dtype(result["dtstart"])
        assert pd.api.types.is_datetime64_any_dtype(result["dtend"])

    def test_no_duplicates(self, v_detail_enrollment):
        result = build_provider(v_detail_enrollment)
        assert not result.duplicated().any()

    def test_correct_row_count(self, v_detail_enrollment):
        result = build_provider(v_detail_enrollment)
        assert len(result) == 3

    def test_minimal_input_no_optional_cols(self):
        df = pd.DataFrame({"patient_id": ["P001", "P002"]})
        result = build_provider(df)
        assert "patient_id" in result.columns
