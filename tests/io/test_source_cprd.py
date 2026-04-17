"""Tests for Phase 4: CPRD adapter and CPRD-specific vocab loaders.

Covers:
- vocab/readcode.py   — load_medical_map, join_readcode_by_medcode
- vocab/prodcode.py   — load_product_map, join_drugsubstance_by_prodcode
- adapters/cprd.py    — build_diagnosis, build_therapy, build_labtest,
                        build_patinfo
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ehrdata.io.source.adapters import cprd as cprd_adapter
from ehrdata.io.source.schema import DIAGNOSIS, LABTEST, PATINFO, THERAPY
from ehrdata.io.source.vocab import prodcode as prodcode_vocab
from ehrdata.io.source.vocab import readcode as readcode_vocab

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_DATA = Path(__file__).parent.parent / "data"
_VOCAB = _DATA / "source_vocab"
_CPRD = _DATA / "source_cprd"

MEDICAL_MAP_PATH = _VOCAB / "medical_map.txt"
PRODUCT_MAP_PATH = _VOCAB / "product_map.txt"

CLINICAL_PATH = _CPRD / "clinical.tsv"
REFERRAL_PATH = _CPRD / "referral.tsv"
TEST_PATH = _CPRD / "test.tsv"
ADDITIONAL_PATH = _CPRD / "additional.tsv"
THERAPY_PATH = _CPRD / "therapy.tsv"
PATIENT_PATH = _CPRD / "patient.tsv"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, **kwargs)


# ---------------------------------------------------------------------------
# TestLoadMedicalMap
# ---------------------------------------------------------------------------


class TestLoadMedicalMap:
    def test_returns_dataframe(self):
        result = readcode_vocab.load_medical_map(MEDICAL_MAP_PATH)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self):
        result = readcode_vocab.load_medical_map(MEDICAL_MAP_PATH)
        assert list(result.columns) == ["medcode", "readcode"]

    def test_row_count(self):
        result = readcode_vocab.load_medical_map(MEDICAL_MAP_PATH)
        assert len(result) == 5

    def test_known_readcode(self):
        result = readcode_vocab.load_medical_map(MEDICAL_MAP_PATH)
        row = result[result["medcode"] == "100"]
        assert row["readcode"].iloc[0] == "A10..00"

    def test_dtype_is_string(self):
        result = readcode_vocab.load_medical_map(MEDICAL_MAP_PATH)
        assert result["medcode"].dtype == object
        assert result["readcode"].dtype == object

    def test_deduplicates_on_medcode(self):
        # feeding a path where the same medcode appears twice
        import io
        data = "medcode\treadcode\tdesc\n100\tA10..00\tFoo\n100\tA10..00\tFoo duplicate\n"
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(data)
            tmp = f.name
        try:
            result = readcode_vocab.load_medical_map(tmp)
            assert result["medcode"].duplicated().sum() == 0
        finally:
            os.unlink(tmp)

    def test_extra_columns_ignored(self):
        result = readcode_vocab.load_medical_map(MEDICAL_MAP_PATH)
        assert "desc" not in result.columns


# ---------------------------------------------------------------------------
# TestJoinReadcodeByMedcode
# ---------------------------------------------------------------------------


class TestJoinReadcodeByMedcode:
    @pytest.fixture
    def medical_map(self):
        return readcode_vocab.load_medical_map(MEDICAL_MAP_PATH)

    def test_matched_row_gets_readcode(self, medical_map):
        df = pd.DataFrame({"medcode": ["100"], "patient_id": ["P1"]})
        result = readcode_vocab.join_readcode_by_medcode(df, medical_map)
        assert result["readcode"].iloc[0] == "A10..00"

    def test_unmatched_row_gets_nan(self, medical_map):
        df = pd.DataFrame({"medcode": ["UNKNOWN"], "patient_id": ["P1"]})
        result = readcode_vocab.join_readcode_by_medcode(df, medical_map)
        assert pd.isna(result["readcode"].iloc[0])

    def test_row_count_preserved(self, medical_map):
        df = pd.DataFrame({"medcode": ["100", "200", "UNKNOWN"]})
        result = readcode_vocab.join_readcode_by_medcode(df, medical_map)
        assert len(result) == 3

    def test_does_not_mutate_input(self, medical_map):
        df = pd.DataFrame({"medcode": ["100"]})
        cols_before = list(df.columns)
        readcode_vocab.join_readcode_by_medcode(df, medical_map)
        assert list(df.columns) == cols_before

    def test_custom_medcode_column(self, medical_map):
        df = pd.DataFrame({"mcode": ["200"]})
        result = readcode_vocab.join_readcode_by_medcode(df, medical_map, medcode_col="mcode")
        assert result["readcode"].iloc[0] == "C10..00"


# ---------------------------------------------------------------------------
# TestLoadProductMap
# ---------------------------------------------------------------------------


class TestLoadProductMap:
    def test_returns_dataframe(self):
        result = prodcode_vocab.load_product_map(PRODUCT_MAP_PATH)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self):
        result = prodcode_vocab.load_product_map(PRODUCT_MAP_PATH)
        assert list(result.columns) == ["prodcode", "drugsubstance"]

    def test_row_count(self):
        result = prodcode_vocab.load_product_map(PRODUCT_MAP_PATH)
        assert len(result) == 3

    def test_known_drugsubstance(self):
        result = prodcode_vocab.load_product_map(PRODUCT_MAP_PATH)
        row = result[result["prodcode"] == "PROD1"]
        assert row["drugsubstance"].iloc[0] == "metformin"

    def test_prefers_updated_column(self):
        import io, tempfile, os
        data = "prodcode\tdrugsubstance\tdrugsubstance.updated\n" "PC1\traw name\tupdated name\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(data)
            tmp = f.name
        try:
            result = prodcode_vocab.load_product_map(tmp)
            assert result["drugsubstance"].iloc[0] == "updated name"
        finally:
            os.unlink(tmp)

    def test_csv_separator(self):
        import tempfile, os
        data = "prodcode,drugsubstance\nPC1,aspirin\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(data)
            tmp = f.name
        try:
            result = prodcode_vocab.load_product_map(tmp)
            assert result["drugsubstance"].iloc[0] == "aspirin"
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# TestJoinDrugsubstanceByProdcode
# ---------------------------------------------------------------------------


class TestJoinDrugsubstanceByProdcode:
    @pytest.fixture
    def product_map(self):
        return prodcode_vocab.load_product_map(PRODUCT_MAP_PATH)

    def test_matched_row_gets_drugsubstance(self, product_map):
        df = pd.DataFrame({"prodcode": ["PROD2"]})
        result = prodcode_vocab.join_drugsubstance_by_prodcode(df, product_map)
        assert result["drugsubstance"].iloc[0] == "insulin glargine"

    def test_unmatched_row_gets_nan(self, product_map):
        df = pd.DataFrame({"prodcode": ["UNKNOWN"]})
        result = prodcode_vocab.join_drugsubstance_by_prodcode(df, product_map)
        assert pd.isna(result["drugsubstance"].iloc[0])

    def test_row_count_preserved(self, product_map):
        df = pd.DataFrame({"prodcode": ["PROD1", "PROD3", "UNKNOWN"]})
        result = prodcode_vocab.join_drugsubstance_by_prodcode(df, product_map)
        assert len(result) == 3

    def test_does_not_mutate_input(self, product_map):
        df = pd.DataFrame({"prodcode": ["PROD1"]})
        cols_before = list(df.columns)
        prodcode_vocab.join_drugsubstance_by_prodcode(df, product_map)
        assert list(df.columns) == cols_before

    def test_custom_prodcode_column(self, product_map):
        df = pd.DataFrame({"pcode": ["PROD3"]})
        result = prodcode_vocab.join_drugsubstance_by_prodcode(df, product_map, prodcode_col="pcode")
        assert result["drugsubstance"].iloc[0] == "atorvastatin"


# ---------------------------------------------------------------------------
# Shared adapter fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clinical():
    return _read_tsv(CLINICAL_PATH)


@pytest.fixture
def referral():
    return _read_tsv(REFERRAL_PATH)


@pytest.fixture
def test_data():
    return _read_tsv(TEST_PATH)


@pytest.fixture
def additional():
    return _read_tsv(ADDITIONAL_PATH)


@pytest.fixture
def therapy_data():
    return _read_tsv(THERAPY_PATH)


@pytest.fixture
def patient():
    return _read_tsv(PATIENT_PATH)


@pytest.fixture
def medical_map():
    return readcode_vocab.load_medical_map(MEDICAL_MAP_PATH)


@pytest.fixture
def product_map():
    return prodcode_vocab.load_product_map(PRODUCT_MAP_PATH)


# ---------------------------------------------------------------------------
# TestCprdBuildDiagnosis
# ---------------------------------------------------------------------------


class TestCprdBuildDiagnosis:
    def test_schema_valid(self, clinical, referral, test_data):
        result = cprd_adapter.build_diagnosis(clinical, referral, test_data)
        errors = DIAGNOSIS.validate(result)
        assert errors == []

    def test_three_sources_contribute(self, clinical, referral, test_data):
        result = cprd_adapter.build_diagnosis(clinical, referral, test_data)
        # clinical: P001/P002/P003, referral: P002/P003, test: P001/P002/P003
        assert set(result["patient_id"]) >= {"P001", "P002", "P003"}

    def test_dxver_is_none_for_all_rows(self, clinical, referral, test_data):
        result = cprd_adapter.build_diagnosis(clinical, referral, test_data)
        assert result["dxver"].isna().all()

    def test_dxver_dtype_is_object(self, clinical, referral, test_data):
        result = cprd_adapter.build_diagnosis(clinical, referral, test_data)
        assert result["dxver"].dtype == object

    def test_medical_map_translates_to_readcode(self, clinical, referral, test_data, medical_map):
        result = cprd_adapter.build_diagnosis(clinical, referral, test_data, medical_map=medical_map)
        # medcode 100 → A10..00; medcode 200 → C10..00
        assert "A10..00" in result["dx"].values
        assert "C10..00" in result["dx"].values

    def test_no_medical_map_uses_raw_medcode(self, clinical, referral, test_data):
        result = cprd_adapter.build_diagnosis(clinical, referral, test_data)
        assert "100" in result["dx"].values

    def test_no_duplicate_rows(self, clinical, referral, test_data):
        result = cprd_adapter.build_diagnosis(clinical, referral, test_data)
        assert not result.duplicated().any()

    def test_sorted_by_patient_then_date(self, clinical, referral, test_data):
        result = cprd_adapter.build_diagnosis(clinical, referral, test_data)
        pids = result["patient_id"].tolist()
        dates = result["eventdate"].tolist()
        for i in range(len(pids) - 1):
            if pids[i] == pids[i + 1]:
                assert dates[i] <= dates[i + 1] or pd.isna(dates[i + 1])

    def test_eventdate_is_datetime(self, clinical, referral, test_data):
        result = cprd_adapter.build_diagnosis(clinical, referral, test_data)
        assert pd.api.types.is_datetime64_any_dtype(result["eventdate"])

    def test_cprd_date_format_parsed(self, clinical, referral, test_data):
        result = cprd_adapter.build_diagnosis(clinical, referral, test_data)
        # 15/03/2019 should parse to 2019-03-15
        p1_dates = result[result["patient_id"] == "P001"]["eventdate"]
        assert (p1_dates == pd.Timestamp("2019-03-15")).any()

    def test_patient_id_is_string(self, clinical, referral, test_data):
        result = cprd_adapter.build_diagnosis(clinical, referral, test_data)
        assert result["patient_id"].dtype == object


# ---------------------------------------------------------------------------
# TestCprdBuildTherapy
# ---------------------------------------------------------------------------


class TestCprdBuildTherapy:
    def test_schema_valid(self, therapy_data):
        result = cprd_adapter.build_therapy(therapy_data)
        errors = THERAPY.validate(result)
        assert errors == []

    def test_fill_date_is_eventdate(self, therapy_data):
        result = cprd_adapter.build_therapy(therapy_data)
        assert pd.api.types.is_datetime64_any_dtype(result["fill_date"])
        # 01/04/2019 → 2019-04-01
        assert (result["fill_date"] == pd.Timestamp("2019-04-01")).any()

    def test_product_map_provides_ingredient(self, therapy_data, product_map):
        result = cprd_adapter.build_therapy(therapy_data, product_map=product_map)
        assert "metformin" in result["ingredient"].values

    def test_no_product_map_gives_null_ingredient(self, therapy_data):
        result = cprd_adapter.build_therapy(therapy_data)
        assert result["ingredient"].isna().all()

    def test_prescription_start_end_dates_are_nat(self, therapy_data):
        result = cprd_adapter.build_therapy(therapy_data)
        for col in ("prescription_date", "start_date", "end_date"):
            assert result[col].isna().all()

    def test_rxcui_and_ndc11_are_null(self, therapy_data):
        result = cprd_adapter.build_therapy(therapy_data)
        assert result["rxcui"].isna().all()
        assert result["ndc11"].isna().all()

    def test_refill_is_nullable_int(self, therapy_data):
        result = cprd_adapter.build_therapy(therapy_data)
        assert result["refill"].dtype == "Int64"

    def test_no_duplicate_rows(self, therapy_data):
        # clinical.tsv has a duplicate therapy row; should be removed
        result = cprd_adapter.build_therapy(therapy_data)
        assert not result.duplicated().any()

    def test_patient_id_is_string(self, therapy_data):
        result = cprd_adapter.build_therapy(therapy_data)
        assert result["patient_id"].dtype == object

    def test_three_unique_drugs_present(self, therapy_data, product_map):
        result = cprd_adapter.build_therapy(therapy_data, product_map=product_map)
        assert set(result["ingredient"].dropna()) >= {"metformin", "insulin glargine", "atorvastatin"}


# ---------------------------------------------------------------------------
# TestCprdBuildLabtest
# ---------------------------------------------------------------------------


class TestCprdBuildLabtest:
    def test_schema_valid(self, clinical, additional, test_data):
        result = cprd_adapter.build_labtest(clinical, additional, test_data)
        errors = LABTEST.validate(result)
        assert errors == []

    def test_both_sources_contribute(self, clinical, additional, test_data):
        result = cprd_adapter.build_labtest(clinical, additional, test_data)
        # clinical+additional contributes P001/P002/P003 via adid join
        # test_data also contributes P001/P002/P003
        assert set(result["patient_id"]) == {"P001", "P002", "P003"}

    def test_data2_mapped_to_value(self, clinical, additional, test_data):
        result = cprd_adapter.build_labtest(clinical, additional, test_data)
        assert "value" in result.columns
        assert "7.2" in result["value"].values or 7.2 in result["value"].values

    def test_data3_mapped_to_unit(self, clinical, additional, test_data):
        result = cprd_adapter.build_labtest(clinical, additional, test_data)
        assert "mmol/L" in result["unit"].values

    def test_data4_mapped_to_valuecat(self, clinical, additional, test_data):
        result = cprd_adapter.build_labtest(clinical, additional, test_data)
        assert "normal" in result["valuecat"].values or "high" in result["valuecat"].values

    def test_loinc_is_null(self, clinical, additional, test_data):
        result = cprd_adapter.build_labtest(clinical, additional, test_data)
        assert result["loinc"].isna().all()

    def test_eventdate_is_datetime(self, clinical, additional, test_data):
        result = cprd_adapter.build_labtest(clinical, additional, test_data)
        assert pd.api.types.is_datetime64_any_dtype(result["eventdate"])

    def test_entity_filter_reduces_rows(self, clinical, additional, test_data):
        all_result = cprd_adapter.build_labtest(clinical, additional, test_data)
        # only enttype 7 (from test_data only since additional+clinical join gives enttype 4/5)
        filtered = cprd_adapter.build_labtest(clinical, additional, test_data, entity_enttypes={"7"})
        assert len(filtered) < len(all_result)

    def test_no_duplicate_rows(self, clinical, additional, test_data):
        result = cprd_adapter.build_labtest(clinical, additional, test_data)
        assert not result.duplicated().any()

    def test_sorted_by_patient_then_date(self, clinical, additional, test_data):
        result = cprd_adapter.build_labtest(clinical, additional, test_data)
        pids = result["patient_id"].tolist()
        dates = result["eventdate"].tolist()
        for i in range(len(pids) - 1):
            if pids[i] == pids[i + 1]:
                assert dates[i] <= dates[i + 1] or pd.isna(dates[i + 1])


# ---------------------------------------------------------------------------
# TestCprdBuildPatinfo
# ---------------------------------------------------------------------------


class TestCprdBuildPatinfo:
    def test_schema_valid(self, patient):
        result = cprd_adapter.build_patinfo(patient)
        errors = PATINFO.validate(result)
        assert errors == []

    def test_three_canonical_columns_only(self, patient):
        result = cprd_adapter.build_patinfo(patient)
        assert list(result.columns) == ["patient_id", "dobyr", "sex"]

    def test_all_patients_present(self, patient):
        result = cprd_adapter.build_patinfo(patient)
        assert set(result["patient_id"]) == {"P001", "P002", "P003"}

    def test_dobyr_is_nullable_int(self, patient):
        result = cprd_adapter.build_patinfo(patient)
        assert result["dobyr"].dtype == "Int64"

    def test_dobyr_values_correct(self, patient):
        result = cprd_adapter.build_patinfo(patient)
        p1 = result[result["patient_id"] == "P001"]["dobyr"].iloc[0]
        assert p1 == 1955

    def test_deduplicates_across_tables(self, patient):
        result = cprd_adapter.build_patinfo(patient, patient.copy())
        assert not result.duplicated().any()
        assert len(result) == 3

    def test_patient_id_is_string(self, patient):
        result = cprd_adapter.build_patinfo(patient)
        assert result["patient_id"].dtype == object

    def test_pracid_column_not_in_output(self, patient):
        result = cprd_adapter.build_patinfo(patient)
        assert "pracid" not in result.columns
