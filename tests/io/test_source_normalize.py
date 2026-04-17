from pathlib import Path

import pandas as pd
import pytest

from ehrdata.io.source.normalize import (
    coerce_date,
    coerce_patient_id,
    deduplicate,
    infer_icd_version,
    normalize_diagnosis,
    normalize_labtest,
    normalize_procedure,
    normalize_therapy,
    sort_events,
)
from ehrdata.io.source.schema import DIAGNOSIS, LABTEST, PROCEDURE, THERAPY

FIXTURE_DIR = Path("tests/data/source_basic")


# ---------------------------------------------------------------------------
# coerce_patient_id
# ---------------------------------------------------------------------------


class TestCoercePatientId:
    def test_strips_whitespace(self):
        s = pd.Series([" 1 ", "  2", "3  "])
        result = coerce_patient_id(s)
        assert list(result) == ["1", "2", "3"]

    def test_converts_int_to_str(self):
        s = pd.Series([1, 2, 3])
        result = coerce_patient_id(s)
        assert result.dtype == object
        assert list(result) == ["1", "2", "3"]

    def test_preserves_index(self):
        s = pd.Series(["a", "b"], index=[10, 20])
        result = coerce_patient_id(s)
        assert list(result.index) == [10, 20]

    def test_large_int_ids(self):
        s = pd.Series([123456789012345])
        result = coerce_patient_id(s)
        assert result[0] == "123456789012345"


# ---------------------------------------------------------------------------
# coerce_date
# ---------------------------------------------------------------------------


class TestCoerceDate:
    def test_iso_dates(self):
        s = pd.Series(["2020-01-15", "2021-06-30"])
        result = coerce_date(s)
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result[0] == pd.Timestamp("2020-01-15")

    def test_invalid_becomes_nat(self):
        s = pd.Series(["not-a-date", "2020-01-01"])
        result = coerce_date(s)
        assert pd.isna(result[0])
        assert result[1] == pd.Timestamp("2020-01-01")

    def test_already_datetime_passthrough(self):
        s = pd.to_datetime(pd.Series(["2020-01-01"]))
        result = coerce_date(s)
        assert pd.api.types.is_datetime64_any_dtype(result)

    def test_explicit_format(self):
        s = pd.Series(["15/01/2020", "01/06/2021"])
        result = coerce_date(s, formats=["%d/%m/%Y"])
        assert result[0] == pd.Timestamp("2020-01-15")
        assert result[1] == pd.Timestamp("2021-06-01")

    def test_fallback_when_format_fails(self):
        # format list has a wrong entry first, auto-inference picks up the date
        s = pd.Series(["2020-01-15"])
        result = coerce_date(s, formats=["%Y/%m/%d", "%Y-%m-%d"])
        assert result[0] == pd.Timestamp("2020-01-15")

    def test_all_null_returns_nat_series(self):
        s = pd.Series([None, None])
        result = coerce_date(s)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# infer_icd_version
# ---------------------------------------------------------------------------


class TestInferIcdVersion:
    def _make_df(self, dxver_values, dx_values):
        return pd.DataFrame({"patient_id": range(len(dx_values)), "dxver": dxver_values, "dx": dx_values})

    def test_e_prefix_inferred_as_icd9(self):
        df = self._make_df([None], ["E930.0"])
        result = infer_icd_version(df)
        assert result.loc[0, "dxver"] == "9"

    def test_v_prefix_inferred_as_icd9(self):
        df = self._make_df([None], ["V58.67"])
        result = infer_icd_version(df)
        assert result.loc[0, "dxver"] == "9"

    def test_existing_dxver_not_overwritten(self):
        df = self._make_df(["0"], ["E11.9"])
        result = infer_icd_version(df)
        assert result.loc[0, "dxver"] == "0"

    def test_non_ev_prefix_stays_null(self):
        df = self._make_df([None], ["I10"])
        result = infer_icd_version(df)
        assert pd.isna(result.loc[0, "dxver"])

    def test_does_not_mutate_input(self):
        df = self._make_df([None], ["E930.0"])
        _ = infer_icd_version(df)
        assert pd.isna(df.loc[0, "dxver"])

    def test_mixed_rows(self):
        df = self._make_df([None, "0", None, None], ["E930.0", "E11.9", "V58.67", "I10"])
        result = infer_icd_version(df)
        assert result.loc[0, "dxver"] == "9"  # E prefix + null → inferred
        assert result.loc[1, "dxver"] == "0"  # explicit, not overwritten
        assert result.loc[2, "dxver"] == "9"  # V prefix + null → inferred
        assert pd.isna(result.loc[3, "dxver"])  # I prefix + null → stays null

    def test_custom_column_names(self):
        df = pd.DataFrame({"pid": [1], "ver": [None], "code": ["E930.0"]})
        result = infer_icd_version(df, dx_col="code", dxver_col="ver")
        assert result.loc[0, "ver"] == "9"


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------


class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result = deduplicate(df)
        assert len(result) == 2

    def test_subset_dedup(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "z", "y"]})
        result = deduplicate(df, subset=["a"])
        assert len(result) == 2

    def test_index_reset(self):
        df = pd.DataFrame({"a": [1, 1, 2]}, index=[10, 10, 20])
        result = deduplicate(df)
        assert list(result.index) == [0, 1]

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = deduplicate(df)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# sort_events
# ---------------------------------------------------------------------------


class TestSortEvents:
    def test_sorted_by_patient_then_date(self):
        df = pd.DataFrame(
            {
                "patient_id": ["2", "1", "1"],
                "eventdate": pd.to_datetime(["2020-02-01", "2020-03-01", "2020-01-01"]),
            }
        )
        result = sort_events(df)
        assert list(result["patient_id"]) == ["1", "1", "2"]
        assert list(result["eventdate"]) == [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-03-01"),
            pd.Timestamp("2020-02-01"),
        ]

    def test_nat_placed_last(self):
        df = pd.DataFrame(
            {
                "patient_id": ["1", "1"],
                "eventdate": pd.to_datetime([None, "2020-01-01"]),
            }
        )
        result = sort_events(df)
        assert result["eventdate"].iloc[0] == pd.Timestamp("2020-01-01")
        assert pd.isna(result["eventdate"].iloc[1])

    def test_custom_column_names(self):
        df = pd.DataFrame(
            {
                "pid": ["b", "a"],
                "dt": pd.to_datetime(["2020-02-01", "2020-01-01"]),
            }
        )
        result = sort_events(df, patient_col="pid", date_col="dt")
        assert result["pid"].iloc[0] == "a"


# ---------------------------------------------------------------------------
# normalize_diagnosis (composite pipeline)
# ---------------------------------------------------------------------------


class TestNormalizeDiagnosis:
    @pytest.fixture
    def raw_df(self):
        return pd.read_csv(FIXTURE_DIR / "diagnosis.csv")

    def test_schema_valid_after_normalize(self, raw_df):
        result = normalize_diagnosis(raw_df)
        errors = DIAGNOSIS.validate(result)
        assert errors == [], errors

    def test_patient_id_is_stripped_string(self, raw_df):
        result = normalize_diagnosis(raw_df)
        assert result["patient_id"].dtype == object
        assert " " not in result["patient_id"].iloc[-1]  # " 4 " → "4"

    def test_eventdate_is_datetime(self, raw_df):
        result = normalize_diagnosis(raw_df)
        assert pd.api.types.is_datetime64_any_dtype(result["eventdate"])

    def test_icd_version_inferred(self, raw_df):
        result = normalize_diagnosis(raw_df)
        # Row with dx=E10.9 had null dxver → should be inferred as "9"
        e10_rows = result[result["dx"] == "E10.9"]
        assert (e10_rows["dxver"] == "9").all()

    def test_duplicate_removed(self, raw_df):
        # CSV has patient_id=1 / E11.9 / 2020-01-15 twice
        result = normalize_diagnosis(raw_df)
        dupes = result[(result["patient_id"] == "1") & (result["dx"] == "E11.9")]
        assert len(dupes) == 1

    def test_sorted_by_patient_then_date(self, raw_df):
        result = normalize_diagnosis(raw_df)
        patients = list(result["patient_id"])
        assert patients == sorted(patients)

    def test_dxver_column_added_when_absent(self):
        df = pd.DataFrame({"patient_id": ["1"], "eventdate": ["2020-01-01"], "dx": ["I10"]})
        result = normalize_diagnosis(df)
        assert "dxver" in result.columns

    def test_does_not_mutate_input(self, raw_df):
        original_len = len(raw_df)
        normalize_diagnosis(raw_df)
        assert len(raw_df) == original_len


# ---------------------------------------------------------------------------
# normalize_therapy
# ---------------------------------------------------------------------------


class TestNormalizeTherapy:
    @pytest.fixture
    def raw_df(self):
        return pd.read_csv(FIXTURE_DIR / "therapy.csv")

    def test_schema_valid_after_normalize(self, raw_df):
        result = normalize_therapy(raw_df)
        errors = THERAPY.validate(result)
        assert errors == [], errors

    def test_date_columns_parsed(self, raw_df):
        result = normalize_therapy(raw_df)
        assert pd.api.types.is_datetime64_any_dtype(result["fill_date"])
        assert pd.api.types.is_datetime64_any_dtype(result["end_date"])

    def test_missing_date_becomes_nat(self, raw_df):
        result = normalize_therapy(raw_df)
        # row 2 has no end_date
        row2 = result[result["patient_id"] == "2"]
        assert pd.isna(row2["end_date"].iloc[0])

    def test_duplicate_removed(self, raw_df):
        result = normalize_therapy(raw_df)
        assert len(result) == 2  # one duplicate row removed

    def test_patient_id_is_string(self, raw_df):
        result = normalize_therapy(raw_df)
        assert result["patient_id"].dtype == object


# ---------------------------------------------------------------------------
# normalize_labtest
# ---------------------------------------------------------------------------


class TestNormalizeLabtest:
    @pytest.fixture
    def raw_df(self):
        return pd.read_csv(FIXTURE_DIR / "labtest.csv")

    def test_schema_valid_after_normalize(self, raw_df):
        result = normalize_labtest(raw_df)
        errors = LABTEST.validate(result)
        assert errors == [], errors

    def test_duplicate_removed(self, raw_df):
        # CSV has patient_id=2 / high duplicate
        result = normalize_labtest(raw_df)
        assert len(result) == 3

    def test_sorted_by_patient_then_date(self, raw_df):
        result = normalize_labtest(raw_df)
        pids = list(result["patient_id"])
        assert pids == sorted(pids)


# ---------------------------------------------------------------------------
# normalize_procedure
# ---------------------------------------------------------------------------


class TestNormalizeProcedure:
    @pytest.fixture
    def raw_df(self):
        return pd.read_csv(FIXTURE_DIR / "procedure.csv")

    def test_schema_valid_after_normalize(self, raw_df):
        result = normalize_procedure(raw_df)
        errors = PROCEDURE.validate(result)
        assert errors == [], errors

    def test_duplicate_removed(self, raw_df):
        result = normalize_procedure(raw_df)
        assert len(result) == 3  # one duplicate removed

    def test_null_proctype_preserved(self, raw_df):
        result = normalize_procedure(raw_df)
        null_proc = result[result["patient_id"] == "3"]
        assert pd.isna(null_proc["proctype"].iloc[0])
