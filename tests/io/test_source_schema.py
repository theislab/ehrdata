import pandas as pd
import pytest

from ehrdata.io.source.schema import (
    ALL_SCHEMAS,
    DIAGNOSIS,
    HABIT,
    INSURANCE,
    LABTEST,
    PATINFO,
    PROCEDURE,
    PROVIDER,
    THERAPY,
    ColumnSpec,
    TableSchema,
)


class TestColumnSpec:
    def test_defaults(self):
        col = ColumnSpec("patient_id", "object")
        assert col.name == "patient_id"
        assert col.dtype == "object"
        assert col.nullable is True

    def test_non_nullable(self):
        col = ColumnSpec("dx", "object", nullable=False)
        assert col.nullable is False

    def test_frozen(self):
        col = ColumnSpec("x", "object")
        with pytest.raises(Exception):
            col.name = "y"  # type: ignore[misc]


class TestTableSchema:
    def test_column_names(self):
        assert DIAGNOSIS.column_names == ["patient_id", "dxver", "eventdate", "dx"]

    def test_empty_returns_zero_row_dataframe(self):
        df = DIAGNOSIS.empty()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == DIAGNOSIS.column_names

    def test_empty_dtype_object(self):
        df = PATINFO.empty()
        assert df["patient_id"].dtype == object
        assert df["sex"].dtype == object

    def test_empty_datetime_dtype(self):
        df = DIAGNOSIS.empty()
        assert pd.api.types.is_datetime64_any_dtype(df["eventdate"])

    def test_validate_valid_dataframe(self):
        df = pd.DataFrame({"patient_id": ["1"], "dxver": ["0"], "eventdate": pd.to_datetime(["2020-01-01"]), "dx": ["E11.9"]})
        assert DIAGNOSIS.validate(df) == []

    def test_validate_missing_column(self):
        df = pd.DataFrame({"patient_id": ["1"], "dxver": ["0"], "eventdate": pd.to_datetime(["2020-01-01"])})
        errors = DIAGNOSIS.validate(df)
        assert len(errors) == 1
        assert "dx" in errors[0]

    def test_validate_multiple_missing_columns(self):
        df = pd.DataFrame({"patient_id": ["1"]})
        errors = DIAGNOSIS.validate(df)
        assert len(errors) == 3

    def test_validate_strict_rejects_extra_columns(self):
        df = pd.DataFrame({
            "patient_id": ["1"], "dxver": ["0"],
            "eventdate": pd.to_datetime(["2020-01-01"]),
            "dx": ["E11.9"], "extra_col": [99],
        })
        assert DIAGNOSIS.validate(df, strict=False) == []
        errors = DIAGNOSIS.validate(df, strict=True)
        assert len(errors) == 1
        assert "extra_col" in errors[0]

    def test_validate_strict_no_false_positives(self):
        df = DIAGNOSIS.empty()
        assert DIAGNOSIS.validate(df, strict=True) == []


class TestAllSchemas:
    def test_contains_all_eight_tables(self):
        assert set(ALL_SCHEMAS.keys()) == {"diagnosis", "therapy", "labtest", "procedure", "patinfo", "insurance", "provider", "habit"}

    def test_all_schemas_start_with_patient_id(self):
        for name, schema in ALL_SCHEMAS.items():
            assert schema.column_names[0] == "patient_id", f"{name} does not start with patient_id"

    def test_all_schemas_have_non_empty_columns(self):
        for name, schema in ALL_SCHEMAS.items():
            assert len(schema.columns) > 0, f"{name} has no columns"

    @pytest.mark.parametrize("schema,expected_cols", [
        (THERAPY, ["patient_id", "prescription_date", "start_date", "fill_date", "end_date", "refill", "rxcui", "ndc11", "ingredient"]),
        (LABTEST, ["patient_id", "eventdate", "value", "valuecat", "unit", "loinc"]),
        (PROCEDURE, ["patient_id", "proctype", "eventdate", "proc"]),
        (PATINFO, ["patient_id", "dobyr", "sex"]),
        (INSURANCE, ["patient_id", "svcdate", "cob", "coins", "copay"]),
        (PROVIDER, ["patient_id", "dtstart", "dtend", "plantyp", "rx", "hlthplan"]),
        (HABIT, ["patient_id", "encounter_date", "mapped_question_answer"]),
    ])
    def test_schema_columns(self, schema, expected_cols):
        assert schema.column_names == expected_cols

    def test_schema_lookup_by_name(self):
        assert ALL_SCHEMAS["diagnosis"] is DIAGNOSIS
        assert ALL_SCHEMAS["therapy"] is THERAPY

    def test_schemas_are_frozen(self):
        with pytest.raises(Exception):
            DIAGNOSIS.name = "other"  # type: ignore[misc]
