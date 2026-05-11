import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from ehrdata.io.source.extract import (
    read_csv_with_duckdb,
    read_zipped_tsv,
    read_zipped_tsvs,
    union_tables,
    unnest_codes,
)

FIXTURE_DIR = Path("tests/data/source_basic")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zip(members: dict[str, str]) -> bytes:
    """Return in-memory zip bytes with the given {filename: tsv_content} map."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# union_tables
# ---------------------------------------------------------------------------


class TestUnionTables:
    def test_basic_concat(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df2 = pd.DataFrame({"a": [3, 4], "b": ["z", "w"]})
        result = union_tables([df1, df2])
        assert len(result) == 4
        assert list(result.columns) == ["a", "b"]

    def test_removes_exact_duplicates(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df2 = pd.DataFrame({"a": [2, 3], "b": ["y", "z"]})
        result = union_tables([df1, df2])
        assert len(result) == 3

    def test_index_reset(self):
        df1 = pd.DataFrame({"a": [10]}, index=[99])
        df2 = pd.DataFrame({"a": [20]}, index=[100])
        result = union_tables([df1, df2])
        assert list(result.index) == [0, 1]

    def test_single_dataframe(self):
        df = pd.DataFrame({"a": [1, 1, 2]})
        result = union_tables([df])
        assert len(result) == 2

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            union_tables([])

    def test_preserves_column_order(self):
        df1 = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        result = union_tables([df1])
        assert list(result.columns) == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# unnest_codes
# ---------------------------------------------------------------------------


class TestUnnestCodes:
    def test_wide_to_long(self):
        df = pd.read_csv(FIXTURE_DIR / "diagnosis_wide.csv")
        result = unnest_codes(df, id_cols=["patient_id", "eventdate"], code_cols=["dx1", "dx2", "dx3"], value_name="dx")
        assert "dx" in result.columns
        assert "variable" not in result.columns
        assert "dx1" not in result.columns

    def test_drops_null_codes(self):
        df = pd.read_csv(FIXTURE_DIR / "diagnosis_wide.csv")
        result = unnest_codes(df, id_cols=["patient_id", "eventdate"], code_cols=["dx1", "dx2", "dx3"], value_name="dx")
        assert result["dx"].notna().all()

    def test_correct_row_count(self):
        # Row1: dx1=E11.9, dx2=I10  → 2 codes
        # Row2: dx1=E10.9            → 1 code
        # Row3: dx1=V58.67, dx2=E11.9, dx3=I10 → 3 codes
        # Total before dedup: 6; after dedup: 6 (all distinct patient×date×dx)
        df = pd.read_csv(FIXTURE_DIR / "diagnosis_wide.csv")
        result = unnest_codes(df, id_cols=["patient_id", "eventdate"], code_cols=["dx1", "dx2", "dx3"], value_name="dx")
        assert len(result) == 6

    def test_deduplicates_identical_long_rows(self):
        df = pd.DataFrame(
            {
                "pid": [1, 1],
                "dx1": ["A01", "A01"],
                "dx2": ["B02", "B02"],
            }
        )
        result = unnest_codes(df, id_cols=["pid"], code_cols=["dx1", "dx2"], value_name="dx")
        assert len(result) == 2  # (pid=1,dx=A01) and (pid=1,dx=B02)

    def test_custom_value_name(self):
        df = pd.DataFrame({"pid": [1], "p1": ["99213"], "p2": [None]})
        result = unnest_codes(df, id_cols=["pid"], code_cols=["p1", "p2"], value_name="proc")
        assert "proc" in result.columns

    def test_all_null_returns_empty(self):
        df = pd.DataFrame({"pid": [1, 2], "dx1": [None, None]})
        result = unnest_codes(df, id_cols=["pid"], code_cols=["dx1"])
        assert len(result) == 0

    def test_index_reset(self):
        df = pd.DataFrame({"pid": [1, 2], "dx1": ["A", "B"]})
        result = unnest_codes(df, id_cols=["pid"], code_cols=["dx1"])
        assert list(result.index) == [0, 1]


# ---------------------------------------------------------------------------
# read_zipped_tsv
# ---------------------------------------------------------------------------


class TestReadZippedTsv:
    @pytest.fixture
    def zip_bytes(self, tmp_path):
        content = "patient_id\teventdate\tdx\n1\t2020-01-15\tE11.9\n2\t2020-02-01\tI10\n"
        zdata = _make_zip({"clinical.txt": content})
        p = tmp_path / "extract.zip"
        p.write_bytes(zdata)
        return p

    def test_reads_member(self, zip_bytes):
        df = read_zipped_tsv(zip_bytes, "clinical.txt")
        assert len(df) == 2
        assert list(df.columns) == ["patient_id", "eventdate", "dx"]

    def test_usecols_filter(self, zip_bytes):
        df = read_zipped_tsv(zip_bytes, "clinical.txt", usecols=["patient_id", "dx"])
        assert list(df.columns) == ["patient_id", "dx"]
        assert "eventdate" not in df.columns

    def test_accepts_string_path(self, zip_bytes):
        df = read_zipped_tsv(str(zip_bytes), "clinical.txt")
        assert len(df) == 2


# ---------------------------------------------------------------------------
# read_zipped_tsvs
# ---------------------------------------------------------------------------


class TestReadZippedTsvs:
    @pytest.fixture
    def zip_bytes(self, tmp_path):
        tsv1 = "patient_id\tdx\n1\tE11.9\n2\tI10\n"
        tsv2 = "patient_id\tdx\n3\tE10.9\n4\tJ45\n"
        zdata = _make_zip({"controls/clinical.txt": tsv1, "main/clinical.txt": tsv2})
        p = tmp_path / "extract.zip"
        p.write_bytes(zdata)
        return p

    def test_unions_all_members(self, zip_bytes):
        df = read_zipped_tsvs(zip_bytes)
        assert len(df) == 4

    def test_pattern_filter(self, zip_bytes):
        df = read_zipped_tsvs(zip_bytes, pattern="controls")
        assert len(df) == 2

    def test_no_match_returns_empty(self, zip_bytes):
        df = read_zipped_tsvs(zip_bytes, pattern="nonexistent")
        assert len(df) == 0

    def test_usecols(self, zip_bytes):
        df = read_zipped_tsvs(zip_bytes, usecols=["patient_id"])
        assert list(df.columns) == ["patient_id"]


# ---------------------------------------------------------------------------
# read_csv_with_duckdb
# ---------------------------------------------------------------------------


class TestReadCsvWithDuckdb:
    def test_reads_csv(self):
        df = read_csv_with_duckdb(FIXTURE_DIR / "diagnosis.csv")
        assert len(df) > 0
        assert "patient_id" in df.columns

    def test_column_projection(self):
        df = read_csv_with_duckdb(FIXTURE_DIR / "diagnosis.csv", columns=["patient_id", "dx"])
        assert list(df.columns) == ["patient_id", "dx"]
        assert "eventdate" not in df.columns

    def test_where_clause(self):
        df = read_csv_with_duckdb(FIXTURE_DIR / "diagnosis.csv", where="dxver = '0'")
        assert all(str(v) == "0" for v in df["dxver"])
