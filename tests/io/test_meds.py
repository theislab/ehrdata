import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from ehrdata import EHRData
from ehrdata.core.constants import ANCHOR_TIME_KEY
from ehrdata.io import from_events, read_h5ed, read_meds, write_h5ed, write_meds

T0 = pd.Timestamp("2020-01-01")
HOUR = pd.Timedelta(1, "h")


def _write_parquet(df: pd.DataFrame, path: Path, select: str) -> None:
    con = duckdb.connect()
    con.register("df", df)
    con.sql(f"SELECT {select} FROM df").write_parquet(str(path))


def _write_data(rows, path: Path) -> None:
    _write_parquet(
        pd.DataFrame(rows, columns=["subject_id", "time", "code", "numeric_value"]),
        path,
        "subject_id::BIGINT AS subject_id, time::TIMESTAMP AS time, code, numeric_value::FLOAT AS numeric_value",
    )


@pytest.fixture
def meds_root(tmp_path) -> Path:
    root = tmp_path / "meds"
    (root / "data" / "train").mkdir(parents=True)
    (root / "metadata").mkdir()
    _write_data(
        [
            (1, pd.NaT, "GENDER//F", None),
            (1, pd.NaT, "HEIGHT", 170.0),
            (1, T0, "HR", 80.0),
            (1, T0 + 0.5 * HOUR, "HR", 90.0),
            (1, T0 + HOUR, "ICD//I10", None),
            (1, T0 + 2 * HOUR, "HR", 70.0),
            (2, pd.NaT, "GENDER//M", None),
            (2, T0 + 24 * HOUR, "HR", 60.0),
            (2, T0 + 27 * HOUR, "HR", 65.0),
        ],
        root / "data" / "train" / "0.parquet",
    )
    _write_data([(3, pd.NaT, "HEIGHT", 180.0), (3, T0, "HR", 100.0)], root / "data" / "1.parquet")
    _write_parquet(
        pd.DataFrame({"subject_id": [1, 2, 3], "split": ["train", "tuning", "held_out"]}),
        root / "metadata" / "subject_splits.parquet",
        "subject_id::BIGINT AS subject_id, split",
    )
    _write_parquet(
        pd.DataFrame({"code": ["HR", "ICD//I10", "HEIGHT"], "description": ["Heart rate", "Hypertension", "Height"]}),
        root / "metadata" / "codes.parquet",
        "code, description, NULL::VARCHAR[] AS parent_codes",
    )
    return root


def test_read_meds(meds_root):
    edata = read_meds(meds_root)

    assert list(edata.obs_names) == ["1", "2", "3"]
    assert list(edata.var_names) == ["HR", "ICD//I10"]
    np.testing.assert_array_equal(
        edata.X,
        [
            [[90.0, np.nan, 70.0, np.nan], [np.nan, 1.0, np.nan, np.nan]],
            [[60.0, np.nan, np.nan, 65.0], [np.nan] * 4],
            [[100.0, np.nan, np.nan, np.nan], [np.nan] * 4],
        ],
    )
    assert list(edata.obs["split"]) == ["train", "tuning", "held_out"]
    assert list(edata.obs["GENDER__F"]) == [True, False, False]
    assert list(edata.obs["GENDER__M"]) == [False, True, False]
    assert edata.uns["meds_static_codes"] == {"GENDER__F": "GENDER//F", "GENDER__M": "GENDER//M", "HEIGHT": "HEIGHT"}
    np.testing.assert_array_equal(edata.obs["HEIGHT"], [170.0, np.nan, 180.0])
    assert list(edata.obs[ANCHOR_TIME_KEY]) == [str(T0), str(T0 + 24 * HOUR), str(T0)]
    assert list(edata.var["description"]) == ["Heart rate", "Hypertension"]


def test_read_meds_binning(meds_root):
    edata = read_meds(
        meds_root, codes=["HR"], interval_length_number=1, interval_length_unit="D", aggregation_strategy="mean"
    )

    np.testing.assert_array_equal(edata.X, [[[80.0]], [[62.5]], [[100.0]]])


@pytest.mark.parametrize(("split", "subjects"), [("train", ["1"]), (["train", "held_out"], ["1", "3"])])
def test_read_meds_split(meds_root, split, subjects):
    edata = read_meds(meds_root, split=split)

    assert list(edata.obs_names) == subjects
    np.testing.assert_array_equal(edata.X, read_meds(meds_root)[subjects, :, : edata.n_t].X)


def test_read_meds_split_without_splits(meds_root):
    (meds_root / "metadata" / "subject_splits.parquet").unlink()

    assert "split" not in read_meds(meds_root).obs.columns
    with pytest.raises(ValueError, match="split"):
        read_meds(meds_root, split="train")


@pytest.mark.parametrize("sparse", [False, True])
def test_meds_roundtrip(meds_root, tmp_path, sparse):
    edata = read_meds(meds_root, sparse=sparse, layer="binned")

    write_meds(edata, tmp_path / "written", layer="binned")
    read = read_meds(tmp_path / "written", sparse=sparse, layer="binned")

    np.testing.assert_array_equal(
        read.layers["binned"].todense() if sparse else read.layers["binned"],
        edata.layers["binned"].todense() if sparse else edata.layers["binned"],
    )
    pd.testing.assert_frame_equal(read.obs, edata.obs)
    pd.testing.assert_frame_equal(read.var.drop(columns="n_events"), edata.var.drop(columns="n_events"))
    pd.testing.assert_frame_equal(read.tem, edata.tem)
    assert read.uns["meds_static_codes"] == edata.uns["meds_static_codes"]


def test_meds_roundtrip_through_h5ed(meds_root, tmp_path):
    edata = read_meds(meds_root)
    write_h5ed(edata, tmp_path / "meds.h5ed")

    write_meds(read_h5ed(tmp_path / "meds.h5ed"), tmp_path / "written")
    read = read_meds(tmp_path / "written")

    np.testing.assert_array_equal(read.X, edata.X)
    pd.testing.assert_frame_equal(read.obs, edata.obs)
    pd.testing.assert_frame_equal(read.var.drop(columns="n_events"), edata.var.drop(columns="n_events"))
    assert read.uns["meds_static_codes"] == edata.uns["meds_static_codes"]
    assert (
        "GENDER//F"
        in duckdb.read_parquet(str(tmp_path / "written" / "metadata" / "codes.parquet")).fetchnumpy()["code"]
    )


def test_write_meds_files(meds_root, tmp_path):
    root = tmp_path / "written"
    write_meds(read_meds(meds_root), root)

    con = duckdb.connect()
    data = con.read_parquet(str(root / "data" / "0.parquet"))
    assert dict(zip(data.columns, map(str, data.types), strict=True)) == {
        "subject_id": "BIGINT",
        "time": "TIMESTAMP",
        "code": "VARCHAR",
        "numeric_value": "FLOAT",
    }
    assert data.fetchall()[:5] == [
        (1, None, "GENDER//F", None),
        (1, None, "HEIGHT", 170.0),
        (1, T0.to_pydatetime(), "HR", 90.0),
        (1, (T0 + HOUR).to_pydatetime(), "ICD//I10", 1.0),
        (1, (T0 + 2 * HOUR).to_pydatetime(), "HR", 70.0),
    ]
    code_metadata = con.read_parquet(str(root / "metadata" / "codes.parquet"))
    assert dict(zip(code_metadata.columns, map(str, code_metadata.types), strict=True)) == {
        "code": "VARCHAR",
        "description": "VARCHAR",
        "parent_codes": "VARCHAR[]",
    }
    assert sorted(code_metadata.fetchnumpy()["code"]) == ["GENDER//F", "GENDER//M", "HEIGHT", "HR", "ICD//I10"]
    splits = con.read_parquet(str(root / "metadata" / "subject_splits.parquet"))
    assert splits.fetchall() == [(1, "train"), (2, "tuning"), (3, "held_out")]
    dataset_metadata = json.loads((root / "metadata" / "dataset.json").read_text())
    assert {"etl_name", "etl_version", "meds_version", "created_at"} <= dataset_metadata.keys()


def test_write_meds_validates_against_meds_schema(meds_root, tmp_path):
    meds = pytest.importorskip("meds")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "written"
    write_meds(read_meds(meds_root), root)

    meds.DataSchema.validate(pq.read_table(root / "data" / "0.parquet"))
    meds.CodeMetadataSchema.validate(pq.read_table(root / "metadata" / "codes.parquet"))
    meds.SubjectSplitSchema.validate(pq.read_table(root / "metadata" / "subject_splits.parquet"))
    meds.DatasetMetadataSchema.validate(json.loads((root / "metadata" / "dataset.json").read_text()))


def test_write_meds_relative_times(tmp_path):
    events = pd.DataFrame(
        {"subject_id": [1, 1], "time": [HOUR, 3 * HOUR], "code": ["HR", "HR"], "numeric_value": [80.0, 70.0]}
    )

    write_meds(from_events(events), tmp_path / "written")

    times = duckdb.read_parquet(str(tmp_path / "written" / "data" / "0.parquet")).fetchnumpy()["time"]
    np.testing.assert_array_equal(times, np.datetime64(0, "us") + np.array([HOUR, 3 * HOUR], dtype="m8[us]"))


def test_write_meds_invalid(meds_root, tmp_path):
    edata = read_meds(meds_root)
    with pytest.raises(FileExistsError):
        write_meds(edata, meds_root)

    edata.obs_names = ["a", "b", "c"]
    with pytest.raises(ValueError, match="integers"):
        write_meds(edata, tmp_path / "written")

    with pytest.raises(ValueError, match="interval_start_offset"):
        write_meds(EHRData(X=np.ones((1, 1, 2)), obs=pd.DataFrame(index=["1"])), tmp_path / "written")


def test_read_meds_anchor_code(meds_root):
    edata = read_meds(meds_root, anchor_code="ICD", interval_length_number=1, interval_length_unit="h")

    assert edata.obs[ANCHOR_TIME_KEY].iloc[0] == str(T0 + HOUR)
    assert edata.obs[ANCHOR_TIME_KEY].iloc[1:].isna().all()
    np.testing.assert_array_equal(edata.X[0], [[np.nan, 70.0], [1.0, np.nan]])
    assert np.isnan(edata.X[1:]).all()
