import pandas as pd
import pytest
from tests.conftest import TEST_DATA_PATH, _assert_shape_matches

from ehrdata.io import read_csv

TEST_PATH = TEST_DATA_PATH / "toy_csv"


@pytest.mark.parametrize(
    ("filename", "target_shape"),
    [
        ("csv_basic.csv", (5, 4)),
        ("csv_non_num_with_missing.csv", (5, 6)),
        ("csv_num_with_missing.csv", (5, 4)),
    ],
)
def test_read_csv(filename, target_shape):
    edata = read_csv(TEST_PATH / filename)

    _assert_shape_matches(edata, (target_shape[0], target_shape[1], 0), check_R_None=True)

    assert edata.obs.shape == (target_shape[0], 0)
    assert edata.var.shape == (target_shape[1], 0)
    assert edata.tem.shape == (0, 0)

    desired_default_index = pd.RangeIndex(start=0, stop=target_shape[0], step=1).astype(str)
    assert edata.obs.index.equals(desired_default_index)


@pytest.mark.parametrize(
    ("filename", "target_shape"),
    [
        ("csv_basic.csv", (5, 4)),
        ("csv_non_num_with_missing.csv", (5, 6)),
        ("csv_num_with_missing.csv", (5, 4)),
    ],
)
@pytest.mark.parametrize("index_column", ["patient_id", 0])
def test_read_csv_index_column(filename, target_shape, index_column):
    edata = read_csv(TEST_PATH / filename, index_column=index_column)

    _assert_shape_matches(edata, (target_shape[0], target_shape[1] - 1, 0), check_R_None=True)
    assert edata.obs.shape == (target_shape[0], 0)
    assert edata.var.shape == (target_shape[1] - 1, 0)
    assert edata.tem.shape == (0, 0)

    assert edata.obs.index.name == "patient_id"


@pytest.mark.parametrize(
    ("filename", "target_shape"),
    [
        ("csv_basic.csv", (5, 4)),
        ("csv_non_num_with_missing.csv", (5, 6)),
        ("csv_num_with_missing.csv", (5, 4)),
    ],
)
def test_read_csv_columns_obs_only(filename, target_shape):
    edata = read_csv(TEST_PATH / filename, columns_obs_only=["patient_id"])

    _assert_shape_matches(edata, (target_shape[0], target_shape[1] - 1, 0), check_R_None=True)
    assert edata.obs.shape == (target_shape[0], 1)
    assert edata.var.shape == (target_shape[1] - 1, 0)
    assert edata.tem.shape == (0, 0)

    desired_default_index = pd.RangeIndex(start=0, stop=target_shape[0], step=1).astype(str)
    assert edata.obs.index.equals(desired_default_index)

    assert "patient_id" in edata.obs.columns
