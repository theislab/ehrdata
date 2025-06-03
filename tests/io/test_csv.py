from pathlib import Path

import pandas as pd
import pytest
from tests.conftest import TEST_DATA_PATH

from ehrdata.io import read_csv

_TEST_PATH = f"{TEST_DATA_PATH}/toy_simple/dataframe/"
_TEST_PATH_H5AD = f"{_TEST_PATH}/toy_simple/h5ad"


@pytest.mark.parametrize(
    ("filename", "target_shape"),
    [
        ("dataset_basic.csv", (5, 4)),
        ("dataset_non_num_with_missing.csv", (5, 6)),
        ("dataset_num_with_missing.csv", (5, 4)),
    ],
)
def test_read_csv(filename, target_shape):
    edata = read_csv(Path(_TEST_PATH) / filename)

    assert edata.shape == (target_shape[0], target_shape[1], 0)
    assert edata.X.shape == (target_shape[0], target_shape[1])
    assert edata.obs.shape == (target_shape[0], 0)
    assert edata.var.shape == (target_shape[1], 0)
    assert edata.tem.shape == (0, 0)

    desired_default_index = pd.RangeIndex(start=0, stop=target_shape[0], step=1).astype(str)
    assert edata.obs.index.equals(desired_default_index)


@pytest.mark.parametrize(
    ("filename", "target_shape"),
    [
        ("dataset_basic.csv", (5, 4)),
        ("dataset_non_num_with_missing.csv", (5, 6)),
        ("dataset_num_with_missing.csv", (5, 4)),
    ],
)
@pytest.mark.parametrize("index_column", ["patient_id", 0])
def test_read_csv_index_column(filename, target_shape, index_column):
    edata = read_csv(Path(_TEST_PATH) / filename, index_column=index_column)

    assert edata.shape == (target_shape[0], target_shape[1] - 1, 0)
    assert edata.X.shape == (target_shape[0], target_shape[1] - 1)
    assert edata.obs.shape == (target_shape[0], 0)
    assert edata.var.shape == (target_shape[1] - 1, 0)
    assert edata.tem.shape == (0, 0)

    assert edata.obs.index.name == "patient_id"


@pytest.mark.parametrize(
    ("filename", "target_shape"),
    [
        ("dataset_basic.csv", (5, 4)),
        ("dataset_non_num_with_missing.csv", (5, 6)),
        ("dataset_num_with_missing.csv", (5, 4)),
    ],
)
def test_read_csv_columns_obs_only(filename, target_shape):
    edata = read_csv(Path(_TEST_PATH) / filename, columns_obs_only=["patient_id"])

    assert edata.shape == (target_shape[0], target_shape[1] - 1, 0)
    assert edata.X.shape == (target_shape[0], target_shape[1] - 1)
    assert edata.obs.shape == (target_shape[0], 1)
    assert edata.var.shape == (target_shape[1] - 1, 0)
    assert edata.tem.shape == (0, 0)

    desired_default_index = pd.RangeIndex(start=0, stop=target_shape[0], step=1).astype(str)
    assert edata.obs.index.equals(desired_default_index)

    assert "patient_id" in edata.obs.columns


# def test_read_h5ad():
#     edata = read_h5ad(dataset_path=f"{_TEST_PATH_H5AD}/dataset9.h5ad")

#     assert edata.X.shape == (4, 3)
#     assert set(edata.var_names) == {"col" + str(i) for i in range(1, 4)}
#     assert set(edata.obs.columns) == set()


# def test_read_multiple_h5ad():
#     edatas = read_h5ad(dataset_path=f"{_TEST_PATH_H5AD}")
#     edata_ids = set(edatas.keys())

#     assert all(edata_id in edata_ids for edata_id in ("dataset8", "dataset9"))
#     assert set(edatas["dataset8"].var_names) == {"indexcol", "intcol", "boolcol", "binary_col", "strcol"}
#     assert set(edatas["dataset9"].var_names) == {"col" + str(i) for i in range(1, 4)}
#     assert all(obs_name in set(edatas["dataset8"].obs.columns) for obs_name in ("datetime",))


# def test_read_csv_without_index_column():
#     edata = read_csv(dataset_path=f"{_TEST_PATH}/dataset_index.csv")
#     matrix = np.array(
#         [[1, 14, 500, False], [2, 7, 330, False], [3, 10, 800, True], [4, 11, 765, True], [5, 3, 800, True]]
#     )
#     assert edata.X.shape == (5, 4)
#     assert (matrix == edata.X).all()
#     assert edata.var_names.to_list() == ["clinic_id", "los_days", "b12_values", "survival"]
#     assert (edata.layers["original"] == matrix).all()
#     assert id(edata.layers["original"]) != id(edata.X)
#     assert list(edata.obs.index) == ["0", "1", "2", "3", "4"]


# def test_read_csv_with_bools_obs_only():
#     edata = read_csv(dataset_path=f"{_TEST_PATH}/dataset_basic.csv", columns_obs_only=["survival", "b12_values"])
#     matrix = np.array([[12, 14], [13, 7], [14, 10], [15, 11], [16, 3]])
#     assert edata.X.shape == (5, 2)
#     assert (matrix == edata.X).all()
#     assert edata.var_names.to_list() == ["patient_id", "los_days"]
#     assert (edata.layers["original"] == matrix).all()
#     assert id(edata.layers["original"]) != id(edata.X)
#     assert set(edata.obs.columns) == {"b12_values", "survival"}
#     assert pd.api.types.is_bool_dtype(edata.obs["survival"].dtype)
#     assert pd.api.types.is_numeric_dtype(edata.obs["b12_values"].dtype)


# def test_read_csv_with_bools_and_cats_obs_only():
#     edata = read_csv(
#         dataset_path=f"{_TEST_PATH}/dataset_bools_and_str.csv", columns_obs_only=["b12_values", "name", "survival"]
#     )
#     matrix = np.array([[1, 14], [2, 7], [3, 10], [4, 11], [5, 3]])
#     with pytest.raises(ValueError):
#         _ = read_csv(
#             dataset_path=f"{_TEST_PATH}",
#             columns_obs_only={
#                 "dataset_non_num_with_missing": ["intcol"],
#                 "dataset_num_with_missing": ["col1", "col2"],
#             },
#             columns_x_only={"dataset_non_num_with_missing": ["indexcol"], "dataset_num_with_missing": ["col3"]},
#         )


# def test_move_single_column_to_x():
#     edata = read_csv(dataset_path=f"{_TEST_PATH}/dataset_basic.csv", columns_x_only=["b12_values"])
#     assert edata.X.shape == (5, 1)
#     assert list(edata.var_names) == ["b12_values"]
#     assert "b12_values" not in list(edata.obs.columns)
#     assert all(obs_names in list(edata.obs.columns) for obs_names in ["los_days", "patient_id", "survival"])


# def test_move_multiple_columns_to_x():
#     edata = read_csv(dataset_path=f"{_TEST_PATH}/dataset_basic.csv", columns_x_only=["b12_values", "survival"])
#     assert edata.X.shape == (5, 2)
#     assert all(var_names in list(edata.var_names) for var_names in ["b12_values", "survival"])
#     assert all(obs_names in list(edata.obs.columns) for obs_names in ["los_days", "patient_id"])
#     assert all(var_names not in list(edata.obs.columns) for var_names in ["b12_values", "survival"])


# def test_read_multiple_csv_with_x_only():
#     edatas = read_csv(
#         dataset_path=f"{_TEST_PATH}",
#         columns_x_only={"dataset_non_num_with_missing": ["strcol"], "dataset_num_with_missing": ["col1"]},
#     )
#     edata_ids = set(edatas.keys())
#     assert all(edata_id in edata_ids for edata_id in ("dataset_non_num_with_missing", "dataset_num_with_missing"))
#     assert set(edatas["dataset_non_num_with_missing"].obs.columns) == {
#         "indexcol",
#         "intcol",
#         "boolcol",
#         "binary_col",
#         "datetime",
#     }
#     assert set(edatas["dataset_num_with_missing"].obs.columns) == {"col" + str(i) for i in range(2, 4)}
#     assert set(edatas["dataset_non_num_with_missing"].var_names) == {"strcol"}
#     assert set(edatas["dataset_num_with_missing"].var_names) == {"col1"}


# def test_read_multiple_csv_with_x_only_2():
#     edatas = read_csv(
#         dataset_path=f"{_TEST_PATH}",
#         columns_x_only={
#             "dataset_non_num_with_missing": ["strcol", "intcol", "boolcol"],
#             "dataset_num_with_missing": ["col1", "col3"],
#         },
#     )
#     edata_ids = set(edatas.keys())
#     assert all(edata_id in edata_ids for edata_id in ("dataset_non_num_with_missing", "dataset_num_with_missing"))
#     assert set(edatas["dataset_non_num_with_missing"].obs.columns) == {"indexcol", "binary_col", "datetime"}
#     assert set(edatas["dataset_num_with_missing"].obs.columns) == {"col2"}
#     assert set(edatas["dataset_non_num_with_missing"].var_names) == {"strcol", "intcol", "boolcol"}
#     assert set(edatas["dataset_num_with_missing"].var_names) == {"col1", "col3"}
