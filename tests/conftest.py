from pathlib import Path

import anndata as ad
import dask.array as da
import duckdb
import numpy as np
import pandas as pd
import pytest
import sparse as sp

from ehrdata import EHRData
from ehrdata.core.constants import R_LAYER_KEY
from ehrdata.io.omop import setup_connection


def _assert_shape_matches(
    edata: EHRData, shape: tuple[int, int, int], *, check_X_None: bool = False, check_R_None: bool = False
):
    assert edata.shape == shape

    if check_X_None:
        assert edata.X is None
    else:
        assert edata.X.shape == shape[0:2]

    if check_R_None:
        assert edata.R is None
    else:
        assert edata.R.shape == shape

    assert isinstance(edata.obs, pd.DataFrame)
    assert len(edata.obs) == shape[0]
    assert edata.n_obs == shape[0]

    assert isinstance(edata.var, pd.DataFrame)
    assert len(edata.var) == shape[1]
    assert edata.n_vars == shape[1]

    assert isinstance(edata.tem, pd.DataFrame)
    assert len(edata.tem) == shape[2]
    assert edata.n_t == shape[2]

    for key in edata.layers:
        if key != R_LAYER_KEY:
            assert edata.layers[key].shape == shape[0:2]


def _assert_dtype_object_array_with_missing_values_equal(a: np.ndarray, b: np.ndarray):
    # need to use pd.isnull to check for np.isnan in dtype object arrays, because np.isnan does not work on dtype object array
    a = a.copy()
    b = b.copy()
    assert np.array_equal(pd.isnull(a), pd.isnull(b))
    # if verified equal position of nan values, replace with 0 and verify the rest of the entries are equal
    a[pd.isnull(a)] = 0
    b[pd.isnull(b)] = 0
    assert np.array_equal(a, b)


@pytest.fixture
def csv_basic():
    return pd.read_csv("tests/data/toy_csv/csv_basic.csv")


@pytest.fixture
def csv_non_num_with_missing():
    return pd.read_csv("tests/data/toy_csv/csv_non_num_with_missing.csv")


@pytest.fixture
def csv_num_with_missing():
    return pd.read_csv("tests/data/toy_csv/csv_num_with_missing.csv")


@pytest.fixture
def X_numpy_32():
    return np.arange(1, 7).reshape(3, 2)


@pytest.fixture
def X_numpy_33():
    return np.arange(1, 10).reshape(3, 3)


@pytest.fixture
def R_numpy_322():
    return np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])


@pytest.fixture
def R_sparse_322():
    return sp.COO(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]))


@pytest.fixture
def R_dask_322():
    return da.from_array(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]), chunks=(3, 2, 2))


@pytest.fixture
def R_numpy_333():
    return np.arange(1, 28).reshape(3, 3, 3)


@pytest.fixture
def obs_31():
    return pd.DataFrame({"obs_col_1": [1, 2, 3]}, index=["obs1", "obs2", "obs3"])


@pytest.fixture
def var_21():
    return pd.DataFrame({"var_col_1": [1, 2]}, index=["var1", "var2"])


@pytest.fixture
def var_31():
    return pd.DataFrame({"var_col_1": [1, 2, 3]}, index=["var1", "var2", "var3"])


@pytest.fixture
def tem_11():
    return pd.DataFrame({"tem_col_1": [1]}, index=["t1"])


@pytest.fixture
def tem_21():
    return pd.DataFrame({"tem_col_1": [1, 2]}, index=["t1", "t2"])


@pytest.fixture
def tem_31():
    return pd.DataFrame({"tem_col_1": [1, 2, 3]}, index=["t1", "t2", "t3"])


@pytest.fixture
def edata_333(X_numpy_33, R_numpy_333, obs_31, var_31, tem_31):
    return EHRData(X=X_numpy_33, R=R_numpy_333, obs=obs_31, var=var_31, tem=tem_31)


@pytest.fixture
def edata_330(X_numpy_33, obs_31, var_31):
    return EHRData(X=X_numpy_33, obs=obs_31, var=var_31)


@pytest.fixture
def adata_33(X_numpy_33, obs_31, var_31):
    return ad.AnnData(X=X_numpy_33, obs=obs_31, var=var_31)


@pytest.fixture
def variable_type_samples():
    column_types = {
        "float_column": np.array([1.1, 1.2, 1.3, 2.1]),
        "float_column_with_missing": np.array([1.1, np.nan, 1.3, 2.1]),
        "int_column": np.array([1, 2, 3, 4]),
        "int_column_with_missing": np.array([1, np.nan, 3, 4]),
        "int_column_irregular": np.array([1, 2, 5, 6]),
        "string_column": np.array(["a", "b", "c", "d"]),
        "string_column_with_missing": np.array(["a", np.nan, "c", "d"]),
        "string_column_with_missing_strings": np.array(["a", "np.nan", "nan", "d"]),
        "bool_column_TrueFalse": np.array([True, False, True, False]),
        "bool_column_01": np.array([1, 0, 1, 0]),
        "bool_column_with_missing": np.array([True, np.nan, True, False]),
    }

    target_types = {
        "float_column": "numeric",
        "float_column_with_missing": "numeric",
        "int_column": "numeric",
        "int_column_with_missing": "numeric",
        "int_column_irregular": "numeric",
        "string_column": "categorical",
        "string_column_with_missing": "categorical",
        "string_column_with_missing_strings": "categorical",
        "bool_column_TrueFalse": "categorical",
        "bool_column_01": "categorical",
        "bool_column_with_missing": "categorical",
    }
    return column_types, target_types


@pytest.fixture
def variable_type_samples_string_format(variable_type_samples):
    # cast entries with .astype(str)
    data, target_types = variable_type_samples
    for key, value in data.items():
        data[key] = value.astype(str)
    return data, target_types


@pytest.fixture
def edata_nonnumeric_missing_330(obs_31, var_31):
    # create X of dtype object - np would create a string array
    X = pd.DataFrame(
        [
            [3, "E10", 12.1],
            [np.nan, "E11", 13.2],
            [14, np.nan, 12.5],
        ]
    ).to_numpy()
    return EHRData(X=X, layers={"other_layer": X}, obs=obs_31, var=var_31)


@pytest.fixture
def edata_basic_with_tem_full():
    edata_basic_with_tem_dict = {
        "X": np.ones((5, 4)),
        "R": np.ones((5, 4, 2)),
        "obs": pd.DataFrame({"survival": [1, 2, 3, 4, 5]}),
        "var": pd.DataFrame({"variables": ["var_1", "var_2", "var_3", "var_4"]}),
        "obsm": {"obs_level_representation": np.ones((5, 2))},
        "varm": {"var_level_representation": np.ones((4, 2))},
        "layers": {"other_layer": np.ones((5, 4))},
        "obsp": {"obs_level_connectivities": np.ones((5, 5))},
        "varp": {"var_level_connectivities": np.random.randn(4, 4)},
        "uns": {"information": ["info1"]},
        "tem": pd.DataFrame({"timestep": ["t1", "t2"]}),
    }
    return EHRData(**edata_basic_with_tem_dict)


@pytest.fixture
def omop_connection_vanilla():
    con = duckdb.connect()
    setup_connection(path="tests/data/toy_omop/vanilla", backend_handle=con)
    yield con
    con.close()


@pytest.fixture
def omop_connection_capital_letters():
    con = duckdb.connect()
    setup_connection(path="tests/data/toy_omop/capital_letters", backend_handle=con)
    yield con
    con.close()


@pytest.fixture
def omop_connection_empty_observation():
    con = duckdb.connect()
    setup_connection(path="tests/data/toy_omop/empty_observation", backend_handle=con)
    yield con
    con.close()


@pytest.fixture
def omop_connection_multiple_units():
    con = duckdb.connect()
    setup_connection(path="tests/data/toy_omop/multiple_units", backend_handle=con)
    yield con
    con.close()


def _assert_io_read(edata: EHRData):
    """Assert the test zarr and h5ad files are read correctly."""
    assert "survival" in edata.obs.columns
    assert all(edata.obs["survival"].values == [1, 2, 3, 4, 5])
    assert "variables" in edata.var.columns
    assert all(edata.var["variables"].values == ["var_1", "var_2", "var_3", "var_4"])
    assert "obs_level_representation" in edata.obsm
    assert edata.obsm["obs_level_representation"].shape == (5, 2)
    assert "var_level_representation" in edata.varm
    assert edata.varm["var_level_representation"].shape == (4, 2)
    # shapes are enforced by AnnData/EHRData for the below, no need to test
    assert "other_layer" in edata.layers
    assert "obs_level_connectivities" in edata.obsp
    assert "var_level_connectivities" in edata.varp
    assert "information" in edata.uns


TEST_DATA_PATH = Path(__file__).parent / "data"
