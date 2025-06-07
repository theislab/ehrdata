from pathlib import Path

import anndata as ad
import dask.array as da
import duckdb
import numpy as np
import pandas as pd
import pytest
import sparse as sp

from ehrdata import EHRData
from ehrdata.io.omop import setup_connection


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
    # duplicate entries with .astype(str)
    for key, value in column_types.items():
        column_types[key + "_str"] = value.astype(str)
    return column_types


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
    return EHRData(X=X, obs=obs_31, var=var_31)


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


TEST_DATA_PATH = Path(__file__).parent / "data"
