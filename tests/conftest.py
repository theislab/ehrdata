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
def t_11():
    return pd.DataFrame({"t_col_1": [1]}, index=["t1"])


@pytest.fixture
def t_21():
    return pd.DataFrame({"t_col_1": [1, 2]}, index=["t1", "t2"])


@pytest.fixture
def t_31():
    return pd.DataFrame({"t_col_1": [1, 2, 3]}, index=["t1", "t2", "t3"])


@pytest.fixture
def edata_333(X_numpy_33, R_numpy_333, obs_31, var_31, t_31):
    return EHRData(X=X_numpy_33, R=R_numpy_333, obs=obs_31, var=var_31, t=t_31)


@pytest.fixture
def adata_33(X_numpy_33, obs_31, var_31):
    return ad.AnnData(X=X_numpy_33, obs=obs_31, var=var_31)


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
