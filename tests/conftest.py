import datetime as dt
from pathlib import Path

import anndata as ad
import dask.array as da
import duckdb
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import sparse

from ehrdata import EHRData
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from ehrdata.dt import mimic_iv_omop
from ehrdata.io.omop import setup_connection


def _assert_shape_matches(
    edata: EHRData,
    shape: tuple[int, int, int],
    *,
    check_X_None: bool = False,
):
    assert edata.shape == shape

    if check_X_None:
        assert edata.X is None
    else:
        # X may be 2D (n_obs, n_vars) or 3D (n_obs, n_vars, n_t); tolerate both like layers below.
        assert edata.X.shape in [shape, shape[0:2], (shape[0], shape[1], 1)]

    assert isinstance(edata.obs, pd.DataFrame)
    assert len(edata.obs) == shape[0]
    assert edata.n_obs == shape[0]

    assert isinstance(edata.var, pd.DataFrame)
    assert len(edata.var) == shape[1]
    assert edata.n_vars == shape[1]

    assert isinstance(edata.tem, pd.DataFrame)
    if len(shape) > 2:
        assert len(edata.tem) == shape[2]
        assert edata.n_t == shape[2]

    for key in edata.layers:
        assert edata.layers[key].shape in [shape, shape[0:2], (shape[0], shape[1], 1)]


def _assert_dtype_object_array_with_missing_values_equal(a: np.ndarray, b: np.ndarray):
    # need to use pd.isnull to check for np.isnan in dtype object arrays, because np.isnan does not work on dtype object array
    a = a.copy()
    b = b.copy()
    assert np.array_equal(pd.isnull(a), pd.isnull(b))
    # if verified equal position of nan values, replace with 0 and verify the rest of the entries are equal
    a[pd.isnull(a)] = 0
    b[pd.isnull(b)] = 0
    assert np.array_equal(a, b)


def _object_column(values: list) -> pd.Series:
    # pandas 3 infers strings in an `object` array as `str`, while a database read can leave them behind as `object`
    return pd.Series(values, dtype=object, index=["0", "1", "2"])


def _edata_with_columns_anndata_cannot_write(X) -> EHRData:
    """An EHRData with `X` of shape `(3, 2, 2)` whose `.obs`, `.var` and `.tem` hold the column dtypes anndata cannot write.

    Datetime and timedelta columns, and `object` columns holding something else than only strings: an all-null column
    as a database read leaves it behind, integers, booleans, floats and strings next to a missing value, and values of
    mixed types. Columns anndata writes itself sit next to them.
    """
    obs = pd.DataFrame(
        {
            "birth_datetime": pd.to_datetime(["2000-01-01 01:02:03", None, "2001-02-03 04:05:06"]),
            # either side of the switch to daylight saving time, so with two offsets
            "visit_start_datetime": pd.to_datetime(["2000-03-26 01:00", "2000-03-26 03:00", None]).tz_localize(
                "Europe/Berlin"
            ),
            "length_of_stay": pd.to_timedelta(["1 days 02:00:00", None, "3h"]),
            "location_id": _object_column([None, None, None]),
            "care_site_id": _object_column([2**53 + 1, None, 3]),
            "is_deceased": _object_column([True, None, False]),
            "value_as_number": _object_column([1.5, None, 2]),
            "gender_source_value": _object_column(["F", None, "M"]),
            "person_source_value": _object_column(["p1", "p2", "p3"]),
            "race_source_value": _object_column(["a", "a", "b"]),
            "death_date": _object_column([dt.date(2001, 1, 1), None, "unknown"]),
            "concept_code": _object_column(["007", 2, None]),
            "huge_id": _object_column([2**63, None, 1]),
            "duration": _object_column([np.timedelta64(1, "h"), None, np.timedelta64(30, "m")]),
            "weight": pd.array([70.5, None, 80.0], dtype="Float64"),
            "year_of_birth": [2000, 1990, 2001],
        },
        index=["0", "1", "2"],
    )
    var = pd.DataFrame({"valid_start_date": pd.to_datetime(["1970-01-01", "1970-01-02"])}, index=["0", "1"])
    tem = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2000-01-01", "2000-01-02"]), "phase": pd.Series(["x", "x"], dtype=object)}
    )

    return EHRData(X=X, obs=obs, var=var, tem=tem)


def _values(column: pd.Series) -> list:
    # missing values as None, however the dtype read back marks them
    return [None if pd.isna(value) else value for value in column.astype(object)]


def _assert_columns_anndata_cannot_write_read_back(edata: EHRData, edata_read: EHRData):
    obs = edata_read.obs
    assert _values(obs["birth_datetime"]) == ["2000-01-01T01:02:03", None, "2001-02-03T04:05:06"]
    assert _values(obs["visit_start_datetime"]) == ["2000-03-26T01:00:00+01:00", "2000-03-26T03:00:00+02:00", None]
    assert _values(obs["length_of_stay"]) == ["P1DT2H0M0S", None, "P0DT3H0M0S"]
    assert obs["location_id"].isna().all()
    # integers and booleans stay exact, beyond the integers a float64 holds
    assert obs["care_site_id"].dtype == "Int64"
    assert _values(obs["care_site_id"]) == [2**53 + 1, None, 3]
    assert obs["is_deceased"].dtype == "boolean"
    assert _values(obs["is_deceased"]) == [True, None, False]
    assert obs["value_as_number"].dtype == np.float64
    assert _values(obs["value_as_number"]) == [1.5, None, 2.0]
    assert _values(obs["gender_source_value"]) == ["F", None, "M"]
    assert _values(obs["person_source_value"]) == ["p1", "p2", "p3"]
    assert _values(obs["race_source_value"]) == ["a", "a", "b"]
    assert _values(obs["death_date"]) == ["2001-01-01", None, "unknown"]
    # strings next to other values stay strings, keeping leading zeros, as do integers no numeric dtype holds exactly
    assert _values(obs["concept_code"]) == ["007", "2", None]
    assert _values(obs["huge_id"]) == [str(2**63), None, "1"]
    assert _values(obs["duration"]) == ["P0DT1H0M0S", None, "P0DT0H30M0S"]
    assert obs["weight"].dtype == np.float64
    assert _values(obs["weight"]) == [70.5, None, 80.0]
    assert obs["year_of_birth"].tolist() == [2000, 1990, 2001]
    assert _values(edata_read.var["valid_start_date"]) == ["1970-01-01T00:00:00", "1970-01-02T00:00:00"]
    assert _values(edata_read.tem["timestamp"]) == ["2000-01-01T00:00:00", "2000-01-02T00:00:00"]
    assert _values(edata_read.tem["phase"]) == ["x", "x"]

    assert type(edata_read.X) is type(edata.X)
    assert np.array_equal(_to_dense(edata_read.X), _to_dense(edata.X), equal_nan=True)

    # the object that was written is left as it was
    assert pd.api.types.is_datetime64_any_dtype(edata.obs["birth_datetime"])
    assert pd.api.types.is_timedelta64_dtype(edata.obs["length_of_stay"])
    assert edata.obs["care_site_id"].dtype == object
    assert edata.obs["race_source_value"].dtype == object
    assert edata.tem["phase"].dtype == object


def _to_dense(X) -> np.ndarray:
    if isinstance(X, sparse.COO):
        return X.todense()
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


@pytest.fixture(params=["numpy_3d", "sparse_coo_3d", "scipy_csr_2d"])
def X_any_array_type_322(request):
    """An `X` of shape `(3, 2, 2)`, or `(3, 2)` as a 2D scipy sparse array, in each array type ehrdata writes."""
    X = np.arange(3 * 2 * 2, dtype=np.float64).reshape(3, 2, 2)
    # in place of the only zero, so that no sparse array holds an implicit zero harmonize_missing_values turns into np.nan
    X[0, 0, 0] = np.nan
    return {
        "numpy_3d": X,
        "sparse_coo_3d": sparse.COO.from_numpy(X),
        "scipy_csr_2d": sp.csr_matrix(X[:, :, 1]),
    }[request.param]


def _edata_with_object_layers(X) -> EHRData:
    """An EHRData with `object` layers of shape `(3, 2, 2)` next to `X`, which can be `None`.

    The layer `numbers` holds integers and a missing value, the layer `strings` holds a variable of numbers as strings
    and a variable of strings, both with a missing value.
    """
    numbers = np.array([[[1, None], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], dtype=object)
    strings = np.array([[["1", None], ["a", "b"]], [["2", "3"], ["c", None]], [["4", "5"], ["d", "e"]]], dtype=object)
    return EHRData(
        X=X,
        layers={"numbers": numbers, "strings": strings},
        obs=pd.DataFrame(index=["0", "1", "2"]),
        var=pd.DataFrame(index=["0", "1"]),
        tem=pd.DataFrame(index=["0", "1"]),
    )


def _assert_object_layers_read_back(edata: EHRData, edata_read: EHRData):
    assert np.array_equal(
        edata_read.layers["numbers"].astype(np.float64),
        np.array([[[1, np.nan], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]),
        equal_nan=True,
    )
    # each variable is cast to float on read where it can be, and its missing values are np.nan again
    strings = edata_read.layers["strings"]
    assert np.array_equal(strings[:, 0, :].astype(np.float64), np.array([[1, np.nan], [2, 3], [4, 5]]), equal_nan=True)
    _assert_dtype_object_array_with_missing_values_equal(
        strings[:, 1, :].astype(object), np.array([["a", "b"], ["c", np.nan], ["d", "e"]], dtype=object)
    )

    if edata.X is None:
        assert edata_read.X is None
    else:
        assert type(edata_read.X) is type(edata.X)
        assert np.array_equal(_to_dense(edata_read.X), _to_dense(edata.X), equal_nan=True)

    # the object that was written is left as it was
    assert edata.layers["numbers"].dtype == object
    assert edata.layers["strings"].dtype == object


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
def X_numpy_322():
    return np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])


@pytest.fixture
def X_sparse_322():
    return sparse.COO.from_numpy(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]))


@pytest.fixture
def X_dask_322():
    return da.from_array(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]), chunks=(3, 2, 2))


@pytest.fixture
def X_numpy_333():
    return np.arange(1, 28).reshape(3, 3, 3)


@pytest.fixture
def obs_31():
    return pd.DataFrame({"obs_col_1": [1, 2, 3]}, index=["obs1", "obs2", "obs3"])


@pytest.fixture
def obs_32():
    df = pd.DataFrame(
        {
            "obs_col_1": [1, 2, 3],
            "obs_col_2": ["a", "a", "c"],
        },
        index=["obs1", "obs2", "obs3"],
    )
    df = df.astype({"obs_col_2": "string"})
    return df


@pytest.fixture
def var_21():
    return pd.DataFrame({"var_col_1": [1, 2]}, index=["var1", "var2"])


@pytest.fixture
def var_31():
    return pd.DataFrame({"var_col_1": [1, 2, 3]}, index=["var1", "var2", "var3"])


@pytest.fixture
def var_32():
    df = pd.DataFrame(
        {
            "var_col_1": [1, 2, 3],
            "var_col_2": ["x", "x", "z"],
        },
        index=["var1", "var2", "var3"],
    )
    df = df.astype({"var_col_2": "string"})
    return df


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
def tem_32():
    df = pd.DataFrame(
        {
            "tem_col_1": [1, 2, 3],
            "tem_col_2": ["l", "l", "n"],
        },
        index=["t1", "t2", "t3"],
    )
    df = df.astype({"tem_col_2": "string"})
    return df


@pytest.fixture
def edata_333(X_numpy_33, X_numpy_333, obs_31, var_31, tem_31):
    return EHRData(X=X_numpy_33, layers={DEFAULT_TEM_LAYER_NAME: X_numpy_333}, obs=obs_31, var=var_31, tem=tem_31)


@pytest.fixture(params=["layer", "X"])
def edata_3d_slot(request, X_numpy_33, X_numpy_333, obs_31, var_31, tem_31):
    """A (3, 3, 3) EHRData with the 3D tensor in the layer or in ``.X``.

    Parametrizes the slicing battery over the tensor slot so a 3D ``.X`` gets the same
    coverage as a 3D layer. Both variants keep a valid ``.X`` (3D in the X case, 2D in the
    layer case). Returns ``(edata, slot)`` where ``slot`` is ``"layer"`` or ``"X"``.
    """
    slot = request.param
    if slot == "X":
        edata = EHRData(X=X_numpy_333, obs=obs_31, var=var_31, tem=tem_31)
    else:
        edata = EHRData(X=X_numpy_33, layers={DEFAULT_TEM_LAYER_NAME: X_numpy_333}, obs=obs_31, var=var_31, tem=tem_31)
    return edata, slot


@pytest.fixture
def edata_333_larger_obs_var_tem(X_numpy_33, X_numpy_333, obs_32, var_32, tem_32):
    return EHRData(X=X_numpy_33, layers={DEFAULT_TEM_LAYER_NAME: X_numpy_333}, obs=obs_32, var=var_32, tem=tem_32)


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
        "obs": pd.DataFrame({"survival": [1, 2, 3, 4, 5]}, index=["obs1", "obs2", "obs3", "obs4", "obs5"]),
        "var": pd.DataFrame(
            {"variables": ["var_1", "var_2", "var_3", "var_4"]}, index=["var1", "var2", "var3", "var4"]
        ),
        "obsm": {"obs_level_representation": np.ones((5, 2))},
        "varm": {"var_level_representation": np.ones((4, 2))},
        "layers": {DEFAULT_TEM_LAYER_NAME: np.ones((5, 4)), "other_layer": np.ones((5, 4))},
        "obsp": {"obs_level_connectivities": np.ones((5, 5))},
        "varp": {"var_level_connectivities": np.random.randn(4, 4)},
        "uns": {"information": ["info1"]},
        "tem": pd.DataFrame({"timestep": ["t1"]}, index=["t1"]),
    }
    return EHRData(**edata_basic_with_tem_dict)


@pytest.fixture
def edata_blobs_small():
    """Create EHRData with 3D layers for testing."""
    import ehrdata as ed

    return ed.dt.ehrdata_blobs(
        layer="tem_data",
        n_observations=20,
        n_variables=10,
        base_timepoints=5,
        n_centers=2,
        random_state=42,
    )


@pytest.fixture
def omop_connection_vanilla():
    con = duckdb.connect()
    setup_connection(path="tests/data/toy_omop/vanilla", backend_handle=con)
    yield con
    con.close()


@pytest.fixture
def omop_connection_vanilla_parquet():
    con = duckdb.connect()
    setup_connection(path="tests/data/toy_omop/vanilla_parquet", backend_handle=con)
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


@pytest.fixture
def omop_connection_multiple_visit_occurrences():
    con = duckdb.connect()
    setup_connection(path="tests/data/toy_omop/multiple_visit_occurrences", backend_handle=con)
    yield con
    con.close()


@pytest.fixture
def omop_connection_mimic_iv():
    con = duckdb.connect(":memory:")
    # Use the default ehrapy_data/ path (shared with test_dt) so the already-seeded CI cache covers it; tmp_path would re-download from physionet every run.
    mimic_iv_omop(backend_handle=con)
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


def _check_aligned_anndata_parts_equal(edata: EHRData, edata_read: EHRData | ad.AnnData):
    pd.testing.assert_frame_equal(edata.obs.iloc[:, :1], edata_read.obs.iloc[:, :1])
    pd.testing.assert_frame_equal(edata.var.iloc[:, :1], edata_read.var.iloc[:, :1])

    for key in edata.obsm:
        assert key in edata_read.obsm
        assert np.array_equal(edata.obsm[key], edata_read.obsm[key])
    for key in edata.varm:
        assert key in edata_read.varm
        assert np.array_equal(edata.varm[key], edata_read.varm[key])
    for key in edata.obsp:
        assert key in edata_read.obsp
        assert np.array_equal(edata.obsp[key], edata_read.obsp[key])
    for key in edata.varp:
        assert key in edata_read.varp
        assert np.array_equal(edata.varp[key], edata_read.varp[key])
    for key in edata.uns:
        assert key in edata_read.uns
        assert np.array_equal(edata.uns[key], edata_read.uns[key])
