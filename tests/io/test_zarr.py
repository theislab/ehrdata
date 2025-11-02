import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import issparse
from tests.conftest import (
    TEST_DATA_PATH,
    _assert_dtype_object_array_with_missing_values_equal,
    _assert_io_read,
    _assert_shape_matches,
)

from ehrdata.core.constants import EHRDATA_ZARR_ENCODING_VERSION
from ehrdata.io import read_zarr, write_zarr

TEST_PATH_ZARR = TEST_DATA_PATH / "toy_zarr"


@pytest.mark.parametrize("harmonize_missing_values", [False, True])
@pytest.mark.parametrize("cast_variables_to_float", [False, True])
def test_read_anndata_zarr_basic(harmonize_missing_values, cast_variables_to_float):
    edata = read_zarr(
        filename=TEST_PATH_ZARR / "adata_basic.zarr",
        harmonize_missing_values=harmonize_missing_values,
        cast_variables_to_float=cast_variables_to_float,
    )

    store = zarr.open(TEST_PATH_ZARR / "adata_basic.zarr")

    # check the store is an anndata zarr store
    assert store.attrs["encoding-type"] == "anndata"

    _assert_shape_matches(edata, (5, 4, 1))
    _assert_io_read(edata)


@pytest.mark.parametrize("harmonize_missing_values", [False, True])
@pytest.mark.parametrize("cast_variables_to_float", [False, True])
def test_read_zarr_basic_with_tem(harmonize_missing_values, cast_variables_to_float):
    edata = read_zarr(
        filename=TEST_PATH_ZARR / "edata_basic_with_tem.zarr",
        harmonize_missing_values=harmonize_missing_values,
        cast_variables_to_float=cast_variables_to_float,
    )
    _assert_shape_matches(edata, (5, 4, 2))
    _assert_io_read(edata)

    assert "timestep" in edata.tem.columns
    assert all(edata.tem["timestep"].values == ["t1", "t2"])

    # check the store is an ehrdata zarr store
    store = zarr.open(TEST_PATH_ZARR / "edata_basic_with_tem.zarr")
    assert store.attrs["encoding-type"] == "ehrdata"
    assert store.attrs["encoding-version"] == EHRDATA_ZARR_ENCODING_VERSION


@pytest.mark.parametrize("harmonize_missing_values", [False, True])
@pytest.mark.parametrize("cast_variables_to_float", [False, True])
def test_read_zarr_sparse_with_tem(harmonize_missing_values, cast_variables_to_float):
    edata = read_zarr(
        filename=TEST_PATH_ZARR / "edata_sparse_with_tem.zarr",
        harmonize_missing_values=harmonize_missing_values,
        cast_variables_to_float=cast_variables_to_float,
    )
    _assert_shape_matches(edata, (5, 4, 2))
    _assert_io_read(edata)

    assert "timestep" in edata.tem.columns
    assert all(edata.tem["timestep"].values == ["t1", "t2"])

    assert issparse(edata.X)
    assert issparse(edata.layers["other_layer"])

    # check the store is an ehrdata zarr store
    store = zarr.open(TEST_PATH_ZARR / "edata_sparse_with_tem.zarr")
    assert store.attrs["encoding-type"] == "ehrdata"
    assert store.attrs["encoding-version"] == EHRDATA_ZARR_ENCODING_VERSION


@pytest.mark.parametrize(
    "edata_name",
    [
        "edata_330",
        "edata_333",
        "edata_333_larger_obs_var_tem",
        "edata_basic_with_tem_full",
        "edata_nonnumeric_missing_330",
    ],
)
def test_write_read_zarr_basic(edata_name, request, tmp_path):
    edata = request.getfixturevalue(edata_name)
    store_path = tmp_path / f"{edata_name}.zarr"

    write_zarr(edata.copy(), store_path)
    edata_read = read_zarr(store_path)

    assert edata.shape == edata_read.shape

    _assert_dtype_object_array_with_missing_values_equal(edata.X, edata_read.X)
    for key in edata.layers:
        _assert_dtype_object_array_with_missing_values_equal(edata.layers[key], edata_read.layers[key])

    pd.testing.assert_frame_equal(edata.obs.iloc[:, :1], edata_read.obs.iloc[:, :1])
    pd.testing.assert_frame_equal(edata.var.iloc[:, :1], edata_read.var.iloc[:, :1])
    pd.testing.assert_frame_equal(edata.tem.iloc[:, :1], edata_read.tem.iloc[:, :1])
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

    # check the test file is an ehrdata zarr store
    store = zarr.open(store_path)
    assert store.attrs["encoding-type"] == "ehrdata"
    assert store.attrs["encoding-version"] == EHRDATA_ZARR_ENCODING_VERSION

    # check success of convert_strings_to_categoricals
    if "obs_col_2" in edata_read.obs.columns:
        assert edata_read.obs["obs_col_2"].dtype == "category"
    if "var_col_2" in edata_read.var.columns:
        assert edata_read.var["var_col_2"].dtype == "category"
    if "tem_col_2" in edata_read.tem.columns:
        assert edata_read.tem["tem_col_2"].dtype == "category"
