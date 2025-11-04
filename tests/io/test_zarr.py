import anndata as ad
import pandas as pd
import pytest
import zarr
from scipy.sparse import issparse
from tests.conftest import (
    TEST_DATA_PATH,
    _assert_dtype_object_array_with_missing_values_equal,
    _assert_io_read,
    _assert_shape_matches,
    _check_aligned_anndata_parts_equal,
)

from ehrdata.core.constants import EHRDATA_ZARR_ENCODING_VERSION
from ehrdata.io import read_zarr, write_zarr

TEST_PATH_ZARR = TEST_DATA_PATH / "toy_zarr"


@pytest.mark.parametrize("harmonize_missing_values", [False, True])
@pytest.mark.parametrize("cast_variables_to_float", [False, True])
def test_read_zarr_anndata_store_as_ehrdata(harmonize_missing_values, cast_variables_to_float):
    # ehrdata should be able to read data from a regular anndata zarr store
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
@pytest.mark.parametrize("chunks", ["auto", "ehrdata_auto"])
def test_write_zarr_anndata_subgroup_of_ehrdata_store(edata_name, chunks, request, tmp_path):
    # ehrdata's write should create a regular anndata group
    # this test uses ad.io.read_zarr as a sanity check to ensure the anndata subgroup is written properly
    edata = request.getfixturevalue(edata_name)
    store_path = tmp_path / f"{edata_name}.zarr"

    write_zarr(edata.copy(), store_path, chunks=chunks)
    adata = ad.io.read_zarr(store_path / "anndata")
    _check_aligned_anndata_parts_equal(edata, adata)


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
@pytest.mark.parametrize("chunks", ["auto", "ehrdata_auto"])
def test_write_read_zarr_basic(edata_name, chunks, request, tmp_path):
    edata = request.getfixturevalue(edata_name)
    store_path = tmp_path / f"{edata_name}.zarr"

    write_zarr(edata.copy(), store_path, chunks=chunks)
    edata_read = read_zarr(store_path)

    assert edata.shape == edata_read.shape

    _assert_dtype_object_array_with_missing_values_equal(edata.X, edata_read.X)
    for key in edata.layers:
        _assert_dtype_object_array_with_missing_values_equal(edata.layers[key], edata_read.layers[key])

    _check_aligned_anndata_parts_equal(edata, edata_read)
    pd.testing.assert_frame_equal(edata.tem.iloc[:, :1], edata_read.tem.iloc[:, :1])

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


def test_write_zarr_chunks_error(edata_333, tmp_path):
    with pytest.raises(NotImplementedError):
        write_zarr(edata_333, tmp_path / "test.zarr", chunks=None)
    with pytest.raises(NotImplementedError):
        write_zarr(edata_333, tmp_path / "test.zarr", chunks=1000)
    with pytest.raises(NotImplementedError):
        write_zarr(edata_333, tmp_path / "test.zarr", chunks="foobar")
