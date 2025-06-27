import anndata as ad
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

from ehrdata.io import read_zarr, write_zarr

TEST_PATH_ZARR = TEST_DATA_PATH / "toy_zarr"


@pytest.mark.parametrize("harmonize_missing_values", [False, True])
@pytest.mark.parametrize("cast_variables_to_float", [False, True])
def test_read_zarr_basic(harmonize_missing_values, cast_variables_to_float):
    edata = read_zarr(
        filename=TEST_PATH_ZARR / "adata_basic.zarr",
        harmonize_missing_values=harmonize_missing_values,
        cast_variables_to_float=cast_variables_to_float,
    )

    _assert_shape_matches(edata, (5, 4, 0), check_R_None=True)
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


@pytest.mark.parametrize("edata_name", ["edata_333", "edata_basic_with_tem_full", "edata_nonnumeric_missing_330"])
def test_write_zarr_basic(edata_name, request, tmp_path):
    edata = request.getfixturevalue(edata_name)

    write_zarr(edata, f"{tmp_path}/{edata_name}.zarr")

    zarr_file = zarr.open(f"{tmp_path}/{edata_name}.zarr", mode="r")
    # Note that R is not included in the list because it is just a value of the .layers field
    assert set(dict(zarr_file).keys()) == {
        "X",
        "obs",
        "var",
        "obsm",
        "varm",
        "layers",
        "obsp",
        "varp",
        "uns",
        "tem",
    }

    assert np.array_equal(ad.io.read_elem(zarr_file["X"]).astype(str), edata.X.astype(str))

    pd.testing.assert_frame_equal(ad.io.read_elem(zarr_file["obs"]), edata.obs)
    pd.testing.assert_frame_equal(ad.io.read_elem(zarr_file["var"]), edata.var)
    pd.testing.assert_frame_equal(ad.io.read_elem(zarr_file["tem"]), edata.tem)
    for key in edata.obsm:
        assert key in ad.io.read_elem(zarr_file["obsm"])
        assert np.array_equal(ad.io.read_elem(zarr_file["obsm"][key]), edata.obsm[key])
    for key in edata.varm:
        assert key in ad.io.read_elem(zarr_file["varm"])
        assert np.array_equal(ad.io.read_elem(zarr_file["varm"][key]), edata.varm[key])
    for key in edata.obsp:
        assert key in ad.io.read_elem(zarr_file["obsp"])
        assert np.array_equal(ad.io.read_elem(zarr_file["obsp"][key]), edata.obsp[key])
    for key in edata.varp:
        assert key in ad.io.read_elem(zarr_file["varp"])
        assert np.array_equal(ad.io.read_elem(zarr_file["varp"][key]), edata.varp[key])
    for key in edata.uns:
        assert key in ad.io.read_elem(zarr_file["uns"])


@pytest.mark.parametrize("edata_name", ["edata_333", "edata_basic_with_tem_full", "edata_nonnumeric_missing_330"])
def test_write_read_zarr_basic(edata_name, request, tmp_path):
    edata = request.getfixturevalue(edata_name)
    write_zarr(edata.copy(), f"{tmp_path}/{edata_name}.zarr")
    edata_read = read_zarr(f"{tmp_path}/{edata_name}.zarr")

    assert edata.shape == edata_read.shape

    _assert_dtype_object_array_with_missing_values_equal(edata.X, edata_read.X)
    for key in edata.layers:
        _assert_dtype_object_array_with_missing_values_equal(edata.layers[key], edata_read.layers[key])

    pd.testing.assert_frame_equal(edata.obs, edata_read.obs)
    pd.testing.assert_frame_equal(edata.var, edata_read.var)
    pd.testing.assert_frame_equal(edata.tem, edata_read.tem)
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
