import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse
from tests.conftest import (
    TEST_DATA_PATH,
    _assert_dtype_object_array_with_missing_values_equal,
    _assert_io_read,
    _assert_shape_matches,
)

from ehrdata.io import read_h5ad, write_h5ad

TEST_PATH_H5AD = TEST_DATA_PATH / "toy_h5ad"


def test_read_h5ad_basic():
    edata = read_h5ad(filename=TEST_PATH_H5AD / "adata_basic.h5ad")

    _assert_shape_matches(edata, (5, 4, 0), check_R_None=True)
    _assert_io_read(edata)


def test_read_h5ad_basic_with_tem():
    edata = read_h5ad(filename=TEST_PATH_H5AD / "edata_basic_with_tem.h5ad")

    _assert_shape_matches(edata, (5, 4, 2))
    _assert_io_read(edata)

    assert "timestep" in edata.tem.columns
    assert all(edata.tem["timestep"].values == ["t1", "t2"])


def test_read_h5ad_sparse_with_tem():
    edata = read_h5ad(filename=TEST_PATH_H5AD / "edata_sparse_with_tem.h5ad")

    _assert_shape_matches(edata, (5, 4, 2))
    _assert_io_read(edata)

    assert "timestep" in edata.tem.columns
    assert all(edata.tem["timestep"].values == ["t1", "t2"])
    assert issparse(edata.X)
    assert issparse(edata.layers["other_layer"])


@pytest.mark.parametrize("edata_name", ["edata_333", "edata_basic_with_tem_full", "edata_nonnumeric_missing_330"])
def test_write_h5ad_basic(edata_name, request, tmp_path):
    edata = request.getfixturevalue(edata_name)

    write_h5ad(edata, f"{tmp_path}/{edata_name}.h5ad")

    with h5py.File(f"{tmp_path}/{edata_name}.h5ad", "r") as h5ad_file:
        # Note that R is not included in the list because it is just a value of the .layers field
        assert set(dict(h5ad_file).keys()) == {
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

        assert np.array_equal(ad.io.read_elem(h5ad_file["X"]), edata.X)

        pd.testing.assert_frame_equal(ad.io.read_elem(h5ad_file["obs"]), edata.obs)
        pd.testing.assert_frame_equal(ad.io.read_elem(h5ad_file["var"]), edata.var)
        pd.testing.assert_frame_equal(ad.io.read_elem(h5ad_file["tem"]), edata.tem)
        for key in edata.obsm:
            assert key in ad.io.read_elem(h5ad_file["obsm"])
            assert np.array_equal(ad.io.read_elem(h5ad_file["obsm"][key]), edata.obsm[key])
        for key in edata.varm:
            assert key in ad.io.read_elem(h5ad_file["varm"])
            assert np.array_equal(ad.io.read_elem(h5ad_file["varm"][key]), edata.varm[key])
        for key in edata.obsp:
            assert key in ad.io.read_elem(h5ad_file["obsp"])
            assert np.array_equal(ad.io.read_elem(h5ad_file["obsp"][key]), edata.obsp[key])
        for key in edata.varp:
            assert key in ad.io.read_elem(h5ad_file["varp"])
            assert np.array_equal(ad.io.read_elem(h5ad_file["varp"][key]), edata.varp[key])
        for key in edata.uns:
            assert key in ad.io.read_elem(h5ad_file["uns"])


@pytest.mark.parametrize("edata_name", ["edata_333", "edata_basic_with_tem_full", "edata_nonnumeric_missing_330"])
def test_write_read_h5ad_basic(edata_name, request, tmp_path):
    edata = request.getfixturevalue(edata_name)
    write_h5ad(edata.copy(), f"{tmp_path}/{edata_name}.h5ad")
    edata_read = read_h5ad(f"{tmp_path}/{edata_name}.h5ad")

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
