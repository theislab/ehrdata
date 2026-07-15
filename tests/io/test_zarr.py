import anndata as ad
import numpy as np
import pandas as pd
import pytest
import sparse
import zarr
from scipy.sparse import issparse
from tests.conftest import (
    _ANNDATA_ALLOWS_COO,
    _ANNDATA_ALLOWS_ND_X,
    TEST_DATA_PATH,
    _assert_dtype_object_array_with_missing_values_equal,
    _assert_io_read,
    _assert_shape_matches,
    _check_aligned_anndata_parts_equal,
)

from ehrdata.core.constants import EHRDATA_ONDISK_VERSION
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
    # legacy committed fixture: written with the old "0.0.1" encoding-version, still readable
    assert store.attrs["encoding-version"] == "0.0.1"


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
    # legacy committed fixture: written with the old "0.0.1" encoding-version, still readable
    assert store.attrs["encoding-version"] == "0.0.1"


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
    assert store.attrs["ehrdata-encoding-version"] == EHRDATA_ONDISK_VERSION

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


def test_write_zarr_v2_relocates_3d_arrays_to_obsm(edata_333, tmp_path):

    # ehrdata writes the v2 layout: 3D arrays move into .obsm so anndata's 2D-only spec is respected.
    path = tmp_path / "edata_333.ehrdata.zarr"
    write_zarr(edata_333.copy(), path)

    store = zarr.open(path)
    assert "_ed_ondisk_layers_tem_data" in store["anndata"]["obsm"]
    assert "tem_data" not in store["anndata"]["layers"]
    assert store.attrs["encoding-type"] == "ehrdata"
    assert store.attrs["ehrdata-encoding-version"] == str(EHRDATA_ONDISK_VERSION)

    edata_read = read_zarr(path)
    _assert_shape_matches(edata_read, (3, 3, 3))
    for key in edata_333.layers:
        assert np.array_equal(edata_333.layers[key], edata_read.layers[key])
    assert not any(k.startswith("_ed_ondisk_") for k in edata_read.obsm)


def test_write_read_zarr_X_none_with_3d_layer(edata_330, tmp_path):

    # X=None is a first-class state (encode_for_disk drops a None X); reading it with harmonization on (the default) must not crash on the None X.
    edata = edata_330.copy()
    edata.layers["tem_data"] = np.arange(3 * 3 * 3).reshape(3, 3, 3).astype(float)
    edata.X = None

    path = tmp_path / "edata_X_none.ehrdata.zarr"
    write_zarr(edata.copy(), path)

    edata_read = read_zarr(path)  # harmonize_missing_values=True by default
    _assert_shape_matches(edata_read, (3, 3, 3), check_X_None=True)
    assert edata_read.X is None
    assert np.array_equal(edata.layers["tem_data"], edata_read.layers["tem_data"])
    assert not any(k.startswith("_ed_ondisk_") for k in edata_read.obsm)


@pytest.mark.skipif(not _ANNDATA_ALLOWS_ND_X, reason="anndata <0.13 does not allow a >2D X in memory")
def test_write_read_zarr_3d_X_relocated_to_obsm(tmp_path):

    from ehrdata import EHRData

    # a 3D X is relocated to the reserved `_ed_ondisk_X` obsm key on write and restored on read, without leaking the unified-X None key in as a spurious "layer".
    X3 = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)
    path = tmp_path / "edata_3dX.ehrdata.zarr"
    write_zarr(EHRData(X=X3), path)

    store = zarr.open(path)
    assert "_ed_ondisk_X" in store["anndata"]["obsm"]
    assert not any(k.startswith("_ed_ondisk_layers_") for k in store["anndata"]["obsm"])

    edata_read = read_zarr(path, harmonize_missing_values=False, cast_variables_to_float=False)
    assert edata_read.shape == (2, 3, 4)
    assert np.array_equal(np.asarray(edata_read.X), X3)
    assert [k for k in edata_read.layers if k is not None] == []


@pytest.mark.skipif(not _ANNDATA_ALLOWS_COO, reason="anndata <0.13.1 rejects sparse.COO in memory")
@pytest.mark.parametrize("chunks", ["auto", "ehrdata_auto"])
@pytest.mark.parametrize("slot", ["X", "layer"])
def test_write_read_zarr_sparse_coo_3d(slot, chunks, tmp_path):
    from ehrdata import EHRData

    dense = np.zeros((3, 2, 4))
    dense[0, 0, 1] = 5.0
    dense[2, 1, 3] = 7.0
    coo = sparse.COO.from_numpy(dense)

    if slot == "X":
        edata = EHRData(X=coo)
        obsm_key = "_ed_ondisk_X"
    else:
        edata = EHRData(X=np.zeros((3, 2)), layers={"tem_data": coo})
        obsm_key = "_ed_ondisk_layers_tem_data"

    path = tmp_path / f"coo_{slot}_{chunks}.ehrdata.zarr"
    write_zarr(edata.copy(), path, chunks=chunks)

    store = zarr.open(path)
    assert store["anndata"]["obsm"][obsm_key].attrs["encoding-type"] == "ehrdata-coo"

    edata_read = read_zarr(path)
    restored = edata_read.X if slot == "X" else edata_read.layers["tem_data"]
    assert isinstance(restored, sparse.COO)
    assert np.array_equal(restored.todense(), dense)


def test_read_minimal_corpus_zarr():
    # Minimal read-test corpus (zarr half): a "version 0" plain-anndata store (no ehrdata stamp)
    # and a committed 0.2.0 ehrdata store (relocated 3D layer + stamp) both read correctly.
    v0 = zarr.open(TEST_PATH_ZARR / "adata_basic.zarr")
    assert "ehrdata-encoding-version" not in v0.attrs
    edata_v0 = read_zarr(TEST_PATH_ZARR / "adata_basic.zarr")
    _assert_shape_matches(edata_v0, (5, 4, 1))

    store_020 = zarr.open(TEST_PATH_ZARR / "edata_minimal_v0_2_0.ehrdata.zarr")
    assert store_020.attrs["ehrdata-encoding-type"] == "ehrdata"
    assert store_020.attrs["ehrdata-encoding-version"] == str(EHRDATA_ONDISK_VERSION)
    edata_020 = read_zarr(TEST_PATH_ZARR / "edata_minimal_v0_2_0.ehrdata.zarr")
    _assert_shape_matches(edata_020, (3, 2, 2))
    assert np.array_equal(np.asarray(edata_020.layers["tem_data"]), np.arange(3 * 2 * 2, dtype=float).reshape(3, 2, 2))
