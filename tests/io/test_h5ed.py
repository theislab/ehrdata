import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy as sp
from scipy.sparse import issparse
from tests.conftest import (
    _ANNDATA_ALLOWS_ND_X,
    TEST_DATA_PATH,
    _assert_dtype_object_array_with_missing_values_equal,
    _assert_io_read,
    _assert_shape_matches,
)

from ehrdata.io import read_h5ed, write_h5ed

TEST_PATH_H5AD = TEST_DATA_PATH / "toy_h5ad"


# TODO: tests with X as None, tests with R as None?
@pytest.mark.parametrize(
    ("backed", "harmonize_missing_values", "cast_variables_to_float"),
    [(False, False, False), (False, False, True), (False, True, False), (True, False, False)],
)
def test_read_h5ed_basic(backed, harmonize_missing_values, cast_variables_to_float):
    edata = read_h5ed(
        filename=TEST_PATH_H5AD / "adata_basic.h5ad",
        backed=backed,
        harmonize_missing_values=harmonize_missing_values,
        cast_variables_to_float=cast_variables_to_float,
    )

    _assert_shape_matches(edata, (5, 4, 1))
    _assert_io_read(edata)


@pytest.mark.parametrize(
    ("backed", "harmonize_missing_values", "cast_variables_to_float"),
    [(False, False, False), (False, False, True), (False, True, False), (False, True, True), (True, False, False)],
)
def test_read_h5ed_basic_with_tem(backed, harmonize_missing_values, cast_variables_to_float):
    edata = read_h5ed(
        filename=TEST_PATH_H5AD / "edata_basic_with_tem.h5ad",
        backed=backed,
        harmonize_missing_values=harmonize_missing_values,
        cast_variables_to_float=cast_variables_to_float,
    )
    assert edata.isbacked == backed

    _assert_shape_matches(edata, (5, 4, 2))
    _assert_io_read(edata)

    assert "timestep" in edata.tem.columns
    assert all(edata.tem["timestep"].values == ["t1", "t2"])


@pytest.mark.parametrize(
    (
        "backed",
        "harmonize_missing_values",
        "cast_variables_to_float",
        "expected_type",
    ),
    [
        (False, False, False, sp.sparse.csr_matrix),
        (False, False, True, sp.sparse.csr_matrix),
        (False, True, False, sp.sparse.csr_matrix),
        (False, True, True, sp.sparse.csr_matrix),
        (True, False, False, ad._core.sparse_dataset._CSRDataset),
    ],
)
def test_read_h5ed_sparse_with_tem(backed, harmonize_missing_values, cast_variables_to_float, expected_type):
    edata = read_h5ed(
        filename=TEST_PATH_H5AD / "edata_sparse_with_tem.h5ad",
        backed=backed,
        harmonize_missing_values=harmonize_missing_values,
        cast_variables_to_float=cast_variables_to_float,
    )

    assert edata.isbacked == backed
    _assert_shape_matches(edata, (5, 4, 2))
    _assert_io_read(edata)

    assert "timestep" in edata.tem.columns
    assert all(edata.tem["timestep"].values == ["t1", "t2"])
    assert isinstance(edata.X, expected_type)
    assert issparse(edata.layers["other_layer"])


def test_read_h5ed_backed_harmonize_missing_values_error():
    with pytest.raises(ValueError):
        read_h5ed(
            filename=TEST_PATH_H5AD / "edata_basic_with_tem.h5ad",
            backed=True,
            harmonize_missing_values=True,
        )
    with pytest.raises(ValueError):
        read_h5ed(
            filename=TEST_PATH_H5AD / "edata_basic_with_tem.h5ad",
            backed=True,
            cast_variables_to_float=True,
        )


@pytest.mark.parametrize("edata_name", ["edata_333", "edata_basic_with_tem_full", "edata_nonnumeric_missing_330"])
def test_write_h5ed_basic(edata_name, request, tmp_path):
    edata = request.getfixturevalue(edata_name)

    write_h5ed(edata, f"{tmp_path}/{edata_name}.h5ed")

    with h5py.File(f"{tmp_path}/{edata_name}.h5ed", "r") as h5ad_file:
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

        assert np.array_equal(ad.io.read_elem(h5ad_file["X"]).astype(str), edata.X.astype(str))

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
def test_write_read_h5ed_basic(edata_name, request, tmp_path):
    edata = request.getfixturevalue(edata_name)
    write_h5ed(edata.copy(), f"{tmp_path}/{edata_name}.h5ed")
    edata_read = read_h5ed(f"{tmp_path}/{edata_name}.h5ed")

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


def test_write_h5ed_v2_relocates_3d_arrays_to_obsm(edata_333, tmp_path):
    # ehrdata writes the v2 layout: 3D arrays move into .obsm so anndata's 2D-only spec is respected.
    path = tmp_path / "edata_333.h5ed"
    write_h5ed(edata_333.copy(), path)

    with h5py.File(path, "r") as f:
        assert "_ed_ondisk_layers_tem_data" in f["obsm"]
        assert "tem_data" not in f["layers"]
        assert f.attrs["ehrdata-encoding-type"] == "ehrdata"
        assert f.attrs["ehrdata-encoding-version"] == "2"

    edata_read = read_h5ed(path)
    _assert_shape_matches(edata_read, (3, 3, 3))
    for key in edata_333.layers:
        assert np.array_equal(edata_333.layers[key], edata_read.layers[key])
    # the relocation must be invisible to the user-facing object
    assert not any(k.startswith("_ed_ondisk_") for k in edata_read.obsm)


def test_read_h5ed_legacy_v1_with_3d_in_layers(edata_333, tmp_path):
    # legacy v1 files store 3D arrays directly in layers, with no reserved obsm keys; they must still
    # read correctly via the self-describing layout (no reserved keys -> nothing to relocate).
    path = tmp_path / "legacy_v1.h5ad"
    edata = edata_333.copy()

    # anndata >=0.13 blocks writing a 3D array in `layers` through the high-level writer, so forge a
    # legacy v1 file by writing the 2D scaffold normally and injecting the 3D layer (and tem) with the
    # low-level `write_elem`, which bypasses the on-write 2D check.
    ad.AnnData(X=np.asarray(edata.X), obs=edata.obs.copy(), var=edata.var.copy()).write_h5ad(path)
    with h5py.File(path, "a") as f:
        ad.io.write_elem(f, "layers", {"tem_data": np.asarray(edata.layers["tem_data"])})
        ad.io.write_elem(f, "tem", edata.tem)

    edata_read = read_h5ed(path)
    _assert_shape_matches(edata_read, (3, 3, 3))
    assert "tem_data" in edata_read.layers
    assert np.array_equal(edata_333.layers["tem_data"], edata_read.layers["tem_data"])


@pytest.mark.skipif(not _ANNDATA_ALLOWS_ND_X, reason="anndata <0.13 does not allow a >2D X in memory")
def test_write_read_h5ed_3d_X_relocated_to_obsm(tmp_path):
    # a 3D X is relocated to the reserved `_ed_ondisk_X` obsm key (and dropped from X) on write, and
    # restored to a 3D X on read.
    from ehrdata import EHRData

    X3 = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)
    edata = EHRData(X=X3)
    assert edata.shape == (2, 3, 4)

    path = tmp_path / "edata_3dX.h5ed"
    write_h5ed(edata.copy(), path)

    with h5py.File(path, "r") as f:
        assert "_ed_ondisk_X" in f["obsm"]
        # the 3D array must not remain in the 2D-only X slot
        assert "X" not in dict(f)
        # the unified-X None key must not leak in as a relocated "layer" (anndata 0.13)
        assert not any(k.startswith("_ed_ondisk_layers_") for k in f["obsm"])

    edata_read = read_h5ed(path)
    assert edata_read.shape == (2, 3, 4)
    assert np.array_equal(np.asarray(edata_read.X), X3)
    assert not any(k.startswith("_ed_ondisk_") for k in edata_read.obsm)
    # no spurious layer (e.g. a "None"-named one) was created from the unified-X key
    assert [k for k in edata_read.layers if k is not None] == []


@pytest.mark.skipif(not _ANNDATA_ALLOWS_ND_X, reason="anndata <0.13 does not allow a >2D X in memory")
def test_read_h5ed_accepts_3d_X_directly_on_disk(tmp_path):
    # ehrdata reads a file that stores a 3D array directly in X (as anndata >=0.13 still can, with a
    # warning) just as naturally as one using the relocated `_ed_ondisk_*` layout.
    X3 = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)

    path = tmp_path / "raw_3d_X.h5ad"
    ad.AnnData(np.ones((2, 3), dtype=float)).write_h5ad(path)  # 2D X scaffold
    with h5py.File(path, "a") as f:
        del f["X"]
        ad.io.write_elem(f, "X", X3)  # replace it with a 3D X, bypassing the on-write 2D check

    edata_read = read_h5ed(path, harmonize_missing_values=False, cast_variables_to_float=False)
    assert edata_read.shape == (2, 3, 4)
    assert np.array_equal(np.asarray(edata_read.X), X3)


def test_h5ad_io_aliases_are_deprecated(edata_333, tmp_path):
    # `read_h5ad`/`write_h5ad` remain as deprecated aliases of the `.h5ed` API and still work.
    from ehrdata.io import read_h5ad, write_h5ad

    path = tmp_path / "edata_alias.h5ed"
    with pytest.warns(DeprecationWarning, match="write_h5ed"):
        write_h5ad(edata_333.copy(), path)
    with pytest.warns(DeprecationWarning, match="read_h5ed"):
        edata_read = read_h5ad(path)

    _assert_shape_matches(edata_read, (3, 3, 3))
    assert np.array_equal(edata_333.layers["tem_data"], edata_read.layers["tem_data"])
