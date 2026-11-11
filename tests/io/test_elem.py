import h5py
import numpy as np
import pandas as pd
import pytest
import sparse
import zarr
from tests.conftest import (
    _assert_columns_anndata_cannot_write_read_back,
    _edata_with_columns_anndata_cannot_write,
    _to_dense,
)

from ehrdata import EHRData
from ehrdata.io import read_elem, read_h5ed, read_zarr, write_elem, write_h5ed, write_zarr


@pytest.fixture(params=["h5", "zarr", "zarr_v2"])
def root(request, tmp_path):
    """Root group of a new HDF5 file, Zarr v3 store or Zarr v2 store."""
    if request.param == "h5":
        with h5py.File(tmp_path / "bundle.h5", "w") as f:
            yield f
    else:
        yield zarr.open_group(tmp_path / "bundle.zarr", mode="w", zarr_format=3 if request.param == "zarr" else 2)


def _assert_edata_equal(edata, edata_read):
    assert edata.shape == edata_read.shape
    for frame in ("obs", "var", "tem"):
        pd.testing.assert_frame_equal(getattr(edata, frame), getattr(edata_read, frame))
    assert type(edata.X) is type(edata_read.X)
    assert np.array_equal(_to_dense(edata.X), _to_dense(edata_read.X), equal_nan=True)
    assert edata.layers.keys() == edata_read.layers.keys()
    for key in edata.layers:
        assert type(edata.layers[key]) is type(edata_read.layers[key])
        assert np.array_equal(_to_dense(edata.layers[key]), _to_dense(edata_read.layers[key]), equal_nan=True)


def _attrs(obj) -> dict:
    return {key: np.asarray(value).tolist() for key, value in obj.attrs.items()}


def _layout(group) -> dict:
    """Attributes, and dtype, shape and compression or shards of arrays, of each member of `group`."""
    if isinstance(group, h5py.Group):
        members = []
        group.visititems(lambda name, obj: members.append((name, obj)))
    else:
        members = list(group.members(max_depth=None))
    layout = {"": _attrs(group)}
    for name, obj in members:
        layout[name] = [_attrs(obj)]
        if isinstance(obj, h5py.Dataset):
            layout[name] += [obj.dtype, obj.shape, obj.compression]
        elif isinstance(obj, zarr.Array):
            layout[name] += [obj.dtype, obj.shape, obj.shards]
    return layout


@pytest.mark.parametrize("nested", [False, True])
def test_write_read_elem(root, X_any_array_type_322, nested):
    edata = _edata_with_columns_anndata_cannot_write(X_any_array_type_322)
    parent = root.create_group("a").create_group("b") if nested else root
    root.create_group("other")

    write_elem(parent, "edata", edata)

    assert "other" in root
    _assert_columns_anndata_cannot_write_read_back(edata, read_elem(parent["edata"], harmonize_missing_values=False))


@pytest.mark.parametrize("convert_strings_to_categoricals", [True, False])
@pytest.mark.parametrize("edata_name", ["edata_333", "edata_string_columns"])
def test_write_elem_same_as_file(root, edata_name, convert_strings_to_categoricals, request, tmp_path):
    edata = request.getfixturevalue(edata_name)
    write_elem(root, "edata", edata, convert_strings_to_categoricals=convert_strings_to_categoricals)

    if isinstance(root, h5py.File):
        write_h5ed(edata, tmp_path / "edata.h5ed", convert_strings_to_categoricals=convert_strings_to_categoricals)
        edata_file = read_h5ed(tmp_path / "edata.h5ed")
        # the group, copied into a file of its own
        with h5py.File(tmp_path / "copy.h5ed", "w") as f:
            for key in root["edata"]:
                root.copy(root["edata"][key], f, key)
            f.attrs.update(root["edata"].attrs)
        _assert_edata_equal(edata_file, read_h5ed(tmp_path / "copy.h5ed"))
    else:
        write_zarr(edata, tmp_path / "edata.zarr", convert_strings_to_categoricals=convert_strings_to_categoricals)
        edata_file = read_zarr(tmp_path / "edata.zarr")
        _assert_edata_equal(edata_file, read_zarr(tmp_path / "bundle.zarr" / "edata"))

    _assert_edata_equal(edata_file, read_elem(root["edata"]))
    if edata_name == "edata_string_columns":
        assert (edata_file.obs["sex"].dtype == "category") is convert_strings_to_categoricals


def test_write_elem_same_layout_as_file(root, X_any_array_type_322, tmp_path):
    edata = _edata_with_columns_anndata_cannot_write(X_any_array_type_322)

    if isinstance(root, h5py.File):
        write_elem(root, "edata", edata, compression="gzip")
        write_h5ed(edata, tmp_path / "edata.h5ed", compression="gzip")
        assert "raw" not in root["edata"]
        with h5py.File(tmp_path / "edata.h5ed", "r") as f:
            assert _layout(root["edata"]) == _layout(f)
    elif root.metadata.zarr_format == 3:
        write_elem(root, "edata", edata, chunks="ehrdata_auto")
        write_zarr(edata, tmp_path / "edata.zarr", chunks="ehrdata_auto")
        assert _layout(root["edata"]) == _layout(zarr.open_group(tmp_path / "edata.zarr", mode="r"))
    else:
        pytest.skip("write_zarr writes Zarr format 3")


def test_write_elem_existing_key(root, edata_333, edata_string_columns):
    # replaced, as by anndata.io.write_elem
    root.create_group("edata").create_group("stale")
    write_elem(root, "edata", edata_333)
    write_elem(root, "edata", edata_string_columns)

    assert "stale" not in root["edata"]
    edata_read = read_elem(root["edata"])
    assert edata_read.shape == edata_string_columns.shape
    assert "tem_data" not in edata_read.layers


def test_read_elem_ignores_other_children(root, edata_333):
    # e.g. metadata of a package bundling the EHRData
    write_elem(root, "edata", edata_333)
    slots = root["edata"] if isinstance(root, h5py.File) else root["edata"]["anndata"]
    slots.create_group("extra")

    _assert_edata_equal(read_elem(root["edata"]), edata_333)


@pytest.mark.parametrize(
    ("edata_unwritable", "error"),
    [
        (EHRData(X=np.zeros((3, 2)), obsm={"coo": sparse.COO.from_numpy(np.ones((3, 2)))}), NotImplementedError),
        (EHRData(X=sparse.COO(np.zeros((3, 1), dtype=int), np.array(["a"]), shape=(3, 2, 2))), ValueError),
    ],
    ids=["2d_coo_in_obsm", "string_coo"],
)
def test_write_elem_failing_keeps_existing_key(root, edata_333, edata_unwritable, error):
    write_elem(root, "edata", edata_333)

    with pytest.raises(error):
        write_elem(root, "edata", edata_unwritable)

    _assert_edata_equal(read_elem(root["edata"]), edata_333)


@pytest.mark.filterwarnings("ignore:Consolidated metadata")
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_write_elem_zarr_consolidated(edata_333, zarr_format, tmp_path):
    zarr.open_group(tmp_path / "bundle.zarr", mode="w", zarr_format=zarr_format).create_group("sub")
    zarr.consolidate_metadata(tmp_path / "bundle.zarr")
    root = zarr.open_group(tmp_path / "bundle.zarr", mode="a")

    for parent in (root, root["sub"]):
        with pytest.raises(ValueError, match="consolidated metadata"):
            write_elem(parent, "edata", edata_333)


def test_write_elem_errors(root, edata_333):
    for key in ["", "a/edata", "/"]:
        with pytest.raises(ValueError, match="write_h5ed or write_zarr" if key == "/" else "non-empty name"):
            write_elem(root, key, edata_333)
    with pytest.raises(TypeError, match="key must be a str"):
        write_elem(root, 1, edata_333)
    # options of the other format
    other_format_option = {"chunks": "ehrdata_auto"} if isinstance(root, h5py.File) else {"compression": "gzip"}
    with pytest.raises(ValueError, match="only appl"):
        write_elem(root, "edata", edata_333, **other_format_option)
    if isinstance(root, zarr.Group) and root.metadata.zarr_format == 2:
        with pytest.raises(ValueError, match="need Zarr format 3"):
            write_elem(root, "edata", edata_333, chunks="ehrdata_auto")
    with pytest.raises(TypeError, match=r"h5py\.Group or a zarr\.Group"):
        write_elem({}, "edata", edata_333)

    # the parent instead of the group written
    write_elem(root, "edata", edata_333)
    with pytest.raises(ValueError, match="'encoding-type'"):
        read_elem(root)
    with pytest.raises(TypeError, match=r"h5py\.Group or a zarr\.Group"):
        read_elem({})
