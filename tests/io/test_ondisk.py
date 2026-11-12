import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import sparse
from tests.conftest import TEST_DATA_PATH, _assert_object_layers_read_back, _edata_with_object_layers, _values

from ehrdata import EHRData
from ehrdata._logger import logger
from ehrdata.io import read_h5ed, read_zarr, write_h5ed

TOY_H5ED = TEST_DATA_PATH / "toy_h5ed"
TOY_ZARR = TEST_DATA_PATH / "toy_zarr"


@pytest.mark.parametrize("with_X", [True, False])
def test_write_read_object_layers(write_read, X_any_array_type_322, with_X):
    # each layer is encoded on its own, whatever `X` is
    edata = _edata_with_object_layers(X_any_array_type_322 if with_X else None)

    _assert_object_layers_read_back(edata, write_read(edata))


@pytest.mark.parametrize("shape", [(3, 2), (3, 2, 2)])
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            ["007", "nan", None, 1, 2.5, "V45", 2**53 + 1, None, "a", "", True, "010"],
            # integers that float64 does not hold exactly stay text; bools become floats
            ["007", "nan", None, 1.0, 2.5, "V45", str(2**53 + 1), None, "a", "", 1.0, "010"],
        ),
        ([1, 2**53 + 1, None, *range(9)], [1.0, str(2**53 + 1), None, *map(float, range(9))]),
    ],
    ids=["text", "integers"],
)
def test_write_read_object_X(write_read, shape, values, expected):
    X = np.array(values, dtype=object)[: np.prod(shape)].reshape(shape)
    edata = EHRData(X=X)

    X_read = write_read(edata).X

    assert X_read.dtype == object
    assert _values(pd.Series(X_read.ravel())) == expected[: X.size]
    assert [type(value) for value in X_read.ravel() if value is not None and not pd.isna(value)] == [
        type(value) for value in expected[: X.size] if value is not None
    ]
    # the input is left unchanged
    assert list(edata.X.ravel()) == values[: X.size]


@pytest.mark.parametrize(
    ("read_kwargs", "zeros_read_as"),
    [({}, 0.0), ({"harmonize_missing_values": True}, np.nan)],
    ids=["default", "harmonize"],
)
def test_write_read_zero_filled_coo(write_read, read_kwargs, zeros_read_as):
    # e.g. counts of OMOP's setup_variables(..., sparse=True)
    dense = np.array([[[0, 1], [2, 0]], [[0, 0], [3, 0]], [[4, 0], [0, 5]]], dtype=np.float64)
    edata = EHRData(X=sparse.COO.from_numpy(dense))

    X_read = write_read(edata, **read_kwargs).X

    assert isinstance(X_read, sparse.COO)
    assert np.array_equal(X_read.todense(), np.where(dense == 0, zeros_read_as, dense), equal_nan=True)


@pytest.mark.parametrize(
    ("read_kwargs", "codes_read_as", "zeros_read_as"),
    [
        ({}, [["007", "nan"], ["250", "V45"]], 0.0),
        (
            {"harmonize_missing_values": True, "cast_variables_to_float": True},
            [[7.0, None], [250.0, "V45"]],
            np.nan,
        ),
    ],
    ids=["default", "heuristics"],
)
def test_write_read_str_layer(write_read, read_kwargs, codes_read_as, zeros_read_as):
    codes = np.array([["007", "nan"], ["250", "V45"]])
    counts = np.array([[[0, 1], [2, 0]], [[0, 0], [3, 0]]], dtype=np.float64)
    edata = EHRData(layers={"codes": codes, "counts": sparse.COO.from_numpy(counts)}, shape=(2, 2))

    edata_read = write_read(edata, **read_kwargs)

    # object from every format
    assert edata_read.layers["codes"].dtype == object
    assert [_values(pd.Series(row)) for row in edata_read.layers["codes"]] == codes_read_as
    assert np.array_equal(
        edata_read.layers["counts"].todense(), np.where(counts == 0, zeros_read_as, counts), equal_nan=True
    )


@pytest.mark.parametrize(
    "path", [TOY_H5ED / "edata_object_layers_v0_2_0.h5ed", TOY_ZARR / "edata_object_layers_v0_2_0.ehrdata.zarr"]
)
@pytest.mark.parametrize(
    ("read_kwargs", "codes_read_as", "zeros_read_as"),
    [
        # strings are harmonized and cast, zeros stay
        ({}, [[7.0, None], [250.0, "V45"], [42.0, None]], 0.0),
        ({"harmonize_missing_values": True}, [[7.0, None], [250.0, "V45"], [42.0, None]], np.nan),
        (
            {"harmonize_missing_values": False, "cast_variables_to_float": False},
            [["007", "nan"], ["250", "V45"], ["042", "nan"]],
            0.0,
        ),
    ],
    ids=["default", "harmonize", "exact"],
)
def test_read_v0_2_0_object_layers(path, read_kwargs, codes_read_as, zeros_read_as):
    # written by ehrdata before on-disk version 0.3.0, see the creation reference script
    edata = (read_h5ed if path.suffix == ".h5ed" else read_zarr)(path, **read_kwargs)

    assert [_values(pd.Series(row)) for row in edata.layers["codes"]] == codes_read_as
    counts = np.zeros((3, 2, 2))
    counts[0, 0, 1], counts[2, 1, 0] = 2, 1
    assert np.array_equal(
        edata.layers["counts"].todense(), np.where(counts == 0, zeros_read_as, counts), equal_nan=True
    )


def test_read_stale_kinds_ignored(write_read, monkeypatch):
    # e.g. a 0.3.0 file read and written again by ehrdata before 0.3.0
    edata = EHRData(X=np.array([["007", None], ["250", 1]], dtype=object))
    messages = []
    monkeypatch.setattr(logger, "warning", lambda msg, **kwargs: messages.append(msg))

    X_read = write_read(edata, version="0.2.0").X

    assert any("Ignoring ['_ed_ondisk_kinds_X']" in message for message in messages)
    assert [_values(pd.Series(row)) for row in X_read] == [[7.0, None], [250.0, 1.0]]


@pytest.mark.parametrize("change", ["layer_missing", "other_shape"])
def test_read_h5ed_mismatched_kinds_ignored(change, tmp_path, monkeypatch):
    edata = EHRData(X=np.zeros((2, 2)), layers={"codes": np.array([["007", None], ["250", 1]], dtype=object)})
    write_h5ed(edata, tmp_path / "edata.h5ed")
    with h5py.File(tmp_path / "edata.h5ed", "a") as f:
        if change == "layer_missing":
            f.move("layers/codes", "layers/other")
        else:
            del f["obsm/_ed_ondisk_kinds_layers_codes"]
            ad.io.write_elem(f["obsm"], "_ed_ondisk_kinds_layers_codes", np.zeros((2, 1), dtype=np.uint8))

    messages = []
    monkeypatch.setattr(logger, "warning", lambda msg, **kwargs: messages.append(msg))

    edata_read = read_h5ed(tmp_path / "edata.h5ed")

    assert messages == [
        "Ignoring '_ed_ondisk_kinds_layers_codes' in .obsm, which does not match the array it belongs to."
    ]
    assert not any(key.startswith("_ed_ondisk_") for key in edata_read.obsm)
    # the stored strings
    layer = edata_read.layers["other" if change == "layer_missing" else "codes"]
    assert [_values(pd.Series(row)) for row in layer] == [["007", "nan"], ["250", "1"]]


def test_write_reserved_obsm_key_raises(write_read):
    edata = EHRData(X=np.zeros((2, 2)), obsm={"_ed_ondisk_kinds_layers_zzz": np.zeros((2, 2), dtype=np.uint8)})

    with pytest.raises(ValueError, match="reserved for ehrdata's on-disk layout"):
        write_read(edata)


def test_read_newer_version_raises(write_read):
    with pytest.raises(ValueError, match="upgrade ehrdata"):
        write_read(EHRData(X=np.zeros((2, 2))), version="99.0.0")


@pytest.mark.parametrize(("write", "read"), [("write_h5ad", read_h5ed), ("write_zarr", read_zarr)])
def test_read_anndata_heuristics(write, read, tmp_path):
    # anndata files carry no on-disk version of ehrdata
    path = tmp_path / "adata"
    getattr(ad.AnnData(X=np.array([["007", "nan"], ["250", "V45"]])), write)(path)

    X_read = read(path).X

    assert [_values(pd.Series(row)) for row in X_read] == [[7.0, None], [250.0, "V45"]]
