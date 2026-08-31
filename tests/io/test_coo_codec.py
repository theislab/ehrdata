"""Unit tests for the binsparse data-type table in :mod:`ehrdata.io._coo_codec`.

These lock ehrdata's on-disk COO dtype handling to the binsparse specification's canonical
data-type strings (https://graphblas.org/binsparse-specification/#key_data_types). They exercise
the pure mapping function directly, so they run regardless of the installed anndata version (no
in-memory ``sparse.COO`` and no I/O involved).
"""

import numpy as np
import pytest

from ehrdata.io._coo_codec import _BINSPARSE_BOOL_STR, _binsparse_dtype_str


@pytest.mark.parametrize(
    ("np_dtype", "binsparse_str"),
    [
        ("int8", "int8"),
        ("int16", "int16"),
        ("int32", "int32"),
        ("int64", "int64"),
        ("uint8", "uint8"),
        ("uint16", "uint16"),
        ("uint32", "uint32"),
        ("uint64", "uint64"),
        ("float32", "float32"),
        ("float64", "float64"),
        ("bool", "bint8"),  # binsparse: unsigned 8-bit integer reinterpreted as boolean
    ],
)
def test_binsparse_dtype_str_supported(np_dtype, binsparse_str):
    assert _binsparse_dtype_str(np.dtype(np_dtype)) == binsparse_str


def test_binsparse_dtype_str_bool_is_bint8():
    assert _binsparse_dtype_str(np.dtype("bool")) == _BINSPARSE_BOOL_STR == "bint8"


@pytest.mark.parametrize(
    "np_dtype",
    [
        np.dtype("<U5"),  # unicode string
        np.dtype(object),
        np.dtype("complex64"),
        np.dtype("complex128"),
        np.dtype("float16"),
        np.dtype("datetime64[ns]"),
    ],
)
def test_binsparse_dtype_str_unsupported_raises(np_dtype):
    # dtypes outside the binsparse table are rejected, never written with a non-spec label
    with pytest.raises(ValueError, match="binsparse layout"):
        _binsparse_dtype_str(np_dtype)
