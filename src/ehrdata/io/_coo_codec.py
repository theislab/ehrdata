"""ehrdata-owned serialization of pydata-sparse :class:`sparse.COO` tensors.

anndata's IO has no writer for n-D ``sparse.COO``, and ehrdata keeps its time-series as 3D
``X``/layers. Rather than register a codec on anndata's *private* IO registry (which would
tie ehrdata to non-public internals), ehrdata serializes these tensors itself, directly on
the h5py/zarr group.

The on-disk layout follows the binsparse convention (https://graphblas.org/binsparse-specification/):
one ``indices_<dim>`` dataset per axis, a ``values`` dataset, a length-1 ``fill_value``, and a
JSON ``binsparse`` descriptor attribute. A namespaced ehrdata marker attribute
(:data:`~ehrdata.core.constants.EHRDATA_ENCODING_TYPE_KEY`) identifies the group on read, so
detection never relies on anndata's ``encoding-type``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import sparse

from ehrdata.core.constants import (
    EHRDATA_ENCODING_TYPE_KEY,
    EHRDATA_ONDISK_VERSION,
    EHRDATA_ONDISK_VERSION_KEY,
)

if TYPE_CHECKING:
    from ehrdata._types import GroupStorageType

# Value of the namespaced ehrdata marker stamped on a COO group
COO_ENCODING_TYPE = "ehrdata-coo"
# binsparse descriptor attribute key and the spec version
_BINSPARSE_KEY = "binsparse"
_BINSPARSE_VERSION = "0.1"

# numpy dtype -> binsparse data-type string (https://graphblas.org/binsparse-specification/#key_data_types).
# see also ivirshup/binsparse-python: https://github.com/ivirshup/binsparse-python/blob/6b286eea239af5eb40c1c4fd0d80c82bc6855c54/src/binsparse/_io/methods.py#L10
_DTYPE_TO_BINSPARSE_STR = {
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("bool"): "bint8",
}
_BINSPARSE_BOOL_STR = "bint8"


# analogous to ivirshup/binsparse-python https://github.com/ivirshup/binsparse-python/blob/6b286eea239af5eb40c1c4fd0d80c82bc6855c54/src/binsparse/_io/methods.py#L25
def _binsparse_dtype_str(dtype: np.dtype) -> str:
    """Map a numpy dtype to its binsparse data-type string, raising on unsupported dtypes."""
    result = _DTYPE_TO_BINSPARSE_STR.get(np.dtype(dtype))
    if result is None:
        msg = f"dtype {dtype!r} are not supported by the binsparse layout."
        raise ValueError(msg)
    return result


def _to_disk_array(arr: np.ndarray) -> np.ndarray:
    """Cast an array to its on-disk dtype (booleans are stored as uint8 per binsparse ``bint8``)."""
    return arr.astype(np.uint8) if arr.dtype == np.bool_ else arr


def _coo_descriptor(shape: tuple[int, ...], data: np.ndarray) -> dict[str, Any]:
    """Build the binsparse JSON descriptor from a COO's shape and its (as-written) values array."""
    ndim = len(shape)
    data_types = {f"indices_{i}": "int64" for i in range(ndim)}  # indices are always written as int64
    data_types["values"] = _binsparse_dtype_str(data.dtype)
    return {
        "version": _BINSPARSE_VERSION,
        "format": "COO",
        "shape": [int(s) for s in shape],
        "number_of_stored_values": int(data.shape[0]),
        "data_types": data_types,
        "fill": True,
    }


def _sorted_coords_and_data(coo: sparse.COO) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates/data in row-major (C) order.

    binsparse requires stored coordinates to be sorted (row-major) with no duplicates. pydata
    ``sparse.COO`` coalesces duplicates at construction, but exposes no attribute to tell
    whether ``.coords`` is currently sorted (``sorted`` is a constructor parameter only), so
    reorder explicitly (``np.lexsort`` with the last axis varying fastest).
    """
    order = np.lexsort(coo.coords[::-1])
    return coo.coords[:, order], coo.data[order]


def _stamp(group: GroupStorageType, descriptor: dict[str, Any]) -> None:
    group.attrs[EHRDATA_ENCODING_TYPE_KEY] = COO_ENCODING_TYPE
    group.attrs[EHRDATA_ONDISK_VERSION_KEY] = str(EHRDATA_ONDISK_VERSION)
    group.attrs[_BINSPARSE_KEY] = json.dumps(descriptor)


def is_coo_group(group: GroupStorageType) -> bool:
    """Return True if ``group`` holds an ehrdata COO tensor."""
    return group.attrs.get(EHRDATA_ENCODING_TYPE_KEY) == COO_ENCODING_TYPE


def write_coo_h5(group, coo: sparse.COO, *, compression=None, compression_opts=None) -> None:
    """Write a :class:`sparse.COO` into an (already-created) h5py group."""
    coords, data = _sorted_coords_and_data(coo)
    kwargs: dict[str, Any] = {}

    coo_description = _coo_descriptor(coo.shape, data)  # checks if valid binsparse format

    if compression is not None:
        kwargs = {"compression": compression, "compression_opts": compression_opts}
    for i in range(coords.shape[0]):
        group.create_dataset(f"indices_{i}", data=np.ascontiguousarray(coords[i], dtype=np.int64), **kwargs)

    group.create_dataset("values", data=np.ascontiguousarray(_to_disk_array(data)), **kwargs)
    group.create_dataset("fill_value", data=_to_disk_array(np.asarray(coo.fill_value).reshape(1)))

    _stamp(group, coo_description)


def write_coo_zarr(group, coo: sparse.COO) -> None:
    """Write a :class:`sparse.COO` into an (already-created) zarr group."""
    coords, data = _sorted_coords_and_data(coo)

    coo_description = _coo_descriptor(coo.shape, data)  # checks if valid binsparse format

    for i in range(coords.shape[0]):
        group[f"indices_{i}"] = np.ascontiguousarray(coords[i], dtype=np.int64)
    group["values"] = np.ascontiguousarray(_to_disk_array(data))
    group["fill_value"] = _to_disk_array(np.asarray(coo.fill_value).reshape(1))

    _stamp(group, coo_description)


def read_coo(group: GroupStorageType) -> sparse.COO:
    """Reconstruct a :class:`sparse.COO` from an ehrdata COO group (h5py or zarr)."""
    descriptor = json.loads(group.attrs[_BINSPARSE_KEY])
    shape = tuple(int(s) for s in descriptor["shape"])
    coords = np.vstack([group[f"indices_{i}"][...] for i in range(len(shape))])
    values = group["values"][...]
    fill_value = group["fill_value"][...][0]
    if descriptor["data_types"]["values"] == _BINSPARSE_BOOL_STR:
        # binsparse bint8: on-disk uint8 reinterpreted as boolean
        values = values.astype(np.bool_)
        fill_value = np.bool_(fill_value)
    return sparse.COO(coords, values, shape=shape, fill_value=fill_value)
