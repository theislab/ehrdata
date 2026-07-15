"""Register anndata IO handlers for pydata-sparse :class:`sparse.COO`.

anndata's IO registry has no writer for n-D ``sparse.COO``. This adds one (and its reader)
under a namespaced spec so a COO tensor relocated into ``.obsm`` round-trips through h5ed/zarr.
Importing this module registers the handlers; registration is idempotent for the same spec.
"""

from __future__ import annotations

import h5py
import numpy as np
import sparse
import zarr
from anndata._io.specs.registry import _REGISTRY, IOSpec

_SPEC = IOSpec("ehrdata-coo", "0.1.0")


def _write_coo(f, k, elem, *, _writer, dataset_kwargs=None):
    dataset_kwargs = dataset_kwargs or {}
    g = f.create_group(k)
    g.attrs["fill_value"] = float(elem.fill_value)
    _writer.write_elem(g, "coords", np.ascontiguousarray(elem.coords), dataset_kwargs=dataset_kwargs)
    _writer.write_elem(g, "data", np.ascontiguousarray(elem.data), dataset_kwargs=dataset_kwargs)
    _writer.write_elem(g, "shape", np.asarray(elem.shape), dataset_kwargs=dataset_kwargs)


def _read_coo(f, *, _reader):
    coords = _reader.read_elem(f["coords"])
    data = _reader.read_elem(f["data"])
    shape = tuple(int(v) for v in _reader.read_elem(f["shape"]))
    fill_value = data.dtype.type(f.attrs.get("fill_value", 0))
    return sparse.COO(coords, data, shape=shape, fill_value=fill_value)


for _group in (h5py.Group, zarr.Group):
    _REGISTRY.register_write(_group, sparse.COO, _SPEC)(_write_coo)
    _REGISTRY.register_read(_group, _SPEC)(_read_coo)
