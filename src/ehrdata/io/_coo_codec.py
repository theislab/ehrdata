"""Register anndata-mimicking IO handlers for pydata-sparse :class:`sparse.COO`.

anndata's IO registry has no writer for n-D ``sparse.COO``.
This adds one (and its reader) so a COO tensor relocated into ``.obsm`` round-trips through h5ed/zarr.
Registration runs on import and is best-effort: it is skipped if anndata's private IO registry moved, or if a ``sparse.COO`` writer is already registered.
"""

from __future__ import annotations

import contextlib

import h5py
import numpy as np
import sparse
import zarr


def _register() -> None:

    from anndata._io.specs.registry import _REGISTRY, IOSpec

    if sparse.COO in _REGISTRY.write_specs:
        return

    spec = IOSpec("ehrdata-coo", "0.1.0")

    def _write_coo(f, k, elem, *, _writer, dataset_kwargs):
        g = f.create_group(k)
        g.attrs["shape"] = list(elem.shape)
        _writer.write_elem(g, "coords", np.ascontiguousarray(elem.coords), dataset_kwargs=dataset_kwargs)
        _writer.write_elem(g, "data", np.ascontiguousarray(elem.data), dataset_kwargs=dataset_kwargs)
        # length-1 (not 0-d)
        _writer.write_elem(g, "fill_value", np.asarray(elem.fill_value).reshape(1), dataset_kwargs=dataset_kwargs)

    def _read_coo(f, *, _reader):
        coords = _reader.read_elem(f["coords"])
        data = _reader.read_elem(f["data"])
        shape = tuple(int(v) for v in f.attrs["shape"])
        fill_value = _reader.read_elem(f["fill_value"])[0]
        return sparse.COO(coords, data, shape=shape, fill_value=fill_value)

    for group in (h5py.Group, zarr.Group):
        _REGISTRY.register_write(group, sparse.COO, spec)(_write_coo)
        _REGISTRY.register_read(group, spec)(_read_coo)


# don't break `import ehrdata` if andata's private registry is absent/changed
with contextlib.suppress(ImportError, AttributeError, TypeError):
    _register()
