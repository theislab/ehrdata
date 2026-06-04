"""Translate :class:`~ehrdata.EHRData` to and from its versioned on-disk layout.

EHRData keeps time-series as 3D arrays in ``X``/``layers``, but anndata only guarantees 2D arrays
there. On write (format v2) these helpers move 3D arrays into ``.obsm`` under reserved keys, and on
read they move them back, so the same logic is shared by the h5ad and zarr readers/writers. The layout
is self-describing: a file with the reserved ``.obsm`` keys is v2, one without them is the legacy v1
layout (3D already in ``X``/``layers``). See :mod:`ehrdata.core.constants` for the version definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anndata as ad

from ehrdata.core.constants import (
    EHRDATA_OBSM_3D_LAYER_PREFIX,
    EHRDATA_OBSM_3D_X_KEY,
)

if TYPE_CHECKING:
    from ehrdata import EHRData


def _is_3d(arr: Any) -> bool:
    """Return whether ``arr`` is a 3D array. Safe to call on anything, including ``None``."""
    return arr is not None and hasattr(arr, "shape") and len(arr.shape) == 3


def encode_for_disk(edata: EHRData) -> ad.AnnData:
    """Build an :class:`~anndata.AnnData` laid out in the current ehrdata on-disk format (v2).

    Every 3D array in ``X``/``layers`` is relocated into ``.obsm`` under a reserved key so the result
    satisfies anndata's 2D-only spec for ``X``/``layers``. The relocation is self-describing via the
    reserved keys, so no extra bookkeeping is written. The ``.tem`` field is written separately by the
    callers and is not part of the returned object.
    """
    obsm = dict(edata.obsm)
    layers = {}

    X = edata.X
    if _is_3d(X):
        obsm[EHRDATA_OBSM_3D_X_KEY] = X
        X = None

    for key, value in edata.layers.items():
        if _is_3d(value):
            obsm[f"{EHRDATA_OBSM_3D_LAYER_PREFIX}{key}"] = value
        else:
            layers[key] = value

    return ad.AnnData(
        X=X,
        obs=edata.obs.copy(),
        var=edata.var.copy(),
        uns=dict(edata.uns),
        obsm=obsm,
        varm=dict(edata.varm),
        layers=layers,
        obsp=dict(edata.obsp),
        varp=dict(edata.varp),
        shape=None if X is not None else (edata.n_obs, edata.n_vars),
    )


def decode_init_dict(init: dict[str, Any]) -> dict[str, Any]:
    """Restore the in-memory EHRData layout from a dict of init kwargs read from disk.

    For the v2 layout, moves the relocated 3D arrays from ``obsm`` (reserved keys) back into
    ``X``/``layers``. Files without the reserved keys (legacy v1 ehrdata files, or plain anndata) are
    returned unchanged, so the dict can be passed straight to ``EHRData(**init)``.
    """
    obsm = dict(init.get("obsm") or {})
    layers = dict(init.get("layers") or {})

    relocated = False
    if EHRDATA_OBSM_3D_X_KEY in obsm:
        init["X"] = obsm.pop(EHRDATA_OBSM_3D_X_KEY)
        relocated = True
    for key in [k for k in obsm if k.startswith(EHRDATA_OBSM_3D_LAYER_PREFIX)]:
        layers[key.removeprefix(EHRDATA_OBSM_3D_LAYER_PREFIX)] = obsm.pop(key)
        relocated = True

    if relocated:
        init["obsm"] = obsm
        init["layers"] = layers

    return init
