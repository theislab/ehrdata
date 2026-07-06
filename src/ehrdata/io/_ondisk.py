"""Translate :class:`~ehrdata.EHRData` to and from its on-disk layout.

EHRData keeps time-series as 3D arrays in ``X``/``layers``, but anndata only guarantees 2D arrays there.
On write, 3D arrays are relocated into ``.obsm`` under reserved keys and restored on read, so the same logic is shared by the h5ed and zarr readers/writers.
The layout is self-describing: a file with the reserved ``.obsm`` keys uses the relocated layout, one without them is the legacy layout (3D directly in ``X``/``layers``).
See :mod:`ehrdata.core.constants` for the version and reserved keys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anndata as ad

from ehrdata.core.constants import EHRDATA_OBSM_3D_LAYER_PREFIX, EHRDATA_OBSM_3D_X_KEY

if TYPE_CHECKING:
    from ehrdata import EHRData


def _is_3d(arr: Any) -> bool:
    return arr is not None and hasattr(arr, "shape") and len(arr.shape) == 3


def _obsm_key_for_layer(name: str) -> str:
    return f"{EHRDATA_OBSM_3D_LAYER_PREFIX}{name}"


def _layer_for_obsm_key(key: str) -> str | None:
    return key.removeprefix(EHRDATA_OBSM_3D_LAYER_PREFIX) if key.startswith(EHRDATA_OBSM_3D_LAYER_PREFIX) else None


def encode_for_disk(edata: EHRData) -> ad.AnnData:
    """Build an :class:`~anndata.AnnData` with 3D ``X``/``layers`` relocated into reserved ``.obsm`` keys.

    The result satisfies anndata's 2D-only ``X``/``layers`` spec.
    ``.tem`` is written separately by the callers and is not part of the returned object.
    """
    obsm = dict(edata.obsm)
    layers = {}

    X = edata.X
    if _is_3d(X):
        obsm[EHRDATA_OBSM_3D_X_KEY] = X
        X = None

    for key, value in edata.layers.items():
        if key is None:  # anndata 0.13: the unified `.X` slot, already handled above
            continue
        if _is_3d(value):
            obsm[_obsm_key_for_layer(key)] = value
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
    """Restore relocated 3D arrays from ``.obsm`` back into ``X``/``layers`` in a dict of init kwargs.

    Files without the reserved keys (legacy ehrdata files or plain anndata) are returned unchanged.
    """
    obsm = dict(init.get("obsm") or {})
    layers = dict(init.get("layers") or {})

    relocated = False
    if EHRDATA_OBSM_3D_X_KEY in obsm:
        init["X"] = obsm.pop(EHRDATA_OBSM_3D_X_KEY)
        relocated = True
    for key in [k for k in obsm if _layer_for_obsm_key(k) is not None]:
        layers[_layer_for_obsm_key(key)] = obsm.pop(key)
        relocated = True

    if relocated:
        init["obsm"] = obsm
        init["layers"] = layers

    return init
