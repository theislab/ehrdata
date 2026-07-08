"""Translate :class:`~ehrdata.EHRData` to and from its on-disk layout.

EHRData keeps time-series as 3D arrays in ``X``/``layers``, but anndata only guarantees 2D arrays there, and enforces this 0.13.0 onwards
EHRData in memory: keeps 3D arrays in ``X``/``layers``. EHRData on disk: moves 3D arrays into ``.obsm``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anndata as ad

from ehrdata.core.constants import (
    EHRDATA_ENCODING_TYPE,
    EHRDATA_ENCODING_TYPE_KEY,
    EHRDATA_ENCODING_TYPE_KEY_ZARR,
    EHRDATA_OBSM_3D_LAYER_PREFIX,
    EHRDATA_OBSM_3D_X_KEY,
    EHRDATA_ONDISK_VERSION,
    EHRDATA_ONDISK_VERSION_KEY,
)

if TYPE_CHECKING:
    from ehrdata import EHRData


def _is_3d(arr: Any) -> bool:
    return arr is not None and hasattr(arr, "shape") and len(arr.shape) == 3


def _obsm_key_for_layer(name: str) -> str:
    return f"{EHRDATA_OBSM_3D_LAYER_PREFIX}{name}"


def _layer_for_obsm_key(key: str) -> str | None:
    return key.removeprefix(EHRDATA_OBSM_3D_LAYER_PREFIX) if key.startswith(EHRDATA_OBSM_3D_LAYER_PREFIX) else None


def encode_for_disk(edata: EHRData) -> ad.AnnData:
    """Build an :class:`~anndata.AnnData` with 3D ``X``/``layers`` relocated into reserved ``.obsm`` keys."""
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
    """Restore relocated 3D arrays from ``.obsm`` back into ``X``/``layers`` in a dict of init kwargs."""
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


def _check_020_ehrdata_on_disk_format(f: Any) -> bool:
    """Return True if the file/store carries the ehrdata 0.2.0 on-disk stamp.

    The type stamp lives under a different key per format (zarr uses ``encoding-type``, h5ed uses the
    namespaced ``ehrdata-encoding-type``), so either key identifies an ehrdata store here.
    """
    stamped_as_ehrdata = EHRDATA_ENCODING_TYPE in (
        f.attrs.get(EHRDATA_ENCODING_TYPE_KEY_ZARR),
        f.attrs.get(EHRDATA_ENCODING_TYPE_KEY),
    )
    return stamped_as_ehrdata and f.attrs.get(EHRDATA_ONDISK_VERSION_KEY) == EHRDATA_ONDISK_VERSION
