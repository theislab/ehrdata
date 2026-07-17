"""Translate :class:`~ehrdata.EHRData` to and from its on-disk layout.

EHRData keeps time-series as 3D arrays in ``X``/``layers``, but anndata only guarantees 2D arrays there, and enforces this 0.13.0 onwards
EHRData in memory: keeps 3D arrays in ``X``/``layers``. EHRData on disk: moves 3D arrays into ``.obsm``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anndata as ad
import sparse

from ehrdata.core.constants import (
    EHRDATA_ENCODING_TYPE,
    EHRDATA_ENCODING_TYPE_KEY,
    EHRDATA_ENCODING_TYPE_KEY_ZARR,
    EHRDATA_OBSM_3D_LAYER_PREFIX,
    EHRDATA_OBSM_3D_X_KEY,
    EHRDATA_ONDISK_VERSION,
    EHRDATA_ONDISK_VERSION_KEY,
)
from ehrdata.io._coo_codec import is_coo_group, read_coo

if TYPE_CHECKING:
    from ehrdata import EHRData
    from ehrdata._types import GroupStorageType


def _is_3d(arr: Any) -> bool:
    return arr is not None and hasattr(arr, "shape") and len(arr.shape) == 3


def _obsm_key_for_layer(name: str) -> str:
    return f"{EHRDATA_OBSM_3D_LAYER_PREFIX}{name}"


def _layer_for_obsm_key(key: str) -> str | None:
    return key.removeprefix(EHRDATA_OBSM_3D_LAYER_PREFIX) if key.startswith(EHRDATA_OBSM_3D_LAYER_PREFIX) else None


def _reject_stray_coo(edata: EHRData) -> None:
    """Reject a :class:`sparse.COO` anywhere ehrdata cannot persist it.

    anndata has no COO writer; ehrdata serializes COO itself (see :mod:`ehrdata.io._coo_codec`),
    but only for its own reserved 3D ``X``/layer slots. A COO in any other slot — or a 2D COO
    in ``X``/a layer — is unsupported and raised here so the failure is clear instead of a
    cryptic anndata "no writer" error later.
    """
    offenders = []
    if isinstance(edata.X, sparse.COO) and not _is_3d(edata.X):
        offenders.append("X (2D sparse.COO)")
    for key, value in edata.layers.items():
        if key is None:  # anndata 0.13: the unified `.X` slot, handled via edata.X above
            continue
        if isinstance(value, sparse.COO) and not _is_3d(value):
            offenders.append(f"layers[{key!r}] (2D sparse.COO)")
    for name, mapping in (("obsm", edata.obsm), ("varm", edata.varm), ("obsp", edata.obsp), ("varp", edata.varp)):
        offenders.extend(f"{name}[{key!r}]" for key, value in mapping.items() if isinstance(value, sparse.COO))
    if offenders:
        msg = (
            "ehrdata can only persist a sparse.COO as a 3D X or a 3D layer; "
            f"found an unsupported sparse.COO in: {', '.join(offenders)}."
        )
        raise NotImplementedError(msg)


def encode_for_disk(edata: EHRData) -> tuple[ad.AnnData, dict[str, sparse.COO]]:
    """Build an :class:`~anndata.AnnData` with 3D ``X``/``layers`` relocated into reserved ``.obsm`` keys.

    Dense 3D arrays are placed into the AnnData's ``.obsm`` under the reserved keys.
    Sparse 3D arrays, not writeable with AnnData, are pulled out into the returned ``sparse_3d_data`` mapping (keyed by the same reserved ``.obsm`` keys)
    """
    _reject_stray_coo(edata)

    obsm = dict(edata.obsm)
    sparse_3d_data: dict[str, sparse.COO] = {}
    layers = {}

    X = edata.X
    if _is_3d(X):
        if isinstance(X, sparse.COO):
            sparse_3d_data[EHRDATA_OBSM_3D_X_KEY] = X
        else:
            obsm[EHRDATA_OBSM_3D_X_KEY] = X
        X = None

    for key, value in edata.layers.items():
        if key is None:  # anndata 0.13: the unified `.X` slot, already handled above
            continue
        if _is_3d(value):
            target = _obsm_key_for_layer(key)
            if isinstance(value, sparse.COO):
                sparse_3d_data[target] = value
            else:
                obsm[target] = value
        else:
            layers[key] = value

    adata = ad.AnnData(
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
    return adata, sparse_3d_data


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


def read_mapping_with_coo(group: GroupStorageType) -> dict[str, Any]:
    """Read an ``.obsm``-like group child-by-child, routing ehrdata COO groups to the COO reader.

    anndata's ``read_elem`` cannot read the ehrdata-owned COO groups, so ``.obsm`` is read
    per child instead of wholesale: COO children go through :func:`~ehrdata.io._coo_codec.read_coo`,
    everything else through the public :func:`anndata.io.read_elem`.
    """
    out: dict[str, Any] = {}
    for key in group:
        child = group[key]
        out[key] = read_coo(child) if is_coo_group(child) else ad.io.read_elem(child)
    return out


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
