"""Translate :class:`~ehrdata.EHRData` to and from its on-disk layout.

EHRData keeps time-series as 3D arrays in ``X``/``layers``, but anndata only guarantees 2D arrays there, and enforces this 0.13.0 onwards
EHRData in memory: keeps 3D arrays in ``X``/``layers``. EHRData on disk: moves 3D arrays into ``.obsm``.
An ``object`` ``X``/layer of text next to numbers or missing values is written as strings, with the kind of each value in ``.obsm``.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import sparse

from ehrdata._feature_types import _harmonize_on_read
from ehrdata._logger import logger
from ehrdata.core.constants import (
    EHRDATA_ENCODING_TYPE,
    EHRDATA_ENCODING_TYPE_KEY,
    EHRDATA_ENCODING_TYPE_KEY_ZARR,
    EHRDATA_OBSM_3D_LAYER_PREFIX,
    EHRDATA_OBSM_3D_X_KEY,
    EHRDATA_OBSM_KINDS_LAYER_PREFIX,
    EHRDATA_OBSM_KINDS_X_KEY,
    EHRDATA_OBSM_RESERVED_PREFIX,
    EHRDATA_ONDISK_VERSION_KEY,
    EHRDATA_ONDISK_VERSIONS_READ,
)
from ehrdata.io._array_casting import (
    _cast_dataframe_columns_to_writable_dtype,
    _cast_variables_to_float,
    _decode_object_array,
    _encode_object_array,
)
from ehrdata.io._coo_codec import _binsparse_dtype_str, is_coo_group, read_coo

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping

    import pandas as pd

    from ehrdata import EHRData
    from ehrdata._types import GroupStorageType


def _allow_write_nullable_strings[T, **P](f: Callable[P, T]) -> Callable[P, T]:
    @wraps(f)
    def wrapped(*args: P.args, **kwargs: P.kwargs):
        with ad.settings.override(allow_write_nullable_strings=True):
            return f(*args, **kwargs)

    return wrapped


def _is_3d(arr: Any) -> bool:
    return arr is not None and hasattr(arr, "shape") and len(arr.shape) == 3


def _obsm_key_for_layer(name: str) -> str:
    return f"{EHRDATA_OBSM_3D_LAYER_PREFIX}{name}"


def _layer_for_obsm_key(key: str) -> str | None:
    return key.removeprefix(EHRDATA_OBSM_3D_LAYER_PREFIX) if key.startswith(EHRDATA_OBSM_3D_LAYER_PREFIX) else None


def _kinds_key(layer: str | None) -> str:
    return EHRDATA_OBSM_KINDS_X_KEY if layer is None else f"{EHRDATA_OBSM_KINDS_LAYER_PREFIX}{layer}"


def _pop_kinds(obsm: MutableMapping[str, Any]) -> dict[str | None, Any]:
    """Pop the kind arrays from `obsm`, keyed by their layer, `None` for `X`."""
    kinds = {}
    if EHRDATA_OBSM_KINDS_X_KEY in obsm:
        kinds[None] = obsm.pop(EHRDATA_OBSM_KINDS_X_KEY)
    for key in [k for k in obsm if k.startswith(EHRDATA_OBSM_KINDS_LAYER_PREFIX)]:
        kinds[key.removeprefix(EHRDATA_OBSM_KINDS_LAYER_PREFIX)] = obsm.pop(key)
    return kinds


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
    In-memory ``object`` arrays are encoded (see :func:`~ehrdata.io._array_casting._encode_object_array`), with their kind
    arrays in reserved ``.obsm`` keys, and the ``.obs``/``.var`` columns anndata cannot write are cast, leaving ``edata`` unchanged.
    """
    _reject_stray_coo(edata)
    reserved = [key for key in edata.obsm if key.startswith(EHRDATA_OBSM_RESERVED_PREFIX)]
    if reserved:
        msg = f"The keys {reserved} of .obsm are reserved for ehrdata's on-disk layout; rename them to write."
        raise ValueError(msg)

    obsm = dict(edata.obsm)
    sparse_3d_data: dict[str, sparse.COO] = {}
    layers = {}

    X, kinds = _encode_object_array(edata.X, "X")
    if kinds is not None:
        obsm[_kinds_key(None)] = kinds
    if _is_3d(X):
        if isinstance(X, sparse.COO):
            sparse_3d_data[EHRDATA_OBSM_3D_X_KEY] = X
        else:
            obsm[EHRDATA_OBSM_3D_X_KEY] = X
        X = None

    for key, value in edata.layers.items():
        if key is None:  # anndata 0.13: the unified `.X` slot, already handled above
            continue
        value, kinds = _encode_object_array(value, f"layers[{key!r}]")
        if kinds is not None:
            obsm[_kinds_key(key)] = kinds
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
        obs=_cast_dataframe_columns_to_writable_dtype(edata.obs, "obs"),
        var=_cast_dataframe_columns_to_writable_dtype(edata.var, "var"),
        uns=dict(edata.uns),
        obsm=obsm,
        varm=dict(edata.varm),
        layers=layers,
        obsp=dict(edata.obsp),
        varp=dict(edata.varp),
        shape=None if X is not None else (edata.n_obs, edata.n_vars),
    )
    return adata, sparse_3d_data


def strings_to_categoricals(adata: ad.AnnData, tem: pd.DataFrame) -> None:
    """Convert the string columns of ``.obs``, ``.var`` and ``.tem`` in which a value repeats or is missing to categoricals.

    Converts in place, so pass the copies built for writing (see :func:`encode_for_disk`), never the caller's data.
    The writers of every format call this, so that the same columns come back as categoricals from every format.
    """
    adata.strings_to_categoricals(adata.obs)
    adata.strings_to_categoricals(adata.var)
    adata.strings_to_categoricals(tem)


def encode_for_writing(
    edata: EHRData, *, convert_strings_to_categoricals: bool
) -> tuple[ad.AnnData, dict[str, sparse.COO], pd.DataFrame]:
    """Encode and check an :class:`~ehrdata.EHRData` for every writer, before anything is written.

    Returns the :class:`~anndata.AnnData` and the sparse 3D arrays of :func:`encode_for_disk`, and the cast `.tem`.
    Raises for data that cannot be written, so that a writer does not replace existing data with a failed write.
    """
    adata, sparse_3d_data = encode_for_disk(edata)
    for coo in sparse_3d_data.values():
        _binsparse_dtype_str(coo.dtype)
    tem = _cast_dataframe_columns_to_writable_dtype(edata.tem, "tem")
    if convert_strings_to_categoricals:
        strings_to_categoricals(adata, tem)
    return adata, sparse_3d_data, tem


def writes_exact_arrays(version: str | None) -> bool:
    """Whether data of on-disk `version` holds `X`/layers exactly, with kind arrays, as since 0.3.0."""
    return version not in {None, "0.2.0"}


def _strings_to_object(array: Any) -> Any:
    # as written; zarr reads strings as StringDType
    return array.astype(object) if isinstance(array, np.ndarray) and array.dtype.kind in "UT" else array


def _decode_arrays(X: Any, layers: MutableMapping[str, Any], obsm: MutableMapping[str, Any], version: str) -> Any:
    """Decode `X` and `layers` with the kind arrays popped from `obsm`, in place for `layers`; return `X`."""
    kinds = _pop_kinds(obsm)
    if not writes_exact_arrays(version):
        if kinds:
            logger.warning(f"Ignoring {[_kinds_key(k) for k in kinds]} in .obsm, of data of on-disk version {version}.")
        return X
    for layer, layer_kinds in kinds.items():
        array = X if layer is None else layers.get(layer)
        if array is None or array.shape != layer_kinds.shape:
            logger.warning(f"Ignoring {_kinds_key(layer)!r} in .obsm, which does not match the array it belongs to.")
            continue
        if layer is None:
            X = _decode_object_array(array, layer_kinds)
        else:
            layers[layer] = _decode_object_array(array, layer_kinds)
    for layer in [k for k in layers if k is not None]:
        layers[layer] = _strings_to_object(layers[layer])
    return _strings_to_object(X)


def decode_init_dict(init: dict[str, Any], version: str) -> dict[str, Any]:
    """Restore relocated 3D arrays from ``.obsm`` back into ``X``/``layers``, and decode the arrays with kind arrays, in a dict of init kwargs."""
    obsm = dict(init.get("obsm") or {})
    layers = dict(init.get("layers") or {})

    if EHRDATA_OBSM_3D_X_KEY in obsm:
        init["X"] = obsm.pop(EHRDATA_OBSM_3D_X_KEY)
    for key in [k for k in obsm if _layer_for_obsm_key(k) is not None]:
        layers[_layer_for_obsm_key(key)] = obsm.pop(key)
    init["X"] = _decode_arrays(init.get("X"), layers, obsm, version)

    init["obsm"] = obsm
    init["layers"] = layers
    return init


def decode_backed_layers(adata: ad.AnnData, version: str) -> None:
    """Decode the layers of a backed :class:`~anndata.AnnData`, which holds its layers in memory; its `X` stays backed."""
    _decode_arrays(None, adata.layers, adata.obsm, version)


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


# other children are ignored, as by anndata
_INIT_KEYS = ("X", "obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns", "raw", "tem")


def read_init_dict(group: GroupStorageType) -> dict[str, Any]:
    """Read the AnnData slots and `.tem` of a group as init kwargs, ignoring other children."""
    return {
        k: read_mapping_with_coo(group[k]) if k == "obsm" else ad.io.read_elem(group[k])
        for k in _INIT_KEYS
        if k in group
    }


def ehrdata_on_disk_version(f: Any) -> str | None:
    """Return the on-disk version of an ehrdata file/store, or `None` for AnnData and ehrdata stores before 0.2.0.

    The type stamp lives under a different key per format (zarr uses ``encoding-type``, h5ed uses the
    namespaced ``ehrdata-encoding-type``), so either key identifies an ehrdata store here.
    Raises for versions this ehrdata cannot read, such as those of a newer ehrdata.
    """
    stamped_as_ehrdata = EHRDATA_ENCODING_TYPE in (
        f.attrs.get(EHRDATA_ENCODING_TYPE_KEY_ZARR),
        f.attrs.get(EHRDATA_ENCODING_TYPE_KEY),
    )
    version = f.attrs.get(EHRDATA_ONDISK_VERSION_KEY) if stamped_as_ehrdata else None
    if version is not None and version not in EHRDATA_ONDISK_VERSIONS_READ:
        msg = (
            f"The data was written in ehrdata's on-disk version {version!r}, but this ehrdata reads versions "
            f"{', '.join(EHRDATA_ONDISK_VERSIONS_READ)}. Please upgrade ehrdata."
        )
        raise ValueError(msg)
    return version


def apply_read_heuristics(
    edata: EHRData, version: str | None, *, harmonize_missing_values: bool | None, cast_variables_to_float: bool | None
) -> None:
    """Apply the read-time heuristics; by default (`None`) only to data written before on-disk version 0.3.0."""
    written_before_030 = not writes_exact_arrays(version)
    if harmonize_missing_values or (harmonize_missing_values is None and written_before_030):
        # zeros of a sparse.COO were written as zeros
        _harmonize_on_read(edata, skip_coo=harmonize_missing_values is None)
    if cast_variables_to_float or (cast_variables_to_float is None and written_before_030):
        _cast_variables_to_float(edata)
