from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import h5py
import zarr

from ehrdata.io.h5ed import _read_h5ed_elem, _write_h5ed_elem
from ehrdata.io.zarr import _read_zarr_elem, _write_zarr_elem

if TYPE_CHECKING:
    from ehrdata import EHRData


def write_elem(
    parent: h5py.Group | zarr.Group,
    key: str,
    edata: EHRData,
    *,
    convert_strings_to_categoricals: bool = False,
    compression: Literal["gzip", "lzf"] | None = None,
    compression_opts: int | None = None,
    chunks: Literal["auto", "ehrdata_auto"] = "auto",
) -> None:
    """Write :class:`~ehrdata.EHRData` objects into a group of an HDF5 file or a Zarr store.

    Use this to write an :class:`~ehrdata.EHRData` as one part of a larger file or store, next to other data.
    `parent[key]` holds the same layout as a file written with :func:`~ehrdata.io.write_h5ed` for an :class:`h5py.Group`,
    and as a store written with :func:`~ehrdata.io.write_zarr` for a :class:`zarr.Group`, and
    the data is cast for writing as these functions do; `X` and `layers` read back exactly.
    An existing `parent[key]` is replaced, as with :func:`anndata.io.write_elem`.
    Zarr stores with consolidated metadata cannot be written to.
    Read it back with :func:`~ehrdata.io.read_elem`.

    Args:
        parent: The group to write into, such as an open :class:`h5py.File`, a :class:`zarr.Group`, or a subgroup of these.
            Zarr groups of Zarr format 2 and 3 are supported.
        key: The name of the new group in `parent`, without `/`.
            To write into a nested group, pass that group as `parent`.
            To write a whole file or store, use :func:`~ehrdata.io.write_h5ed` or :func:`~ehrdata.io.write_zarr`.
        edata: Central data object.
        convert_strings_to_categoricals: Convert the string columns of `.obs`, `.var` and `.tem` in which a value repeats or is missing to `categorical` dtype.
            By default, strings are kept.
            The :class:`~ehrdata.EHRData` passed is left unchanged.
        compression: Optional compression for an :class:`h5py.Group`, see :func:`~ehrdata.io.write_h5ed`.
            Raises for a :class:`zarr.Group`.
        compression_opts: For [`lzf`, `gzip`], see the h5py :ref:`dataset_compression`.
            Raises for a :class:`zarr.Group`.
        chunks: Chunking strategy for a :class:`zarr.Group`, see :func:`~ehrdata.io.write_zarr`.
            `"ehrdata_auto"` needs a group of Zarr format 3, and raises for an :class:`h5py.Group`.

    Examples:
        >>> import h5py
        >>> import ehrdata as ed
        >>> edata = ed.dt.ehrdata_blobs(n_variables=3, n_observations=5)
        >>> with h5py.File("bundle.h5", "w") as f:
        ...     ed.io.write_elem(f, "edata", edata)
        ...     f["other"] = [1, 2, 3]
    """
    if not isinstance(key, str):
        msg = f"key must be a str, not {type(key).__name__}."
        raise TypeError(msg)
    if key == "/":
        msg = "key='/' would write the whole file or store; use write_h5ed or write_zarr instead."
        raise ValueError(msg)
    if not key or "/" in key:
        msg = f"key {key!r} must be a non-empty name without '/'; pass a nested group as `parent` instead."
        raise ValueError(msg)

    if isinstance(parent, h5py.Group):
        if chunks != "auto":
            msg = "`chunks` only applies to a zarr.Group; use `compression` for an h5py.Group."
            raise ValueError(msg)
        _write_h5ed_elem(
            parent,
            key,
            edata,
            compression=compression,
            compression_opts=compression_opts,
            convert_strings_to_categoricals=convert_strings_to_categoricals,
        )
    elif isinstance(parent, zarr.Group):
        if compression is not None or compression_opts is not None:
            msg = "`compression` and `compression_opts` only apply to an h5py.Group; use `chunks` for a zarr.Group."
            raise ValueError(msg)
        _write_zarr_elem(
            parent, key, edata, chunks=chunks, convert_strings_to_categoricals=convert_strings_to_categoricals
        )
    else:
        msg = f"parent must be an h5py.Group or a zarr.Group, not {type(parent).__name__}."
        raise TypeError(msg)


def read_elem(
    group: h5py.Group | zarr.Group,
    *,
    harmonize_missing_values: bool | None = None,
    cast_variables_to_float: bool | None = None,
) -> EHRData:
    """Read an :class:`~ehrdata.EHRData` from a group of an HDF5 file or a Zarr store.

    Reads a group written with :func:`~ehrdata.io.write_elem`, :func:`~ehrdata.io.write_h5ed` or :func:`~ehrdata.io.write_zarr`,
    fully into memory.
    Like :func:`~ehrdata.io.read_h5ed` and :func:`~ehrdata.io.read_zarr`, it also reads :class:`~anndata.AnnData` groups,
    and reads `X` and `layers` back exactly, applying the heuristics below by default only to groups written before
    ehrdata's on-disk version 0.3.0 and to :class:`~anndata.AnnData` groups.

    Args:
        group: The group to read, such as `f["edata"]` of an open :class:`h5py.File` or a :class:`zarr.Group`.
        harmonize_missing_values: Whether to call :func:`~ehrdata.harmonize_missing_values` on `X` and all layers.
            By default (`None`), only for groups written before on-disk version 0.3.0 and :class:`~anndata.AnnData` groups,
            and leaving `sparse.COO` arrays untouched. `True` also applies it to `sparse.COO` arrays, treating their zeros as missing.
        cast_variables_to_float: For non-numeric arrays, try to cast the values for each variable to dtype `np.float64`.
            If the cast fails for the values of one variable, then the values of these variable remain unaltered.
            This can be helpful to recover arrays that were of dtype `object` when they were written to disk.
            By default (`None`), only for groups written before on-disk version 0.3.0 and :class:`~anndata.AnnData` groups.

    Examples:
        >>> import h5py
        >>> import ehrdata as ed
        >>> edata = ed.dt.ehrdata_blobs(n_variables=3, n_observations=5)
        >>> with h5py.File("bundle.h5", "w") as f:
        ...     ed.io.write_elem(f, "edata", edata)
        >>> with h5py.File("bundle.h5", "r") as f:
        ...     edata_2 = ed.io.read_elem(f["edata"])
    """
    if isinstance(group, h5py.Group):
        if group.attrs.get("encoding-type") != "anndata":
            msg = (
                f"The group {group.name!r} holds no EHRData or AnnData: its 'encoding-type' attribute is not 'anndata'."
            )
            raise ValueError(msg)
        return _read_h5ed_elem(
            group, harmonize_missing_values=harmonize_missing_values, cast_variables_to_float=cast_variables_to_float
        )
    if isinstance(group, zarr.Group):
        return _read_zarr_elem(
            group, harmonize_missing_values=harmonize_missing_values, cast_variables_to_float=cast_variables_to_float
        )
    msg = f"group must be an h5py.Group or a zarr.Group, not {type(group).__name__}."
    raise TypeError(msg)
