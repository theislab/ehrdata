from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anndata as ad
import h5py

from ehrdata._logger import logger
from ehrdata.core.constants import (
    EHRDATA_DEFAULT_FORMAT_VERSION,
    EHRDATA_ENCODING_TYPE,
    EHRDATA_OBSM_3D_LAYER_PREFIX,
    EHRDATA_OBSM_3D_X_KEY,
)
from ehrdata.io._array_casting import _cast_arrays_dtype_to_float_or_str_if_nonnumeric_object, _cast_variables_to_float
from ehrdata.io._ondisk import decode_init_dict, encode_for_disk

if TYPE_CHECKING:
    from ehrdata import EHRData


def _restore_3d_from_obsm_backed(edata: EHRData) -> None:
    """Restore format-v2 3D arrays from obsm back into layers for a backed read.

    The v2 layout is self-describing via the reserved obsm keys. Backed reads only support updates to X,
    so a 3D X cannot be safely restored here and is left in obsm with a warning; 3D layers (the common
    time-series case) are restored.
    """
    for obsm_key in [k for k in edata.obsm if k.startswith(EHRDATA_OBSM_3D_LAYER_PREFIX)]:
        edata.layers[obsm_key.removeprefix(EHRDATA_OBSM_3D_LAYER_PREFIX)] = edata.obsm[obsm_key]
        del edata.obsm[obsm_key]

    if EHRDATA_OBSM_3D_X_KEY in edata.obsm:
        logger.warning(
            "This file stores a 3D X in obsm (ehrdata format v2). Restoring it to X is not supported in "
            "backed mode; it remains accessible under obsm. Read without backed=... to restore X."
        )


def read_h5ed(
    filename: Path | str,
    *,
    backed: Literal["r", "r+"] | bool | None = None,
    harmonize_missing_values: bool = True,
    cast_variables_to_float: bool = True,
) -> EHRData:
    """Read an ehrdata hdf5 file (`.h5ed`) into an :class:`~ehrdata.EHRData` object.

    Also reads plain anndata `.h5ad` files. 3D arrays are restored to `X`/`layers` whether they were
    relocated into `.obsm` (ehrdata v2 format) or stored directly in `X`/`layers` (legacy ehrdata files
    or anndata files that still contain higher-dimensional arrays).

    Args:
        filename: Path to the file or directory to read.
        backed: If 'r', load :class:`~ehrdata.EHRData` in backed mode instead of fully loading it into memory (memory mode).
            If you want to modify backed attributes of the :class:`~ehrdata.EHRData` object, you need to choose 'r+'.
            Currently, backed only support updates to `X`.
            That means any changes to other slots like obs will not be written to disk in backed mode.
            If you would like save changes made to these slots of a backed EHRData,
            write them to a new file (see :func:`~ehrdata.io.write_h5ed`).
        harmonize_missing_values: Whether to call `ehrdata.harmonize_missing_values` on all detected layers.
            Cannot be called if `backed`.
        cast_variables_to_float: For non-numeric arrays, try to cast the values for each variable to dtype `np.float64`.
            If the cast fails for the values of one variable, then the values of these variable remain unaltered.
            This can be helpful to recover arrays that were of dtype `object` when they were written to disk.
            Cannot be called if `backed`.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.io.write_h5ed(edata, "mimic_2.h5ed")
        >>> edata_2 = ed.io.read_h5ed("mimic_2.h5ed")
    """
    import ehrdata as ed
    from ehrdata import EHRData

    if backed and harmonize_missing_values:
        msg = "backed reading is not available with 'harmonize_missing_values=True'."
        raise ValueError(msg)

    if backed and cast_variables_to_float:
        msg = "backed reading is not available with 'cast_variables_to_float=True'."
        raise ValueError(msg)

    if backed in {None, False}:
        dictionary_for_init = {}

        with h5py.File(filename, "r") as f:
            dictionary_for_init.update(
                {k: ad.io.read_elem(f[k]) for k, _ in dict(f).items() if not k.startswith("raw.")}
            )

        # Restore the versioned on-disk layout (e.g. move format-v2 3D arrays from obsm back to layers).
        dictionary_for_init = decode_init_dict(dictionary_for_init)
        edata = EHRData(**dictionary_for_init)

    else:
        mode = backed
        if mode is True:
            mode = "r+"
        adata = ad.read_h5ad(filename, backed=mode)

        with h5py.File(filename, "r") as f:
            tem = ad.io.read_elem(f["tem"]) if "tem" in f else None

            edata = EHRData.from_adata(adata, tem=tem)

        _restore_3d_from_obsm_backed(edata)

    if harmonize_missing_values:
        if edata.X is not None:
            ed.harmonize_missing_values(edata)
            logger.info("Harmonizing missing values of X")

        for key in edata.layers:
            ed.harmonize_missing_values(edata, layer=key)
            logger.info(f"Harmonizing missing values of layer {key}")

    if cast_variables_to_float:
        _cast_variables_to_float(edata)

    return edata


def write_h5ed(
    edata: EHRData,
    filename: str | Path,
    *,
    compression: Literal["gzip", "lzf"] | None = None,
    compression_opts: int | None = None,
) -> None:
    """Write :class:`~ehrdata.EHRData` objects to an ehrdata hdf5 file (`.h5ed`).

    `.h5ed` is the ehrdata on-disk format, marking it distinct from anndata's `.h5ad`. To write the
    file, `X` and `layers` cannot be written as `object` dtype. If any of these fields is of `object`
    dtype, this function will attempt to cast it to a numeric dtype; if this fails, the field will be
    casted to a string dtype.

    The data is written in the ehrdata v2 on-disk format: since EHRData stores time-series as 3D arrays
    in `X`/`layers` but anndata only guarantees 2D arrays there, any 3D arrays are relocated into
    `.obsm` (under reserved `_ed_ondisk_*` keys) and dropped from `X`/`layers`;
    :func:`~ehrdata.io.read_h5ed` restores them automatically. Files written by older ehrdata versions
    (3D arrays directly in `X`/`layers`) remain readable.

    Args:
        edata: Central data object.
        filename: Name of the output file, can also be prefixed with relative or absolute path to save the file to.
        compression: Optional file compression.
            Setting compression to 'gzip' can save disk space but will slow down writing and subsequent reading.
        compression_opts: See http://docs.h5py.org/en/latest/high/dataset.html.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.io.write_h5ed(edata, "mimic_2.h5ed")
    """
    filename = Path(filename)

    edata = _cast_arrays_dtype_to_float_or_str_if_nonnumeric_object(edata)

    encode_for_disk(edata).write_h5ad(
        filename,
        compression=compression,
        compression_opts=compression_opts,
    )
    with h5py.File(filename, "a") as f:
        ad.io.write_elem(f, "tem", edata.tem)
        # Stamp the ehrdata format version for discoverability, namespaced so it does not clash with
        # anndata's own root encoding-type/-version attrs. Reading does not depend on these.
        f.attrs["ehrdata-encoding-type"] = EHRDATA_ENCODING_TYPE
        f.attrs["ehrdata-encoding-version"] = str(EHRDATA_DEFAULT_FORMAT_VERSION)


def read_h5ad(
    filename: Path | str,
    *,
    backed: Literal["r", "r+"] | bool | None = None,
    harmonize_missing_values: bool = True,
    cast_variables_to_float: bool = True,
) -> EHRData:
    """Deprecated alias for :func:`read_h5ed`.

    ehrdata's on-disk format is now `.h5ed`; use :func:`~ehrdata.io.read_h5ed` instead.
    """
    warnings.warn(
        "`ehrdata.io.read_h5ad` is deprecated and will be removed in a future release; "
        "use `ehrdata.io.read_h5ed` instead (ehrdata's on-disk format is `.h5ed`).",
        DeprecationWarning,
        stacklevel=2,
    )
    return read_h5ed(
        filename,
        backed=backed,
        harmonize_missing_values=harmonize_missing_values,
        cast_variables_to_float=cast_variables_to_float,
    )


def write_h5ad(
    edata: EHRData,
    filename: str | Path,
    *,
    compression: Literal["gzip", "lzf"] | None = None,
    compression_opts: int | None = None,
) -> None:
    """Deprecated alias for :func:`write_h5ed`.

    ehrdata's on-disk format is now `.h5ed`; use :func:`~ehrdata.io.write_h5ed` instead.
    """
    warnings.warn(
        "`ehrdata.io.write_h5ad` is deprecated and will be removed in a future release; "
        "use `ehrdata.io.write_h5ed` instead (ehrdata's on-disk format is `.h5ed`).",
        DeprecationWarning,
        stacklevel=2,
    )
    write_h5ed(edata, filename, compression=compression, compression_opts=compression_opts)
