from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anndata as ad
import h5py
from lamin_utils import logger

from ehrdata.io._array_casting import _cast_arrays_dtype_to_float_or_str_if_nonnumeric_object, _cast_variables_to_float

if TYPE_CHECKING:
    from ehrdata import EHRData


def read_h5ad(
    filename: Path | str,
    *,
    backed: Literal["r", "r+"] | bool | None = None,
    harmonize_missing_values: bool = True,
    cast_variables_to_float: bool = True,
) -> EHRData:
    """Read a hdf5 (h5ad) file into an :class:`~ehrdata.EHRData` object.

    Args:
        filename: Path to the file or directory to read.
        backed: If 'r', load :class:`~ehrdata.EHRData` in backed mode instead of fully loading it into memory (memory mode).
            If you want to modify backed attributes of the :class:`~ehrdata.EHRData` object, you need to choose 'r+'.
            Currently, backed only support updates to `X`.
            That means any changes to other slots like obs will not be written to disk in backed mode.
            If you would like save changes made to these slots of a backed EHRData,
            write them to a new file (see :func:`~ehrdata.io.write_h5ad`).
        harmonize_missing_values: Whether to call `ehrdata.harmonize_missing_values` on all detected layers.
            Cannot be called if `backed`.
        cast_variables_to_float: For non-numeric arrays, try to cast the values for each variable to dtype `np.float64`.
            If the cast fails for the values of one variable, then the values of these variable remain unaltered.
            This can be helpful to recover arrays that were of dtype `object` when they were written to disk.
            Cannot be called if `backed`.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.io.write_h5ad("mimic_2.h5ad", edata)
        >>> edata_2 = ed.io.read_h5ad("mimic_2.h5ad")
    """
    import ehrdata as ed
    from ehrdata import EHRData

    if backed and harmonize_missing_values:
        msg = "backed reading is not available with 'harmonize_missing_values=True'."
        raise ValueError(msg)

    if backed and cast_variables_to_float:
        msg = "backed reading is not available with 'cast_variables_to_float=True'."
        raise ValueError(msg)

    if backed not in {None, False}:
        mode = backed
        if mode is True:
            mode = "r+"
        assert mode in {"r", "r+"}
        dictionary_for_init = {"filemode": mode, "filename": filename}
    else:
        dictionary_for_init = {}

    with h5py.File(filename, "r") as f:
        dictionary_for_init.update({k: ad.io.read_elem(f[k]) for k, _ in dict(f).items() if not k.startswith("raw.")})

    edata = EHRData(**dictionary_for_init)

    if harmonize_missing_values:
        ed.harmonize_missing_values(edata)
        logger.info("Harmonizing missing values of X")

        for key in edata.layers:
            ed.harmonize_missing_values(edata, layer=key)
            logger.info(f"Harmonizing missing values of layer {key}")

    if cast_variables_to_float:
        _cast_variables_to_float(edata)

    return edata


def write_h5ad(
    edata: EHRData,
    filename: str | Path,
    *,
    compression: Literal["gzip", "lzf"] | None = None,
    compression_opts: int | None = None,
) -> None:
    """Write :class:`~ehrdata.EHRData` objects to an hdf5 file.

    It is possible to either write an :class:`~ehrdata.EHRData` object to an `.h5ad` or a compressed `.gzip` or `lzf` file.
    To write to an h5ad file, `X`, `R`, and `layers` cannot be written as `object` dtype.
    If any of these fields is of `object` dtype, it this function will attempt to cast it to a numeric dtype;
    if this fails, the field will be casted to a string dtype.

    Args:
        filename: Name of the output file, can also be prefixed with relative or absolute path to save the file to.
        edata: Data object.
        compression: Optional file compression.
            Setting compression to 'gzip' can save disk space but will slow down writing and subsequent reading.
        compression_opts: See http://docs.h5py.org/en/latest/high/dataset.html.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.io.write_h5ad("mimic_2.h5ad", edata)
    """
    filename = Path(filename)

    edata = _cast_arrays_dtype_to_float_or_str_if_nonnumeric_object(edata)

    ad.AnnData(edata).write_h5ad(
        filename,
        compression=compression,
        compression_opts=compression_opts,
    )
    with h5py.File(filename, "a") as f:
        ad.io.write_elem(f, "tem", edata.tem)
