from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import zarr
from lamin_utils import logger

from ehrdata.io._array_casting import _cast_arrays_dtype_to_float_or_str_if_nonnumeric_object, _cast_variables_to_float

if TYPE_CHECKING:
    from os import PathLike

    from ehrdata import EHRData


def read_zarr(
    filename: PathLike[str] | zarr.group.Group | str,
    *,
    harmonize_missing_values: bool = True,
    cast_variables_to_float: bool = True,
) -> EHRData:
    """Read a zarr store into an :class:`~ehrdata.EHRData` object.

    Args:
        filename: The filename, or a Zarr storage class.
        harmonize_missing_values: Whether to call `ehrdata.harmonize_missing_values` on all detected layers.
        cast_variables_to_float: For non-numeric arrays, try to cast the values for each variable to dtype `np.float64`.
            If the cast fails for the values of one variable, then the values of these variable remain unaltered.
            This can be helpful to recover arrays that were of dtype `object` when they were written to disk.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.io.write_zarr("mimic_2.zarr", edata)
        >>> edata_from_zarr = ed.io.read_zarr("mimic_2.zarr")
    """
    import ehrdata as ed
    from ehrdata import EHRData

    if isinstance(filename, Path):
        filename = str(filename)

    f = filename if isinstance(filename, zarr.Group) else zarr.open(filename, mode="r")

    dictionary_for_init = {k: ad.io.read_elem(f[k]) for k, v in dict(f).items() if not k.startswith("raw.")}

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


def write_zarr(
    edata: EHRData,
    filename: str | Path,
    *,
    chunks: bool | int | tuple[int, ...] | None = None,
    convert_strings_to_categoricals: bool = True,
) -> None:
    """Write :class:`~ehrdata.EHRData` objects to disk.

    To write to a `.zarr` file, `X`, `R`, and `layers` cannot be written as  `object` dtype.
    If any of these fields is of `object` dtype, it this function will attempt to cast it to a numeric dtype; if this fails, the field will be casted to a `str` dtype.

    Args:
        edata: Data object.
        filename: Name of the output file, can also be prefixed with relative or absolute path to save the file to.
        chunks: Chunk shape, passed to :meth:`zarr.Group.create_dataset` for `Zarr` version 2, or to :meth:`zarr.Group.create_array` for `Zarr` version 3.
        convert_strings_to_categoricals: Convert columns of `str` dtype in `.obs` and `.var` to `categorical` dtype.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.io.write_zarr("mimic_2.zarr", edata)
    """
    filename = Path(filename)

    edata = _cast_arrays_dtype_to_float_or_str_if_nonnumeric_object(edata)

    ad.AnnData(edata).write_zarr(
        filename,
        chunks=chunks,
        convert_strings_to_categoricals=convert_strings_to_categoricals,
    )
    f = zarr.open(filename, mode="a")
    ad.io.write_elem(f, "tem", edata.tem)
