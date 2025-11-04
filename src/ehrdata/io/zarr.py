from __future__ import annotations

import warnings
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import zarr

import ehrdata as ed
from ehrdata._logger import logger
from ehrdata.core.constants import EHRDATA_ZARR_ENCODING_VERSION
from ehrdata.io._array_casting import _cast_arrays_dtype_to_float_or_str_if_nonnumeric_object, _cast_variables_to_float

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from os import PathLike

    from ehrdata import EHRData


def read_zarr(
    filename: PathLike[str] | zarr.group.Group | str,
    *,
    harmonize_missing_values: bool = True,
    cast_variables_to_float: bool = True,
) -> EHRData:
    """Read a zarr store into an :class:`~ehrdata.EHRData` object.

    Can also read :class:`~anndata.AnnData` Zarr stores. In this case, a default `.tem` field is created in the `ehrdata object`.

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
    from ehrdata import EHRData

    if isinstance(filename, Path):
        filename = str(filename)

    f = filename if isinstance(filename, zarr.Group) else zarr.open(filename, mode="r")

    if "encoding-type" not in f.attrs:
        err = "The zarr store does not contain an encoding-type attribute."
        raise ValueError(err)

    if f.attrs["encoding-type"] == "ehrdata":
        if "anndata" in f:
            dictionary_for_init = {
                k: ad.io.read_elem(f["anndata"][k]) for k, v in dict(f["anndata"]).items() if not k.startswith("raw.")
            }
        else:
            err = "The zarr store does not contain the 'anndata' group."
            raise ValueError(err)
        if "tem" in f:
            dictionary_for_init["tem"] = ad.io.read_elem(f["tem"])
        else:
            warnings.warn("The zarr store does not contain the 'tem' group.", stacklevel=2)

    elif f.attrs["encoding-type"] == "anndata":
        dictionary_for_init = {k: ad.io.read_elem(f[k]) for k, v in dict(f).items() if not k.startswith("raw.")}

    else:
        err = f"Unkown encoding-type '{f.attrs['encoding-type']}'."
        raise ValueError(err)

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


def _allow_write_nullable_strings[T, **P](f: Callable[P, T]) -> Callable[P, T]:
    @wraps(f)
    def wrapped(*args: P.args, **kwargs: P.kwargs):
        with ad.settings.override(allow_write_nullable_strings=True):
            return f(*args, **kwargs)

    return wrapped


@_allow_write_nullable_strings
def write_zarr(
    edata: EHRData,
    filename: str | Path,
    *,
    chunks: Literal["auto" | "ehrdata_auto"] = "auto",
    convert_strings_to_categoricals: bool = True,
) -> None:
    """Write :class:`~ehrdata.EHRData` objects to disk.

    To write to a `.zarr` file, `X`, and `layers` cannot be written as `object` dtype.
    If any of these fields is of `object` dtype, it this function will attempt to cast it to a numeric dtype; if this fails, the field will be casted to a `str` dtype.

    Args:
        edata: Central data object.
        filename: Name of the output file, can also be prefixed with relative or absolute path to save the file to.
        chunks: Specify strategy of how data should be chunked. For simplicity, currently only 2 options are available: "auto" will write the data with :func:`~anndata.io.write_elem`'s default settings. `chunks='ehrdata_auto'` will write the data chunked (and sharded) based on a heuristic that loosely speaking writes slightly smaller chunks.
        convert_strings_to_categoricals: Convert columns of `str` dtype in `.obs` and `.var` and `.tem` to `categorical` dtype.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.io.write_zarr("mimic_2.zarr", edata)
    """
    filename = Path(filename)
    edata = _cast_arrays_dtype_to_float_or_str_if_nonnumeric_object(edata)

    store = zarr.open_group(filename, mode="a", use_consolidated=False, zarr_format=3)

    adata = ad.AnnData(edata)

    if convert_strings_to_categoricals:
        adata.strings_to_categoricals(adata.obs)
        adata.strings_to_categoricals(adata.var)
        adata.strings_to_categoricals(edata.tem)

    # write_sharded this is a slightly modified version from https://anndata.readthedocs.io/en/stable/tutorials/zarr-v3.html
    # write_sharded is intended as a future blueprint of implementing better chunking defaults for ehrdata based based on real usecases
    def write_sharded(group: zarr.Group, adata: ad.AnnData):
        def callback(
            func: ad.experimental.Write,
            g: zarr.Group,
            k: str,
            elem: ad.typing.RWAble,
            dataset_kwargs: Mapping[str, Any],
            iospec: ad.experimental.IOSpec,
        ):
            if iospec.encoding_type in {"array"} and not isinstance(elem, list):
                dataset_kwargs = {
                    "shards": tuple(int(2 ** (16 / len(elem.shape))) for _ in elem.shape),
                    **dataset_kwargs,
                }
                dataset_kwargs["chunks"] = tuple(i // 2 for i in dataset_kwargs["shards"])
            elif iospec.encoding_type in {"csr_matrix", "csc_matrix"}:
                dataset_kwargs = {"shards": (2**16,), "chunks": (2**8,), **dataset_kwargs}
            func(g, k, elem, dataset_kwargs=dataset_kwargs)

        return ad.experimental.write_dispatched(group, "/", adata, callback=callback)

    if chunks == "auto":
        ad.io.write_elem(store, "anndata", adata)
    elif chunks == "ehrdata_auto":
        anndata_group = store.create_group("anndata")
        write_sharded(anndata_group, adata)
    else:
        err = (
            f"chunks={chunks} is not implemented. Currently, only chunks='auto' and chunks='ehrdata_auto' is supported."
        )
        raise NotImplementedError(err)

    ad.io.write_elem(store, "tem", edata.tem)

    store.attrs["encoding-version"] = EHRDATA_ZARR_ENCODING_VERSION
    store.attrs["encoding-type"] = "ehrdata"
