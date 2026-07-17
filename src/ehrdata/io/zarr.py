from __future__ import annotations

import warnings
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import zarr

from ehrdata._feature_types import _harmonize_on_read
from ehrdata.core.constants import (
    EHRDATA_ENCODING_TYPE,
    EHRDATA_ENCODING_TYPE_KEY_ZARR,
    EHRDATA_ONDISK_VERSION,
    EHRDATA_ONDISK_VERSION_KEY,
)
from ehrdata.io._array_casting import _cast_arrays_dtype_to_float_or_str_if_nonnumeric_object, _cast_variables_to_float
from ehrdata.io._coo_codec import write_coo_zarr
from ehrdata.io._ondisk import (
    _check_020_ehrdata_on_disk_format,
    decode_init_dict,
    encode_for_disk,
    read_mapping_with_coo,
)

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

    if EHRDATA_ENCODING_TYPE_KEY_ZARR not in f.attrs:
        err = f"The zarr store does not contain the required '{EHRDATA_ENCODING_TYPE_KEY_ZARR}' attribute."
        raise ValueError(err)

    encoding_type = f.attrs[EHRDATA_ENCODING_TYPE_KEY_ZARR]
    if encoding_type == EHRDATA_ENCODING_TYPE:
        if "anndata" in f:
            dictionary_for_init = {}
            for k in dict(f["anndata"]):
                if k.startswith("raw."):
                    continue
                # obsm may hold ehrdata COO groups anndata can't read, so read it child-by-child.
                dictionary_for_init[k] = (
                    read_mapping_with_coo(f["anndata"]["obsm"]) if k == "obsm" else ad.io.read_elem(f["anndata"][k])
                )
        else:
            err = "The zarr store does not contain the 'anndata' group."
            raise ValueError(err)
        if "tem" in f:
            dictionary_for_init["tem"] = ad.io.read_elem(f["tem"])
        else:
            warnings.warn("The zarr store does not contain the 'tem' group.", stacklevel=2)

    elif encoding_type == "anndata":
        dictionary_for_init = {}
        for k in dict(f):
            if k.startswith("raw."):
                continue
            dictionary_for_init[k] = read_mapping_with_coo(f["obsm"]) if k == "obsm" else ad.io.read_elem(f[k])

    else:
        err = f"Unknown encoding-type '{encoding_type}'."
        raise ValueError(err)

    if _check_020_ehrdata_on_disk_format(f):
        dictionary_for_init = decode_init_dict(dictionary_for_init)
    edata = EHRData(**dictionary_for_init)

    if harmonize_missing_values:
        _harmonize_on_read(edata)

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

    To write to a `.zarr` store, `X`, and `layers` cannot be written as `object` dtype.
    If any of these fields is of `object` dtype, this function will attempt to cast it to a numeric dtype; if this fails, the field will be casted to a `str` dtype.


    Args:
        edata: Central data object.
        filename: Name of the output file, can also be prefixed with relative or absolute path to save the file to.
        chunks: Specify strategy of how data should be chunked. For simplicity, currently only 2 options are available: `"auto"` will write the data with :func:`~anndata.io.write_elem`'s default settings. `"ehrdata_auto"` will write the data chunked (and sharded) based on a heuristic that loosely speaking writes slightly smaller chunks.
        convert_strings_to_categoricals: Convert columns of `str` dtype in `.obs` and `.var` and `.tem` to `categorical` dtype.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.io.write_zarr("mimic_2.ehrdata.zarr", edata)
    """
    filename = Path(filename)
    edata = _cast_arrays_dtype_to_float_or_str_if_nonnumeric_object(edata)

    store = zarr.open_group(filename, mode="a", use_consolidated=False, zarr_format=3)

    adata, coo_obsm = encode_for_disk(edata)

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

        # anndata 0.13 rejects writing with key "/" into a non-root subgroup, so dispatch from the root store under the "anndata" key (matching the chunks="auto" write_elem path).
        return ad.experimental.write_dispatched(group, "anndata", adata, callback=callback)

    if chunks == "auto":
        ad.io.write_elem(store, "anndata", adata)
    elif chunks == "ehrdata_auto":
        write_sharded(store, adata)
    else:
        err = (
            f"chunks={chunks} is not implemented. Currently, only chunks='auto' and chunks='ehrdata_auto' is supported."
        )
        raise NotImplementedError(err)

    # ehrdata serializes sparse.COO tensors itself (anndata has no COO writer); see io._coo_codec.
    if coo_obsm:
        obsm_group = store["anndata"].require_group("obsm")
        for key, coo in coo_obsm.items():
            write_coo_zarr(obsm_group.require_group(key), coo)

    ad.io.write_elem(store, "tem", edata.tem)

    store.attrs[EHRDATA_ENCODING_TYPE_KEY_ZARR] = EHRDATA_ENCODING_TYPE
    store.attrs[EHRDATA_ONDISK_VERSION_KEY] = EHRDATA_ONDISK_VERSION
