from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import zarr

import ehrdata as ed
from ehrdata._logger import logger
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
    from ehrdata import EHRData

    # TODO: check that anndata can be read
    # TODO: check for backwards compatibility and announce clear version of when it stops
    # TODO: check that ehrdata can be read

    if isinstance(filename, Path):
        filename = str(filename)

    f = filename if isinstance(filename, zarr.Group) else zarr.open(filename, mode="r")

    if "ehrdata" in f.attrs:
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
    else:
        warnings.warn(
            "The zarr store does not contain an ehrdata attribute. This is might not be a valid ehrdata Zarr store, and the store might not be readable or be wrongly interpeted.",
            stacklevel=2,
        )
        if "anndata" in f.attrs:
            warnings.warn(
                "The zarr store is an AnnData store, which can be read but might not support all ehrdata features.",
                stacklevel=2,
            )

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

    To write to a `.zarr` file, `X`, and `layers` cannot be written as `object` dtype.
    If any of these fields is of `object` dtype, it this function will attempt to cast it to a numeric dtype; if this fails, the field will be casted to a `str` dtype.

    Args:
        edata: Central data object.
        filename: Name of the output file, can also be prefixed with relative or absolute path to save the file to.
        chunks: Chunk shape, passed to :meth:`zarr.Group.create_array` for `Zarr` version 3.
        convert_strings_to_categoricals: Convert columns of `str` dtype in `.obs` and `.var` and `.tem` to `categorical` dtype.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.io.write_zarr("mimic_2.zarr", edata)
    """
    filename = Path(filename)
    edata = _cast_arrays_dtype_to_float_or_str_if_nonnumeric_object(edata)

    # TODO: add test that checks these fields
    # TODO: ensure this is "canonical" and what anndata is doing
    store = zarr.open(filename, mode="w")
    store.attrs["ehrdata_version"] = ed.__version__
    store.attrs["ehrdata_type"] = "ehrdata"

    # while adata.write_zarr supports convert_strings_to_categoricals, ad.io.write_elem does not
    # so we need to convert the strings to categoricals ourselves
    # TODO: figure out what anndata is doing
    # map all columns of edata.obs to categorical dtype if they are of str dtype
    # def _convert_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    #     for column in df.columns:
    #         if df[column].dtype == "str":
    #             df[column] = df[column].astype("category")
    #     return df

    adata = ad.AnnData(edata)

    # TODO: test
    if convert_strings_to_categoricals:
        # inplace conversion
        adata.strings_to_categoricals(adata.obs)
        adata.strings_to_categoricals(adata.var)
        adata.strings_to_categoricals(edata.tem)
    # adata.obs = edata.obs if not convert_strings_to_categoricals else _convert_to_categorical(edata.obs)
    # adata.var = edata.var if not convert_strings_to_categoricals else _convert_to_categorical(edata.var)
    # adata.tem = edata.tem if not convert_strings_to_categoricals else _convert_to_categorical(edata.tem)

    ad.io.write_elem(
        store,
        "anndata",
        adata,  # this will store everything but the .tem field
        dataset_kwargs={"chunks": chunks},
    )

    ad.io.write_elem(store, "tem", edata.tem)
