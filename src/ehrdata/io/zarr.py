from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import zarr
from scipy.sparse import issparse

if TYPE_CHECKING:
    from ehrdata import EHRData


def read_zarr(
    file_name: Path | str,
) -> EHRData:
    """Reads an zarr store.

    Args:
        file_name: Path to the file or directory to read.
        backed: If 'r', load AnnData in backed mode instead of fully loading it into memory (memory mode). If you want to modify backed attributes of the AnnData object, you need to choose 'r+'.
            Currently, backed only support updates to X. That means any changes to other slots like obs will not be written to disk in backed mode. If you would like save changes made to these slots of a backed AnnData, write them to a new file (see write()). For an example, see Partial reading of large data.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ep.dt.mimic_2()
        >>> ed.io.write_zarr("mimic_2.zarr", edata)
        >>> edata_2 = ed.io.read_zarr("mimic_2.zarr")
    """
    from ehrdata import EHRData
    from ehrdata.tl import harmonize_missing_values

    f = file_name if isinstance(file_name, zarr.Group) else zarr.open(file_name, mode="r")

    dictionary_for_init = {k: ad.io.read_elem(f[k]) for k, v in dict(f).items() if not k.startswith("raw.")}

    # If X, layers is str; convert to object dtype. First, try to cast each column to float64 in pandas.
    if "X" in dictionary_for_init:
        dictionary_for_init["X"] = dictionary_for_init["X"].astype(object)
    if "layers" in dictionary_for_init:
        for key in dictionary_for_init["layers"]:
            dictionary_for_init["layers"][key] = dictionary_for_init["layers"][key].astype(object)

    edata = EHRData(**dictionary_for_init)

    # cast "nan" and other designated missing value strings to np.nan to enable to-float casting if only numbers and missing values
    harmonize_missing_values(edata)
    for column in range(edata.X.shape[1]):
        with contextlib.suppress(ValueError):
            edata.X[:, column] = edata.X[:, column].astype(np.float64)
    for key in edata.layers:
        harmonize_missing_values(edata, layer=key)
        for column in range(edata.layers[key].shape[1]):
            with contextlib.suppress(ValueError):
                edata.layers[key][:, column] = edata.layers[key][:, column].astype(np.float64)

    return edata


def write_zarr(
    edata: EHRData,
    filename: str | Path,
    *,
    chunks: bool | int | tuple[int, ...] | None = None,
    convert_strings_to_categoricals: bool = True,
) -> None:
    """Write :class:`~ehrdata.EHRData` objects to file.

    It is possbile to either write an :class:`~ehrdata.EHRData` object to an .zarr file.
    The .zarr file can be used as a cache to save the current state of the object and to retrieve it faster once needed.
    This preserves the object state at the time of writing.

    Args:
        edata: Annotated data matrix.
        filename: File name or path to write the file to.
        chunks: Chunk shape.
        convert_strings_to_categoricals: Convert string columns to categorical.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.io.write_zarr("mimic_2.zarr", edata)
    """
    filename = Path(filename)  # allow passing strings

    if not issparse(edata.X) and edata.X.dtype == np.object_:
        try:
            edata.X = edata.X.astype(np.float64)
        except ValueError:
            edata.X = edata.X.astype(str)
    for layer, array in edata.layers.items():
        if not issparse(array) and array.dtype == np.object_:
            try:
                edata.layers[layer] = array.astype(np.float64)
            except ValueError:
                edata.layers[layer] = array.astype(str)
    ad.AnnData(edata).write_zarr(
        filename,
        chunks=chunks,
        convert_strings_to_categoricals=convert_strings_to_categoricals,
    )
    f = zarr.open(filename, mode="a")
    ad.io.write_elem(f, "tem", edata.tem)
