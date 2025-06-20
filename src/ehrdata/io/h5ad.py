from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anndata as ad
import h5py
import numpy as np
from scipy.sparse import issparse

if TYPE_CHECKING:
    from ehrdata import EHRData


def read_h5ad(
    filename: Path | str,
    backed: Literal["r", "r+"] | bool | None = None,
) -> EHRData:
    """Reads an h5ad file.

    Args:
        filename: Path to the file or directory to read.
        backed: If 'r', load :class:`~ehrdata.EHRData` in backed mode instead of fully loading it into memory (memory mode). If you want to modify backed attributes of the :class:`~ehrdata.EHRData` object, you need to choose 'r+'.
            Currently, backed only support updates to `X`. That means any changes to other slots like obs will not be written to disk in backed mode. If you would like save changes made to these slots of a backed EHRData, write them to a new file (see write()). For an example, see Partial reading of large data.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ep.dt.mimic_2()
        >>> ed.io.write("mimic_2.h5ad", edata)
        >>> edata_2 = ed.io.read_h5ad("mimic_2.h5ad")
    """
    from ehrdata import EHRData
    from ehrdata.tl import harmonize_missing_values

    with h5py.File(filename, "r") as f:
        dictionary_for_init = {k: ad.io.read_elem(f[k]) for k, v in dict(f).items() if not k.startswith("raw.")}

    # If X, layers is str; convert to object dtype. First, try to cast each column to float64 in pandas.
    if "X" in dictionary_for_init and dictionary_for_init["X"].dtype == str:
        dictionary_for_init["X"] = dictionary_for_init["X"].astype(object)
    if "layers" in dictionary_for_init:
        for key in dictionary_for_init["layers"]:
            if dictionary_for_init["layers"][key].dtype == str:
                dictionary_for_init["layers"][key] = dictionary_for_init["layers"][key].astype(object)

    edata = EHRData(**dictionary_for_init)

    # cast "nan" and other designated missing value strings to np.nan to enable to-float casting if only numbers and missing values
    harmonize_missing_values(edata)
    if not issparse(edata.X):
        for column in range(edata.X.shape[1]):
            with contextlib.suppress(ValueError):
                edata.X[:, column] = edata.X[:, column].astype(np.float64)
    for key in edata.layers:
        harmonize_missing_values(edata, layer=key)
        if not issparse(edata.layers[key]):
            for column in range(edata.layers[key].shape[1]):
                with contextlib.suppress(ValueError):
                    edata.layers[key][:, column] = edata.layers[key][:, column].astype(np.float64)

    return edata


def write_h5ad(
    edata: EHRData,
    filename: str | Path,
    *,
    compression: Literal["gzip", "lzf"] | None = None,
    compression_opts: int | None = None,
) -> None:
    """Write :class:`~ehrdata.EHRData` objects to disk.

    It is possible to either write an :class:`~ehrdata.EHRData` object to an `.h5ad` or a compressed `.gzip` or `lzf` file.

    Args:
        filename: File name or path to write the file to.
        edata: Data object.
        compression: Optional file compression. Setting compression to 'gzip' can save disk space but will slow down writing and subsequent reading.
        compression_opts: See http://docs.h5py.org/en/latest/high/dataset.html.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.io.write_h5ad("mimic_2.h5ad", edata)
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
    ad.AnnData(edata).write_h5ad(
        filename,
        compression=compression,
        compression_opts=compression_opts,
    )
    with h5py.File(filename, "a") as f:
        ad.io.write_elem(f, "tem", edata.tem)
