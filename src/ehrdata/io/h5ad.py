from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.sparse import issparse

from ehrdata.tl import infer_feature_types

if TYPE_CHECKING:
    from ehrdata import EHRData


def read_h5ad(
    file_name: Path | str,
    backed: Literal["r", "r+"] | bool | None = None,
) -> EHRData:
    """Reads an h5ad file.

    Args:
        file_name: Path to the file or directory to read.
        backed: If 'r', load AnnData in backed mode instead of fully loading it into memory (memory mode). If you want to modify backed attributes of the AnnData object, you need to choose 'r+'.
            Currently, backed only support updates to X. That means any changes to other slots like obs will not be written to disk in backed mode. If you would like save changes made to these slots of a backed AnnData, write them to a new file (see write()). For an example, see Partial reading of large data.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ep.dt.mimic_2()
        >>> ed.io.write("mimic_2.h5ad", edata)
        >>> edata_2 = ed.io.read_h5ad("mimic_2.h5ad")
    """
    import anndata as ad

    from ehrdata import EHRData

    # TODO: support temp files
    # TODO: cast to object dtype if string dtype
    edata = EHRData.from_adata(ad.read_h5ad(f"{file_name}", backed=backed))

    return edata


def write_h5ad(
    edata: EHRData,
    filename: str | Path,
    *,
    compression: Literal["gzip", "lzf"] | None = None,
    compression_opts: int | None = None,
) -> None:
    """Write :class:`~ehrdata.EHRData` objects to file.

    It is possbile to either write an :class:`~ehrdata.EHRData` object to an .h5ad file.
    The .h5ad file can be used as a cache to save the current state of the object and to retrieve it faster once needed.
    This preserves the object state at the time of writing.

    Args:
        filename: File name or path to write the file to.
        edata: Annotated data matrix.
        compression: Optional file compression.
        compression_opts: See http://docs.h5py.org/en/latest/high/dataset.html.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.io.write_h5ad("mimic_2.h5ad", edata)
    """
    filename = Path(filename)  # allow passing strings

    infer_feature_types(edata)
    # TODO: support tem
    # numpy object dtype is not supported by h5ad;
    # if numeric, store as numeric;
    # if object, convert to (expensive) string
    # sparse matrices can't be dtype object, don't need to worry about them
    if not issparse(edata.X) and edata.X.dtype == np.object_:
        edata.X = edata.X.astype(str)
    for layer, array in edata.layers.items():
        if not issparse(array) and array.dtype == np.object_:
            edata.layers[layer] = array.astype(str)

    edata.write_h5ad(filename, compression=compression, compression_opts=compression_opts)
