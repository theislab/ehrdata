from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import zarr

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
        >>> ed.io.write_zarr("mimic_2.h5ad", edata)
        >>> edata_2 = ed.io.read_zarr("mimic_2.h5ad")
    """
    from ehrdata import EHRData

    f = file_name if isinstance(file_name, zarr.Group) else zarr.open(file_name, mode="r")

    dictionary_for_init = {k: ad.io.read_elem(f[k]) for k, v in dict(f).items() if not k.startswith("raw.")}
    edata = EHRData(**dictionary_for_init)

    return edata


def write_zarr(
    edata: EHRData,
    filename: str | Path,
    *,
    chunks: bool | int | tuple[int, ...] | None = None,
    convert_strings_to_categoricals: bool = True,
) -> None:
    """Write :class:`~ehrdata.EHRData` objects to file.

    It is possbile to either write an :class:`~ehrdata.EHRData` object to an .h5ad file.
    The .h5ad file can be used as a cache to save the current state of the object and to retrieve it faster once needed.
    This preserves the object state at the time of writing.

    Args:
        edata: Annotated data matrix.
        filename: File name or path to write the file to.
        chunks: Chunk shape.
        convert_strings_to_categoricals: Convert string columns to categorical.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.io.write_h5ad("mimic_2.h5ad", edata)
    """
    filename = Path(filename)  # allow passing strings
    # TODO: support temp files

    # if not issparse(edata.X) and edata.X.dtype == np.object_:
    #     edata.X = edata.X.astype(str)
    # for layer, array in edata.layers.items():
    #     if not issparse(array) and array.dtype == np.object_:
    #         edata.layers[layer] = array.astype(str)
    ad.AnnData(edata.X).write_zarr(
        filename,
        chunks=chunks,
    )
    # edata.write_zarr()
    # adata = EHRData.to_adata(edata)
    # adata.write_zarr(filename, chunks=chunks, convert_strings_to_categoricals=convert_strings_to_categoricals)
