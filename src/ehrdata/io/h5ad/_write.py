from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ehrdata import EHRData


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

    # if not issparse(edata.X) and edata.X.dtype == np.object_:
    #     edata.X = edata.X.astype(str)
    # for layer, array in edata.layers.items():
    #     if not issparse(array) and array.dtype == np.object_:
    #         edata.layers[layer] = array.astype(str)

    edata.write_h5ad(filename, compression=compression, compression_opts=compression_opts)
