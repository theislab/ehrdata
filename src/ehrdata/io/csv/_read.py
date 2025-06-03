from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from ehrdata import EHRData


def read_csv(
    filename: Path | str,
    sep: str = ",",
    index_column: str | None = None,
    columns_obs_only: list[str] | None = None,
    **kwargs,
) -> EHRData:
    """Reads a csv file.

    This function reads a csv file, and creates an :class:`ehrdata.EHRData` object.
    It first reads the csv file using :func:`pandas.read_csv`, and then passes the resulting DataFrame to :func:`ehrdata.tl.from_dataframe`.

    Args:
        filename: Path to the file or directory to read.
        sep: Separator in the file. Delegates to pandas.read_csv().
        index_column: The index column of obs. Usually the patient visit ID or the patient ID.
        columns_obs_only: These columns will be added to the `obs` DataFrame only.

        **kwargs: Passed to :func:`pandas.read_csv`

    Returns:
        The dataset in the form of a data object.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.io.read_csv("myfile.csv")
    """
    from ehrdata.tl import from_dataframe

    df = pd.read_csv(filename, sep=sep, index_col=index_column, **kwargs)
    edata = from_dataframe(df, columns_obs_only=columns_obs_only)

    return edata


# def read_h5ad(
#     file_name: Path | str,
#     backed: Literal["r", "r+"] | bool | None = None,
# ) -> EHRData:
#     """Reads an h5ad file.

#     Args:
#         file_name: Path to the file or directory to read.
#         backed: If 'r', load AnnData in backed mode instead of fully loading it into memory (memory mode). If you want to modify backed attributes of the AnnData object, you need to choose 'r+'.
#             Currently, backed only support updates to X. That means any changes to other slots like obs will not be written to disk in backed mode. If you would like save changes made to these slots of a backed AnnData, write them to a new file (see write()). For an example, see Partial reading of large data.

#     Returns:
#         Returns the data as a data object.

#     Examples:
#         >>> import ehrdata as ed
#         >>> edata = ed.dt.mimic_2()
#         >>> ed.io.write("mimic_2.h5ad", edata)
#         >>> edata_2 = ed.io.read_h5ad("mimic_2.h5ad")
#     """
#     file_path = Path(file_name)

#     import anndata as ad

#     edata = EHRData.from_anndata(ad.read_h5ad(file_path, backed=backed))
#     # if "ehrapy_dummy_encoding" in edata.uns.keys():
#     #     # if dummy encoding was needed, the original dtype of X could not be numerical, so cast it to object
#     #     edata.X = edata.X.astype("object")
#     #     decoded_edata = _decode_cached_edata(edata, list(edata.uns["columns_obs_only"]))
#     #     return decoded_edata
#     return edata
