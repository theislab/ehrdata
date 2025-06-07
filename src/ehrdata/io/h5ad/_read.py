from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

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

    # TODO: support tem
    edata = EHRData.from_adata(ad.read_h5ad(f"{file_name}", backed=backed))
    edata = EHRData(
        X=edata.X,
        R=edata.R,
        obs=edata.obs,
        var=edata.var,
        uns=edata.uns,
        obsm=edata.obsm,
        varm=edata.varm,
        obsp=edata.obsp,
        varp=edata.varp,
    )

    return edata
