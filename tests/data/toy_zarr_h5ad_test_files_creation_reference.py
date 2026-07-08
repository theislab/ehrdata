#########################
# Toy zarr / h5ad / h5ed files creation
#########################
# Reference script that (re)creates the committed toy fixtures. Paths are resolved relative to
# THIS file's directory (`tests/data/`), so it can be run from any working directory without
# scattering stores into the wrong place (e.g. a stray `tests/toy_zarr/`).
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy as sp

import ehrdata as ed
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

HERE = Path(__file__).parent
TOY_ZARR = HERE / "toy_zarr"
TOY_H5AD = HERE / "toy_h5ad"
TOY_H5ED = HERE / "toy_h5ed"

#########################
# adata_basic.zarr/h5ad:  basic AnnData object ("version 0" = plain anndata, pre-ehrdata-format)

adata_basic = ad.AnnData(
    X=np.ones((5, 4)),
    obs=pd.DataFrame({"survival": [1, 2, 3, 4, 5]}, index=[str(i) for i in range(5)]),
    var=pd.DataFrame({"variables": ["var_1", "var_2", "var_3", "var_4"]}, index=[str(i) for i in range(4)]),
    obsm={"obs_level_representation": np.ones((5, 2))},
    varm={"var_level_representation": np.ones((4, 2))},
    layers={"other_layer": np.ones((5, 4))},
    obsp={"obs_level_connectivities": np.ones((5, 5))},
    varp={"var_level_connectivities": np.random.randn(4, 4)},
    uns={"information": ["info1"]},
)
adata_basic.write_zarr(TOY_ZARR / "adata_basic.zarr")
adata_basic.write_h5ad(TOY_H5AD / "adata_basic.h5ad")

#########################
# edata_basic_with_tem.zarr/h5ad:  basic EHRData object with tem, 3dlayer


edata_basic_with_tem_dict = {
    "X": np.ones((5, 4)),
    "obs": pd.DataFrame({"survival": [1, 2, 3, 4, 5]}, index=[str(i) for i in range(5)]),
    "var": pd.DataFrame({"variables": ["var_1", "var_2", "var_3", "var_4"]}, index=[str(i) for i in range(4)]),
    "obsm": {"obs_level_representation": np.ones((5, 2))},
    "varm": {"var_level_representation": np.ones((4, 2))},
    "layers": {DEFAULT_TEM_LAYER_NAME: np.ones((5, 4, 2)), "other_layer": np.ones((5, 4))},
    "obsp": {"obs_level_connectivities": np.ones((5, 5))},
    "varp": {"var_level_connectivities": np.random.randn(4, 4)},
    "uns": {"information": ["info1"]},
    "tem": pd.DataFrame({"timestep": ["t1", "t2"]}, index=[str(i) for i in range(2)]),
}
ed.io.write_zarr(ed.EHRData(**edata_basic_with_tem_dict), TOY_ZARR / "edata_basic_with_tem.zarr")

with h5py.File(TOY_H5AD / "edata_basic_with_tem.h5ad", "w") as h5ad_file:
    for k, v in edata_basic_with_tem_dict.items():
        ad.io.write_elem(h5ad_file, k, v)


#########################
# edata_sparse_with_tem.zarr/h5ad:  basic EHRData object with tem, 3dlayer, sparse X
edata_sparse_with_tem_dict = edata_basic_with_tem_dict.copy()
edata_sparse_with_tem_dict["X"] = sp.sparse.csr_matrix(edata_sparse_with_tem_dict["X"])
edata_sparse_with_tem_dict["layers"]["other_layer"] = sp.sparse.csr_matrix(
    edata_sparse_with_tem_dict["layers"]["other_layer"]
)
ed.io.write_zarr(ed.EHRData(**edata_sparse_with_tem_dict), TOY_ZARR / "edata_sparse_with_tem.zarr")

with h5py.File(TOY_H5AD / "edata_sparse_with_tem.h5ad", "w") as h5ad_file:
    for k, v in edata_sparse_with_tem_dict.items():
        ad.io.write_elem(h5ad_file, k, v)


#########################
# edata_minimal_v0_2_0.h5ed / .ehrdata.zarr:  tiny EHRData written in the current 0.2.0 format
# (3D arrays relocated into .obsm + `ehrdata-encoding-version="0.2.0"` stamp). Together with the
# plain-anndata "version 0" adata_basic.* above, these form the minimal read-test corpus
# (2x2: {h5, zarr} x {version-0 plain-anndata, 0.2.0 ehrdata}). Deterministic (no RNG).
TOY_H5ED.mkdir(exist_ok=True)
edata_minimal = ed.EHRData(
    X=np.arange(6, dtype=float).reshape(3, 2),
    layers={DEFAULT_TEM_LAYER_NAME: np.arange(3 * 2 * 2, dtype=float).reshape(3, 2, 2)},
    obs=pd.DataFrame({"survival": [1, 2, 3]}, index=[str(i) for i in range(3)]),
    var=pd.DataFrame({"variables": ["var_1", "var_2"]}, index=[str(i) for i in range(2)]),
    tem=pd.DataFrame({"timestep": ["t1", "t2"]}, index=[str(i) for i in range(2)]),
)
ed.io.write_h5ed(edata_minimal.copy(), TOY_H5ED / "edata_minimal_v0_2_0.h5ed")
ed.io.write_zarr(edata_minimal.copy(), TOY_ZARR / "edata_minimal_v0_2_0.ehrdata.zarr")
