#########################
# Toy zarr files creation
#########################
import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy as sp
import zarr

#########################
# adata_basic.zarr/h5ad:  basic AnnData object

adata_basic = ad.AnnData(
    X=np.ones((5, 4)),
    obs=pd.DataFrame({"survival": [1, 2, 3, 4, 5]}),
    var=pd.DataFrame({"variables": ["var_1", "var_2", "var_3", "var_4"]}),
    obsm={"obs_level_representation": np.ones((5, 2))},
    varm={"var_level_representation": np.ones((4, 2))},
    layers={"other_layer": np.ones((5, 4))},
    obsp={"obs_level_connectivities": np.ones((5, 5))},
    varp={"var_level_connectivities": np.random.randn(4, 4)},
    uns={"information": ["info1"]},
)
# adata_basic.write_zarr("toy_zarr/adata_basic.zarr")
# adata_basic.write_h5ad("toy_h5ad/adata_basic.h5ad")

#########################
# edata_basic_with_tem.zarr/h5ad:  basic EHRData object with tem, R


edata_basic_with_tem_dict = {
    "X": np.ones((5, 4)),
    "R": np.ones((5, 4, 2)),
    "obs": pd.DataFrame({"survival": [1, 2, 3, 4, 5]}),
    "var": pd.DataFrame({"variables": ["var_1", "var_2", "var_3", "var_4"]}),
    "obsm": {"obs_level_representation": np.ones((5, 2))},
    "varm": {"var_level_representation": np.ones((4, 2))},
    "layers": {"other_layer": np.ones((5, 4))},
    "obsp": {"obs_level_connectivities": np.ones((5, 5))},
    "varp": {"var_level_connectivities": np.random.randn(4, 4)},
    "uns": {"information": ["info1"]},
    "tem": pd.DataFrame({"timestep": ["t1", "t2"]}),
}
with zarr.open("toy_zarr/edata_basic_with_tem.zarr", "w") as zarr_file:
    for k, v in edata_basic_with_tem_dict.items():
        ad.io.write_elem(zarr_file, k, v)


with h5py.File("toy_h5ad/edata_basic_with_tem.h5ad", "w") as h5ad_file:
    for k, v in edata_basic_with_tem_dict.items():
        ad.io.write_elem(h5ad_file, k, v)


#########################
# edata_sparse_with_tem.zarr/h5ad:  basic EHRData object with tem, R, sparse X
edata_sparse_with_tem_dict = edata_basic_with_tem_dict.copy()
edata_sparse_with_tem_dict["X"] = sp.sparse.csr_matrix(edata_sparse_with_tem_dict["X"])
edata_sparse_with_tem_dict["layers"]["other_layer"] = sp.sparse.csr_matrix(
    edata_sparse_with_tem_dict["layers"]["other_layer"]
)
with zarr.open("toy_zarr/edata_sparse_with_tem.zarr", "w") as zarr_file:
    for k, v in edata_sparse_with_tem_dict.items():
        ad.io.write_elem(zarr_file, k, v)


with h5py.File("toy_h5ad/edata_sparse_with_tem.h5ad", "w") as h5ad_file:
    for k, v in edata_sparse_with_tem_dict.items():
        ad.io.write_elem(h5ad_file, k, v)
