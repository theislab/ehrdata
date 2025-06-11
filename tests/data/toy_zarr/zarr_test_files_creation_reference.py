#########################
# Toy zarr files creation
#########################
import anndata as ad
import numpy as np
import pandas as pd
import zarr

#########################
# adata_basic.zarr:  basic AnnData object


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
adata_basic.write_zarr("adata_basic.zarr")

#########################
# adata_basic.zarr:  basic EHRData object with tem, R

f = zarr.open("edata_basic_with_tem.zarr", mode="w")
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
for k, v in edata_basic_with_tem_dict.items():
    ad.io.write_elem(f, k, v)
