import numpy as np
from anndata import AnnData

from ehrdata.pl.vitessce import gen_config


def test_gen_config(tmp_path):
    adata = AnnData(
        X=np.array([[1, 2, 3], [4, 5, 6]]),
        obs={"gender_concept_id": ["M", "F"]},
        obsm={"X_pca": np.array([[1, 2], [3, 4]])},
    )
    adata.write_zarr(path := tmp_path / "test.zarr")
    gen_config(path)
