import numpy as np
import pytest
from anndata import AnnData

from ehrdata.pl.vitessce import gen_config


@pytest.fixture
def adata() -> AnnData:
    return AnnData(
        X=np.array([[1, 2, 3], [4, 5, 6]]),
        obs={"gender_concept_id": ["M", "F"]},
        obsm={"X_pca": np.array([[1, 2], [3, 4]])},
    )


def test_gen_config(adata, tmp_path):
    adata.write_zarr(path := tmp_path / "test.zarr")
    gen_config(path)


# needs more setup until it works
# def test_gen_config_lamin(adata):
#    artifact = ln.Artifact.from_anndata(adata, description="Test AnnData")
#    gen_config(artifact=artifact)
