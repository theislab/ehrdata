import numpy as np
import pytest
from anndata import AnnData

pytest.importorskip("vitessce")

from ehrdata.integrations.vitessce import gen_config


@pytest.fixture
def adata() -> AnnData:
    return AnnData(
        X=np.array([[1, 2, 3], [4, 5, 6]]),
        obs={"gender_concept_id": ["M", "F"]},
        obsm={"X_pca": np.array([[1, 2], [3, 4]])},
    )


def test_gen_config(adata, tmp_path):
    adata.write_zarr(path := tmp_path / "test.zarr")
    with pytest.warns(FutureWarning):
        vc = gen_config(path)
    assert vc is not None
    assert hasattr(vc, "widget")
    assert (tmp_path / "test.zarr").exists()


def test_gen_default_config_basic(edata_blobs_small, tmp_path):
    """Test gen_default_config with basic obs_columns only."""
    import ehrdata as ed

    # Add categorical columns to .obs
    np.random.seed(42)
    edata_blobs_small.obs["Gender"] = np.random.choice(["M", "F"], size=edata_blobs_small.n_obs)

    # Generate config with just obs_columns
    vc = ed.integrations.vitessce.gen_default_config(
        edata_blobs_small,
        zarr_filepath=tmp_path / "test_basic.zarr",
        obs_columns=["Gender"],
        layer="tem_data",
        timestep=3,
    )

    # Verify config was created
    assert vc is not None
    assert hasattr(vc, "widget")
    assert (tmp_path / "test_basic.zarr").exists()


def test_gen_default_config_with_embedding(edata_blobs_small, tmp_path):
    """Test gen_default_config with obs_columns and obs_embedding."""
    import ehrdata as ed

    # Add categorical columns and embedding
    np.random.seed(42)
    edata_blobs_small.obs["Gender"] = np.random.choice(["M", "F"], size=edata_blobs_small.n_obs)
    edata_blobs_small.obs["Age_Group"] = np.random.choice(["<30", "30-60", ">60"], size=edata_blobs_small.n_obs)
    edata_blobs_small.obsm["X_pca"] = np.random.randn(edata_blobs_small.n_obs, 2)

    # Generate config with obs_columns and embedding
    vc = ed.integrations.vitessce.gen_default_config(
        edata_blobs_small,
        zarr_filepath=tmp_path / "test_embedding.zarr",
        obs_columns=["Gender", "Age_Group"],
        obs_embedding="X_pca",
        layer="tem_data",
        timestep=4,
    )

    # Verify config was created
    assert vc is not None
    assert hasattr(vc, "widget")
    assert (tmp_path / "test_embedding.zarr").exists()


def test_gen_default_config_with_scatter(edata_blobs_small, tmp_path):
    """Test gen_default_config with obs_columns and scatter plot variables."""
    import ehrdata as ed

    # Add categorical column
    np.random.seed(42)
    edata_blobs_small.obs["Cluster"] = edata_blobs_small.obs["cluster"].astype(str)

    # Get first two variables for scatter
    scatter_vars = list(edata_blobs_small.var_names[:2])

    # Generate config with obs_columns and scatter vars
    vc = ed.integrations.vitessce.gen_default_config(
        edata_blobs_small,
        zarr_filepath=tmp_path / "test_scatter.zarr",
        obs_columns=["Cluster"],
        scatter_var_cols=scatter_vars,
        layer="tem_data",
        timestep=1,
    )

    # Verify config was created
    assert vc is not None
    assert hasattr(vc, "widget")
    assert (tmp_path / "test_scatter.zarr").exists()


def test_gen_default_config_with_embedding_and_scatter(edata_blobs_small, tmp_path):
    """Test gen_default_config with all options: obs_columns, embedding, and scatter."""
    import ehrdata as ed

    # Add all necessary fields
    np.random.seed(42)
    edata_blobs_small.obs["Gender"] = np.random.choice(["M", "F"], size=edata_blobs_small.n_obs)
    edata_blobs_small.obs["Age_Group"] = np.random.choice(["<30", "30-60", ">60"], size=edata_blobs_small.n_obs)
    edata_blobs_small.obsm["X_pca"] = np.random.randn(edata_blobs_small.n_obs, 2)

    # Get first two variables for scatter
    scatter_vars = list(edata_blobs_small.var_names[:2])

    # Generate config with both embedding and scatter
    vc = ed.integrations.vitessce.gen_default_config(
        edata_blobs_small,
        zarr_filepath=tmp_path / "test_embedding_and_scatter.zarr",
        obs_columns=["Gender", "Age_Group"],
        obs_embedding="X_pca",
        scatter_var_cols=scatter_vars,
        layer="tem_data",
        timestep=4,
    )

    # Verify config was created
    assert vc is not None
    assert hasattr(vc, "widget")
    assert (tmp_path / "test_embedding_and_scatter.zarr").exists()


def test_gen_default_config_illegal_arguments(edata_blobs_small, tmp_path):
    """Test gen_default_config with various illegal arguments."""
    import ehrdata as ed

    np.random.seed(42)
    edata_blobs_small.obs["Gender"] = np.random.choice(["M", "F"], size=edata_blobs_small.n_obs)
    edata_blobs_small.obsm["X_pca"] = np.random.randn(edata_blobs_small.n_obs, 2)

    # Test 1: Invalid obs_columns
    with pytest.raises(ValueError, match=r"not found in edata\.obs"):
        ed.integrations.vitessce.gen_default_config(
            edata_blobs_small,
            zarr_filepath=tmp_path / "test_invalid_obs.zarr",
            obs_columns=["NonexistentColumn"],
            layer="tem_data",
            timestep=0,
        )

    # Test 2: Invalid scatter_var_cols - wrong number of variables
    with pytest.raises(ValueError, match="scatter_var_cols must be an Iterable of 2 variables"):
        ed.integrations.vitessce.gen_default_config(
            edata_blobs_small,
            zarr_filepath=tmp_path / "test_invalid_scatter_count.zarr",
            obs_columns=["Gender"],
            scatter_var_cols=[edata_blobs_small.var_names[0]],  # Only 1 variable
            layer="tem_data",
            timestep=0,
        )

    # Test 3: Invalid scatter_var_cols - nonexistent variables
    with pytest.raises(ValueError, match=r"not found in edata\.var\.index"):
        ed.integrations.vitessce.gen_default_config(
            edata_blobs_small,
            zarr_filepath=tmp_path / "test_invalid_scatter_vars.zarr",
            obs_columns=["Gender"],
            scatter_var_cols=["nonexistent_var1", "nonexistent_var2"],
            layer="tem_data",
            timestep=0,
        )

    # Test 4: Invalid obs_embedding
    with pytest.raises(ValueError, match=r"Embedding .* not found in edata.obsm"):
        ed.integrations.vitessce.gen_default_config(
            edata_blobs_small,
            zarr_filepath=tmp_path / "test_invalid_embedding.zarr",
            obs_columns=["Gender"],
            obs_embedding="NonexistentEmbedding",
            layer="tem_data",
            timestep=0,
        )


# needs more setup until it works
# def test_gen_config_lamin(adata):
#    artifact = ln.Artifact.from_anndata(adata, description="Test AnnData")
#    gen_config(artifact=artifact)
