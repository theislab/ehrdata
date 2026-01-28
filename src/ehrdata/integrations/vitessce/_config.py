from __future__ import annotations

import warnings
from functools import reduce
from operator import or_, truediv
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import zarr
    from lamindb import Artifact
    from vitessce import VitessceConfig
    from zarr.storage import StoreLike

    import ehrdata as ed


def gen_config(
    path: Path | None = None,
    *,
    store: Path | StoreLike | None = None,
    url: str | None = None,
    artifact: Artifact | None = None,
    name: str | None = None,
    obs_sets: Mapping[str, str] = MappingProxyType({"obs/gender_concept_id": "Gender Concept ID"}),
    obs_embeddings: Mapping[str, str] = MappingProxyType({"obsm/X_pca": "PCA"}),
    description: str = "",
) -> VitessceConfig:
    """Generate a VitessceConfig for EHRData.

    Args:
        path: Path to the data's Zarr store directory.
        store: The data's Zarr store or a path to it.
        url: URL pointing to the data's remote Zarr store.
        artifact: Lamin artifact representing the data.
        name: Name of the dataset. If None, derived from path.
        obs_sets: Mapping of observation set paths to names, e.g.
            {"obs/some_annotation": "My cool annotation"}
        obs_embeddings: Mapping of observation embedding paths to names, e.g.
            {"obsm/X_pca": "PCA"}
        description: Description of the dataset, to be displayed.

    Returns:
        A :doc:`Vitessce <vitessce:index>` configuration object.
        Call :meth:`~vitessce.config.VitessceConfig.widget` on it to display it.
    """
    warnings.warn(
        "gen_config is deprecated and will be removed in a future version. Please use gen_default_config instead.",
        category=FutureWarning,
        stacklevel=2,
    )
    vc = _gen_config(path, store, url, artifact, name, obs_sets, obs_embeddings, description)
    return vc


def _gen_config(path, store, url, artifact, name, obs_sets, obs_embeddings, description):
    """Generate a VitessceConfig for EHRData.

    Args:
        path: Path to the data's Zarr store directory.
        store: The data's Zarr store or a path to it.
        url: URL pointing to the data's remote Zarr store.
        artifact: Lamin artifact representing the data.
        name: Name of the dataset. If None, derived from path.
        obs_sets: Mapping of observation set paths to names, e.g.
            {"obs/some_annotation": "My cool annotation"}
        obs_embeddings: Mapping of observation embedding paths to names, e.g.
            {"obsm/X_pca": "PCA"}
        description: Description of the dataset, to be displayed.

    Returns:
        A :doc:`Vitessce <vitessce:index>` configuration object.
    """
    obs_type = "person"
    feature_type = "variable"

    if name is None:
        if artifact is not None:
            name = artifact.description
        elif path is not None:
            name = path.stem
        else:
            msg = "`name` needs to be specified or derived from `path` or `artifact`."
            raise ValueError(msg)

    coordination = {
        "obsType": obs_type,
        "featureType": feature_type,
    }

    from vitessce import AnnDataWrapper, VitessceConfig
    from vitessce import Component as cm

    # # Auto-detect EHRData zarr stores and use the anndata subdirectory
    # if path is not None and (path / "anndata").exists() and (path / "tem").exists():
    #     # This is an EHRData zarr store, use the anndata subdirectory
    #     import zarr

    #     store = zarr.open_group(str(path / "anndata"), mode="r")
    if path is not None:
        # Convert Path to string for vitessce compatibility
        path = str(path)

    wrapper = AnnDataWrapper(
        adata_path=path if isinstance(path, str) else None,
        adata_url=url,
        # vitessce is old and doesn't deal with proper Paths
        adata_store=str(store) if isinstance(store, Path) else store,
        adata_artifact=artifact,
        obs_set_paths=list(obs_sets.keys()),
        obs_set_names=list(obs_sets.values()),
        obs_embedding_paths=list(obs_embeddings.keys()),
        obs_embedding_names=list(obs_embeddings.values()),
        obs_feature_matrix_path="X",
        coordination_values=coordination,
    )

    vc = VitessceConfig(schema_version="1.0.15", name=name, description=description)
    dataset = vc.add_dataset(name=name).add_object(wrapper)

    if len(obs_embeddings) == 2:
        view1 = vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping=next(iter(obs_embeddings.values())))
        view2 = vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping=list(obs_embeddings.values())[1])
    elif len(obs_embeddings) == 1:
        view1 = vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping=next(iter(obs_embeddings.values())))
        view2 = vc.add_view(cm.STATUS, dataset=dataset)
    else:
        view1 = vc.add_view(cm.STATUS, dataset=dataset)
        view2 = None

    views = (
        (
            vc.add_view(cm.OBS_SETS, dataset=dataset),
            vc.add_view(cm.OBS_SET_SIZES, dataset=dataset),
            vc.add_view(cm.OBS_SET_FEATURE_VALUE_DISTRIBUTION, dataset=dataset),
        ),
        (
            vc.add_view(cm.FEATURE_LIST, dataset=dataset),
            view1,
            vc.add_view(cm.FEATURE_VALUE_HISTOGRAM, dataset=dataset),
        ),
        (
            vc.add_view(cm.DESCRIPTION, dataset=dataset, mapping="uns/vitessce_description"),
            view2,
            vc.add_view(cm.HEATMAP, dataset=dataset),
        ),
    )

    filtered_views = tuple(tuple(view for view in row if view is not None) for row in views)

    vc.link_views(
        [view for row in filtered_views for view in row],
        list(coordination.keys()),
        list(coordination.values()),
    )

    # (a / b / c) | (d / e / f) | ...
    vc.layout(reduce(or_, (reduce(truediv, row) for row in filtered_views)))

    return vc


def gen_default_config(
    edata: ed.EHRData,
    zarr_filepath: zarr.storage.StoreLike | Path | str = Path("adata_for_vitessce.zarr"),
    *,
    obs_columns: Iterable[str] | None = None,
    obs_embedding: str | None = None,
    scatter_var_cols: Iterable[str] | None = None,
    layer="tem_data",
    timestep=0,
    return_lamin_artifact: bool = False,
):
    """Quickstart interactive Vitessce generator.

    `Vitessce <https://vitessce.io/>`_  :cite:`keller2025vitessce` is a tool for interactive exploration of high-dimensional data, and compatible with `EHRData`.

    While Vitessce has many features, this function provides a convenient way for an opinionated set of illustrations for an `EHRData` to explore interactively together(called "views").
    Specifically, this function will create a Vitessce widget with multiple views:

    - A view with patient groups of selected columns in `edata.obs` (``obs_columns``)
    - A list of variables to display values for (``obs_embedding``)
    - A bar plot for the number of categories of groups selected during the interactive exploration
    - A scatterplot of the selected ``obs_embedding`` if provided
    - A scatterplot of 2 variables if provided in ``scatter_var_cols``
    - A violin plot of selected variables across groups selected during the interactive exploration
    - A histogram for selected variables during exploration
    - A heatmap of the variables selected in ``var_cols``

    See the `vitessce-python <https://python-docs.vitessce.io/index.html#>`_ documentation for more details and examples.

    Args:
        edata: EHRData object to visualize
        zarr_filepath: Path to save the prepared zarr file that Vitessce can read from.
        obs_columns: List of observation column names (without 'obs/' prefix)
        obs_labels: Optional dict mapping column names to display labels
        obs_embedding: Embedding key in edata.obsm
        obs_embedding_labels: Optional dict mapping embedding keys to display labels
        scatter_var_cols: Optional list of 2 variable columns to create ascatterplot from
        layer: Name of the layer to use for visualization. If the layer is 3D (temporal),
               a timestep must be selected. Default is "tem_data"
        timestep: For 3D layers, the timestep index to extract. Default is 0
        return_lamin_artifact: If `True`, return a Lamin `Artifact` of the generated .zarr file.

    Returns:
        VitessceConfig object

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.physionet2019(
        ...     layer="tem_data",
        ...     n_samples=4000,
        ... )
        >>> vc = ed.integrations.vitessce.gen_default_config(
        ...     edata,
        ...     obs_columns=["Gender", "Age", "training_Set"],
        ...     scatter_var_cols=["HR", "MAP"],
        ...     layer="tem_data",
        ...     timestep=10,
        ... )
        >>> vc.widget()

        .. image:: ../_static/tutorial_images/vitessce_preview.png
    """
    import anndata as ad

    if obs_columns is not None and not all(col in edata.obs for col in obs_columns):
        err = f"Columns {[col for col in obs_columns if col not in edata.obs]} not found in edata.obs"
        raise ValueError(err)
    if scatter_var_cols is not None:
        if len(scatter_var_cols) != 2:
            err = "scatter_var_cols must be an Iterable of 2 variables"
            raise ValueError(err)
        if not all(col in edata.var.index for col in scatter_var_cols):
            err = f"Columns {[col for col in scatter_var_cols if col not in edata.var.index]} not found in edata.var.index"
            raise ValueError(err)
    if obs_embedding is not None and obs_embedding not in edata.obsm:
        err = f"Embedding {obs_embedding} not found in edata.obsm"
        raise ValueError(err)

    if layer is not None and layer in edata.layers:
        layer_data = edata.layers[layer]
        X = layer_data[:, :, timestep].reshape(edata.n_obs, -1) if len(layer_data.shape) == 3 else layer_data
    else:
        X = edata.X

    obsm = {}
    if obs_embedding is not None and obs_embedding in edata.obsm:
        obsm[obs_embedding] = edata.obsm[obs_embedding]

    if scatter_var_cols is not None:
        var_indices = [list(edata.var_names).index(var) for var in scatter_var_cols]
        scatter_data = X[:, var_indices]
        obsm[f"{scatter_var_cols[0]}_vs_{scatter_var_cols[1]}"] = scatter_data

    # Create AnnData with only the required components
    adata = ad.AnnData(
        X=X,
        obs=edata.obs.copy(),
        var=edata.var.copy(),
        obsm=obsm,
    )
    adata.write_zarr(zarr_filepath)

    obs_sets = {f"obs/{col}": col for col in obs_columns}

    obs_embeddings_dict = {}
    if obs_embedding is not None:
        obs_embeddings_dict[f"obsm/{obs_embedding}"] = obs_embedding
    if scatter_var_cols is not None:
        obs_embeddings_dict[f"obsm/{scatter_var_cols[0]}_vs_{scatter_var_cols[1]}"] = (
            f"{scatter_var_cols[0]}_vs_{scatter_var_cols[1]}"
        )

    description = f"""
    Displaying {edata.n_obs} patients with {edata.n_vars} variables.
    The displayed values of the variables are from layer `{layer if layer is not None else "X"}` at timestep `{
        timestep if layer is not None and layer in edata.layers else 0
    }`.


    {
        f'The scatterplot called "{scatter_var_cols[0]} vs {scatter_var_cols[1]}" is based on these two variables'
        if scatter_var_cols
        else ""
    }

    Hint: if the heatmap is unicolored, select its gearwheel and adjust the colormap range.
    """

    if return_lamin_artifact:
        import lamindb as ln

        artifact = ln.Artifact(
            zarr_filepath,
            kind="dataset",
        )

    vc = _gen_config(
        path=zarr_filepath if not return_lamin_artifact else None,
        store=None,
        url=None,
        artifact=artifact if return_lamin_artifact else None,
        name=None,
        obs_sets=obs_sets,
        obs_embeddings=obs_embeddings_dict,
        description=description,
    )

    if return_lamin_artifact:
        return vc, artifact
    else:
        return vc
