from __future__ import annotations

from functools import reduce
from operator import or_, truediv
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lamindb import Artifact
    from vitessce import VitessceConfig
    from zarr.storage import Store


def gen_config(
    path: Path | None = None,
    *,
    store: Path | Store | None = None,
    url: str | None = None,
    artifact: Artifact | None = None,
    # arguments not about how the store goes in:
    name: str | None = None,
    obs_sets: Mapping[str, str] = MappingProxyType({"obs/gender_concept_id": "Gender Concept ID"}),
    obs_embeddings: Mapping[str, str] = MappingProxyType({"obsm/X_pca": "PCA"}),
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

    Returns:
        A :doc:`Vitessce <vitessce:index>` configuration object.
        Call :meth:`~vitessce.config.VitessceConfig.widget` on it to display it.
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

    wrapper = AnnDataWrapper(
        adata_path=path,
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

    vc = VitessceConfig(schema_version="1.0.15", name=name)
    dataset = vc.add_dataset(name=name).add_object(wrapper)

    views = (
        (
            vc.add_view(cm.OBS_SETS, dataset=dataset),
            vc.add_view(cm.OBS_SET_SIZES, dataset=dataset),
            vc.add_view(cm.OBS_SET_FEATURE_VALUE_DISTRIBUTION, dataset=dataset),
        ),
        (
            vc.add_view(cm.FEATURE_LIST, dataset=dataset),
            vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping="PCA"),
            vc.add_view(cm.FEATURE_VALUE_HISTOGRAM, dataset=dataset),
        ),
        (
            vc.add_view(cm.DESCRIPTION, dataset=dataset),
            vc.add_view(cm.STATUS, dataset=dataset),
            vc.add_view(cm.HEATMAP, dataset=dataset),
        ),
    )

    vc.link_views(
        [view for row in views for view in row],
        list(coordination.keys()),
        list(coordination.values()),
    )

    # (a / b / c) | (d / e / f) | ...
    vc.layout(reduce(or_, (reduce(truediv, row) for row in views)))

    return vc
