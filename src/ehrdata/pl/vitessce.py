from __future__ import annotations

from functools import reduce
from operator import or_, truediv
from pathlib import Path
from typing import TYPE_CHECKING

from vitessce import AnnDataWrapper, VitessceConfig
from vitessce import Component as cm

if TYPE_CHECKING:
    from lamindb import Artifact
    from zarr.storage import Store


def gen_config(
    path: Path | None = None,
    *,
    store: Path | Store | None = None,
    url: str | None = None,
    artifact: Artifact | None = None,
    name: str | None = None,
) -> VitessceConfig:
    """Generate a VitessceConfig for EHRData.

    Parameters
    ----------
    path
        Path to the data’s Zarr store directory.
    store
        The data’s Zarr store or a path to it.
    url
        URL pointing to the data’s remote Zarr store.
    name
        Name of the dataset.
        If `None`, derived from `path`.

    Returns
    -------
    VitessceConfig
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

    wrapper = AnnDataWrapper(
        adata_path=path,
        adata_url=url,
        # vitessce is old and doesn’t deal with proper Paths
        adata_store=str(store) if isinstance(store, Path) else store,
        adata_artifact=artifact,
        obs_set_paths=["obs/gender_concept_id"],
        obs_set_names=["Gender Concept ID"],
        obs_embedding_paths=["obsm/X_pca"],
        obs_embedding_names=["PCA"],
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
        coordination.keys(),
        coordination.values(),
    )

    # (a / b / c) | (d / e / f) | ...
    vc.layout(reduce(or_, (reduce(truediv, row) for row in views)))

    return vc
