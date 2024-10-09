from functools import reduce
from operator import or_, truediv
from pathlib import Path

from vitessce import AnnDataWrapper, VitessceConfig
from vitessce import Component as cm


def gen_config(path: Path, name: str = "Dummy EHRData") -> VitessceConfig:
    """Generate a VitessceConfig for EHRData.

    Parameters
    ----------
    path
        Path to the data
    name
        Name of the dataset

    Returns
    -------
    VitessceConfig
    """
    obs_type = "person"
    feature_type = "variable"

    wrapper = AnnDataWrapper(
        adata_store=path,
        obs_set_paths=["obs/gender_concept_id"],
        obs_set_names=["Gender Concept ID"],
        obs_embedding_paths=["obsm/X_pca"],
        obs_embedding_names=["PCA"],
        obs_feature_matrix_path="X",
        coordination_values={
            "obsType": obs_type,
            "featureType": feature_type,
        },
    )

    vc = VitessceConfig(schema_version="1.0.15", name=name)
    dataset = vc.add_dataset(name="Dummy EHRData").add_object(wrapper)

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
        ["obsType", "featureType"],
        [obs_type, feature_type],
    )

    # (a / b / c) | (d / e / f) | ...
    vc.layout(reduce(or_, (reduce(truediv, row) for row in views)))

    return vc
