from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns

from ehrdata.tl import get_concept_name
from ehrdata.utils.omop_utils import get_column_types, map_concept_id, read_table


# TODO allow users to pass features
def feature_counts(
    adata,
    source: Literal[
        "observation",
        "measurement",
        "procedure_occurrence",
        "specimen",
        "device_exposure",
        "drug_exposure",
        "condition_occurrence",
    ],
    number=20,
    key=None,
):
    # if source == 'measurement':
    #     columns = ["value_as_number", "time", "visit_occurrence_id", "measurement_concept_id"]
    # elif source == 'observation':
    #     columns = ["value_as_number", "value_as_string", "measurement_datetime"]
    # elif source == 'condition_occurrence':
    #     columns = None
    # else:
    #     raise KeyError(f"Extracting data from {source} is not supported yet")

    column_types = get_column_types(adata.uns, table_name=source)
    df_source = read_table(adata.uns, table_name=source, dtype=column_types, usecols=[f"{source}_concept_id"])
    feature_counts = df_source[f"{source}_concept_id"].value_counts()
    if adata.uns["use_dask"]:
        feature_counts = feature_counts.compute()
    feature_counts = feature_counts.to_frame().reset_index(drop=False)[0:number]

    feature_counts[f"{source}_concept_id_1"], feature_counts[f"{source}_concept_id_2"] = map_concept_id(
        adata.uns, concept_id=feature_counts[f"{source}_concept_id"], verbose=False
    )
    feature_counts["feature_name"] = get_concept_name(adata, concept_id=feature_counts[f"{source}_concept_id_1"])
    if feature_counts[f"{source}_concept_id_1"].equals(feature_counts[f"{source}_concept_id_2"]):
        feature_counts.drop(f"{source}_concept_id_2", axis=1, inplace=True)
        feature_counts.rename(columns={f"{source}_concept_id_1": f"{source}_concept_id"})
        feature_counts = feature_counts.reindex(columns=["feature_name", f"{source}_concept_id", "count"])
    else:
        feature_counts = feature_counts.reindex(
            columns=["feature_name", f"{source}_concept_id_1", f"{source}_concept_id_2", "count"]
        )

    ax = sns.barplot(feature_counts, x="feature_name", y="count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return feature_counts
