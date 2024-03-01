import numbers
from typing import Literal, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from pandas.tseries.offsets import DateOffset as Offset
from rich import print as rprint

from ehrdata.io._omop import from_dataframe, to_dataframe
from ehrdata.utils._omop_utils import df_to_dict, get_column_types, read_table


def get_concept_name(adata: Union[AnnData, dict], concept_id: Union[str, list], raise_error=False, verbose=True):
    if isinstance(concept_id, numbers.Integral):
        concept_id = [concept_id]

    if isinstance(adata, AnnData):
        adata_dict = adata.uns
    else:
        adata_dict = adata

    column_types = get_column_types(adata_dict, table_name="concept")
    df_concept = read_table(adata_dict, table_name="concept", dtype=column_types)
    # TODO dask Support
    # df_concept.compute().dropna(subset=["concept_id", "concept_name"], inplace=True, ignore_index=True)  # usecols=vocabularies_tables_columns["concept"]
    df_concept.dropna(
        subset=["concept_id", "concept_name"], inplace=True, ignore_index=True
    )  # usecols=vocabularies_tables_columns["concept"]
    concept_dict = df_to_dict(df=df_concept, key="concept_id", value="concept_name")
    concept_name = []
    concept_name_not_found = []
    for id in concept_id:
        try:
            concept_name.append(concept_dict[id])
        except KeyError:
            concept_name.append(id)
            concept_name_not_found.append(id)
    if len(concept_name_not_found) > 0:
        # warnings.warn(f"Couldn't find concept {id} in concept table!")
        if verbose:
            rprint(f"Couldn't find concept {concept_name_not_found} in concept table!")
        if raise_error:
            raise KeyError
    if len(concept_name) == 1:
        return concept_name[0]
    else:
        return concept_name


# downsampling
def aggregate_timeseries_in_bins(
    adata,
    features: Union[str, list[str]],
    slot: Union[str, None] = "obsm",
    value_key: str = "value_as_number",
    time_key: str = "measurement_datetime",
    time_binning_method: Literal["floor", "ceil", "round"] = "floor",
    bin_size: Union[str, Offset] = "h",
    aggregation_method: Literal["median", "mean", "min", "max"] = "median",
):
    if isinstance(features, str):
        features_list = [features]
    else:
        features_list = features

    # Ensure the time_binning_method provided is one of the expected methods
    if time_binning_method not in ["floor", "ceil", "round"]:
        raise ValueError(
            f"time_binning_method {time_binning_method} is not supported. Choose from 'floor', 'ceil', or 'round'."
        )

    if aggregation_method not in {"median", "mean", "min", "max"}:
        raise ValueError(
            f"aggregation_method {aggregation_method} is not supported. Choose from 'median', 'mean', 'min', or 'max'."
        )

    if slot == "obsm":
        for feature in features_list:
            print(f"processing feature [{feature}]")
            df = to_dataframe(adata, features)
            if pd.api.types.is_datetime64_any_dtype(df[time_key]):
                func = getattr(df[time_key].dt, time_binning_method, None)
                if func is not None:
                    df[time_key] = func(bin_size)
            else:
                # TODO need to take care of this if it doesn't follow omop standard
                if bin_size == "h":
                    df[time_key] = df[time_key] / 3600
                    func = getattr(np, time_binning_method)
                    df[time_key] = func(df[time_key])

            df[time_key] = df[time_key].astype(str)
            # Adjust time values that are equal to the time_upper_bound
            # df.loc[df[time_key] == time_upper_bound, time_key] = time_upper_bound - 1

            # Group and aggregate data
            df = (
                df.groupby(["visit_occurrence_id", time_key])[value_key].agg(aggregation_method).reset_index(drop=False)
            )
            adata = from_dataframe(adata, feature, df)

    return adata


# TODO
def get_concept_id():
    pass


# TODO
def note_nlp_map(
    self,
):
    # Got some inspirations from: https://github.com/aws-samples/amazon-comprehend-medical-omop-notes-mapping
    # connect with existing functions
    pass
