import warnings
from typing import Literal, Union

import ehrapy as ep
import pandas as pd
from rich import print as rprint

from ehrdata.utils._omop_utils import get_column_types, get_feature_info, read_table


def get_feature_statistics(
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
    features: Union[str, int, list[Union[str, int]]] = None,
    level="stay_level",
    value_col: str = None,
    aggregation_methods: Union[
        Literal["min", "max", "mean", "std", "count"], list[Literal["min", "max", "mean", "std", "count"]]
    ] = None,
    add_aggregation_to_X: bool = True,
    verbose: bool = False,
    use_dask: bool = None,
):
    if source in ["measurement", "observation", "specimen"]:
        key = f"{source}_concept_id"
    elif source in ["device_exposure", "procedure_occurrence", "drug_exposure", "condition_occurrence"]:
        key = f"{source.split('_')[0]}_concept_id"
    else:
        raise KeyError(f"Extracting data from {source} is not supported yet")

    if source == "measurement":
        value_col = "value_as_number"
        warnings.warn(
            f"Extracting values from {value_col}. Value in measurement table could be saved in these columns: value_as_number, value_source_value.\nSpecify value_col to extract value from desired column.",
            stacklevel=2,
        )
        source_table_columns = ["visit_occurrence_id", "measurement_datetime", key, value_col]
    elif source == "observation":
        value_col = "value_as_number"
        warnings.warn(
            f"Extracting values from {value_col}. Value in observation table could be saved in these columns: value_as_number, value_as_string, value_source_value.\nSpecify value_col to extract value from desired column.",
            stacklevel=2,
        )
        source_table_columns = ["visit_occurrence_id", "observation_datetime", key, value_col]
    elif source == "condition_occurrence":
        source_table_columns = None
    else:
        raise KeyError(f"Extracting data from {source} is not supported yet")
    if isinstance(features, str):
        features = [features]
        rprint(f"Trying to extarct the following features: {features}")

    if use_dask is None:
        use_dask = True

    column_types = get_column_types(adata.uns, table_name=source)
    df_source = read_table(
        adata.uns, table_name=source, dtype=column_types, usecols=source_table_columns, use_dask=use_dask
    )

    info_df = get_feature_info(adata.uns, features=features, verbose=verbose)
    info_dict = info_df[["feature_id", "feature_name"]].set_index("feature_id").to_dict()["feature_name"]

    # Select featrues
    df_source = df_source[df_source[key].isin(list(info_df.feature_id))]
    # TODO Select time
    # da_measurement = da_measurement[(da_measurement.time >= 0) & (da_measurement.time <= 48*60*60)]
    # df_source[f'{source}_name'] = df_source[key].map(info_dict)
    if aggregation_methods is None:
        aggregation_methods = ["min", "max", "mean", "std", "count"]
    if level == "stay_level":
        result = df_source.groupby(["visit_occurrence_id", key]).agg({value_col: aggregation_methods})

        if use_dask:
            result = result.compute()
        result = result.reset_index(drop=False)
        result.columns = ["_".join(a) for a in result.columns.to_flat_index()]
        result.columns = result.columns.str.removesuffix("_")
        result.columns = result.columns.str.removeprefix(f"{value_col}_")
        result[f"{source}_name"] = result[key].map(info_dict)

        df_statistics = result.pivot(index="visit_occurrence_id", columns=f"{source}_name", values=aggregation_methods)
        df_statistics.columns = df_statistics.columns.swaplevel()
        df_statistics.columns = ["_".join(a) for a in df_statistics.columns.to_flat_index()]

        # TODO
        sort_columns = True
        if sort_columns:
            new_column_order = []
            for feature in features:
                for suffix in (f"_{aggregation_method}" for aggregation_method in aggregation_methods):
                    col_name = f"{feature}{suffix}"
                    if col_name in df_statistics.columns:
                        new_column_order.append(col_name)

            df_statistics.columns = new_column_order

    df_statistics.index = df_statistics.index.astype(str)

    adata.obs = pd.merge(adata.obs, df_statistics, how="left", left_index=True, right_index=True)

    if add_aggregation_to_X:
        uns = adata.uns
        obsm = adata.obsm
        varm = adata.varm
        # layers = adata.layers
        adata = ep.ad.move_to_x(adata, list(df_statistics.columns))
        adata.uns = uns
        adata.obsm = obsm
        adata.varm = varm
        # It will change
        # adata.layers = layers
    return adata
