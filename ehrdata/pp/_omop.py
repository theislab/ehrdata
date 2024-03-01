import warnings
from typing import Literal, Union

import awkward as ak
import ehrapy as ep
import numpy as np
import pandas as pd
from anndata import AnnData
from rich import print as rprint
from thefuzz import process

from ehrdata.utils._omop_utils import get_column_types, get_feature_info, read_table


# TODO should be able to also extract data from .obsm
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
    if isinstance(aggregation_methods, str):
        aggregation_methods = [aggregation_methods]
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


def qc_lab_measurements(
    adata: AnnData,
    reference_table: pd.DataFrame = None,
    measurements: list[str] = None,
    obsm_measurements: list[str] = None,
    action: Literal["remove"] = "remove",
    unit: Literal["traditional", "SI"] = None,
    layer: str = None,
    threshold: int = 20,
    age_col: str = None,
    age_range: str = None,
    sex_col: str = None,
    sex: str = None,
    ethnicity_col: str = None,
    ethnicity: str = None,
    copy: bool = False,
    verbose: bool = False,
) -> AnnData:
    if copy:
        adata = adata.copy()

    preprocessing_dir = "/Users/xinyuezhang/ehrapy/ehrapy/preprocessing"
    if reference_table is None:
        reference_table = pd.read_csv(
            f"{preprocessing_dir}/laboratory_reference_tables/laposata.tsv", sep="\t", index_col="Measurement"
        )
    if obsm_measurements:
        measurements = obsm_measurements
    for measurement in measurements:
        best_column_match, score = process.extractOne(
            query=measurement, choices=reference_table.index, score_cutoff=threshold
        )
        if best_column_match is None:
            rprint(f"[bold yellow]Unable to find a match for {measurement}")
            continue
        if verbose:
            rprint(
                f"[bold blue]Detected [green]{best_column_match}[blue] for [green]{measurement}[blue] with score [green]{score}."
            )

        reference_column = "SI Reference Interval" if unit == "SI" else "Traditional Reference Interval"

        # Fetch all non None columns from the reference statistics
        not_none_columns = [col for col in [sex_col, age_col, ethnicity_col] if col is not None]
        not_none_columns.append(reference_column)
        reference_values = reference_table.loc[[best_column_match], not_none_columns]

        additional_columns = False
        if sex_col or age_col or ethnicity_col:  # check if additional columns were provided
            additional_columns = True

        # Check if multiple reference values occur and no additional information is available:
        if reference_values.shape[0] > 1 and additional_columns is False:
            raise ValueError(
                f"Several options for {best_column_match} reference value are available. Please specify sex, age or "
                f"ethnicity columns and their values."
            )

        try:
            if age_col:
                min_age, max_age = age_range.split("-")
                reference_values = reference_values[
                    (reference_values[age_col].str.split("-").str[0].astype(int) >= int(min_age))
                    and (reference_values[age_col].str.split("-").str[1].astype(int) <= int(max_age))
                ]
            if sex_col:
                sexes = "U|M" if sex is None else sex
                reference_values = reference_values[reference_values[sex_col].str.contains(sexes)]
            if ethnicity_col:
                reference_values = reference_values[reference_values[ethnicity_col].isin([ethnicity])]

            if layer is not None:
                actual_measurements = adata[:, measurement].layers[layer]
            else:
                if obsm_measurements:
                    actual_measurements = adata.obsm[measurement]["value_as_number"]
                    ak_measurements = adata.obsm[measurement]
                else:
                    actual_measurements = adata[:, measurement].X
        except TypeError:
            rprint(f"[bold yellow]Unable to find specified reference values for {measurement}.")

        check = reference_values[reference_column].values
        check_str: str = np.array2string(check)
        check_str = check_str.replace("[", "").replace("]", "").replace("'", "")
        if "<" in check_str:
            upperbound = float(check_str.replace("<", ""))
            if verbose:
                rprint(f"[bold blue]Using upperbound [green]{upperbound}")
            upperbound_check_results = actual_measurements < upperbound
            if isinstance(actual_measurements, ak.Array):
                if action == "remove":
                    if verbose:
                        rprint(
                            f"Removing {ak.count(actual_measurements) - ak.count(actual_measurements[upperbound_check_results])} outliers"
                        )
                    adata.obsm[measurement] = ak_measurements[upperbound_check_results]
            else:
                upperbound_check_results_array: np.ndarray = upperbound_check_results.copy()
                adata.obs[f"{measurement} normal"] = upperbound_check_results_array

        elif ">" in check_str:
            lower_bound = float(check_str.replace(">", ""))
            if verbose:
                rprint(f"[bold blue]Using lowerbound [green]{lower_bound}")

            lower_bound_check_results = actual_measurements > lower_bound
            if isinstance(actual_measurements, ak.Array):
                if action == "remove":
                    adata.obsm[measurement] = ak_measurements[lower_bound_check_results]
            else:
                lower_bound_check_results_array = lower_bound_check_results.copy()
                adata.obs[f"{measurement} normal"] = lower_bound_check_results_array
        else:  # "-" range case
            min_value = float(check_str.split("-")[0])
            max_value = float(check_str.split("-")[1])
            if verbose:
                rprint(f"[bold blue]Using minimum of [green]{min_value}[blue] and maximum of [green]{max_value}")

            range_check_results = (actual_measurements >= min_value) & (actual_measurements <= max_value)
            if isinstance(actual_measurements, ak.Array):
                if action == "remove":
                    adata.obsm[measurement] = ak_measurements[range_check_results]
            else:
                range_check_results_array: np.ndarray = range_check_results.copy()
                adata.obs[f"{measurement} normal"] = range_check_results_array

    if copy:
        return adata


def drop_nan(
    adata,
    key: Union[str, list[str]],
    slot: Union[str, None] = "obsm",
):
    if isinstance(key, str):
        key_list = [key]
    else:
        key_list = key
    if slot == "obsm":
        for key in key_list:
            ak_array = adata.obsm[key]

            # Update the combined mask based on the presence of None in each field
            for i, field in enumerate(ak_array.fields):
                field_mask = ak.is_none(ak.nan_to_none(ak_array[field]), axis=1)
                if i == 0:
                    combined_mask = ak.full_like(field_mask, fill_value=False, dtype=bool)
                combined_mask = combined_mask | field_mask
            ak_array = ak_array[~combined_mask]
            adata.obsm[key] = ak_array

    return adata
