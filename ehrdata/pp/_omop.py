import warnings
from typing import Literal, Union

import awkward as ak
import numpy as np
import pandas as pd
from anndata import AnnData
from ehrapy.anndata import move_to_x
from rich import print as rprint
from thefuzz import process

from ehrdata.utils._omop_utils import get_column_types, get_feature_info, read_table


# TODO should be able to also extract data from .obsm
def get_feature_statistics(
    adata: AnnData,
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
    level: Literal["stay_level", "patient_level"] = "stay_level",
    value_col: str = None,
    aggregation_methods: Union[
        Literal["min", "max", "mean", "std", "count"], list[Literal["min", "max", "mean", "std", "count"]]
    ] = None,
    add_aggregation_to: Literal["obs", "X", "return"] = "return",
    verbose: bool = False,
    use_dask: bool = None,
) -> AnnData:
    """Calculate statistics for the specified features from the OMOP tables and adds them to the AnnData object.

    Args:
        adata (AnnData): Anndata object
        source (Literal[ &quot;observation&quot;, &quot;measurement&quot;, &quot;procedure_occurrence&quot;, &quot;specimen&quot;, &quot;device_exposure&quot;, &quot;drug_exposure&quot;, &quot;condition_occurrence&quot;, ]): source table name. Defaults to None.
        features (Union[str, int, list[Union[str, int]]], optional): concept_id or concept_name, or list of concept_id or concept_name. Defaults to None.
        level (Literal[&quot;stay_level&quot;, &quot;patient_level&quot;], optional): For stay level, statistics are calculated for each stay. For patient level, statistics are calculated for each patient. It should be aligned with the setting of the adata object. Defaults to &quot;stay_level&quot;.
        value_col (str, optional): column name in source table to extract value from. Defaults to None.
        aggregation_methods (Union[ Literal[&quot;min&quot;, &quot;max&quot;, &quot;mean&quot;, &quot;std&quot;, &quot;count&quot;], list[Literal[&quot;min&quot;, &quot;max&quot;, &quot;mean&quot;, &quot;std&quot;, &quot;count&quot;]] ], optional): aggregation methods to calculate statistics. Defaults to [&quot;min&quot;, &quot;max&quot;, &quot;mean&quot;, &quot;std&quot;, &quot;count&quot;].
        add_aggregation_to_X (bool, optional): add the calculated statistics to adata.X. If False, the statistics will be added to adata.obs. Defaults to True.
        verbose (bool, optional): print verbose information. Defaults to False.
        use_dask (bool, optional): If True, dask will be used to read the tables. For large tables, it is highly recommended to use dask. If None, it will be set to adata.uns[&quot;use_dask&quot;]. Defaults to None.

    Returns
    -------
        AnnData: Anndata object with added statistics either in adata.obs (if add_aggregation_to_X=False) or adata.X (if add_aggregation_to_X=True)
    """
    if source in ["measurement", "observation", "specimen"]:
        key = f"{source}_concept_id"
    elif source in ["device_exposure", "procedure_occurrence", "drug_exposure", "condition_occurrence"]:
        key = f"{source.split('_')[0]}_concept_id"
    else:
        raise KeyError(f"Extracting data from {source} is not supported yet")

    if source == "measurement":
        if value_col is None:
            value_col = "value_as_number"
            warnings.warn(
                f"Extracting values from {value_col}. Value in measurement table could be saved in these columns: value_as_number, value_source_value.\nSpecify value_col to extract value from desired column.",
                stacklevel=2,
            )
        source_table_columns = ["visit_occurrence_id", "measurement_datetime", key, value_col]
    elif source == "observation":
        if value_col is None:
            value_col = "value_as_number"
            warnings.warn(
                f"Extracting values from {value_col}. Value in observation table could be saved in these columns: value_as_number, value_as_string, value_source_value.\nSpecify value_col to extract value from desired column.",
                stacklevel=2,
            )
        source_table_columns = ["visit_occurrence_id", "observation_datetime", key, value_col]
    elif source in ["procedure_occurrence", "specimen", "device_exposure", "drug_exposure", "condition_occurrence"]:
        if source_table_columns is None:
            raise KeyError(f"Please specify value_col for {source} table.")
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
    info_dict = info_df[["feature_id_2", "feature_name"]].set_index("feature_id_2").to_dict()["feature_name"]

    # Select featrues
    df_source = df_source[df_source[key].isin(list(info_df.feature_id_2))]
    # TODO Select time
    # da_measurement = da_measurement[(da_measurement.time >= 0) & (da_measurement.time <= 48*60*60)]
    # df_source[f'{source}_name'] = df_source[key].map(info_dict)
    if aggregation_methods is None:
        aggregation_methods = ["min", "max", "mean", "std", "count"]
    if isinstance(aggregation_methods, str):
        aggregation_methods = [aggregation_methods]
    if level == "stay_level":
        result = df_source.groupby(["visit_occurrence_id", key]).agg({value_col: aggregation_methods})

        print("Calculating statistics")
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
            for feature in list(info_dict.values()):
                for suffix in (f"_{aggregation_method}" for aggregation_method in aggregation_methods):
                    col_name = f"{feature}{suffix}"
                    if col_name in df_statistics.columns:
                        new_column_order.append(col_name)

            df_statistics.columns = new_column_order

    df_statistics.index = df_statistics.index.astype(str)
    if add_aggregation_to == "return":
        return df_statistics
    elif add_aggregation_to == "obs":
        adata.obs = pd.merge(adata.obs, df_statistics, how="left", left_index=True, right_index=True)
        return adata
    elif add_aggregation_to == "X":
        adata.obs = pd.merge(adata.obs, df_statistics, how="left", left_index=True, right_index=True)
        uns = adata.uns
        obsm = adata.obsm
        varm = adata.varm
        # layers = adata.layers
        adata = move_to_x(adata, list(df_statistics.columns))
        adata.uns = uns
        adata.obsm = obsm
        adata.varm = varm
        return adata
    else:
        raise ValueError(f"add_aggregation_to should be one of ['obs', 'X', 'return'], not {add_aggregation_to}")


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
    """Examines lab measurements for reference ranges and outliers.

    Source:
        The used reference values were obtained from https://accessmedicine.mhmedical.com/content.aspx?bookid=1069&sectionid=60775149 .
        This table is compiled from data in the following sources:

        * Tietz NW, ed. Clinical Guide to Laboratory Tests. 3rd ed. Philadelphia: WB Saunders Co; 1995;
        * Laposata M. SI Unit Conversion Guide. Boston: NEJM Books; 1992;
        * American Medical Association Manual of Style: A Guide for Authors and Editors. 9th ed. Chicago: AMA; 1998:486–503. Copyright 1998, American Medical Association;
        * Jacobs DS, DeMott WR, Oxley DK, eds. Jacobs & DeMott Laboratory Test Handbook With Key Word Index. 5th ed. Hudson, OH: Lexi-Comp Inc; 2001;
        * Henry JB, ed. Clinical Diagnosis and Management by Laboratory Methods. 20th ed. Philadelphia: WB Saunders Co; 2001;
        * Kratz A, et al. Laboratory reference values. N Engl J Med. 2006;351:1548–1563; 7) Burtis CA, ed. Tietz Textbook of Clinical Chemistry and Molecular Diagnostics. 5th ed. St. Louis: Elsevier; 2012.

        This version of the table of reference ranges was reviewed and updated by Jessica Franco-Colon, PhD, and Kay Brooks.

    Limitations:
        * Reference ranges differ between continents, countries and even laboratories (https://informatics.bmj.com/content/28/1/e100419).
          The default values used here are only one of many options.
        * Ensure that the values used as input are provided with the correct units. We recommend the usage of SI values.
        * The reference values pertain to adults. Many of the reference ranges need to be adapted for children.
        * By default if no gender is provided and no unisex values are available, we use the **male** reference ranges.
        * The used reference ranges may be biased for ethnicity. Please examine the primary sources if required.
        * We recommend a glance at https://www.nature.com/articles/s41591-021-01468-6 for the effect of such covariates.

    Additional values:
        * Interleukin-6 based on https://pubmed.ncbi.nlm.nih.gov/33155686/

    If you want to specify your own table as a Pandas DataFrame please examine the existing default table.
    Ethnicity and age columns can be added.
    https://github.com/theislab/ehrapy/blob/main/ehrapy/preprocessing/laboratory_reference_tables/laposata.tsv

    Args:
        adata: Annotated data matrix.
        reference_table: A custom DataFrame with reference values. Defaults to the laposata table if not specified.
        measurements: A list of measurements to check.
        obsm_measurements: A list of measurements to check from the obsm. If specified, measurements will be ignored.
        action: The action to take if a measurement is outside the reference range. Defaults to 'remove'.
        unit: The unit of the measurements. Defaults to 'traditional'.
        layer: Layer containing the matrix to calculate the metrics for.
        threshold: Minimum required matching confidence score of the fuzzysearch.
                   0 = no matches, 100 = all must match. Defaults to 20.
        age_col: Column containing age values.
        age_range: The inclusive age-range to filter for such as 5-99.
        sex_col: Column containing sex values. Column must contain 'U', 'M' or 'F'.
        sex: Sex to filter the reference values for. Use U for unisex which uses male values when male and female conflict.
             Defaults to 'U|M'.
        ethnicity_col: Column containing ethnicity values.
        ethnicity: Ethnicity to filter for.
        copy: Whether to return a copy. Defaults to False.
        verbose: Whether to have verbose stdout. Notifies user of matched columns and value ranges.

    Returns
    -------
        A modified AnnData object (copy if specified).

    Examples
    --------
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.qc_lab_measurements(adata, measurements=["potassium_first"], verbose=True)
    """
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
