from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import awkward as ak
import duckdb
import numpy as np
import pandas as pd
from duckdb.duckdb import DuckDBPyConnection

from ehrdata.io.omop._check_arguments import (
    VALID_OBSERVATION_TABLES_JOIN,
    VALID_OBSERVATION_TABLES_SINGLE,
    _check_valid_aggregation_strategy,
    _check_valid_backend_handle,
    _check_valid_concept_ids,
    _check_valid_data_field_to_keep,
    _check_valid_death_table,
    _check_valid_edata,
    _check_valid_enrich_var_with_feature_info,
    _check_valid_enrich_var_with_unit_info,
    _check_valid_interval_length_number,
    _check_valid_interval_length_unit,
    _check_valid_interval_variable_data_tables,
    _check_valid_keep_date,
    _check_valid_num_intervals,
    _check_valid_observation_table,
    _check_valid_variable_data_tables,
)
from ehrdata.io.omop._queries import (
    time_interval_table_for_interval_tables_query_long_format,
    time_interval_table_query_long_format,
)
from ehrdata.utils._omop_utils import get_table_catalog_dict

DOWNLOAD_VERIFICATION_TAG = "download_verification_tag"


def _get_table_list() -> list:
    flat_table_list = []
    for _, value_list in get_table_catalog_dict().items():
        for value in value_list:
            flat_table_list.append(value)
    return flat_table_list


def _set_up_duckdb(path: Path, backend_handle: DuckDBPyConnection, prefix: str = "") -> str:
    """Create tables in the backend from the CSV files in the path from datasets in the OMOP Common Data model."""
    tables = _get_table_list()

    used_tables = []
    missing_tables = []
    unused_files = []
    for file_name in os.listdir(path):
        file_name_trunk = file_name.split(".")[0].lower()
        regular_omop_table_name = file_name_trunk.replace(prefix, "")

        if regular_omop_table_name in tables:
            used_tables.append(regular_omop_table_name)

            if regular_omop_table_name == "measurement":
                dtype = {"measurement_source_value": str}
            else:
                dtype = None

            # read raw csv as temporary table
            temp_relation = backend_handle.read_csv(path / file_name, dtype=dtype, escapechar="%")  # noqa: F841
            backend_handle.execute("CREATE OR REPLACE TABLE temp_table AS SELECT * FROM temp_relation")

            # make query to create table with lowercase column names
            column_names = backend_handle.execute("DESCRIBE temp_table").df()["column_name"].values
            select_columns = ", ".join([f'"{col}" AS "{col.lower()}"' for col in column_names])
            create_table_with_lowercase_columns_query = (
                f"CREATE TABLE {regular_omop_table_name} AS SELECT {select_columns} FROM temp_table"
            )

            # write proper table
            existing_tables = backend_handle.execute("SHOW TABLES").df()["name"].values
            if regular_omop_table_name in existing_tables:
                logging.info(f"Table {regular_omop_table_name} already exists. Dropping and recreating...")
                backend_handle.execute(f"DROP TABLE {regular_omop_table_name}")

            backend_handle.execute(create_table_with_lowercase_columns_query)

            backend_handle.execute("DROP TABLE temp_table")

        elif file_name_trunk != DOWNLOAD_VERIFICATION_TAG:
            unused_files.append(file_name)

    for table in tables:
        if table not in used_tables:
            missing_tables.append(table)

    logging.info(f"missing tables: {missing_tables}")
    logging.info(f"unused files: {unused_files}")


def _collect_units_per_feature(ds, unit_key="unit_concept_id") -> dict:
    feature_units = {}
    for i in range(ds[unit_key].shape[1]):
        single_feature_units = ds[unit_key].isel({ds[unit_key].dims[1]: i})
        single_feature_units_flat = np.array(single_feature_units).flatten()
        single_feature_units_unique = pd.unique(single_feature_units_flat[~pd.isna(single_feature_units_flat)])
        feature_units[ds["data_table_concept_id"][i].item()] = single_feature_units_unique
    return feature_units


def _check_one_unit_per_feature(ds, unit_key="unit_concept_id") -> None:
    feature_units = _collect_units_per_feature(ds, unit_key=unit_key)
    num_units = np.array([len(units) for _, units in feature_units.items()])

    # print(f"no units for features: {np.argwhere(num_units == 0)}")
    print(f"multiple units for features: {np.argwhere(num_units > 1)}")


def _create_feature_unit_concept_id_report(backend_handle, ds) -> pd.DataFrame:
    feature_units_concept = _collect_units_per_feature(ds, unit_key="unit_concept_id")

    feature_units_long_format = []
    for feature, units in feature_units_concept.items():
        if len(units) == 0:
            feature_units_long_format.append({"concept_id": feature, "no_units": True, "multiple_units": False})
        elif len(units) > 1:
            for unit in units:
                feature_units_long_format.append(
                    {
                        "concept_id": feature,
                        "unit_concept_id": unit,
                        "no_units": False,
                        "multiple_units": True,
                    }
                )
        else:
            feature_units_long_format.append(
                {
                    "concept_id": feature,
                    "unit_concept_id": units[0],
                    "no_units": False,
                    "multiple_units": False,
                }
            )

    df = pd.DataFrame(
        feature_units_long_format, columns=["concept_id", "unit_concept_id", "no_units", "multiple_units"]
    )
    df["unit_concept_id"] = df["unit_concept_id"].astype("Int64")

    return df


def _create_enriched_var_with_unit_info(backend_handle, ds, var, unit_report) -> pd.DataFrame:
    feature_concept_id_table = var  # ds["data_table_concept_id"].to_dataframe()

    feature_concept_id_unit_table = pd.merge(
        feature_concept_id_table, unit_report, how="left", left_index=True, right_on="concept_id"
    )

    concepts = backend_handle.sql("SELECT * FROM concept").df()

    feature_concept_id_unit_info_table = pd.merge(
        feature_concept_id_unit_table,
        concepts,
        how="left",
        left_on="unit_concept_id",
        right_on="concept_id",
    )

    return feature_concept_id_unit_info_table


def setup_connection(path: Path | str, backend_handle: DuckDBPyConnection, prefix: str = "") -> None:
    """Setup a connection to the OMOP CDM database.

    This function sets up a connection to the OMOP CDM database.
    It checks the capitalization of the 'person' table, and assumes the same capitalization style is used for all other tables.


    Parameters
    ----------
    path
        The path to the folder containing the CSV files.
    backend_handle
        The backend handle to the database.
    prefix
        The prefix to be removed from the CSV filenames.

    Returns
    -------
    An EHRData object with populated .uns["omop_table_capitalization"] field.

    """
    _set_up_duckdb(Path(path), backend_handle, prefix)


def setup_obs(
    backend_handle: Literal[str, duckdb, Path],
    observation_table: Literal["person", "person_cohort", "person_observation_period", "person_visit_occurrence"],
    death_table: bool = False,
):
    """Setup the observation table.

    This function sets up the observation table for the EHRData object.
    For this, a table from the OMOP CDM which represents the "observed unit" via an id should be selected.
    A unit can be a person, or the data of a person together with either the information from cohort, observation_period, or visit_occurrence.
    Notice a single person can have multiple of the latter, and as such can appear multiple times.
    For person_cohort, the subject_id of the cohort is considered to be the person_id for a join.

    Parameters
    ----------
    backend_handle
        The backend handle to the database.
    observation_table
        The observation table to be used.
    death_table
        Whether to include the death table. The observation_table created will be left joined with the death table as the right table.

    Returns
    -------
    An EHRData object with populated .obs field.
    """
    _check_valid_backend_handle(backend_handle)
    _check_valid_observation_table(observation_table)
    _check_valid_death_table(death_table)

    from ehrdata import EHRData

    if observation_table in VALID_OBSERVATION_TABLES_SINGLE:
        obs = get_table(backend_handle, observation_table)

    elif observation_table in VALID_OBSERVATION_TABLES_JOIN:
        if observation_table == "person_cohort":
            obs = _get_table_join(backend_handle, "person", "cohort", right_key="subject_id")
        elif observation_table == "person_observation_period":
            obs = _get_table_join(backend_handle, "person", "observation_period")
        elif observation_table == "person_visit_occurrence":
            obs = _get_table_join(backend_handle, "person", "visit_occurrence")

    if death_table:
        death = get_table(backend_handle, "death")
        obs = obs.merge(death, how="left", on="person_id")

    return EHRData(obs=obs, uns={"omop_io_observation_table": observation_table.split("person_")[-1]})


def setup_variables(
    edata,
    *,
    backend_handle: duckdb.duckdb.DuckDBPyConnection,
    data_tables: Sequence[Literal["measurement", "observation", "specimen"]]
    | Literal["measurement", "observation", "specimen"],
    data_field_to_keep: str | Sequence[str],
    interval_length_number: int,
    interval_length_unit: str,
    num_intervals: int,
    concept_ids: Literal["all"] | Sequence = "all",
    aggregation_strategy: str = "last",
    enrich_var_with_feature_info: bool = False,
    enrich_var_with_unit_info: bool = False,
):
    """Setup the variables.

    This function sets up the variables for the EHRData object.
    It will fail if there is more than one unit_concept_id per feature.
    Writes a unit report of the features to edata.uns["unit_report_<data_tables>"].

    Parameters
    ----------
    backend_handle
        The backend handle to the database.
    edata
        The EHRData object to which the variables should be added.
    data_tables
        The table to be used. Only a single table can be used.
    data_field_to_keep
        The CDM Field in the data table to be kept. Can be e.g. "value_as_number" or "value_as_concept_id". Importantly, can be "is_present" to have a one-hot encoding of the presence of the feature in a patient in an interval.
    start_time
        Starting time for values to be included.
    interval_length_number
        Numeric value of the length of one interval.
    interval_length_unit
        Unit belonging to the interval length.
    num_intervals
        Number of intervals.
    concept_ids
        Concept IDs to use from this data table. If not specified, 'all' are used.
    aggregation_strategy
        Strategy to use when aggregating multiple data points within one interval.
    enrich_var_with_feature_info
        Whether to enrich the var table with feature information. If a concept_id is not found in the concept table, the feature information will be NaN.
    enrich_var_with_unit_info
        Whether to enrich the var table with unit information. Raises an Error if a) multiple units per feature are found for at least one feature. If a concept_id is not found in the concept table, the feature information will be NaN.

    Returns
    -------
    An EHRData object with populated .r and .var field.
    """
    from ehrdata import EHRData

    _check_valid_edata(edata)
    _check_valid_backend_handle(backend_handle)
    data_tables = _check_valid_variable_data_tables(data_tables)
    data_field_to_keep = _check_valid_data_field_to_keep(data_field_to_keep)
    _check_valid_interval_length_number(interval_length_number)
    _check_valid_interval_length_unit(interval_length_unit)
    _check_valid_num_intervals(num_intervals)
    _check_valid_concept_ids(concept_ids)
    _check_valid_aggregation_strategy(aggregation_strategy)
    _check_valid_enrich_var_with_feature_info(enrich_var_with_feature_info)
    _check_valid_enrich_var_with_unit_info(enrich_var_with_unit_info)

    time_defining_table = edata.uns.get("omop_io_observation_table", None)
    if time_defining_table is None:
        raise ValueError("The observation table must be set up first, use the `setup_obs` function.")

    if data_tables[0] in ["measurement", "observation", "specimen"]:
        # also keep unit_concept_id and unit_source_value;
        if isinstance(data_field_to_keep, list):
            data_field_to_keep = list(data_field_to_keep) + ["unit_concept_id", "unit_source_value"]
        # TODO: use in future version when more than one data table can be used
        # elif isinstance(data_field_to_keep, dict):
        #     data_field_to_keep = {
        #         k: v + ["unit_concept_id", "unit_source_value"] for k, v in data_field_to_keep.items()
        #     }
        else:
            raise ValueError

    # dbms complains about our queries, which sometimes need a column to be of type e.g. datetime, when it can't infer types from data
    count = backend_handle.execute(f"SELECT COUNT(*) as count FROM {data_tables[0]}").df()["count"].item()
    if count == 0:
        logging.info(f"No data found in {data_tables[0]}. Returning edata without additional variables.")
        return edata

    ds = (
        time_interval_table_query_long_format(
            backend_handle=backend_handle,
            time_defining_table=time_defining_table,
            data_table=data_tables[0],
            data_field_to_keep=data_field_to_keep,
            interval_length_number=interval_length_number,
            interval_length_unit=interval_length_unit,
            num_intervals=num_intervals,
            aggregation_strategy=aggregation_strategy,
        )
        .set_index(["person_id", "data_table_concept_id", "interval_step"])
        .to_xarray()
    )

    _check_one_unit_per_feature(ds)
    # TODO ignore? go with more vanilla omop style. _check_one_unit_per_feature(ds, unit_key="unit_source_value")

    unit_report = _create_feature_unit_concept_id_report(backend_handle, ds)

    var = ds["data_table_concept_id"].to_dataframe()

    if enrich_var_with_feature_info or enrich_var_with_unit_info:
        concepts = backend_handle.sql("SELECT * FROM concept").df()
        concepts.columns = concepts.columns.str.lower()

    if enrich_var_with_feature_info:
        var = pd.merge(var, concepts, how="left", left_index=True, right_on="concept_id")

    if enrich_var_with_unit_info:
        if unit_report["multiple_units"].sum() > 0:
            raise ValueError("Multiple units per feature found. Enrichment with feature information not possible.")
        else:
            var = pd.merge(
                var,
                unit_report,
                how="left",
                left_index=True,
                right_on="unit_concept_id",
                suffixes=("", "_unit"),
            )
            var = pd.merge(
                var,
                concepts,
                how="left",
                left_on="unit_concept_id",
                right_on="concept_id",
                suffixes=("", "_unit"),
            )

    t = ds["interval_step"].to_dataframe()

    edata = EHRData(r=ds[data_field_to_keep[0]].values, obs=edata.obs, var=var, uns=edata.uns, t=t)
    edata.uns[f"unit_report_{data_tables[0]}"] = unit_report

    return edata


def setup_interval_variables(
    edata,
    *,
    backend_handle: duckdb.duckdb.DuckDBPyConnection,
    data_tables: Sequence[Literal["drug_exposure"]] | Literal["drug_exposure"],
    data_field_to_keep: str | Sequence[str],
    interval_length_number: int,
    interval_length_unit: str,
    num_intervals: int,
    concept_ids: Literal["all"] | Sequence = "all",
    aggregation_strategy: str = "last",
    enrich_var_with_feature_info: bool = False,
    enrich_var_with_unit_info: bool = False,
    keep_date: Literal["start", "end", "interval"] = "start",
):
    """Setup the interval variables

    This function sets up the variables that are stored as interval in OMOP for the EHRData object.
    It will fail if there is more than one unit_concept_id per feature.
    Writes a unit report of the features to edata.uns["unit_report_<data_tables>"].

    Parameters
    ----------
    backend_handle
        The backend handle to the database.
    edata
        The EHRData object to which the variables should be added.
    data_tables
        The table to be used. Only a single table can be used.
    data_field_to_keep
        The CDM Field in the data table to be kept. Can be e.g. "value_as_number" or "value_as_concept_id".  Importantly, can be "is_present" to have a one-hot encoding of the presence of the feature in a patient in an interval.
    start_time
        Starting time for values to be included.
    interval_length_number
        Numeric value of the length of one interval.
    interval_length_unit
        Unit belonging to the interval length.
    num_intervals
        Number of intervals.
    concept_ids
        Concept IDs to use from this data table. If not specified, 'all' are used.
    aggregation_strategy
        Strategy to use when aggregating multiple data points within one interval.
    enrich_var_with_feature_info
        Whether to enrich the var table with feature information. If a concept_id is not found in the concept table, the feature information will be NaN.
    keep_date
        Whether to keep the start or end date, or the interval span.

    Returns
    -------
    An EHRData object with populated .r and .var field.
    """
    from ehrdata import EHRData

    _check_valid_edata(edata)
    _check_valid_backend_handle(backend_handle)
    data_tables = _check_valid_interval_variable_data_tables(data_tables)
    data_field_to_keep = _check_valid_data_field_to_keep(data_field_to_keep)
    _check_valid_interval_length_number(interval_length_number)
    _check_valid_interval_length_unit(interval_length_unit)
    _check_valid_num_intervals(num_intervals)
    _check_valid_concept_ids(concept_ids)
    _check_valid_aggregation_strategy(aggregation_strategy)
    _check_valid_enrich_var_with_feature_info(enrich_var_with_feature_info)
    _check_valid_enrich_var_with_unit_info(enrich_var_with_unit_info)
    _check_valid_keep_date(keep_date)

    time_defining_table = edata.uns.get("omop_io_observation_table", None)
    if time_defining_table is None:
        raise ValueError("The observation table must be set up first, use the `setup_obs` function.")

    # dbms complains about our queries, which sometimes need a column to be of type e.g. datetime, when it can't infer types from data
    count = backend_handle.execute(f"SELECT COUNT(*) as count FROM {data_tables[0]}").df()["count"].item()
    if count == 0:
        logging.info(f"No data in {data_tables}.")
        return edata

    if keep_date == "start" or keep_date == "end":
        ds = (
            time_interval_table_for_interval_tables_query_long_format(
                backend_handle=backend_handle,
                time_defining_table=time_defining_table,
                data_table=data_tables[0],
                data_field_to_keep=data_field_to_keep,
                interval_length_number=interval_length_number,
                interval_length_unit=interval_length_unit,
                num_intervals=num_intervals,
                aggregation_strategy=aggregation_strategy,
                date_prefix=keep_date,
            )
            .set_index(["person_id", "data_table_concept_id", "interval_step"])
            .to_xarray()
        )
    elif keep_date == "interval":
        ds = (
            time_interval_table_for_interval_tables_query_long_format(
                backend_handle=backend_handle,
                time_defining_table=time_defining_table,
                data_table=data_tables[0],
                data_field_to_keep=data_field_to_keep,
                interval_length_number=interval_length_number,
                interval_length_unit=interval_length_unit,
                num_intervals=num_intervals,
                aggregation_strategy=aggregation_strategy,
                date_prefix=keep_date,
            )
            .set_index(["person_id", "data_table_concept_id", "interval_step"])
            .to_xarray()
        )

    var = ds["data_table_concept_id"].to_dataframe()

    if enrich_var_with_feature_info or enrich_var_with_unit_info:
        concepts = backend_handle.sql("SELECT * FROM concept").df()
        concepts.columns = concepts.columns.str.lower()

    if enrich_var_with_feature_info:
        var = pd.merge(var, concepts, how="left", left_index=True, right_on="concept_id")

    t = ds["interval_step"].to_dataframe()

    edata = EHRData(r=ds[data_field_to_keep[0]].values, obs=edata.obs, var=var, uns=edata.uns, t=t)

    return edata


def get_table(duckdb_instance, table_name: str) -> pd.DataFrame:
    """Extract a table of an OMOP CDM Database."""
    return _lowercase_column_names(duckdb_instance.sql(f"SELECT * FROM {table_name}").df())


def _get_table_join(
    duckdb_instance, table1: str, table2: str, left_key: str = "person_id", right_key: str = "person_id"
) -> pd.DataFrame:
    """Extract a table of an OMOP CDM Database."""
    return _lowercase_column_names(
        duckdb_instance.sql(
            f"SELECT * \
        FROM {table1} as t1 \
        JOIN {table2} as t2 ON t1.{left_key} = t2.{right_key} \
        "
        ).df()
    )


def extract_measurement(duckdb_instance):
    """Extract a table of an OMOP CDM Database."""
    return get_table(
        duckdb_instance,
        table_name="measurement",
        concept_id_col="measurement_concept_id",
        value_col="value_as_number",
        timestamp_col="measurement_datetime",
    )


def extract_observation(duckdb_instance):
    """Extract a table of an OMOP CDM Database."""
    return get_table(
        duckdb_instance,
        table_name="observation",
        concept_id_col="observation_concept_id",
        value_col="value_as_number",
        timestamp_col="observation_datetime",
    )


def extract_procedure_occurrence(duckdb_instance):
    """Extract a table of an OMOP CDM Database."""
    return get_table(
        duckdb_instance,
        table_name="procedure_occurrence",
        concept_id_col="procedure_concept_id",
        value_col="procedure_type_concept_id",  # Assuming `procedure_type_concept_id` is a suitable value field
        timestamp_col="procedure_datetime",
    )


def extract_device_exposure(duckdb_instance):
    """Extract a table of an OMOP CDM Database."""
    # return get_table(
    #     duckdb_instance,
    #     table_name="device_exposure",
    #     concept_id_col="device_concept_id",
    #     value_col="device_type_concept_id",  # Assuming this as value
    #     timestamp_col="device_exposure_start_date"
    # )
    # NEEDS IMPLEMENTATION
    return None


def extract_drug_exposure(duckdb_instance):
    """Extract a table of an OMOP CDM Database."""
    # return get_table(
    #     duckdb_instance,
    #     table_name="drug_exposure",
    #     concept_id_col="drug_concept_id",
    #     value_col="dose_unit_concept_id",  # Assuming `dose_unit_concept_id` as value
    #     timestamp_col="drug_exposure_start_datetime"
    # )
    # NEEDS IMPLEMENTATION
    return None


def extract_note(duckdb_instance):
    """Extract a table of an OMOP CDM Database."""
    return get_table(
        duckdb_instance,
        table_name="note",
        concept_id_col="note_type_concept_id",
        value_col="note_class_concept_id",  # Assuming `note_class_concept_id` as value
        timestamp_col="note_datetime",
    )


def _get_interval_table_from_awkward_array(
    # self,#person_feature_measurement: ak.Array,
    person_ts: ak.Array,
    start_time: str,
    interval_length_number: int,
    interval_length_unit: str,
    num_intervals: int,
    concept_id_index: int,
    aggregation_strategy: str,
    type: str = "measurement",
    which_value: str = "value_as_number",  # alternative value_as_concept_id
):
    timedelta = pd.to_timedelta(interval_length_number, interval_length_unit)

    time_intervals = pd.Series([timedelta] * num_intervals).cumsum()

    patient_measurements = pd.to_datetime(np.array(list(person_ts[concept_id_index][1])))
    start_time_offset = pd.to_datetime(start_time)

    patient_time_deltas = (patient_measurements - start_time_offset).total_seconds()

    index_table = np.searchsorted(time_intervals, patient_time_deltas)

    index_table[index_table >= len(time_intervals)] = len(time_intervals) - 1

    time_frame = pd.DataFrame({"time": np.arange(0, num_intervals), "value": np.nan})

    # TODO: currently, if multiple measurements in 1 interval,
    # indexing like this takes the last value of this interval
    if aggregation_strategy == "last":
        time_frame.iloc[index_table, 1] = np.array(list(person_ts[concept_id_index][0]))
    else:
        raise ValueError("currently, only aggregation_strategy='last' is supported")

    return time_frame


def get_time_interval_table(
    # self,
    con,
    ts: ak.Array,
    obs: pd.DataFrame,
    # duckdb_instance,
    start_time: Literal["observation_period_start"]
    | pd.Timestamp
    | str,  # observation_period_start, birthdate, specific date as popular options?
    interval_length_number: int,
    interval_length_unit: str,
    num_intervals: str | int = "max_observation_duration",
    concept_ids: Literal["all"] | Sequence = "all",
    aggregation_strategy: str = "first",  # what to do if multiple obs. in 1 interval. first, last, mean, median, most_frequent for categories
    # strategy="locf",
) -> np.ndarray:
    """Extract measurement table of an OMOP CDM Database.

    Parameters
    ----------
    con
        Connection to a database where the tables are stored.
    ts
        A specific awkwkard array tree structure.
    obs
        A dataframe of the observation-type table to be used.
    start_time
        Starting time for values to be included. Can be 'observation_period' start, which takes the 'observation_period_start' value from obs, or a specific Timestamp.
    interval_length_number
        Numeric value of the length of one interval.
    interval_length_unit
        Unit belonging to the interval length. See the units of `pandas.to_timedelta <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html>`_
    num_intervals
        Numer of intervals
    concept_ids
        The features to use, that is the OMOP concept_id in ts to be used.
    aggregation_strategy


    Returns
    -------
    A time interval table of shape obs.shape[0] x number of concept_ids x num_intervals
    """
    if concept_ids == "all":
        concept_id_list = range(len(ts[0]))
    else:
        # TODO: like check if they are around, etc
        concept_id_list = concept_ids

    if num_intervals == "max_observation_duration":
        observation_period_df = con.execute("SELECT * from observation_period").df()
        observation_period_df = _lowercase_column_names(observation_period_df)

        # Calculate the duration of observation periods
        num_intervals = np.max(
            observation_period_df["observation_period_end_date"]
            - observation_period_df["observation_period_start_date"]
        ) / pd.to_timedelta(interval_length_number, interval_length_unit)
        num_intervals = int(np.ceil(num_intervals))
        # num_intervals = np.max(
        #     con.execute("SELECT * from observation_period").df()["observation_period_end_date"]
        #     - con.execute("SELECT * from observation_period").df()["observation_period_start_date"]
        # ) / pd.to_timedelta(interval_length_number, interval_length_unit)
        # num_intervals = int(np.ceil(num_intervals))

    tables = []
    for person, person_ts in zip(obs.iterrows(), ts, strict=False):
        if start_time == "observation_period_start":
            person_start_time = person[1]["observation_period_start_date"]
        else:
            raise NotImplementedError("start_time currently only supports 'observation_period_start'")

        person_table = []
        for feature in concept_id_list:
            feature_table = _get_interval_table_from_awkward_array(
                person_ts,
                start_time=person_start_time,
                interval_length_number=interval_length_number,
                interval_length_unit=interval_length_unit,
                num_intervals=num_intervals,
                concept_id_index=feature,
                aggregation_strategy=aggregation_strategy,
            )
            person_table.append(feature_table["value"])
        tables.append(pd.concat(person_table, axis=1))

    # df = pd.concat(tables, axis=1)
    # df.index = feature_table["time"]
    # self.it = tables

    return np.array(tables).transpose(0, 2, 1)  # TODO: store in self, np


def _lowercase_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all column names to lowercase."""
    df.columns = map(str.lower, df.columns)  # Convert all column names to lowercase
    return df


def extract_condition_occurrence():
    """Extract a table of an OMOP CDM Database."""
    pass


def extract_observation_period():
    """Extract a table of an OMOP CDM Database."""
    pass
