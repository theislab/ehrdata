from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import awkward as ak
import duckdb
import numpy as np
import pandas as pd
from ehrdata import EHRData

def _check_sanity_of_folder(folder_path: str | Path):
    pass


def _check_sanity_of_database(backend_handle: duckdb.DuckDB):
    pass


def setup_obs(
    backend_handle: Literal[str, duckdb, Path],
    observation_table: Literal["person", "observation_period", "person_observation_period", "condition_occurrence"],
):
    """Setup the observation table.

    This function sets up the observation table for the EHRData project.
    For this, a table from the OMOP CDM which represents to observed unit should be selected.
    A unit can be a person, an observation period, the join of these two tables, or a condition occurrence.

    Parameters
    ----------
    backend_handle
        The backend handle to the database.
    observation_table
        The observation table to be used.

    Returns
    -------
    An EHRData object with populated .obs field.
    """
    from ehrdata import EHRData

    if observation_table == "person":
        obs = extract_person(backend_handle)
    elif observation_table == "observation_period":
        obs = extract_observation_period(backend_handle)
    elif observation_table == "person_observation_period":
        obs = extract_person_observation_period(backend_handle)
    elif observation_table == "condition_occurrence":
        obs = extract_condition_occurrence(backend_handle)
    else:
        raise ValueError("observation_table must be either 'person', 'observation_period', or 'condition_occurrence'.")

    return EHRData(obs=obs)

def setup_variables(
    backend_handle: Literal[str, duckdb, Path],
    edata,
    tables: Sequence[
        Literal[
            "measurement", "observation", "procedure_occurrence", "specimen", "device_exposure", "drug_exposure", "note"
        ]
    ],
    start_time: Literal["observation_period_start"] | pd.Timestamp | str,
    interval_length_number: int,
    interval_length_unit: str,
    num_intervals: int,
    concept_ids: Literal["all"] | Sequence = "all",
    aggregation_strategy: str = "last",
):
    """Setup the variables.

    This function sets up the variables for the EHRData project.

    Parameters
    ----------
    backend_handle
        The backend handle to the database.
    edata
        The EHRData object to which the variables should be added.
    tables
        The tables to be used.
    start_time
        Starting time for values to be included.
    interval_length_number
        Numeric value of the length of one interval.
    interval_length_unit
        Unit belonging to the interval length.
    num_intervals
        Number of intervals.
    concept_ids
        Concept IDs to filter on or 'all'.
    aggregation_strategy
        Strategy to use when aggregating data within intervals.

    Returns
    -------
    An EHRData object with populated .var field.
    """
    # Mapping of table names to extraction functions and concept ID column names
    table_info = {
        "measurement": {"extract_func": extract_measurement, "concept_id_col": "measurement_concept_id"},
        "observation": {"extract_func": extract_observation, "concept_id_col": "observation_concept_id"},
        "procedure_occurrence": {"extract_func": extract_procedure_occurrence, "concept_id_col": "procedure_concept_id"},
        "specimen": {"extract_func": extract_specimen, "concept_id_col": "specimen_concept_id"},
        "device_exposure": {"extract_func": extract_device_exposure, "concept_id_col": "device_concept_id"},
        "drug_exposure": {"extract_func": extract_drug_exposure, "concept_id_col": "drug_concept_id"},
        "note": {"extract_func": extract_note, "concept_id_col": "note_concept_id"},
    }

    concept_ids_present_list = []
    time_interval_tables = []

    for table in tables:
        if table not in table_info:
            raise ValueError(
                "tables must be a sequence of 'measurement', 'observation', 'procedure_occurrence', 'specimen', 'device_exposure', 'drug_exposure', or 'note'."
            )

        # Get extract function and concept_id column for the table
        extract_func = table_info[table]["extract_func"]
        concept_id_col = table_info[table]["concept_id_col"]
        concept_ids_present_df = normalize_column_names(backend_handle.sql(f"SELECT * FROM {table}").df())
        concept_ids_present = concept_ids_present_df[concept_id_col].unique()
        extracted_awkward = extract_func(backend_handle)

        # Create the time interval table
        time_interval_table = get_time_interval_table(
            backend_handle,
            extracted_awkward,
            edata.obs,
            start_time="observation_period_start",
            interval_length_number=interval_length_number,
            interval_length_unit=interval_length_unit,
            num_intervals=num_intervals,
            concept_ids=concept_ids,
            aggregation_strategy=aggregation_strategy,
        )

        # Append 
        concept_ids_present_list.append(concept_ids_present)
        time_interval_tables.append(time_interval_table)

    # Combine time interval tables
    if len(time_interval_tables) > 1:
        time_interval_table = np.concatenate([time_interval_table, time_interval_table], axis=1)
        concept_ids_present = pd.concat(concept_ids_present_list)
    else:
        time_interval_table = time_interval_tables[0]
        concept_ids_present = concept_ids_present_list[0]

    # Update edata with the new variables
    edata = EHRData(r=time_interval_table, obs=edata.obs, var=concept_ids_present)

    return edata

def load(
    backend_handle: Literal[str, duckdb, Path],
    # folder_path: str,
    # delimiter: str = ",",
    # make_filename_lowercase: bool = True,
    # use_dask: bool = False,
    # level: Literal["stay_level", "patient_level"] = "stay_level",
    # load_tables: str | list[str] | tuple[str] | Literal["auto"] | None = None,
    # remove_empty_column: bool = True,
) -> None:
    """Initialize a connection to the OMOP CDM Database."""
    if isinstance(backend_handle, str) or isinstance(backend_handle, Path):
        _check_sanity_of_folder(backend_handle)
    elif isinstance(backend_handle, duckdb.DuckDB):
        _check_sanity_of_database(backend_handle)
    else:
        raise NotImplementedError(f"Backend {backend_handle} not supported. Choose a valid backend.")


def extract_person(duckdb_instance):
    """Extract person table of an OMOP CDM Database."""
    return normalize_column_names(duckdb_instance.sql("SELECT * FROM person").df())


def extract_observation_period(duckdb_instance):
    """Extract person table of an OMOP CDM Database."""
    return normalize_column_names(duckdb_instance.sql("SELECT * FROM observation_period").df())


def extract_person_observation_period(duckdb_instance):
    """Extract observation table of an OMOP CDM Database."""
    return normalize_column_names(duckdb_instance.sql(
        "SELECT * \
        FROM person \
        LEFT JOIN observation_period USING(person_id) \
        "
    ).df())

def extract_table(duckdb_instance, table_name: str, concept_id_col: str, value_col: str, timestamp_col: str):
    """
    Generalized extraction function to extract data from an OMOP CDM table.

    Parameters
    ----------
    duckdb_instance: duckdb.DuckDB
        The DuckDB instance for querying the database.
    table_name: str
        The name of the table to extract data from (e.g., "measurement", "observation").
    concept_id_col: str
        The name of the column that contains the concept IDs (e.g., "measurement_concept_id").
    value_col: str
        The name of the column that contains the values (e.g., "value_as_number").
    timestamp_col: str
        The name of the column that contains the timestamps (e.g., "measurement_datetime").

    Returns
    -------
    ak.Array
        An Awkward Array with the structure: n_person x n_features x 2 (value, time).
    """
    # Load the specified table
    table_df = duckdb_instance.sql(f"SELECT * FROM {table_name}").df()
    table_df = normalize_column_names(table_df)

    # Load the person table to get unique person IDs
    person_id_df = normalize_column_names(duckdb_instance.sql("SELECT * FROM person").df())
    person_ids = person_id_df["person_id"].unique()

    # Get unique features (concept IDs) for the table
    features = table_df[concept_id_col].unique()

    # Initialize the collection for all persons
    person_collection = []

    for person in person_ids:
        person_as_list = []
        # Get rows for the current person
        person_data = table_df[table_df["person_id"] == person]

        # For each feature, get values and timestamps
        for feature in features:
            feature_data = person_data[person_data[concept_id_col] == feature]

            # Extract the values and timestamps
            feature_values = feature_data[value_col]
            feature_timestamps = feature_data[timestamp_col]

            # Append values and timestamps for this feature
            person_as_list.append([feature_values, feature_timestamps])

        # Append this person's data to the collection
        person_collection.append(person_as_list)

    return ak.Array(person_collection)


def extract_measurement(duckdb_instance):
    return extract_table(
        duckdb_instance,
        table_name="measurement",
        concept_id_col="measurement_concept_id",
        value_col="value_as_number",
        timestamp_col="measurement_datetime"
    )

def extract_observation(duckdb_instance):
    return extract_table(
        duckdb_instance,
        table_name="observation",
        concept_id_col="observation_concept_id",
        value_col="value_as_number",
        timestamp_col="observation_datetime"
    )

def extract_procedure_occurrence(duckdb_instance):
    return extract_table(
        duckdb_instance,
        table_name="procedure_occurrence",
        concept_id_col="procedure_concept_id",
        value_col="procedure_type_concept_id",  # Assuming `procedure_type_concept_id` is a suitable value field
        timestamp_col="procedure_datetime"
    )

def extract_specimen(duckdb_instance):
    return extract_table(
        duckdb_instance,
        table_name="specimen",
        concept_id_col="specimen_concept_id",
        value_col="unit_concept_id",  # Assuming `unit_concept_id` is a suitable value field
        timestamp_col="specimen_datetime"
    )

def extract_device_exposure(duckdb_instance):
    return extract_table(
        duckdb_instance,
        table_name="device_exposure",
        concept_id_col="device_concept_id",
        value_col="device_exposure_type_concept_id",  # Assuming this as value
        timestamp_col="device_exposure_start_datetime"
    )

def extract_drug_exposure(duckdb_instance):
    return extract_table(
        duckdb_instance,
        table_name="drug_exposure",
        concept_id_col="drug_concept_id",
        value_col="dose_unit_concept_id",  # Assuming `dose_unit_concept_id` as value
        timestamp_col="drug_exposure_start_datetime"
    )

def extract_note(duckdb_instance):
    return extract_table(
        duckdb_instance,
        table_name="note",
        concept_id_col="note_concept_id",
        value_col="note_class_concept_id",  # Assuming `note_class_concept_id` as value
        timestamp_col="note_datetime"
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
) -> np.array:
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
        observation_period_df = normalize_column_names(observation_period_df)

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

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all column names to lowercase."""
    df.columns = map(str.lower, df.columns)  # Convert all column names to lowercase
    return df

def extract_procedure_occurrence():
    """Extract procedure_occurrence table of an OMOP CDM Database."""
    pass


def extract_specimen():
    """Extract specimen table of an OMOP CDM Database."""
    pass


def extract_device_exposure():
    """Extract device_exposure table of an OMOP CDM Database."""
    pass


def extract_drug_exposure():
    """Extract drug_exposure table of an OMOP CDM Database."""
    pass


def extract_condition_occurrence():
    """Extract condition_occurrence table of an OMOP CDM Database."""
    pass


def extract_note():
    """Extract note table of an OMOP CDM Database."""
    pass
