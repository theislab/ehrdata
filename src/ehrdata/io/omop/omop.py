from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import awkward as ak
import duckdb
import numpy as np
import pandas as pd


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
    For this, a selection of tables from the OMOP CDM which represents the variables should be selected.
    The tables can be measurement, observation, procedure_occurrence, specimen, device_exposure, drug_exposure, or note.

    Parameters
    ----------
    backend_handle
        The backend handle to the database.
    edata
        The EHRData object to which the variables should be added.
    tables
        The tables to be used.
    start_time
        Starting time for values to be included. Can be 'observation_period' start, which takes the 'observation_period_start' value from obs, or a specific Timestamp.
    interval_length_number
        Numeric value of the length of one interval.
    interval_length_unit
        Unit belonging to the interval length. See the units of `pandas.to_timedelta <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html>`_
    num_intervals
        Numer of intervals

    Returns
    -------
    An EHRData object with populated .var field.
    """
    from ehrdata import EHRData

    concept_ids_present_list = []
    time_interval_tables = []
    for table in tables:
        if table == "measurement":
            concept_ids_present = (
                backend_handle.sql("SELECT * FROM measurement").df()["measurement_concept_id"].unique()
            )
            extracted_awkward = extract_measurement(backend_handle)
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
        # TODO: implement the following
        # elif table == "observation":
        #     var = extract_observation(backend_handle)
        # elif table == "procedure_occurrence":
        #     var = extract_procedure_occurrence(backend_handle)
        # elif table == "specimen":
        #     var = extract_specimen(backend_handle)
        # elif table == "device_exposure":
        #     var = extract_device_exposure(backend_handle)
        # elif table == "drug_exposure":
        #     var = extract_drug_exposure(backend_handle)
        # elif table == "note":
        #     var = extract_note(backend_handle)
        else:
            raise ValueError(
                "tables must be a sequence of 'measurement', 'observation', 'procedure_occurrence', 'specimen', 'device_exposure', 'drug_exposure', or 'note'."
            )
        concept_ids_present_list.append(concept_ids_present)
        time_interval_tables.append(time_interval_table)
    if len(time_interval_tables) > 1:
        time_interval_table = np.concatenate([time_interval_table, time_interval_table], axis=1)
        concept_ids_present = pd.concat(concept_ids_present_list)
    else:
        time_interval_table = time_interval_tables[0]
        concept_ids_present = concept_ids_present_list[0]

    # TODO: copy other fields too. or other way? is is somewhat scverse-y by taking and returing anndata object...
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
    return duckdb_instance.sql("SELECT * FROM person").df()


def extract_observation_period(duckdb_instance):
    """Extract person table of an OMOP CDM Database."""
    return duckdb_instance.sql("SELECT * FROM observation_period").df()


def extract_person_observation_period(duckdb_instance):
    """Extract observation table of an OMOP CDM Database."""
    return duckdb_instance.sql(
        "SELECT * \
        FROM person \
        LEFT JOIN observation_period USING(person_id) \
        "
    ).df()


def extract_measurement(duckdb_instance=None):
    """Extract measurement table of an OMOP CDM Database."""
    measurement_table = duckdb_instance.sql("SELECT * FROM measurement").df()

    # get an array n_person x n_features x 2, one for value, one for time
    person_id = (
        duckdb_instance.sql("SELECT * FROM person").df()["person_id"].unique()
    )  # TODO: in anndata? w.r.t database? for now this
    features = measurement_table["measurement_concept_id"].unique()
    person_collection = []

    for person in person_id:
        person_as_list = []
        person_measurements = measurement_table[
            measurement_table["person_id"] == person
        ]  # or ofc sql in rdbms - lazy, on disk, first step towards huge memory reduction of this prototype if only load this selection
        # person_measurements = person_measurements.sort_values(by="measurement_date")
        # person_measurements = person_measurements[["measurement_date", "value_as_number"]]
        # print(person_measurements)
        for feature in features:
            person_feature = []

            # person_measurements_value = []
            # person_measurements_timestamp = []

            person_feature_measurements = person_measurements["measurement_concept_id"] == feature

            person_feature_measurements_value = person_measurements[person_feature_measurements][
                "value_as_number"
            ]  # again, rdbms/spark backend big time scalable here
            person_feature_measurements_timestamp = person_measurements[person_feature_measurements][
                "measurement_datetime"
            ]

            person_feature.append(person_feature_measurements_value)
            person_feature.append(person_feature_measurements_timestamp)

            person_as_list.append(person_feature)

        person_collection.append(person_as_list)

    return ak.Array(person_collection)


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
        num_intervals = np.max(
            con.execute("SELECT * from observation_period").df()["observation_period_end_date"]
            - con.execute("SELECT * from observation_period").df()["observation_period_start_date"]
        ) / pd.to_timedelta(interval_length_number, interval_length_unit)
        num_intervals = int(np.ceil(num_intervals))

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


def extract_observation():
    """Extract observation table of an OMOP CDM Database."""
    pass


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
