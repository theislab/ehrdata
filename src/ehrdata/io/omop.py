from __future__ import annotations

from collections.abc import Iterable
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


def extract_tables():
    """Extract tables of an OMOP CDM Database."""
    # TODO for all of this; iterate potential API. should this be functions or object methods?
    # extract person, measurements, ....
    # define features
    # make vars and corresponding .obsm, with specific key
    # then can get "static" into .X with
    # ep.ts.aggregate(adata, metric="counts")
    # or
    # ep.ts.aggregate(adata, metrics={counts: [Antibiotic treatment, BP measruement], "average": [Heart Rate]})
    pass


def extract_person(duckdb_instance):
    """Extract person table of an OMOP CDM Database."""
    return duckdb_instance.sql("SELECT * FROM person").df()

    # TODO: check if every person has an observation (can happen)
    # TODO: check if every observation has a person (would be data quality issue)
    # TODO: figure out how to handle multiple observation periods per person; here the person vs view discussion comes into play


def extract_observation_period(duckdb_instance):
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
    t: int = 60,
    start_time: str = "2100-01-01 00:00:00",
    observation_duration: int = 48,
    #  person_id: int = 132592,
    concept_id_index: int = 1,
    type: str = "measurement",
    which_value: str = "value_as_number",  # alternative value_as_concept_id
    strategy: str = "locf",
):
    # TODO: allow units
    time_seconds = t * (60 * 60 * 24)

    time_intervals = np.arange(0, observation_duration * 60 * 60, step=time_seconds)

    patient_measurements = pd.to_datetime(np.array(list(person_ts[concept_id_index][1])))
    start_time_offset = pd.to_datetime(start_time)  # .dt.total_seconds()

    patient_time_deltas = (patient_measurements - start_time_offset).total_seconds()

    index_table = np.searchsorted(time_intervals, patient_time_deltas)

    index_table[index_table >= len(time_intervals)] = len(time_intervals) - 1

    time_frame = pd.DataFrame(
        {"time": np.arange(0, observation_duration * 60 * 60, step=time_seconds), "value": np.nan}
    )
    time_frame.iloc[index_table, 1] = np.array(list(person_ts[concept_id_index][0]))

    if strategy is None:
        pass
    elif strategy == "locf":
        time_frame["value"] = time_frame["value"].ffill()
    else:
        raise NotImplementedError(f"strategy {strategy} not implemented!")

    return time_frame


def time_interval_table(
    # self,
    ts: ak.Array,
    obs: pd.DataFrame,
    # duckdb_instance,
    start_time: str = "patient_hospital_entry",
    observation_duration: int = 48,
    interval_length: float = 60,
    concept_ids: str | Iterable = "all",
    interval_unit="minutes",
    strategy="locf",
):
    """Extract measurement table of an OMOP CDM Database."""
    if concept_ids == "all":
        concept_id_list = range(len(ts[0]))
    else:
        # TODO: like check if they are around, etc
        concept_id_list = concept_ids

    tables = []
    for person, person_ts in zip(obs.iterrows(), ts, strict=False):
        start_time = person[1]["observation_period_start_date"]
        # end_time = person[1]["observation_period_end_date"]
        person_table = []
        for feature in concept_id_list:
            feature_table = _get_interval_table_from_awkward_array(
                person_ts,
                start_time=start_time,
                observation_duration=observation_duration,
                t=interval_length,
                concept_id_index=feature,
                strategy=strategy,
            )
            person_table.append(feature_table["value"])
        tables.append(pd.concat(person_table, axis=1))

    # df = pd.concat(tables, axis=1)
    # df.index = feature_table["time"]
    # self.it = tables

    return tables  # TODO: store in self, np


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
