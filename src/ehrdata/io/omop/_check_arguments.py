from __future__ import annotations

from collections.abc import Sequence

import duckdb

from ehrdata.io.omop._queries import (
    AGGREGATION_STRATEGY_KEY,
)

DOWNLOAD_VERIFICATION_TAG = "download_verification_tag"
VALID_OBSERVATION_TABLES_SINGLE = ["person"]
VALID_OBSERVATION_TABLES_JOIN = ["person_cohort", "person_observation_period", "person_visit_occurrence"]
VALID_VARIABLE_TABLES = ["measurement", "observation", "specimen"]


def _check_valid_backend_handle(backend_handle) -> None:
    if not isinstance(backend_handle, duckdb.duckdb.DuckDBPyConnection):
        raise TypeError("Expected backend_handle to be of type DuckDBPyConnection.")


def _check_valid_observation_table(observation_table) -> None:
    if not isinstance(observation_table, str):
        raise TypeError("Expected observation_table to be a string.")
    if observation_table not in VALID_OBSERVATION_TABLES_SINGLE + VALID_OBSERVATION_TABLES_JOIN:
        raise ValueError(
            f"observation_table must be one of {VALID_OBSERVATION_TABLES_SINGLE+VALID_OBSERVATION_TABLES_JOIN}."
        )


def _check_valid_death_table(death_table) -> None:
    if not isinstance(death_table, bool):
        raise TypeError("Expected death_table to be a boolean.")


def _check_valid_edata(edata) -> None:
    from ehrdata import EHRData

    if not isinstance(edata, EHRData):
        raise TypeError("Expected edata to be of type EHRData.")


def _check_valid_data_tables(data_tables) -> Sequence:
    if isinstance(data_tables, str):
        data_tables = [data_tables]
    if not isinstance(data_tables, Sequence):
        raise TypeError("Expected data_tables to be a string or Sequence.")
    if not all(table in VALID_VARIABLE_TABLES for table in data_tables):
        raise ValueError(f"data_tables must be a subset of {VALID_VARIABLE_TABLES}.")
    return data_tables


def _check_valid_data_field_to_keep(data_field_to_keep) -> Sequence:
    if isinstance(data_field_to_keep, str):
        data_field_to_keep = [data_field_to_keep]
    if not isinstance(data_field_to_keep, Sequence):
        raise TypeError("Expected data_field_to_keep to be a string or Sequence.")
    return data_field_to_keep


def _check_valid_interval_length_number(interval_length_number) -> None:
    if not isinstance(interval_length_number, int):
        raise TypeError("Expected interval_length_number to be an integer.")


def _check_valid_interval_length_unit(interval_length_unit) -> None:
    # TODO: maybe check if it is a valid unit from pandas.to_timedelta
    if not isinstance(interval_length_unit, str):
        raise TypeError("Expected interval_length_unit to be a string.")


def _check_valid_num_intervals(num_intervals) -> None:
    if not isinstance(num_intervals, int):
        raise TypeError("Expected num_intervals to be an integer.")


def _check_valid_concept_ids(concept_ids) -> None:
    if concept_ids != "all" and not isinstance(concept_ids, Sequence):
        raise TypeError("concept_ids must be a sequence of integers or 'all'.")


def _check_valid_aggregation_strategy(aggregation_strategy) -> None:
    if aggregation_strategy not in AGGREGATION_STRATEGY_KEY.keys():
        raise TypeError(f"aggregation_strategy must be one of {AGGREGATION_STRATEGY_KEY.keys()}.")


def _check_valid_enrich_var_with_feature_info(enrich_var_with_feature_info) -> None:
    if not isinstance(enrich_var_with_feature_info, bool):
        raise TypeError("Expected enrich_var_with_feature_info to be a boolean.")


def _check_valid_enrich_var_with_unit_info(enrich_var_with_unit_info) -> None:
    if not isinstance(enrich_var_with_unit_info, bool):
        raise TypeError("Expected enrich_var_with_unit_info to be a boolean.")
