from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd

from ehrdata.io.omop._queries import (
    AGGREGATION_STRATEGY_KEY,
)

DOWNLOAD_VERIFICATION_TAG = "download_verification_tag"
VALID_OBSERVATION_TABLES_SINGLE = ["person"]
VALID_OBSERVATION_TABLES_JOIN = ["person_cohort", "person_observation_period", "person_visit_occurrence"]
VALID_VARIABLE_TABLES = ["measurement", "observation", "specimen"]
VALID_INTERVAL_VARIABLE_TABLES = [
    "drug_exposure",
    "condition_occurrence",
    "procedure_occurrence",
    "device_exposure",
    "drug_era",
    "dose_era",
    "condition_era",
    "episode",
]
VALID_KEEP_DATES = ["start", "end", "interval"]
VALID_TIME_PRECISIONS = ["date", "datetime"]


def _check_valid_backend_handle(backend_handle) -> None:
    if not isinstance(backend_handle, duckdb.DuckDBPyConnection):
        msg = "Expected backend_handle to be of type DuckDBPyConnection."
        raise TypeError(msg)


def _check_valid_observation_table(observation_table) -> None:
    if not isinstance(observation_table, str):
        msg = "Expected observation_table to be a string."
        raise TypeError(msg)
    if observation_table not in VALID_OBSERVATION_TABLES_SINGLE + VALID_OBSERVATION_TABLES_JOIN:
        msg = f"observation_table must be one of {VALID_OBSERVATION_TABLES_SINGLE + VALID_OBSERVATION_TABLES_JOIN}."
        raise ValueError(msg)


def _check_valid_death_table(death_table) -> None:
    if not isinstance(death_table, bool):
        msg = "Expected death_table to be a boolean."
        raise TypeError(msg)


def _check_valid_edata(edata) -> None:
    from ehrdata import EHRData

    if not isinstance(edata, EHRData):
        msg = "Expected edata to be of type EHRData."
        raise TypeError(msg)


def _check_valid_variable_data_tables(data_tables) -> Sequence:
    if isinstance(data_tables, str):
        data_tables = [data_tables]
    if not isinstance(data_tables, Sequence):
        msg = "Expected data_tables to be a string or Sequence."
        raise TypeError(msg)
    if not all(table in VALID_VARIABLE_TABLES for table in data_tables):
        msg = f"data_tables must be a subset of {VALID_VARIABLE_TABLES}."
        raise ValueError(msg)
    return data_tables


def _check_valid_interval_variable_data_tables(data_tables) -> Sequence:
    if isinstance(data_tables, str):
        data_tables = [data_tables]
    if not isinstance(data_tables, Sequence):
        msg = "Expected data_tables to be a string or Sequence."
        raise TypeError(msg)
    if not all(table in VALID_INTERVAL_VARIABLE_TABLES for table in data_tables):
        msg = f"data_tables must be a subset of {VALID_INTERVAL_VARIABLE_TABLES}."
        raise ValueError(msg)
    return data_tables


def _check_valid_data_field_to_keep(data_field_to_keep, data_tables) -> dict[str, Sequence[str]]:
    valid_type = False
    if isinstance(data_field_to_keep, str):
        valid_type = True
        if len(data_tables) > 1:
            msg = "data_field_to_keep must be a dictionary if more than one data table is used."
            raise TypeError(msg)
        else:
            data_field_to_keep = {data_tables[0]: [data_field_to_keep]}

    if isinstance(data_field_to_keep, Sequence):
        valid_type = True
        if len(data_tables) > 1:
            msg = "data_field_to_keep must be a dictionary if more than one data table is used."
            raise TypeError(msg)
        else:
            data_field_to_keep = {data_tables[0]: data_field_to_keep}

    if isinstance(data_field_to_keep, dict):
        valid_type = True
        if set(data_field_to_keep.keys()) != set(data_tables):
            msg = "data_field_to_keep keys must be equal to data_tables."
            raise ValueError(msg)
        if not all(isinstance(value, Sequence) for value in data_field_to_keep.values()):
            msg = "data_field_to_keep values must be a string or Sequence."
            raise TypeError(msg)
        # a new dict, so that the caller's argument is left as it is
        data_field_to_keep = {
            key: [value] if isinstance(value, str) else value for key, value in data_field_to_keep.items()
        }
    if not valid_type:
        msg = f"Expected data_field_to_keep to be a string, Sequence, or dict, but is {type(data_field_to_keep)}"
        raise TypeError(msg)

    return data_field_to_keep


def _check_valid_interval_length_number(interval_length_number) -> None:
    if not isinstance(interval_length_number, int):
        msg = "Expected interval_length_number to be an integer."
        raise TypeError(msg)


def _check_valid_interval_length_unit(interval_length_unit) -> None:
    # TODO: maybe check if it is a valid unit from pandas.to_timedelta
    if not isinstance(interval_length_unit, str):
        msg = "Expected interval_length_unit to be a string."
        raise TypeError(msg)


def _check_valid_num_intervals(num_intervals) -> None:
    if not isinstance(num_intervals, int):
        msg = "Expected num_intervals to be an integer."
        raise TypeError(msg)


def _check_valid_concept_sets(concept_sets, data_tables) -> dict[str, list[int]]:
    """Check a mapping of names to concept ids, and return it as a dict of lists."""
    if len(data_tables) != 1:
        msg = f"Concept sets take one data table, but data_tables is {list(data_tables)}."
        raise ValueError(msg)
    if len(concept_sets) == 0:
        msg = "concept_ids must not be empty."
        raise ValueError(msg)
    checked: dict[str, list[int]] = {}
    set_of_concept: dict[int, str] = {}
    for name, ids in concept_sets.items():
        if not isinstance(name, str) or name == "":
            msg = f"The names of concept sets must be non-empty strings, but one is {name!r}."
            raise TypeError(msg)
        if isinstance(ids, np.ndarray | pd.Series | pd.Index) and np.ndim(ids) == 1:
            ids = list(ids)
        if isinstance(ids, str) or not isinstance(ids, Sequence) or len(ids) == 0:
            msg = f"The concept set '{name}' must be a non-empty sequence of integers."
            raise TypeError(msg)
        if not all(isinstance(concept_id, int | np.integer) and not isinstance(concept_id, bool) for concept_id in ids):
            msg = f"The concept set '{name}' must hold integer concept ids."
            raise TypeError(msg)
        checked[name] = list(dict.fromkeys(int(concept_id) for concept_id in ids))
        for concept_id in checked[name]:
            if concept_id in set_of_concept:
                msg = f"The concept {concept_id} belongs to the concept sets '{set_of_concept[concept_id]}' and '{name}', but may belong to one only."
                raise ValueError(msg)
            set_of_concept[concept_id] = name
    return checked


def _check_valid_concept_ids(concept_ids) -> None:
    if concept_ids != "all" and not isinstance(concept_ids, Sequence):
        msg = "concept_ids must be a sequence of integers or 'all'."
        raise TypeError(msg)
    if len(concept_ids) == 0:
        msg = "concept_ids must not be empty."
        raise ValueError(msg)


def _check_valid_aggregation_strategy(aggregation_strategy) -> None:
    if not isinstance(aggregation_strategy, str) or aggregation_strategy not in AGGREGATION_STRATEGY_KEY:
        msg = f"aggregation_strategy must be one of {AGGREGATION_STRATEGY_KEY.keys()}."
        raise TypeError(msg)


def _check_valid_aggregation_strategy_mapping(aggregation_strategy, *, concept_sets: bool) -> dict:
    """Check the keys and strategies of an aggregation_strategy mapping, and return it with integer concept ids as keys."""
    checked = {}
    for key, strategy in aggregation_strategy.items():
        _check_valid_aggregation_strategy(strategy)
        if concept_sets:
            if not isinstance(key, str):
                msg = f"With concept sets, aggregation_strategy is keyed by the names of the sets, but has the key {key!r}."
                raise TypeError(msg)
        elif not isinstance(key, int | np.integer) or isinstance(key, bool):
            msg = f"aggregation_strategy is keyed by integer concept ids, but has the key {key!r}."
            raise TypeError(msg)
        checked[key if concept_sets else int(key)] = strategy
    return checked


def _check_valid_aggregation_fill_values(table_strategies, data_field_to_keep, numeric_fields, *, sparse: bool) -> None:
    """Check that the strategies of each data table can be instantiated together.

    'count' fills the empty cells of its variables with 0, every other strategy with NaN, and data_field_to_keep='is_present' always with NaN.
    A table that mixes 'count' with other strategies is a float tensor, which a non-numeric data field cannot be, and sparse=True needs one fill value for all tables.
    """
    fill_values = set()
    for table, strategies in table_strategies.items():
        if not strategies or data_field_to_keep[table][0] == "is_present":
            fill_values.add("nan")
            continue
        counted = {strategy == "count" for strategy in strategies.values()}
        if counted == {True, False} and not numeric_fields[table]:
            msg = (
                f"aggregation_strategy mixes 'count' with other strategies, which needs a numeric data field, "
                f"but '{data_field_to_keep[table][0]}' of '{table}' is not."
            )
            raise TypeError(msg)
        fill_values.add("mixed" if counted == {True, False} else ("zero" if counted == {True} else "nan"))
    if sparse and (len(fill_values) > 1 or "mixed" in fill_values):
        msg = (
            "sparse=True needs one fill value, but aggregation_strategy 'count' fills empty cells with 0 and the "
            "other strategies, as well as data_field_to_keep='is_present', with NaN. Set up these variables in separate calls."
        )
        raise ValueError(msg)


def _as_float(number) -> float:
    """Number as a float, infinite if too large."""
    try:
        return float(number)
    except OverflowError:
        return float("inf") if number > 0 else float("-inf")


def _check_valid_value_range(value_range, *, concept_sets: bool) -> dict | None:
    """Check a value_range mapping of variables to (lower, upper), and return it with integer concept ids as keys."""
    if not isinstance(value_range, Mapping | None):
        msg = "value_range must be a mapping of variables to (lower, upper)."
        raise TypeError(msg)
    if not value_range:
        # no range filters nothing, as no value_range does
        return None
    checked = {}
    for key, bounds in value_range.items():
        if concept_sets:
            if not isinstance(key, str):
                msg = f"With concept sets, value_range is keyed by the names of the sets, but has the key {key!r}."
                raise TypeError(msg)
        elif not isinstance(key, int | np.integer) or isinstance(key, bool):
            msg = f"value_range is keyed by integer concept ids, but has the key {key!r}."
            raise TypeError(msg)
        if isinstance(bounds, str) or not isinstance(bounds, Sequence) or len(bounds) != 2:
            msg = f"The value_range of {key!r} must be a pair (lower, upper), but is {bounds!r}."
            raise TypeError(msg)
        lower, upper = bounds
        for bound in (lower, upper):
            if bound is not None and (
                not isinstance(bound, int | float | np.integer | np.floating)
                or isinstance(bound, bool)
                or np.isnan(_as_float(bound))
            ):
                msg = f"The bounds of the value_range of {key!r} must be numbers or None, but are {bounds!r}."
                raise TypeError(msg)
        if lower is not None and upper is not None and lower > upper:
            msg = f"The value_range of {key!r} has a lower bound above its upper bound: {bounds!r}."
            raise ValueError(msg)
        checked[key if concept_sets else int(key)] = tuple(
            None if bound is None else _as_float(bound) for bound in (lower, upper)
        )
    return checked


UNIT_CONVERSION_COLUMNS = ("unit_concept_id", "to_unit_concept_id", "multiplier")


def _integer_column(frame: pd.DataFrame, column: str, *, nullable: bool) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = (values.isna() & frame[column].notna()) | (values.notna() & (values % 1 != 0))
    if invalid.any() or (not nullable and values.isna().any()):
        msg = f"The column '{column}' of unit_conversions must hold integer concept ids{', or missing values' if nullable else ''}."
        raise TypeError(msg)
    return values.astype("Int64")


def _check_valid_unit_conversions(unit_conversions, concept_sets) -> pd.DataFrame | None:
    """Check unit_conversions and resolve it to concepts.

    Returns one row per (concept_id, unit_concept_id): to_unit_concept_id, to_unit_source_value, multiplier and addend, the offset.
    """
    if unit_conversions is None:
        return None
    if not isinstance(unit_conversions, pd.DataFrame):
        msg = "unit_conversions must be a pandas DataFrame."
        raise TypeError(msg)
    keys = [column for column in ("concept_id", "variable") if column in unit_conversions.columns]
    if len(keys) != 1:
        msg = "unit_conversions needs exactly one of the columns 'concept_id' and 'variable'."
        raise ValueError(msg)
    missing = [column for column in UNIT_CONVERSION_COLUMNS if column not in unit_conversions.columns]
    if missing:
        msg = f"unit_conversions lacks the column(s) {missing}."
        raise ValueError(msg)
    frame = unit_conversions.reset_index(drop=True)

    if keys[0] == "variable":
        if not concept_sets:
            msg = "unit_conversions keyed by 'variable' needs concept sets; key it by 'concept_id' otherwise."
            raise ValueError(msg)
        unknown = sorted(set(frame["variable"]) - set(concept_sets), key=str)
        if unknown:
            msg = f"unit_conversions has conversions for {unknown[:5]}, which are no concept sets."
            raise ValueError(msg)
        frame = frame.assign(concept_id=frame["variable"].map(concept_sets)).explode("concept_id")
    concept_id = _integer_column(frame, "concept_id", nullable=False)

    multiplier = pd.to_numeric(frame["multiplier"], errors="coerce")
    addend = (
        pd.to_numeric(frame["offset"], errors="coerce")
        if "offset" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    for name, values in (("multiplier", multiplier), ("offset", addend)):
        if not np.isfinite(values.astype("float64")).all():
            msg = f"The column '{name}' of unit_conversions must hold finite numbers."
            raise TypeError(msg)
    resolved = pd.DataFrame(
        {
            "concept_id": concept_id.to_numpy(),
            "unit_concept_id": _integer_column(frame, "unit_concept_id", nullable=True).to_numpy(),
            "to_unit_concept_id": _integer_column(frame, "to_unit_concept_id", nullable=False).to_numpy(),
            "to_unit_source_value": (
                frame["to_unit_source_value"].astype("string").to_numpy()
                if "to_unit_source_value" in frame.columns
                else pd.array([pd.NA] * len(frame), dtype="string")
            ),
            "multiplier": pd.array(multiplier.astype("float64").to_numpy(), dtype="Float64"),
            "addend": pd.array(addend.astype("float64").to_numpy(), dtype="Float64"),
        }
    ).astype({"concept_id": "Int64", "unit_concept_id": "Int64", "to_unit_concept_id": "Int64"})
    duplicated = resolved.duplicated(["concept_id", "unit_concept_id"], keep=False)
    if duplicated.any():
        pairs = resolved.loc[duplicated, ["concept_id", "unit_concept_id"]].drop_duplicates().to_numpy().tolist()
        msg = f"unit_conversions converts the (concept_id, unit_concept_id) pair(s) {pairs[:5]} more than once."
        raise ValueError(msg)
    targets = resolved.groupby("concept_id")["to_unit_concept_id"].nunique()
    if (targets > 1).any():
        msg = (
            f"unit_conversions converts the concept(s) {targets[targets > 1].index.tolist()[:5]} to more than one unit."
        )
        raise ValueError(msg)
    return resolved


def _check_valid_value_range_variables(
    value_range, variables_of_table, requested_concept_ids, data_field_to_keep, numeric_fields
):
    """Check that value_range is keyed by variables, and that their data fields are numeric values."""
    known = {variable for table_variables in variables_of_table.values() for variable in table_variables}
    unknown = [key for key in value_range if key not in known and key not in set(requested_concept_ids)]
    if unknown:
        msg = f"value_range has ranges for {unknown[:5]}, which are no variables of the data tables."
        raise ValueError(msg)
    for table, table_variables in variables_of_table.items():
        if any(variable in value_range for variable in table_variables) and (
            data_field_to_keep[table][0] == "is_present" or not numeric_fields[table]
        ):
            msg = f"value_range filters by value, which needs a numeric data field, but '{data_field_to_keep[table][0]}' of '{table}' is not."
            raise TypeError(msg)


def _check_valid_aggregation_strategies(aggregation_strategy, variables_of_table, requested_concept_ids) -> dict:
    """Check a mapping of variables to aggregation strategies against the variables of the data tables.

    Every variable needs a strategy. A key that is no variable raises, unless it is one of requested_concept_ids, which can be without data.
    Returns the strategies of the variables of each data table.
    """
    variables = [variable for table_variables in variables_of_table.values() for variable in table_variables]
    missing = [variable for variable in variables if variable not in aggregation_strategy]
    if missing:
        msg = f"aggregation_strategy has no strategy for the variable(s) {missing[:5]}{' ...' if len(missing) > 5 else ''}."
        raise ValueError(msg)
    known = set(variables) | set(requested_concept_ids)
    unknown = [key for key in aggregation_strategy if key not in known]
    if unknown:
        msg = f"aggregation_strategy has strategies for {unknown[:5]}, which are no variables of the data tables."
        raise ValueError(msg)
    return {
        table: {variable: aggregation_strategy[variable] for variable in table_variables}
        for table, table_variables in variables_of_table.items()
    }


def _check_valid_enrich_var_with_feature_info(enrich_var_with_feature_info) -> None:
    if not isinstance(enrich_var_with_feature_info, bool):
        msg = "Expected enrich_var_with_feature_info to be a boolean."
        raise TypeError(msg)


def _check_valid_enrich_var_with_unit_info(enrich_var_with_unit_info) -> None:
    if not isinstance(enrich_var_with_unit_info, bool):
        msg = "Expected enrich_var_with_unit_info to be a boolean."
        raise TypeError(msg)


def _check_valid_sparse(sparse, instantiate_tensor) -> None:
    if not isinstance(sparse, bool):
        msg = "Expected sparse to be a boolean."
        raise TypeError(msg)
    if sparse and not instantiate_tensor:
        msg = "sparse=True sets the type of the tensor, which instantiate_tensor=False does not create."
        raise ValueError(msg)


def _check_valid_keep_date(keep_date: str) -> None:
    if not isinstance(keep_date, str):
        msg = "Expected keep_date to be a string."
        raise TypeError(msg)
    if keep_date not in VALID_KEEP_DATES:
        msg = f"keep_date must be one of {VALID_KEEP_DATES}."
        raise ValueError(msg)


def _check_valid_time_precision(time_precision: str) -> None:
    if not isinstance(time_precision, str):
        msg = "Expected time_precision to be a string."
        raise TypeError(msg)
    if time_precision not in VALID_TIME_PRECISIONS:
        msg = f"time_precision must be one of {VALID_TIME_PRECISIONS}."
        raise ValueError(msg)


def _warn_time_precision_interval_mismatch(interval_length_unit: str, time_precision: str) -> None:
    """Warn if using fine-grained time intervals with date-only precision.

    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html

    Args:
        interval_length_unit: The unit for interval length (e.g., 'day', 'hour', 'minute', 'second').
        time_precision: The time precision ('date' or 'datetime').
    """
    fine_grained_units = [
        "hours",
        "hour",
        "hr",
        "h",
        "minutes",
        "minute",
        "min",
        "m",
        "seconds",
        "second",
        "sec",
        "s",
        "milliseconds",
        "millisecond",
        "millis",
        "milli",
        "ms",
        "microseconds",
        "microsecond",
        "micros",
        "micro",
        "us",
        "nanoseconds",
        "nanosecond",
        "nanos",
        "nano",
        "ns",
    ]
    if time_precision == "date" and interval_length_unit.lower() in fine_grained_units:
        logging.warning(
            f"Using interval_length_unit='{interval_length_unit}' with time_precision='date' may lead to "
            f"unexpected results. Consider using time_precision='datetime' for fine-grained time intervals. ",
        )


def _check_valid_clip_to_end_date(clip_to_end_date) -> None:
    if not isinstance(clip_to_end_date, bool):
        msg = "Expected clip_to_end_date to be a boolean."
        raise TypeError(msg)


def _check_valid_clip_to_end_date_for_observation_table(clip_to_end_date, time_defining_table: str) -> None:
    if not clip_to_end_date and time_defining_table == "person":
        msg = "clip_to_end_date=False is not available for observation_table='person', whose observations have no end date."
        raise ValueError(msg)


def _check_valid_birthdates_for_person_table(start_dates, time_defining_table: str) -> None:
    """Check that all persons in .obs have valid birthdates when using person as observation table.

    Args:
        start_dates: The birth_datetime column of .obs, parsed as datetimes.
        time_defining_table: The observation table being used.

    Raises:
        ValueError: If the observation table is 'person' and any persons have NULL or invalid birthdates.
    """
    if time_defining_table == "person":
        null_count = int(start_dates.isna().sum())

        if null_count > 0:
            msg = (
                f"Found {null_count} person(s) with NULL birth_datetime. "
                f"When using observation_table='person', all persons must have valid birthdates. "
                f"Please ensure birth_datetime is populated for all persons, filter edata to the persons with a "
                f"birth_datetime, or consider using 'person_observation_period' or 'person_visit_occurrence' in "
                f"ehrdata.io.omop.setup_obs."
            )
            raise ValueError(msg)
