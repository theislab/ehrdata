import logging
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Literal

import duckdb
import numpy as np
import pandas as pd
from sparse import COO

TIME_DEFINING_TABLE_SUBJECT_KEY = {
    "person": "person_id",
    "visit_occurrence": "person_id",
    "observation_period": "person_id",
    "cohort": "subject_id",
}

TIME_DEFINING_TABLE_ID_KEY = {
    "person": "person_id",
    "cohort": "subject_id",
    "observation_period": "observation_period_id",
    "visit_occurrence": "visit_occurrence_id",
}

DATA_TABLE_CONCEPT_ID_TRUNK = {
    "measurement": "measurement",
    "observation": "observation",
    "specimen": "specimen",
    "drug_exposure": "drug",
    "condition_occurrence": "condition",
    "procedure_occurrence": "procedure",
    "device_exposure": "device",
    "drug_era": "drug",
    "dose_era": "drug",
    "condition_era": "condition",
    "episode": "episode",
}

DATA_TABLE_DATE_KEYS = {
    "timepoint": {
        "measurement": {"date": "measurement_date", "datetime": "measurement_datetime"},
        "observation": {"date": "observation_date", "datetime": "observation_datetime"},
        "specimen": {"date": "specimen_date", "datetime": "specimen_datetime"},
    },
    "start": {
        "person": {"date": "birth_datetime", "datetime": "birth_datetime"},
        "visit_occurrence": {"date": "visit_start_date", "datetime": "visit_start_datetime"},
        "observation_period": {"date": "observation_period_start_date"},
        "cohort": {"date": "cohort_start_date"},
        "drug_exposure": {"date": "drug_exposure_start_date", "datetime": "drug_exposure_start_datetime"},
        "condition_occurrence": {"date": "condition_start_date", "datetime": "condition_start_datetime"},
        "procedure_occurrence": {
            "date": "procedure_date",
            "datetime": "procedure_datetime",
        },  # in v5.3, procedure didnt have end date
        "device_exposure": {"date": "device_exposure_start_date", "datetime": "device_exposure_start_datetime"},
        "drug_era": {"date": "drug_era_start_date"},
        "dose_era": {"date": "dose_era_start_date"},
        "condition_era": {"date": "condition_era_start_date"},
        "episode": {"date": "episode_start_date", "datetime": "episode_start_datetime"},
    },
    "end": {
        "visit_occurrence": {"date": "visit_end_date", "datetime": "visit_end_datetime"},
        "observation_period": {"date": "observation_period_end_date"},
        "cohort": {"date": "cohort_end_date"},
        "drug_exposure": {"date": "drug_exposure_end_date", "datetime": "drug_exposure_end_datetime"},
        "condition_occurrence": {"date": "condition_end_date", "datetime": "condition_end_datetime"},
        "procedure_occurrence": {
            "date": "procedure_end_date",
            "datetime": "procedure_end_datetime",
        },  # in v5.3, procedure didnt have end date TODO v5.3 support
        "device_exposure": {"date": "device_exposure_end_date", "datetime": "device_exposure_end_datetime"},
        "drug_era": {"date": "drug_era_end_date"},
        "dose_era": {"date": "dose_era_end_date"},
        "condition_era": {"date": "condition_era_end_date"},
        "episode": {"date": "episode_end_date", "datetime": "episode_end_datetime"},
    },
}


def _get_datetime_key(
    key: Literal["timepoint", "start", "end"], data_table: str, time_precision: Literal["date", "datetime"]
) -> str:
    """Helper function to get datetime precision if avilable; and falls back to date if a table doesn't provide it."""
    value = DATA_TABLE_DATE_KEYS[key][data_table].get(time_precision)
    if not value:
        logging.warning(
            f"Time precision {time_precision} not available for data table {data_table}. Using '...date' and midnight (00:00:00) as '...datetime' instead. Consider using time_precision='date' and less fine-grained intervals when working with this table."
        )
        value = DATA_TABLE_DATE_KEYS[key][data_table]["date"]
    return value


def _finer_datetime_keys(key: Literal["timepoint", "start", "end"], data_table: str, time_precision: str) -> list[str]:
    """The datetime column of a table read at date precision, to break ties within a day chronologically."""
    keys = DATA_TABLE_DATE_KEYS[key][data_table]
    datetime_key = keys.get("datetime")
    return [datetime_key] if time_precision == "date" and datetime_key and datetime_key != keys["date"] else []


# The unit fields carried along the aggregation, if the data table has them - see _get_unit_fields.
UNIT_FIELDS = ("unit_concept_id", "unit_source_value")

AGGREGATION_STRATEGY_KEY = {
    "last": "LAST",
    "first": "FIRST",
    "mean": "MEAN",
    "median": "MEDIAN",
    "mode": "MODE",
    "sum": "SUM",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
    "std": "STDDEV_SAMP",
}

# Strategies that keep the value of a single row, rather than combining the values of several rows.
SINGLE_ROW_AGGREGATION_STRATEGIES = ("last", "first")

# Strategies whose result is a number of data points, and hence carries no unit.
UNITLESS_AGGREGATION_STRATEGIES = ("count",)


def _links_rows_to_visits(backend_handle: duckdb.DuckDBPyConnection, time_defining_table: str, data_table: str) -> bool:
    """Whether the rows of data_table are attributed to visit observations by their visit_occurrence_id.

    That is the case if the observations are visits, data_table has a visit_occurrence_id, which 'specimen' and the era tables do not,
    and the database has a visit_occurrence table to tell which visits the ids name.
    """
    if time_defining_table != "visit_occurrence":
        return False

    def columns(table: str) -> set[str]:
        return {
            row[0].lower()
            for row in backend_handle.execute(
                "SELECT column_name FROM information_schema.columns WHERE lower(table_name) = ?", [table]
            ).fetchall()
        }

    return "visit_occurrence_id" in columns(data_table) and {"visit_occurrence_id", "person_id"} <= columns(
        "visit_occurrence"
    )


LINKED_VISIT = "ehrdata_linked_visit_occurrence_id"


def _data_rows_query(data_table: str, *, link_visits: bool) -> str:
    """The rows of data_table, with link_visits also the visit each row names in LINKED_VISIT.

    A row names a visit if its visit_occurrence_id is a visit of the same person in the visit_occurrence table; otherwise LINKED_VISIT is NULL.
    """
    if not link_visits:
        return f"SELECT * FROM {data_table}"
    return f"""
        SELECT d.*, v.visit_occurrence_id AS {LINKED_VISIT}
        FROM {data_table} AS d
        LEFT JOIN (SELECT DISTINCT TRY_CAST(visit_occurrence_id AS BIGINT) AS visit_occurrence_id, person_id FROM visit_occurrence) AS v
            ON TRY_CAST(d.visit_occurrence_id AS BIGINT) = v.visit_occurrence_id AND d.person_id = v.person_id
    """


def _visit_condition(data_alias: str, obs_alias: str, *, link_visits: bool) -> str:
    """The SQL condition that a row of a data table may belong to a visit observation, besides lying in its window.

    A row that names a visit of its person belongs to that visit only; any other row to every visit of its person whose window holds it.
    """
    if not link_visits:
        return "TRUE"
    return (
        f"({data_alias}.{LINKED_VISIT} IS NULL OR {data_alias}.{LINKED_VISIT} = TRY_CAST({obs_alias}.obs_id AS BIGINT))"
    )


def _get_unit_fields(backend_handle: duckdb.DuckDBPyConnection, data_table: str) -> tuple[str, ...]:
    """Get the fields of UNIT_FIELDS that data_table has.

    Not every data table has a unit, and those that do don't have the same unit fields across OMOP CDM versions: 'device_exposure' has them in 5.4, but not in 5.3.
    """
    columns = set(backend_handle.execute(f"DESCRIBE {data_table}").df()["column_name"])
    return tuple(field for field in UNIT_FIELDS if field in columns)


def _concept_ids_condition(data_table: str, concept_ids: Literal["all"] | Sequence[int]) -> str:
    """Get the SQL condition that restricts the rows of data_table to concept_ids."""
    if concept_ids == "all":
        return "TRUE"
    if len(concept_ids) == 0:
        return "FALSE"
    return f"{DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id IN ({', '.join(str(int(concept_id)) for concept_id in concept_ids)})"


def _generate_timedeltas(interval_length_number: int, interval_length_unit: str, num_intervals: int) -> pd.DataFrame:
    timedeltas_dataframe = pd.DataFrame(
        {
            "interval_start_offset": [
                pd.to_timedelta(i * interval_length_number, interval_length_unit) for i in range(num_intervals)
            ],
            "interval_end_offset": [
                pd.to_timedelta(i * interval_length_number, interval_length_unit) for i in range(1, num_intervals + 1)
            ],
            "interval_step": list(range(num_intervals)),
        }
    )
    return timedeltas_dataframe


def _write_timedeltas_to_db(
    backend_handle: duckdb.DuckDBPyConnection,
    timedeltas_dataframe,
) -> None:
    backend_handle.execute("DROP TABLE IF EXISTS timedeltas")
    backend_handle.execute(
        """
        CREATE TABLE timedeltas (
            interval_start_offset INTERVAL,
            interval_end_offset INTERVAL,
            interval_step INTEGER
        )
        """
    )
    # DuckDB reads a pandas Timedelta of 30 days or more as calendar months, so the offsets are passed as microseconds.
    offsets_in_microseconds = timedeltas_dataframe.assign(  # noqa: F841
        interval_start_offset=timedeltas_dataframe["interval_start_offset"] // pd.Timedelta(microseconds=1),
        interval_end_offset=timedeltas_dataframe["interval_end_offset"] // pd.Timedelta(microseconds=1),
    )
    backend_handle.execute(
        """
        INSERT INTO timedeltas
        SELECT to_microseconds(interval_start_offset), to_microseconds(interval_end_offset), interval_step
        FROM offsets_in_microseconds
        """
    )


def _drop_timedeltas(backend_handle: duckdb.DuckDBPyConnection):
    backend_handle.execute("DROP TABLE IF EXISTS timedeltas")


def _generate_value_query(
    data_table: str,
    data_field_to_keep: Sequence,
    aggregation_strategy: str,
    datetime_column: str,
    *,
    unit_fields: Sequence[str] = (),
    finer_datetime_columns: Sequence[str] = (),
) -> str:
    # A row whose data_field_to_keep is NULL carries no value.
    # Such a row must neither mask a value observed in the same interval, nor have a say in the unit of the kept value.
    # Hence the filter below on is_present and on the unit fields; the values themselves need none, as SQL aggregates ignore NULL inputs anyway.
    value_field = data_field_to_keep[0]
    row_filter = f"FILTER (WHERE {value_field} IS NOT NULL)"

    single_row_aggregation = aggregation_strategy in [
        AGGREGATION_STRATEGY_KEY[strategy] for strategy in SINGLE_ROW_AGGREGATION_STRATEGIES
    ]

    # is_present is 1 in all rows of the data_table; but need an aggregation operation, so use LAST.
    # LAST/FIRST need ORDER BY for chronological order; for the other strategies, ordering doesn't matter.
    # Break the rare timestamp ties by the full datetime (at date precision), then the kept columns.
    tie_break = "".join(f", {column}" for column in [*finer_datetime_columns, *data_field_to_keep, *unit_fields])
    order_by = f" ORDER BY {datetime_column}{tie_break}" if single_row_aggregation else ""
    is_present_query = f"LAST(is_present{order_by}) {row_filter} as is_present, "

    if single_row_aggregation:
        # One single row is kept, so all kept columns, the units included, are read from that very row.
        value_query = ", ".join(
            f"{aggregation_strategy}({column}{order_by}) {row_filter} AS {column}"
            for column in [*data_field_to_keep, *unit_fields]
        )
    else:
        # Of several equally frequent values, MODE keeps the first it reads; ordering by the value makes that the smallest.
        value_query = ", ".join(
            f"{aggregation_strategy}({column}{f' ORDER BY {column}' if aggregation_strategy == 'MODE' else ''}) AS {column}"
            for column in data_field_to_keep
        )

        # These strategies combine several rows into one value, so there is no single row to read the unit from.
        # A unit is carried along if the rows carrying a value agree on it, and is NULL if they do not: no unit describes a value that mixes units.
        # A row without a unit is no disagreement, it simply does not tell one - COUNT(DISTINCT) and MIN ignore NULL.
        # Differing unit_concept_id is ruled out up front by _check_one_unit_per_feature_for_aggregation, but unit_source_value can still differ within one unit_concept_id (e.g. 'bpm' and 'beats/min').
        unit_query = ", ".join(
            f"CASE WHEN COUNT(DISTINCT {column}) {row_filter} > 1 THEN NULL ELSE MIN({column}) {row_filter} END AS {column}"
            for column in unit_fields
        )
        value_query = f"{value_query}, {unit_query}" if unit_query else value_query

    return is_present_query + value_query


def _write_long_time_interval_table(
    backend_handle: duckdb.DuckDBPyConnection,
    *,
    data_table: str,
    time_precision: Literal["date", "datetime"],
    interval_length_number: int,
    interval_length_unit: str,
    num_intervals: int,
    aggregation_strategy: str,
    data_field_to_keep: Sequence[str] | str,
    unit_fields: Sequence[str] = (),
    concept_ids: Literal["all"] | Sequence[int] = "all",
    keep_date: str = "",
    link_visits: bool = False,
) -> None:
    if isinstance(data_field_to_keep, str):
        data_field_to_keep = [data_field_to_keep]

    if keep_date == "":
        keep_date = "timepoint"

    timedeltas_dataframe = _generate_timedeltas(interval_length_number, interval_length_unit, num_intervals)

    _write_timedeltas_to_db(
        backend_handle,
        timedeltas_dataframe,
    )

    create_long_table_query = f"CREATE TABLE long_person_timestamp_feature_value_{data_table} AS\n"

    # multi-step query
    # 1. Create person_time_defining_table, the rows of .obs from OBS_WINDOWS_TABLE: the obs_index (unique row identifier, the position in .obs), obs_id, person_id, and the start/end dates.
    # 2. Create person_data_table (data_table is typically measurement), which contains the cross product of obs_id and the distinct concept_id s.
    # 3. Create long_format_backbone, which is the left join of person_time_defining_table and person_data_table.
    # 4. Create long_format_intervals, which is the cross product of long_format_backbone and timedeltas. This table contains most notably the obs_id, the concept_id, the interval start and end dates.
    # 5. Create the final table long_person_timestamp_feature_value_<data_table>, which is the join with the data_table (typically measurement); each measurement is assigned to its obs_id, its concept_id, and the interval it fits into.

    person_time_defining_cte = f"WITH person_time_defining_table AS (SELECT obs_index, obs_id, person_id, start_date, end_date FROM {OBS_WINDOWS_TABLE}), "

    prepare_alias_query = (
        person_time_defining_cte
        + f"""person_data_table AS( \
            WITH distinct_data_table_concept_ids AS ( \
                SELECT DISTINCT {DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id
                FROM {data_table} \
                WHERE {_concept_ids_condition(data_table, concept_ids)} \
            )
            SELECT obs_index, obs_id, person_id, {DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id as data_table_concept_id, start_date, end_date \
            FROM person_time_defining_table \
            CROSS JOIN distinct_data_table_concept_ids \
        ), \
        long_format_backbone as ( \
            SELECT obs_index, obs_id, person_id, data_table_concept_id, start_date, end_date \
            FROM person_data_table \
        ), \
        long_format_intervals as ( \
            SELECT obs_index, obs_id, person_id, data_table_concept_id, interval_step, start_date, end_date, start_date + interval_start_offset as interval_start, start_date + interval_end_offset as interval_end \
            FROM long_format_backbone \
            CROSS JOIN timedeltas \
        ), \
        data_table_with_presence_indicator as( \
            SELECT *, 1 as is_present \
            FROM ({_data_rows_query(data_table, link_visits=link_visits)}) \
        ) \
        """
    )

    if keep_date in ["timepoint", "start", "end"]:
        datetime_col = _get_datetime_key(keep_date, data_table, time_precision)
        select_query = f"""
        SELECT lfi.obs_id, lfi.person_id, lfi.data_table_concept_id, interval_step, interval_start, interval_end, {_generate_value_query("data_table_with_presence_indicator", data_field_to_keep, AGGREGATION_STRATEGY_KEY[aggregation_strategy], datetime_col, unit_fields=unit_fields, finer_datetime_columns=_finer_datetime_keys(keep_date, data_table, time_precision))}, lfi.obs_index \
        FROM long_format_intervals as lfi \
        LEFT JOIN data_table_with_presence_indicator ON lfi.person_id = data_table_with_presence_indicator.person_id AND {_visit_condition("data_table_with_presence_indicator", "lfi", link_visits=link_visits)} AND lfi.data_table_concept_id = data_table_with_presence_indicator.{DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id AND data_table_with_presence_indicator.{datetime_col} >= lfi.interval_start AND data_table_with_presence_indicator.{datetime_col} < lfi.interval_end \
                AND CAST(data_table_with_presence_indicator.{datetime_col} AS DATE) <= lfi.end_date \
        GROUP BY lfi.obs_index, lfi.obs_id, lfi.person_id, lfi.data_table_concept_id, interval_step, interval_start, interval_end
        """

    elif keep_date == "interval":
        datetime_col_start = _get_datetime_key("start", data_table, time_precision)
        datetime_col_end = _get_datetime_key("end", data_table, time_precision)
        select_query = f"""
        SELECT lfi.obs_id, lfi.person_id, lfi.data_table_concept_id, interval_step, interval_start, interval_end, {_generate_value_query("data_table_with_presence_indicator", data_field_to_keep, AGGREGATION_STRATEGY_KEY[aggregation_strategy], datetime_col_start, unit_fields=unit_fields, finer_datetime_columns=_finer_datetime_keys("start", data_table, time_precision))}, lfi.obs_index \
        FROM long_format_intervals as lfi \
        LEFT JOIN data_table_with_presence_indicator ON lfi.person_id = data_table_with_presence_indicator.person_id \
                AND {_visit_condition("data_table_with_presence_indicator", "lfi", link_visits=link_visits)} \
                AND lfi.data_table_concept_id = data_table_with_presence_indicator.{DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id \
                AND (data_table_with_presence_indicator.{datetime_col_start} >= lfi.interval_start AND data_table_with_presence_indicator.{datetime_col_start} < lfi.interval_end \
                    OR data_table_with_presence_indicator.{datetime_col_end} >= lfi.interval_start AND data_table_with_presence_indicator.{datetime_col_end} < lfi.interval_end \
                    OR (data_table_with_presence_indicator.{datetime_col_start} < lfi.interval_start AND data_table_with_presence_indicator.{datetime_col_end} >= lfi.interval_end)) \
                AND CAST(data_table_with_presence_indicator.{datetime_col_start} AS DATE) <= lfi.end_date \
                AND CAST(lfi.interval_start AS DATE) <= lfi.end_date \
        GROUP BY lfi.obs_index, lfi.obs_id, lfi.person_id, lfi.data_table_concept_id, interval_step, interval_start, interval_end
        """

    query = create_long_table_query + prepare_alias_query + select_query

    backend_handle.execute(f"DROP TABLE IF EXISTS long_person_timestamp_feature_value_{data_table}")
    backend_handle.execute(query)


OBS_WINDOWS_TABLE = "ehrdata_obs_windows"
# The concept sets of setup_variables, registered while the long format table is written: variable, concept_id.
VARIABLE_MAP_TABLE = "_ehrdata_variable_map"
# The value ranges of setup_variables, resolved to concepts and registered while the queries that filter by them run: concept_id, lower, upper.
VALUE_RANGE_TABLE = "_ehrdata_value_range"


@contextmanager
def _registered(backend_handle: duckdb.DuckDBPyConnection, name: str, frame: pd.DataFrame | None):
    """Register frame under name for the duration of the block, if frame is given."""
    if frame is None:
        yield
        return
    backend_handle.register(name, frame)
    try:
        yield
    finally:
        backend_handle.unregister(name)


def _value_range_filter(
    data_alias: str, concept_key: str, value_field: str, *, value_ranges, value_type: str = "DOUBLE"
) -> tuple[str, str]:
    """The SQL join and condition that keep the rows whose value lies in the range of their concept, bounds included.

    Rows of concepts without a range, and rows without a value, are kept; NaN lies in no range.
    The bounds are compared in the type of the value field, so that a bound equal to a stored single-precision value includes it.
    """
    if value_ranges is None:
        return "", "TRUE"
    value = f"{data_alias}.{value_field}"
    bound_type = "FLOAT" if value_type.upper() in {"FLOAT", "FLOAT4", "REAL"} else "DOUBLE"
    lower, upper = f"CAST(vr.lower AS {bound_type})", f"CAST(vr.upper AS {bound_type})"
    join = f"LEFT JOIN {VALUE_RANGE_TABLE} AS vr ON TRY_CAST({data_alias}.{concept_key} AS BIGINT) = vr.concept_id"
    condition = (
        f"(vr.concept_id IS NULL OR {value} IS NULL OR "
        f"(NOT isnan(CAST({value} AS DOUBLE)) AND ({value} >= {lower} OR vr.lower IS NULL) AND ({value} <= {upper} OR vr.upper IS NULL)))"
    )
    return join, condition


def _column_type(backend_handle: duckdb.DuckDBPyConnection, table: str, column: str) -> str:
    """The type of column of table, or an empty string."""
    types = {row[0].lower(): row[1] for row in backend_handle.execute(f"DESCRIBE {table}").fetchall()}
    return types.get(column.lower(), "")


def _write_obs_windows(
    backend_handle: duckdb.DuckDBPyConnection,
    obs: pd.DataFrame,
    time_defining_table: str,
    time_precision: Literal["date", "datetime"],
    *,
    clip_to_end_date: bool = True,
) -> None:
    """Write the time window of every row of `obs` to OBS_WINDOWS_TABLE: obs_index, obs_name, obs_id, person_id, start_date and end_date.

    obs_index is the position of the row in `obs`, so that the variables follow the rows of `.obs`, also after they were filtered or reordered since :func:`~ehrdata.io.omop.setup_obs`.
    The start and end dates are those of the columns written by :func:`~ehrdata.io.omop.setup_obs`; a row without a start date, or without an end date if clip_to_end_date, gets no data.
    Without clip_to_end_date, and for the person table, the windows have no end date, which an end far in the future stands for.
    The long format tables are written from OBS_WINDOWS_TABLE, so this runs first.
    """
    from ehrdata.io.omop._check_arguments import _check_valid_birthdates_for_person_table

    columns = {column.lower(): column for column in obs.columns}
    id_key = TIME_DEFINING_TABLE_ID_KEY[time_defining_table]
    start_key = _get_datetime_key("start", time_defining_table, time_precision)
    # The end date rather than datetime, as OMOP end dates are inclusive and end datetimes are optional.
    end_key = (
        None
        if time_defining_table == "person" or not clip_to_end_date
        else _get_datetime_key("end", time_defining_table, "date")
    )
    missing = [key for key in dict.fromkeys([id_key, "person_id", start_key, end_key]) if key and key not in columns]
    if missing:
        msg = f"The time window of each observation is read from .obs, which lacks the column(s) {missing} that setup_obs writes."
        raise ValueError(msg)

    start_dates = pd.to_datetime(obs[columns[start_key]])
    _check_valid_birthdates_for_person_table(start_dates, time_defining_table)
    windows = pd.DataFrame(
        {
            "obs_index": np.arange(len(obs), dtype=np.int32),
            "obs_name": obs.index.astype(str).to_numpy(),
            "obs_id": obs[columns[id_key]].to_numpy(),
            "person_id": obs[columns["person_id"]].to_numpy(),
            "start_date": start_dates.to_numpy(),
            "end_date": pd.to_datetime(obs[columns[end_key]]).to_numpy() if end_key else pd.NaT,
        }
    )
    end_date = "CAST('2999-12-31 23:59:59' AS TIMESTAMP)" if end_key is None else "CAST(end_date AS DATE)"
    # registered under a name of its own, as a table of the database would shadow a Python variable
    backend_handle.register("_ehrdata_obs_windows_frame", windows)
    try:
        backend_handle.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {OBS_WINDOWS_TABLE} AS
            SELECT obs_index, obs_name, obs_id, person_id, CAST(start_date AS TIMESTAMP) AS start_date, {end_date} AS end_date
            FROM _ehrdata_obs_windows_frame
            """
        )
    finally:
        backend_handle.unregister("_ehrdata_obs_windows_frame")


def _write_obs_names(backend_handle: duckdb.DuckDBPyConnection, data_table: str) -> None:
    """Write `long_person_timestamp_feature_value_<data_table>_obs`: the obs_index and obs_name of every row of `.obs` the long format table was set up for.

    The obs_index of the long format table is a position in that `.obs`, which an `EHRData` filtered or reordered later no longer shares; its obs_names map its rows back.
    """
    backend_handle.execute(
        f"CREATE OR REPLACE TABLE long_person_timestamp_feature_value_{data_table}_obs AS SELECT obs_index, obs_name FROM {OBS_WINDOWS_TABLE}"
    )


def _sql_literal(value: int | str) -> str:
    """The SQL literal of value: an integer, or a string in single quotes."""
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(int(value))


def _write_long_observed_table(
    backend_handle: duckdb.DuckDBPyConnection,
    *,
    data_table: str,
    time_precision: Literal["date", "datetime"],
    interval_length_number: int,
    interval_length_unit: str,
    num_intervals: int,
    aggregation_strategy: str,
    data_field_to_keep: Sequence[str],
    unit_fields: Sequence[str] = (),
    concept_ids: Literal["all"] | Sequence[int] = "all",
    link_visits: bool = False,
    concept_sets: Mapping[str, Sequence[int]] | None = None,
    strategies: Mapping[int | str, str] | None = None,
    value_ranges: pd.DataFrame | None = None,
) -> None:
    """Write `long_person_timestamp_feature_value_<data_table>` with one row per observed (observation, concept, interval).

    With value_ranges, the rows whose value lies outside the range of their concept are left out before the aggregation, see `_value_range_filter`.
    With strategies, the aggregation_strategy of each variable, keyed like the variables: by concept id, or by the name of the concept set.
    All strategies aggregate the same observed rows, which are read from the data table once.
    With concept_sets, a row per observed (observation, variable, interval) instead: the rows of the concepts of a set are aggregated together, and the column `variable` holds the name of the set in place of `data_table_concept_id`.

    Each row of the data_table is joined to the observations of its person whose time window holds it, and binned into its interval there.
    With link_visits, a row that names a visit of its person is joined to that visit only, see `_visit_condition`.
    Only the cells that receive a row carrying a value are aggregated and stored, so the table grows with the data rather than with observations x concepts x intervals.
    """
    interval_us = int(pd.Timedelta(interval_length_number, interval_length_unit) // pd.Timedelta(microseconds=1))
    datetime_col = _get_datetime_key("timepoint", data_table, time_precision)
    finer_datetime_cols = _finer_datetime_keys("timepoint", data_table, time_precision)
    concept_key = f"{DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id"

    row_time = f"epoch_us(CAST(d.{datetime_col} AS TIMESTAMP))"
    start_time = "epoch_us(CAST(o.start_date AS TIMESTAMP))"
    kept_columns = [
        column
        for column in dict.fromkeys([*data_field_to_keep, *unit_fields, datetime_col, *finer_datetime_cols])
        if column != "is_present"
    ]
    kept_select = "".join(f", d.{column}" for column in kept_columns)

    variables_of_strategy: dict[str, list] = {}
    for variable, strategy in (strategies or {}).items():
        variables_of_strategy.setdefault(strategy, []).append(variable)
    if not variables_of_strategy:
        variables_of_strategy = {aggregation_strategy: []}

    variable_map = None
    if concept_sets is None:
        key_column, key_select, variable_join = "data_table_concept_id", f"d.{concept_key}", ""
        concept_condition = _concept_ids_condition(data_table, concept_ids)
    else:
        key_column, key_select = "variable", "vm.variable"
        variable_join = f"JOIN {VARIABLE_MAP_TABLE} AS vm ON d.{concept_key} = vm.concept_id"
        concept_condition = "TRUE"
        variable_map = pd.DataFrame(
            [(str(name), int(concept_id)) for name, ids in concept_sets.items() for concept_id in ids],
            columns=["variable", "concept_id"],
        )

    groups = []
    for strategy, variables in variables_of_strategy.items():
        value_query = _generate_value_query(
            "observed_rows",
            data_field_to_keep,
            AGGREGATION_STRATEGY_KEY[strategy],
            datetime_col,
            unit_fields=unit_fields,
            finer_datetime_columns=finer_datetime_cols,
        )
        condition = (
            f"WHERE {key_column} IN ({', '.join(_sql_literal(variable) for variable in variables)})"
            if len(variables_of_strategy) > 1
            else ""
        )
        groups.append(
            f"""
            SELECT obs_id, person_id, {key_column}, interval_step,
                start_date + to_microseconds(interval_step * {interval_us}) AS interval_start,
                start_date + to_microseconds((interval_step + 1) * {interval_us}) AS interval_end,
                {value_query}, obs_index
            FROM observed_rows
            {condition}
            GROUP BY obs_id, obs_index, person_id, start_date, {key_column}, interval_step"""
        )
    observed_cells = " UNION ALL ".join(groups)

    range_join, range_condition = _value_range_filter(
        "d",
        concept_key,
        data_field_to_keep[0],
        value_ranges=value_ranges,
        value_type=_column_type(backend_handle, data_table, data_field_to_keep[0]) if value_ranges is not None else "",
    )

    table = f"long_person_timestamp_feature_value_{data_table}"
    with (
        _registered(backend_handle, VARIABLE_MAP_TABLE, variable_map),
        _registered(backend_handle, VALUE_RANGE_TABLE, value_ranges),
    ):
        backend_handle.execute(f"DROP TABLE IF EXISTS {table}")
        backend_handle.execute(
            f"""
        CREATE TABLE {table} AS
        WITH observations AS (SELECT obs_index, obs_id, person_id, start_date, end_date FROM {OBS_WINDOWS_TABLE}),
        observed_rows AS MATERIALIZED (
            SELECT o.obs_id, o.obs_index, o.person_id, o.start_date, {key_select} AS {key_column},
                CAST(({row_time} - {start_time}) // {interval_us} AS INTEGER) AS interval_step, 1 AS is_present{kept_select}
            FROM ({_data_rows_query(data_table, link_visits=link_visits)}) AS d
            {variable_join}
            {range_join}
            JOIN observations AS o ON d.person_id = o.person_id AND {_visit_condition("d", "o", link_visits=link_visits)}
            WHERE {concept_condition}
                AND {range_condition}
                AND {row_time} >= {start_time}
                AND {row_time} - {start_time} < {interval_us * num_intervals}
                AND CAST(d.{datetime_col} AS DATE) <= o.end_date
        ),
        observed_cells AS ({observed_cells})
        SELECT * FROM observed_cells WHERE is_present IS NOT NULL
        """,
        )


def _instantiate_long_observed_table(
    backend_handle: duckdb.DuckDBPyConnection,
    data_table: str,
    value_field: str,
    *,
    variables: Sequence,
    n_obs: int,
    num_intervals: int,
    aggregation_strategy: str | Sequence[str],
    sparse: bool,
    key_column: str = "data_table_concept_id",
):
    """The (n_obs, len(variables), num_intervals) array of `long_person_timestamp_feature_value_<data_table>`.

    variables are the values of key_column in the order of the variables: the concept ids, or the names of the concept sets.
    aggregation_strategy is one strategy for all variables, or one per variable.

    Cells without a row hold 0 for `aggregation_strategy="count"`, and a missing value otherwise.
    The value field 'is_present' is 1 or missing whatever the aggregation_strategy, as its column holds the presence rather than the aggregate.
    """
    cells = backend_handle.execute(
        f"SELECT obs_index, {key_column} AS key, interval_step, {value_field} AS value FROM long_person_timestamp_feature_value_{data_table}"
    ).df()
    var_index = pd.Index(variables).get_indexer(cells["key"])
    coords = (cells["obs_index"].to_numpy(), var_index, cells["interval_step"].to_numpy())
    shape = (n_obs, len(variables), num_intervals)
    values = cells["value"]

    strategies = (
        [aggregation_strategy] * len(variables) if isinstance(aggregation_strategy, str) else list(aggregation_strategy)
    )
    counted = np.array([strategy == "count" for strategy in strategies]) & (value_field != "is_present")
    if counted.any() and not counted.all():
        # the variables that count data points hold 0 in their empty cells, the others a missing value
        if sparse:
            msg = "sparse=True needs one fill value, but aggregation_strategy 'count' fills empty cells with 0 and the other strategies with NaN."
            raise ValueError(msg)
        array = np.full(shape, np.nan)
        array[:, counted] = 0
        array[coords] = values.to_numpy(dtype=np.float64, na_value=np.nan)
        return array
    if counted.all() and len(strategies) > 0:
        dtype, fill_value = np.int64, 0
    elif pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        dtype, fill_value = np.float64, np.nan
    elif sparse:
        msg = f"sparse=True requires a numeric data field, but '{value_field}' of '{data_table}' is of type {values.dtype}."
        raise TypeError(msg)
    elif isinstance(values.dtype, np.dtype) and values.dtype.kind == "M":
        dtype, fill_value = values.dtype, np.datetime64("NaT")
    else:
        dtype, fill_value = object, np.nan

    values = values.to_numpy(dtype=dtype, na_value=fill_value)
    if sparse:
        return COO(np.stack(coords), values, shape=shape, fill_value=fill_value, has_duplicates=False)
    array = np.full(shape, fill_value, dtype=dtype)
    array[coords] = values
    return array


def _get_ordered_table(duckdb_instance, table_name: str) -> pd.DataFrame:
    """Extract a table of an OMOP CDM Database."""
    if table_name not in TIME_DEFINING_TABLE_ID_KEY:
        err = f"Table {table_name} cannot be used as table of _get_ordered_table"
        raise ValueError(err)

    return _lowercase_column_names(
        duckdb_instance.sql(f"SELECT * FROM {table_name} ORDER BY {TIME_DEFINING_TABLE_ID_KEY[table_name]}").df()
    )


def _get_table_join(
    duckdb_instance, table1: str, table2: str, left_key: str = "person_id", right_key: str = "person_id"
) -> pd.DataFrame:
    """Extract a table of an OMOP CDM Database."""
    if table2 not in TIME_DEFINING_TABLE_ID_KEY:
        err = f"Table {table2} cannot be used as table2 of _get_table_join"
        raise ValueError(err)

    return _lowercase_column_names(
        duckdb_instance.sql(
            f"SELECT * \
        FROM {table1} as t1 \
        JOIN {table2} as t2 ON t1.{left_key} = t2.{right_key} \
        ORDER BY t2.{TIME_DEFINING_TABLE_ID_KEY[table2]} \
        "
        ).df()
    )


def _lowercase_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all column names to lowercase."""
    df.columns = map(str.lower, df.columns)  # Convert all column names to lowercase
    return df
