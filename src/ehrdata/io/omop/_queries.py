import logging
from collections.abc import Sequence
from typing import Literal

import duckdb
import pandas as pd

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


def _get_unit_fields(backend_handle: duckdb.DuckDBPyConnection, data_table: str) -> tuple[str, ...]:
    """Get the fields of UNIT_FIELDS that data_table has.

    Not every data table has a unit, and those that do don't have the same unit fields across OMOP CDM versions: 'device_exposure' has them in 5.4, but not in 5.3.
    """
    columns = set(backend_handle.execute(f"DESCRIBE {data_table}").df()["column_name"])
    return tuple(field for field in UNIT_FIELDS if field in columns)


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
    backend_handle.execute("INSERT INTO timedeltas SELECT * FROM timedeltas_dataframe")


def _drop_timedeltas(backend_handle: duckdb.DuckDBPyConnection):
    backend_handle.execute("DROP TABLE IF EXISTS timedeltas")


def _generate_value_query(
    data_table: str,
    data_field_to_keep: Sequence,
    aggregation_strategy: str,
    datetime_column: str,
    unit_fields: Sequence[str] = (),
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
    order_by = f" ORDER BY {datetime_column}" if single_row_aggregation else ""
    is_present_query = f"LAST(is_present{order_by}) {row_filter} as is_present, "

    if single_row_aggregation:
        # One single row is kept, so all kept columns, the units included, are read from that very row.
        value_query = ", ".join(
            f"{aggregation_strategy}({column}{order_by}) {row_filter} AS {column}"
            for column in [*data_field_to_keep, *unit_fields]
        )
    else:
        value_query = ", ".join(f"{aggregation_strategy}({column}) AS {column}" for column in data_field_to_keep)

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
    time_defining_table: str,
    data_table: str,
    time_precision: Literal["date", "datetime"],
    interval_length_number: int,
    interval_length_unit: str,
    num_intervals: int,
    aggregation_strategy: str,
    data_field_to_keep: Sequence[str] | str,
    unit_fields: Sequence[str] = (),
    keep_date: str = "",
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
    # 1. Create person_time_defining_table, which matches the one created for obs. Needs to contain the obs_id (unique row identifier), person_id, and the start/end dates.
    # 2. Create person_data_table (data_table is typically measurement), which contains the cross product of obs_id and the distinct concept_id s.
    # 3. Create long_format_backbone, which is the left join of person_time_defining_table and person_data_table.
    # 4. Create long_format_intervals, which is the cross product of long_format_backbone and timedeltas. This table contains most notably the obs_id, the concept_id, the interval start and end dates.
    # 5. Create the final table long_person_timestamp_feature_value_<data_table>, which is the join with the data_table (typically measurement); each measurement is assigned to its obs_id, its concept_id, and the interval it fits into.

    time_defining_id_key = TIME_DEFINING_TABLE_ID_KEY.get(time_defining_table, "person_id")

    # Special handling for person table - no join needed, use far future date as end_date
    if time_defining_table == "person":
        person_time_defining_cte = f"""WITH person_time_defining_table AS ( \
            SELECT person_id, person_id as obs_id, {_get_datetime_key("start", time_defining_table, time_precision)} as start_date, CAST('2999-12-31 23:59:59' AS TIMESTAMP) as end_date \
            FROM person \
        ), \
        """
    else:
        person_time_defining_cte = f"""WITH person_time_defining_table AS ( \
            SELECT person.person_id as person_id, {time_defining_table}.{time_defining_id_key} as obs_id, {_get_datetime_key("start", time_defining_table, time_precision)} as start_date, {_get_datetime_key("end", time_defining_table, time_precision)} as end_date \
            FROM person \
            JOIN {time_defining_table} ON person.person_id = {time_defining_table}.{TIME_DEFINING_TABLE_SUBJECT_KEY[time_defining_table]} \
        ), \
        """

    prepare_alias_query = (
        person_time_defining_cte
        + f"""person_data_table AS( \
            WITH distinct_data_table_concept_ids AS ( \
                SELECT DISTINCT {DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id
                FROM {data_table} \
            )
            SELECT obs_id, person_id, {DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id as data_table_concept_id, start_date, end_date \
            FROM person_time_defining_table \
            CROSS JOIN distinct_data_table_concept_ids \
        ), \
        long_format_backbone as ( \
            SELECT obs_id, person_id, data_table_concept_id, start_date, end_date \
            FROM person_data_table \
        ), \
        long_format_intervals as ( \
            SELECT obs_id, person_id, data_table_concept_id, interval_step, start_date, start_date + interval_start_offset as interval_start, start_date + interval_end_offset as interval_end \
            FROM long_format_backbone \
            CROSS JOIN timedeltas \
        ), \
        data_table_with_presence_indicator as( \
            SELECT *, 1 as is_present \
            FROM {data_table} \
        ) \
        """
    )

    if keep_date in ["timepoint", "start", "end"]:
        datetime_col = _get_datetime_key(keep_date, data_table, time_precision)
        select_query = f"""
        SELECT lfi.obs_id, lfi.person_id, lfi.data_table_concept_id, interval_step, interval_start, interval_end, {_generate_value_query("data_table_with_presence_indicator", data_field_to_keep, AGGREGATION_STRATEGY_KEY[aggregation_strategy], datetime_col, unit_fields)} \
        FROM long_format_intervals as lfi \
        LEFT JOIN data_table_with_presence_indicator ON lfi.person_id = data_table_with_presence_indicator.person_id AND lfi.data_table_concept_id = data_table_with_presence_indicator.{DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id AND data_table_with_presence_indicator.{datetime_col} >= lfi.interval_start AND data_table_with_presence_indicator.{datetime_col} < lfi.interval_end \
        GROUP BY lfi.obs_id, lfi.person_id, lfi.data_table_concept_id, interval_step, interval_start, interval_end
        """

    elif keep_date == "interval":
        datetime_col_start = _get_datetime_key("start", data_table, time_precision)
        datetime_col_end = _get_datetime_key("end", data_table, time_precision)
        select_query = f"""
        SELECT lfi.obs_id, lfi.person_id, lfi.data_table_concept_id, interval_step, interval_start, interval_end, {_generate_value_query("data_table_with_presence_indicator", data_field_to_keep, AGGREGATION_STRATEGY_KEY[aggregation_strategy], datetime_col_start, unit_fields)} \
        FROM long_format_intervals as lfi \
        LEFT JOIN data_table_with_presence_indicator ON lfi.person_id = data_table_with_presence_indicator.person_id \
                AND lfi.data_table_concept_id = data_table_with_presence_indicator.{DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id \
                AND (data_table_with_presence_indicator.{datetime_col_start} >= lfi.interval_start AND data_table_with_presence_indicator.{datetime_col_start} < lfi.interval_end \
                    OR data_table_with_presence_indicator.{datetime_col_end} >= lfi.interval_start AND data_table_with_presence_indicator.{datetime_col_end} < lfi.interval_end \
                    OR (data_table_with_presence_indicator.{datetime_col_start} < lfi.interval_start AND data_table_with_presence_indicator.{datetime_col_end} >= lfi.interval_end)) \
        GROUP BY lfi.obs_id, lfi.person_id, lfi.data_table_concept_id, interval_step, interval_start, interval_end
        """

    query = create_long_table_query + prepare_alias_query + select_query

    backend_handle.execute(f"DROP TABLE IF EXISTS long_person_timestamp_feature_value_{data_table}")
    backend_handle.execute(query)

    add_obs_range_index_query = f"""
        ALTER TABLE long_person_timestamp_feature_value_{data_table}
        ADD COLUMN obs_index INTEGER;

        WITH RankedObs AS (
            SELECT obs_id,
                ROW_NUMBER() OVER (ORDER BY obs_id) - 1 AS idx
            FROM (SELECT DISTINCT obs_id FROM long_person_timestamp_feature_value_{data_table}) AS unique_obs
        )
        UPDATE long_person_timestamp_feature_value_{data_table}
        SET obs_index = RO.idx
        FROM RankedObs RO
        WHERE long_person_timestamp_feature_value_{data_table}.obs_id = RO.obs_id;
    """
    backend_handle.execute(add_obs_range_index_query)

    rename_table_query = f"ALTER TABLE long_person_timestamp_feature_value_{data_table} RENAME TO long_person_timestamp_feature_value_{data_table}"
    backend_handle.execute(rename_table_query)


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
