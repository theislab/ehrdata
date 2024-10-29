from collections.abc import Sequence

import duckdb
import pandas as pd

START_DATE_KEY = {
    "visit_occurrence": "visit_start_date",
    "observation_period": "observation_period_start_date",
    "cohort": "cohort_start_date",
}
END_DATE_KEY = {
    "visit_occurrence": "visit_end_date",
    "observation_period": "observation_period_end_date",
    "cohort": "cohort_end_date",
}
TIME_DEFINING_TABLE_SUBJECT_KEY = {
    "visit_occurrence": "person_id",
    "observation_period": "person_id",
    "cohort": "subject_id",
}

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
    "std": "STD",
}


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
    backend_handle: duckdb.duckdb.DuckDBPyConnection,
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


def _drop_timedeltas(backend_handle: duckdb.duckdb.DuckDBPyConnection):
    backend_handle.execute("DROP TABLE IF EXISTS timedeltas")


def _generate_value_query(data_table: str, data_field_to_keep: Sequence, aggregation_strategy: str) -> str:
    query = f"{', ' .join([f'CASE WHEN COUNT(*) = 0 THEN NULL ELSE {aggregation_strategy}({column}) END AS {column}' for column in data_field_to_keep])}"
    return query


def time_interval_table_query_long_format(
    backend_handle: duckdb.duckdb.DuckDBPyConnection,
    time_defining_table: str,
    data_table: str,
    interval_length_number: int,
    interval_length_unit: str,
    num_intervals: int,
    aggregation_strategy: str,
    data_field_to_keep: Sequence[str] | str,
) -> pd.DataFrame:
    """Returns a long format DataFrame from the data_table. The following columns should be considered the indices of this long format: person_id, data_table_concept_id, interval_step. The other columns, except for start_date and end_date, should be considered the values."""
    if isinstance(data_field_to_keep, str):
        data_field_to_keep = [data_field_to_keep]

    timedeltas_dataframe = _generate_timedeltas(interval_length_number, interval_length_unit, num_intervals)

    _write_timedeltas_to_db(
        backend_handle,
        timedeltas_dataframe,
    )

    # multi-step query
    # 1. Create person_time_defining_table, which matches the one created for obs. Needs to contain the person_id, and the start date in particular.
    # 2. Create person_data_table (data_table is typically measurement), which contains the cross product of person_id and the distinct concept_id s.
    # 3. Create long_format_backbone, which is the left join of person_time_defining_table and person_data_table.
    # 4. Create long_format_intervals, which is the cross product of long_format_backbone and timedeltas. This table contains most notably the person_id, the concept_id, the interval start and end dates.
    # 5. Create the final table, which is the join with the data_table (typically measurement); each measurement is assigned to its person_id, its concept_id, and the interval it fits into.
    df = backend_handle.execute(
        f"""
        WITH person_time_defining_table AS ( \
            SELECT person.person_id as person_id, {START_DATE_KEY[time_defining_table]} as start_date, {END_DATE_KEY[time_defining_table]} as end_date \
            FROM person \
            JOIN {time_defining_table} ON person.person_id = {time_defining_table}.{TIME_DEFINING_TABLE_SUBJECT_KEY[time_defining_table]} \
        ), \
        person_data_table AS( \
            WITH distinct_data_table_concept_ids AS ( \
                SELECT DISTINCT {data_table}_concept_id
                FROM {data_table} \
            )
            SELECT person.person_id, {data_table}_concept_id as data_table_concept_id \
            FROM person \
            CROSS JOIN distinct_data_table_concept_ids \
        ), \
        long_format_backbone as ( \
            SELECT person_time_defining_table.person_id, data_table_concept_id, start_date, end_date \
            FROM person_time_defining_table \
            LEFT JOIN person_data_table USING(person_id)\
        ), \
        long_format_intervals as ( \
            SELECT person_id, data_table_concept_id, interval_step, start_date, start_date + interval_start_offset as interval_start, start_date + interval_end_offset as interval_end \
            FROM long_format_backbone \
            CROSS JOIN timedeltas \
        ) \
        SELECT lfi.person_id, lfi.data_table_concept_id, interval_step, interval_start, interval_end, {_generate_value_query(data_table, data_field_to_keep, AGGREGATION_STRATEGY_KEY[aggregation_strategy])} \
        FROM long_format_intervals as lfi \
        LEFT JOIN {data_table} ON lfi.person_id = {data_table}.person_id AND lfi.data_table_concept_id = {data_table}.{data_table}_concept_id AND {data_table}.{data_table}_date BETWEEN lfi.interval_start AND lfi.interval_end \
        GROUP BY lfi.person_id, lfi.data_table_concept_id, interval_step, interval_start, interval_end
        """
    ).df()

    _drop_timedeltas(backend_handle)

    return df
