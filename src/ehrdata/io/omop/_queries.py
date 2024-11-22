from collections.abc import Sequence

import duckdb
import pandas as pd

# START_DATE_KEY = {
#     "visit_occurrence": "visit_start_date",
#     "observation_period": "observation_period_start_date",
#     "cohort": "cohort_start_date",
#     "drug_exposure": "drug_exposure_start_date",
#     "condition_occurrence": "condition_start_date",
#     "procedure_occurrence": "procedure_date",  # v5.4, as v5.3 had no end! TODO: allow v5.3 too
# }
# END_DATE_KEY = {
#     "visit_occurrence": "visit_end_date",
#     "observation_period": "observation_period_end_date",
#     "cohort": "cohort_end_date",
#     "drug_exposure": "drug_exposure_end_date",
#     "condition_occurrence": "condition_end_date",
#     "procedure_occurrence": "procedure_end_date",  # v5.4, as v5.3 had no end! TODO: allow v5.3 too
# }
TIME_DEFINING_TABLE_SUBJECT_KEY = {
    "visit_occurrence": "person_id",
    "observation_period": "person_id",
    "cohort": "subject_id",
}

DATA_TABLE_CONCEPT_ID_TRUNK = {
    "measurement": "measurement",
    "observation": "observation",
    "specimen": "specimen",
    "drug_exposure": "drug",
    "condition_occurrence": "condition",
    "procedure_occurrence": "procedure",
}
# DATA_TABLE_DATE_TRUNK = {
#     "measurement": "measurement",
#     "observation": "observation",
#     "specimen": "specimen",
#     "drug_exposure": "drug_exposure",
#     "condition_occurrence": "condition",
#     "procedure_occurrence": "procedure",
# }

DATA_TABLE_DATE_KEYS = {
    "timepoint": {
        "measurement": "measurement_date",
        "observation": "observation_date",
        "specimen": "specimen_date",
    },
    "start": {
        "visit_occurrence": "visit_start_date",
        "observation_period": "observation_period_start_date",
        "cohort": "cohort_start_date",
        "drug_exposure": "drug_exposure_start_date",
        "condition_occurrence": "condition_start_date",
        "procedure_occurrence": "procedure_date",  # in v5.3, procedure didnt have end date
    },
    "end": {
        "visit_occurrence": "visit_end_date",
        "observation_period": "observation_period_end_date",
        "cohort": "cohort_end_date",
        "drug_exposure": "drug_exposure_end_date",
        "condition_occurrence": "condition_end_date",
        "procedure_occurrence": "procedure_end_date",  # in v5.3, procedure didnt have end date TODO v5.3 support
    },
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
    # is_present is 1 in all rows of the data_table; but need an aggregation operation, so use LAST
    is_present_query = "LAST(is_present) as is_present, "
    value_query = f"{', ' .join([f'{aggregation_strategy}({column}) AS {column}' for column in data_field_to_keep])}"

    return is_present_query + value_query


def time_interval_table_query_long_format(
    backend_handle: duckdb.duckdb.DuckDBPyConnection,
    time_defining_table: str,
    data_table: str,
    interval_length_number: int,
    interval_length_unit: str,
    num_intervals: int,
    aggregation_strategy: str,
    data_field_to_keep: Sequence[str] | str,
    date_prefix: str = "",
) -> pd.DataFrame:
    """Returns a long format DataFrame from the data_table. The following columns should be considered the indices of this long format: person_id, data_table_concept_id, interval_step. The other columns, except for start_date and end_date, should be considered the values."""
    if isinstance(data_field_to_keep, str):
        data_field_to_keep = [data_field_to_keep]

    if date_prefix == "":
        date_prefix = "timepoint"

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
            SELECT person.person_id as person_id, {DATA_TABLE_DATE_KEYS["start"][time_defining_table]} as start_date, {DATA_TABLE_DATE_KEYS["end"][time_defining_table]} as end_date \
            FROM person \
            JOIN {time_defining_table} ON person.person_id = {time_defining_table}.{TIME_DEFINING_TABLE_SUBJECT_KEY[time_defining_table]} \
        ), \
        person_data_table AS( \
            WITH distinct_data_table_concept_ids AS ( \
                SELECT DISTINCT {DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id
                FROM {data_table} \
            )
            SELECT person.person_id, {DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id as data_table_concept_id \
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
        ), \
        data_table_with_presence_indicator as( \
            SELECT *, 1 as is_present \
            FROM {data_table} \
        ) \
        SELECT lfi.person_id, lfi.data_table_concept_id, interval_step, interval_start, interval_end, {_generate_value_query("data_table_with_presence_indicator", data_field_to_keep, AGGREGATION_STRATEGY_KEY[aggregation_strategy])} \
        FROM long_format_intervals as lfi \
        LEFT JOIN data_table_with_presence_indicator ON lfi.person_id = data_table_with_presence_indicator.person_id AND lfi.data_table_concept_id = data_table_with_presence_indicator.{DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id AND data_table_with_presence_indicator.{DATA_TABLE_DATE_KEYS[date_prefix][data_table]} BETWEEN lfi.interval_start AND lfi.interval_end \
        GROUP BY lfi.person_id, lfi.data_table_concept_id, interval_step, interval_start, interval_end
        """
    ).df()

    _drop_timedeltas(backend_handle)

    return df


def time_interval_table_for_interval_tables_query_long_format(
    backend_handle: duckdb.duckdb.DuckDBPyConnection,
    time_defining_table: str,
    data_table: str,
    interval_length_number: int,
    interval_length_unit: str,
    num_intervals: int,
    aggregation_strategy: str,
    data_field_to_keep: Sequence[str] | str,
    date_prefix: str = "",
) -> pd.DataFrame:
    """Returns a long format DataFrame from the data_table. The following columns should be considered the indices of this long format: person_id, data_table_concept_id, interval_step. The other columns, except for start_date and end_date, should be considered the values."""
    if isinstance(data_field_to_keep, str):
        data_field_to_keep = [data_field_to_keep]

    if date_prefix != "":
        date_prefix = date_prefix + "_"

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
            SELECT person.person_id as person_id, {DATA_TABLE_DATE_KEYS["start"][time_defining_table]} as start_date, {DATA_TABLE_DATE_KEYS["end"][time_defining_table]} as end_date \
            FROM person \
            JOIN {time_defining_table} ON person.person_id = {time_defining_table}.{TIME_DEFINING_TABLE_SUBJECT_KEY[time_defining_table]} \
        ), \
        person_data_table AS( \
            WITH distinct_data_table_concept_ids AS ( \
                SELECT DISTINCT {DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id
                FROM {data_table} \
            )
            SELECT person.person_id, {DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id as data_table_concept_id \
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
        ), \
        data_table_with_presence_indicator as( \
            SELECT *, 1 as is_present \
            FROM {data_table} \
        ) \
        SELECT lfi.person_id, lfi.data_table_concept_id, interval_step, interval_start, interval_end, {_generate_value_query("data_table_with_presence_indicator", data_field_to_keep, AGGREGATION_STRATEGY_KEY[aggregation_strategy])} \
        FROM long_format_intervals as lfi \
        LEFT JOIN data_table_with_presence_indicator ON lfi.person_id = data_table_with_presence_indicator.person_id \
                AND lfi.data_table_concept_id = data_table_with_presence_indicator.{DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id \
                AND (data_table_with_presence_indicator.{DATA_TABLE_DATE_KEYS["start"][data_table]} BETWEEN lfi.interval_start AND lfi.interval_end \
                    OR data_table_with_presence_indicator.{DATA_TABLE_DATE_KEYS["end"][data_table]} BETWEEN lfi.interval_start AND lfi.interval_end \
                    OR (data_table_with_presence_indicator.{DATA_TABLE_DATE_KEYS["start"][data_table]} < lfi.interval_start AND data_table_with_presence_indicator.{DATA_TABLE_DATE_KEYS["end"][data_table]} > lfi.interval_end)) \
        GROUP BY lfi.person_id, lfi.data_table_concept_id, interval_step, interval_start, interval_end
        """
    ).df()

    _drop_timedeltas(backend_handle)

    return df
