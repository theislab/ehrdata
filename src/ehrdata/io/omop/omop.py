from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import sparse as sp

from ehrdata.io._omop_utils import get_table_catalog_dict
from ehrdata.io.omop._check_arguments import (
    VALID_OBSERVATION_TABLES_JOIN,
    VALID_OBSERVATION_TABLES_SINGLE,
    _check_valid_aggregation_fill_values,
    _check_valid_aggregation_strategies,
    _check_valid_aggregation_strategy,
    _check_valid_aggregation_strategy_mapping,
    _check_valid_backend_handle,
    _check_valid_clip_to_end_date,
    _check_valid_clip_to_end_date_for_observation_table,
    _check_valid_concept_ids,
    _check_valid_concept_sets,
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
    _check_valid_sparse,
    _check_valid_time_precision,
    _check_valid_value_range,
    _check_valid_value_range_variables,
    _check_valid_variable_data_tables,
    _warn_time_precision_interval_mismatch,
)
from ehrdata.io.omop._queries import (
    DATA_TABLE_CONCEPT_ID_TRUNK,
    SINGLE_ROW_AGGREGATION_STRATEGIES,
    UNITLESS_AGGREGATION_STRATEGIES,
    VALUE_RANGE_TABLE,
    _concept_ids_condition,
    _get_ordered_table,
    _get_table_join,
    _get_unit_fields,
    _instantiate_long_observed_table,
    _links_rows_to_visits,
    _registered,
    _value_range_filter,
    _write_long_observed_table,
    _write_long_time_interval_table,
    _write_obs_names,
    _write_obs_windows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import duckdb
    from duckdb import DuckDBPyConnection

DOWNLOAD_VERIFICATION_TAG = "download_verification_tag"


def _get_table_list() -> list:
    flat_table_list = [value for value_list in get_table_catalog_dict().values() for value in value_list]
    return flat_table_list


def _set_up_duckdb(path: Path, backend_handle: DuckDBPyConnection, prefix: str = "") -> None:
    """Create tables in the backend from CSV or Parquet files in the path from datasets in the OMOP Common Data model."""
    tables = _get_table_list()

    used_tables = []
    missing_tables = []
    unused_files = []
    for filename in os.listdir(path):  # noqa: PTH208
        filename_trunk = filename.split(".")[0].lower()
        regular_omop_table_name = filename_trunk.replace(prefix, "")

        if regular_omop_table_name in tables:
            used_tables.append(regular_omop_table_name)

            dtype = {"measurement_source_value": str} if regular_omop_table_name == "measurement" else None

            # Detect file format and read accordingly
            file_extension = filename.split(".")[-1].lower()
            if file_extension == "parquet":
                # Read parquet file
                temp_relation = backend_handle.read_parquet(str(path / filename))
            elif file_extension == "csv":
                # Read CSV file
                temp_relation = backend_handle.read_csv(path / filename, dtype=dtype)  # noqa: F841
            else:
                logging.warning(f"Unsupported file format for {filename}. Skipping.")
                continue

            # reading from csv/parquet bears risk of not getting correct schema:
            # enforce the critical date and datetime columns, prone to be read as VARCHAR
            schema_df = backend_handle.execute("DESCRIBE temp_relation").df()
            select_parts = []
            for _, row in schema_df.iterrows():
                col = row["column_name"]
                col_lower = col.lower()
                if col_lower.endswith("_datetime"):
                    # Cast to TIMESTAMP (handles YYYY-MM-DD HH:MM:SS)
                    select_parts.append(f'TRY_CAST("{col}" AS TIMESTAMP) AS "{col}"')
                elif col_lower.endswith("_date"):
                    # Cast to DATE (handles YYYY-MM-DD)
                    select_parts.append(f'TRY_CAST("{col}" AS DATE) AS "{col}"')
                else:
                    select_parts.append(f'"{col}" AS "{col}"')
            select_columns = ", ".join(select_parts)
            create_table_query = f"CREATE OR REPLACE VIEW temp_table AS SELECT {select_columns} FROM temp_relation"
            backend_handle.execute(create_table_query)

            # make query to create table with lowercase column names
            column_names = backend_handle.execute("DESCRIBE temp_table").df()["column_name"].values
            select_columns = ", ".join([f'"{col}" AS "{col.lower()}"' for col in column_names])
            create_table_with_lowercase_columns_query = (
                f"CREATE OR REPLACE TABLE {regular_omop_table_name} AS SELECT {select_columns} FROM temp_table"
            )

            backend_handle.execute(create_table_with_lowercase_columns_query)

            backend_handle.execute("DROP VIEW temp_table")

        elif filename_trunk != DOWNLOAD_VERIFICATION_TAG:
            unused_files.append(filename)

    missing_tables = [table for table in tables if table not in used_tables]

    logging.info(f"missing tables: {missing_tables}")
    logging.info(f"unused files: {unused_files}")


def _collect_units_per_feature(
    backend_handle, data_table, unit_key="unit_concept_id", key_column="data_table_concept_id"
) -> dict:
    query = f"""
    SELECT DISTINCT {key_column}, {unit_key} FROM long_person_timestamp_feature_value_{data_table}
    WHERE is_present = 1
    """
    result = backend_handle.execute(query).fetchall()

    feature_units: dict = {}
    for feature, unit in result:
        if feature in feature_units:
            feature_units[feature].append(unit)
        else:
            feature_units[feature] = [unit]
    return feature_units


def _check_one_unit_per_feature(
    backend_handle, data_table, unit_key="unit_concept_id", key_column="data_table_concept_id"
) -> None:
    feature_units = _collect_units_per_feature(backend_handle, data_table, unit_key=unit_key, key_column=key_column)
    num_units = np.array([len(units) for _, units in feature_units.items()])

    # print(f"no units for features: {np.argwhere(num_units == 0)}")
    logging.warning(f"multiple units for features: {np.argwhere(num_units > 1)}")


def _check_one_unit_per_feature_for_aggregation(
    backend_handle,
    data_table,
    value_field,
    aggregation_strategy,
    unit_fields,
    *,
    concept_ids,
    concept_sets=None,
    value_ranges=None,
) -> None:
    """Verify that the data table allows the chosen aggregation strategy to combine values.

    With concept_sets, the features are the concept sets, whose rows are combined across their concepts.

    A strategy that is not one of SINGLE_ROW_AGGREGATION_STRATEGIES combines the values of several rows into one, which is only meaningful if these values share a unit.
    Only the rows of the concept_ids carrying a value are considered, as only those enter the aggregation; among these, a missing unit (NULL) is a unit of its own, as in the unit report.
    Unitless aggregations are exempt: 'is_present' has no unit to begin with, and UNITLESS_AGGREGATION_STRATEGIES yield a number of data points.
    Data tables that do not have a 'unit_concept_id' are exempt, too: they tell no unit that could disagree.
    """
    if (
        aggregation_strategy in SINGLE_ROW_AGGREGATION_STRATEGIES
        or aggregation_strategy in UNITLESS_AGGREGATION_STRATEGIES
        or value_field == "is_present"
        or "unit_concept_id" not in unit_fields
    ):
        return

    concept_id_key = f"{DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id"
    # the rows outside the value range of their concept are left out of the aggregation, and so of the check
    range_join, range_condition = _value_range_filter("d", concept_id_key, value_field, value_ranges=value_ranges)
    with _registered(backend_handle, VALUE_RANGE_TABLE, value_ranges):
        feature_unit_pairs = backend_handle.execute(
            f"""
            SELECT DISTINCT d.{concept_id_key}, d.unit_concept_id FROM {data_table} AS d
            {range_join}
            WHERE d.{value_field} IS NOT NULL AND {_concept_ids_condition(data_table, concept_ids)} AND {range_condition}
            """
        ).fetchall()

    set_of_concept = (
        {concept_id: name for name, ids in concept_sets.items() for concept_id in ids} if concept_sets else None
    )
    feature_units: dict = {}
    for concept_id, unit in feature_unit_pairs:
        feature = set_of_concept[int(concept_id)] if set_of_concept else concept_id
        if unit not in feature_units.setdefault(feature, []):
            feature_units[feature].append(unit)
    multiple_units = {feature: sorted(units, key=str) for feature, units in feature_units.items() if len(units) > 1}

    if multiple_units:
        shown = list(multiple_units.items())[:5]
        listing = ", ".join(f"{feature}: {units}" for feature, units in shown)
        if len(multiple_units) > len(shown):
            listing += f", ... ({len(multiple_units) - len(shown)} more)"
        msg = (
            f"aggregation_strategy='{aggregation_strategy}' combines the values of several rows into one "
            f"value, which requires a single unit per feature. In '{data_table}', {len(multiple_units)} "
            f"feature(s) have more than one unit_concept_id among the rows carrying a '{value_field}' "
            f"({listing}). Use aggregation_strategy='last' or 'first', which keep the value of a single "
            f"row, or harmonize the units of these features in the database."
        )
        raise NotImplementedError(msg)


def _create_feature_unit_concept_id_report(
    backend_handle, data_table, key_column="data_table_concept_id", report_key="concept_id"
) -> pd.DataFrame:
    """The units of each feature among its observed cells, one row per (feature, unit) in the column report_key."""
    feature_units_concept = _collect_units_per_feature(
        backend_handle, data_table, unit_key="unit_concept_id", key_column=key_column
    )
    feature_units_long_format = []
    for feature, units in feature_units_concept.items():
        if len(units) == 0:
            feature_units_long_format.append({report_key: feature, "no_units": True, "multiple_units": False})
        elif len(units) > 1:
            feature_units_long_format.extend(
                [
                    {
                        report_key: feature,
                        "unit_concept_id": unit,
                        "no_units": False,
                        "multiple_units": True,
                    }
                    for unit in units
                ]
            )
        else:
            feature_units_long_format.append(
                {
                    report_key: feature,
                    "unit_concept_id": units[0],
                    "no_units": False,
                    "multiple_units": False,
                }
            )

    df = pd.DataFrame(feature_units_long_format, columns=[report_key, "unit_concept_id", "no_units", "multiple_units"])
    df["unit_concept_id"] = df["unit_concept_id"].astype("Int64")

    return df


def _concept_set_feature_info(backend_handle, concept_sets, concepts) -> pd.DataFrame:
    """One row per concept set: each field of the concept table for the concepts of the set, joined by '; ' in the order of the set.

    As for single concepts, a concept id that the concept table lacks is looked up by its 'Mapped from' relationship; a concept found in neither gives an empty value.
    """
    by_id = concepts.drop_duplicates("concept_id").set_index("concept_id")
    ids = [concept_id for set_ids in concept_sets.values() for concept_id in set_ids]
    mapped: dict = {}
    if any(concept_id not in by_id.index for concept_id in ids):
        logging.warning("Concept_ids are only partially matching. Mapping concept_ids where applicable.")
        relationships = backend_handle.sql(
            "SELECT concept_id_1, concept_id_2 FROM concept_relationship WHERE (relationship_id = 'Mapped from') AND (concept_id_1 <> concept_id_2)"
        ).df()
        mapped = relationships.drop_duplicates("concept_id_1").set_index("concept_id_1")["concept_id_2"].to_dict()

    def record(concept_id):
        concept_id = concept_id if concept_id in by_id.index else mapped.get(concept_id)
        return by_id.loc[concept_id] if concept_id in by_id.index else None

    rows = []
    for set_ids in concept_sets.values():
        records = [record(concept_id) for concept_id in set_ids]
        rows.append(
            {
                field: "; ".join("" if r is None or pd.isna(r[field]) else str(r[field]) for r in records)
                for field in by_id.columns
            }
        )
    return pd.DataFrame(rows, columns=by_id.columns)


def _create_enriched_var_with_unit_info(backend_handle, ds, var, unit_report) -> pd.DataFrame:
    feature_concept_id_table = var  # ds["data_table_concept_id"].to_pandas()

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


def _is_numeric_field(backend_handle, data_table: str, field: str) -> bool:
    """Whether field of data_table is of a numeric type; 'is_present' is."""
    if field == "is_present":
        return True
    column_types = {row[0].lower(): row[1] for row in backend_handle.execute(f"DESCRIBE {data_table}").fetchall()}
    return column_types.get(field.lower(), "").startswith(
        (
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
            "UHUGEINT",
            "FLOAT",
            "DOUBLE",
            "DECIMAL",
        )
    )


def setup_connection(path: Path | str, backend_handle: DuckDBPyConnection, prefix: str = "") -> None:
    """Setup a connection to the OMOP CDM database.

    This function sets up a connection to the OMOP CDM database.
    It checks the capitalization of the 'person' table, and assumes the same capitalization style is used for all other tables.


    Args:
        path: The path to the folder containing the CSV or Parquet files.
        backend_handle: The backend handle to the database.
        prefix: The prefix to be removed from the CSV/Parquet filenames.

    Returns:
        An EHRData object with populated .uns["omop_table_capitalization"] field.

    """
    _set_up_duckdb(Path(path), backend_handle, prefix)


def setup_obs(
    backend_handle: duckdb.DuckDBPyConnection,
    observation_table: Literal["person", "person_cohort", "person_observation_period", "person_visit_occurrence"],
    *,
    death_table: bool = False,
):
    """Setup the observation table for :class:`~ehrdata.EHRData` object.

    For this, a table from the OMOP CDM which represents what a "row" in the `EHRData` object should represent.

    This will be used to set that start timepoint for the time series data in the `EHRData` object.

    Possible choices for the `observation_table` parameter are:

    - `"person"`: Create one row per person `person_id` in the `person` table
    - `"person_cohort"`: Create one row per person `subject_id` in a cohort in the `cohort` table
    - `"person_observation_period"`: Create one row per observation_period_id in the `observation_period` table
    - `"person_visit_occurrence"`: Create one row per visit_occurrence_id in the `visit_occurrence` table

    Data characterizing the rows such as person demographics, and e.g. visit start and end dates are stored in the `.obs` field of the created `EHRData` object.
    Notice a single `person_id` can have multiple rows for e.g. `"person_visit_occurrence"` if a person has multiple visit occurrences.

    Example: 1000 person_id in the `person_table`, with 100 people having 1 visit occurrence, 100 people having 2 visit occurrences, and 800 people having no visit occurrences.
    If `observation_table = "person"`, the created `EHRData` object will have 1000 rows.
    If `observation_table = "person_visit_occurrence"`, the created `EHRData` object will have 300 rows.

    The possible choices affect what is taken as the "time 0" in the :func:`~ehrdata.io.omop.setup_variables` and :func:`~ehrdata.io.omop.setup_interval_variables` functions:

    - If `"person"`, the `birth_datetime` will be used as "time 0". **Note:** All persons must have valid (non-NULL) `birth_datetime` values when using this option with :func:`~ehrdata.io.omop.setup_variables` and :func:`~ehrdata.io.omop.setup_interval_variables`.
    - If `"person_cohort"`, the `cohort_start_date(time)` will be used as "time 0" in the :func:`~ehrdata.io.omop.setup_variables` and :func:`~ehrdata.io.omop.setup_interval_variables` functions.
    - If `"person_observation_period"`, the `observation_period_start_date(time)` will be used as "time 0" in the :func:`~ehrdata.io.omop.setup_variables` and :func:`~ehrdata.io.omop.setup_interval_variables` functions.
    - If `"person_visit_occurrence"`, the `visit_start_date(time)` will be used as "time 0" in the :func:`~ehrdata.io.omop.setup_variables` and :func:`~ehrdata.io.omop.setup_interval_variables` functions.

    For the tables other than `"person"`, the respective end date (e.g. `observation_period_end_date`) ends the time series: data points after this day are not considered, unless `clip_to_end_date=False`.

    For `"person_cohort"`, the `subject_id` of the cohort is considered to be the `person_id` for a join.

    Args:
        backend_handle: The backend handle to the database.
        observation_table: The observation table to be used.
        death_table: Whether to include the `death` table.
            The `observation_table` created will be left joined with the `death` table as the right table.

    Returns:
            An :class:`~ehrdata.EHRData` object with populated `.obs` field.

    Examples:
        >>> import ehrdata as ed
        >>> import duckdb
        >>> con_gi = duckdb.connect(database=":memory:", read_only=False)
        >>> ed.dt.gibleed_omop(
        ...     con_gi,
        ... )
        >>> edata_gi = ed.io.omop.setup_obs(
        >>>     con_gi,
        >>>     observation_table="person_observation_period",
        >>> )
        >>> edata_gi
    """
    _check_valid_backend_handle(backend_handle)
    _check_valid_observation_table(observation_table)
    _check_valid_death_table(death_table)

    from ehrdata import EHRData

    if observation_table in VALID_OBSERVATION_TABLES_SINGLE:
        obs = _get_ordered_table(backend_handle, observation_table)

    elif observation_table in VALID_OBSERVATION_TABLES_JOIN:
        if observation_table == "person_cohort":
            obs = _get_table_join(backend_handle, "person", "cohort", right_key="subject_id")
        elif observation_table == "person_observation_period":
            obs = _get_table_join(backend_handle, "person", "observation_period")
        elif observation_table == "person_visit_occurrence":
            obs = _get_table_join(backend_handle, "person", "visit_occurrence")

    if death_table:
        death = backend_handle.sql("SELECT * FROM death").df()
        obs = obs.merge(death, how="left", on="person_id")

    # AnnData will make this conversion and raise a Warning.
    # Since we have no reason for the RangeIndex, beyond it being introduced by the to_df function of the RDBMS, we can do it immediately and omit a confusing warning
    obs.index = obs.index.astype(str)

    return EHRData(obs=obs, uns={"omop_io_observation_table": observation_table.split("person_")[-1]})


def setup_variables(
    edata,
    *,
    backend_handle: duckdb.DuckDBPyConnection,
    layer: str | None = None,
    data_tables: Sequence[Literal["measurement", "observation", "specimen"]]
    | Literal["measurement", "observation", "specimen"],
    data_field_to_keep: str | Sequence[str] | dict[str, str | Sequence[str]],
    interval_length_number: int,
    interval_length_unit: str,
    time_precision: Literal["date", "datetime"] = "date",
    clip_to_end_date: bool = True,
    value_range: Mapping[int | str, tuple[float | None, float | None]] | None = None,
    num_intervals: int,
    concept_ids: Literal["all"] | Sequence[int] | Mapping[str, Sequence[int]] = "all",
    aggregation_strategy: Literal["last", "first", "mean", "median", "mode", "sum", "count", "min", "max", "std"]
    | Mapping[
        int | str, Literal["last", "first", "mean", "median", "mode", "sum", "count", "min", "max", "std"]
    ] = "last",
    enrich_var_with_feature_info: bool = False,
    enrich_var_with_unit_info: bool = False,
    instantiate_tensor: bool = True,
    sparse: bool = False,
):
    """Extracts selected tables of a data-point character from the OMOP CDM.

    The distinct `concept_id`s encountered in the selected tables form the variables in the EHRData object.
    The variables are sorted by the `concept_id` for each `data_table` in ascending order, and stacked together in the order that the `data_tables` are specified.
    With concept sets (see `concept_ids`), the variables are the sets instead.

    The `data_field_to_keep` parameter specifies which Field in the selected table is to be used for the read-out of the value of a variable.

    The variables are set up for the rows of `edata.obs`, in their order: the time window of each row is read from the columns that :func:`~ehrdata.io.omop.setup_obs` writes to `.obs`.
    Hence `edata` can be filtered or reordered between :func:`~ehrdata.io.omop.setup_obs` and this function.
    With `observation_table="person_visit_occurrence"`, a data point whose `visit_occurrence_id` names a visit of its person in the `visit_occurrence` table belongs to that visit only, and to no visit if it lies outside that visit's window.
    Any other data point belongs to every visit of its person whose window holds it; the data tables without a `visit_occurrence_id`, such as `specimen`, have only such data points.
    Visit ids are compared as integers, as OMOP defines them.
    A row without a start date in `.obs`, or without an end date unless `clip_to_end_date=False`, gets no data; with `observation_table="person"`, every row needs a `birth_datetime`.

    Writes a unit report of the features to `edata.uns['unit_report_<data_tables>']`.
    Writes the setup arguments into `edata.uns['omop_io_variable_setup']`.

    Stores a table(s) named `long_person_timestamp_feature_value_<data_table>` in long format in the RDBMS.
    Its `obs_index` is the position of the row in `edata.obs`; the table `long_person_timestamp_feature_value_<data_table>_obs` holds the `obs_names` of these rows.
    It holds one row per observation, variable and interval that has data, so its size follows the number of data points.
    This table is instantiated into `.X`/`.layers` if `instantiate_tensor` is set to `True`;
    otherwise, the table is only stored in the RDBMS for later use.

    Args:
        edata: Data object to which the variables should be added.
        backend_handle: The backend handle to the database.
        layer: The layer to store the data in. If not specified, uses `X`.
        data_tables: The tables to be used.
        data_field_to_keep: The CDM Field in the data tables to be kept. Can be e.g.
            'value_as_number' or 'value_as_concept_id'. Importantly, can be 'is_present'
            to have a one-hot encoding of the presence of the feature in a patient in an
            interval. Should be a dictionary to specify the data fields to keep per table
            if multiple data tables are used. For example, if data_tables=['measurement',
            'observation'], data_field_to_keep={'measurement': 'value_as_number',
            'observation': 'value_as_number'}.
        time_precision: The precision of the timestamp used in the table indicated in :func:`~ehrdata.io.omop.setup_obs`. If `"date"`, uses the `date` field (e.g. `visit_start_date` for `"person_visit_occurrence"`). If `"datetime"`, uses the `datetime` field (e.g. `visit_start_datetime` for `"person_visit_occurrence"`).
        clip_to_end_date: Whether the data points after the end date of an observation, such as the `visit_end_date`, are left out.
            If `False`, the data points are kept in all `num_intervals` intervals, also after the end date.
            Not available for `observation_table="person"`, whose observations have no end date.
        value_range: The range of plausible values of each variable, as `{variable: (lower, upper)}`, keyed like a mapping of `aggregation_strategy`.
            The data points outside the range, bounds included, are left out before the aggregation, so that they neither count nor tell a unit.
            A bound of `None` leaves that side open, and the variables without a range are not filtered.
            The values are those of the first field of `data_field_to_keep`, which must be numeric.
        interval_length_number: Numeric value of the length of one interval.
        interval_length_unit: Unit of the interval length, needs to be a unit of :class:`pandas.Timedelta`.
        num_intervals: Number of intervals.
        concept_ids: Concept IDs to use from the data tables. If not specified, 'all' are used.
            A mapping of names to concept IDs, such as `{"name": [concept_id_1, concept_id_2]}`, defines concept sets:
            each set is one variable, named by its key and in the order of the mapping, that combines the data points of its concepts.
            The variables are the sets, also those without data points, and `.var["concept_ids"]` lists the concept IDs of each.
            Concept sets take one data table, and a concept may belong to one set only.
            With `enrich_var_with_feature_info`, each field of the `concept` table, such as `concept_name`, lists the values of the concepts of a set, joined by `"; "` in the order of the set.
            With `enrich_var_with_unit_info`, the columns of the unit's concept are prefixed with `unit_`.
            The long format table then holds the column `variable` in place of `data_table_concept_id`, and the unit report is keyed by `variable` in place of `concept_id`.
        aggregation_strategy: Strategy to use when aggregating multiple data points within one interval.
            For `'last'` and `'first'`, only data points that have a value in the data field of interest (the first field of `data_field_to_keep`) are considered.
            Data points at the same time, for instance of two concepts of a concept set, are ordered by their value.
            All kept fields are read from that very data point, so that e.g. the units describe the value that is kept.
            Every other strategy combines the values of several data points into one, which requires a single unit per feature.
            A `NotImplementedError` is raised for features with more than one `unit_concept_id` among their data points carrying a value (a missing unit is a unit of its own here).
            Exempt are `'count'` and `data_field_to_keep='is_present'`, whose values are numbers of data points and hence carry no unit.
            For `'mode'`, of several equally frequent values the smallest is kept.
            Intervals without data points hold 0 for `'count'`, and a missing value for every other strategy and for `data_field_to_keep='is_present'`.
            A mapping sets a strategy per variable, keyed by its integer concept ID, or by its name for concept sets, such as `{"heart_rate": "median", "urine_output": "sum"}`.
            Every variable needs a strategy, and a key that is no variable raises, unless it is one of `concept_ids` without data points.
            Rows without a concept ID then belong to no variable, and a concept in several data tables has one strategy in all of them.
            A data table whose variables mix `'count'` with other strategies is a float tensor with 0 in the empty intervals of the counted variables, and needs a numeric data field;
            `sparse=True` raises for it, as a sparse array has one fill value.
        enrich_var_with_feature_info: Whether to enrich the var table with feature
           information. If a concept_id is not found in the concept table, their respective alternate `concept_id` included in the concept_relationship table is retrieved to add the available feature information.
           Otherwise the feature information will be NaN.
        enrich_var_with_unit_info: Whether to enrich the var table with unit information.
            Raises an Error if multiple units per feature are found for at least one
            feature. For entire missing data points, and for data points that do not
            contribute a value (see `aggregation_strategy`), the units are ignored. For
            observed data points with missing unit information (NULL in either
            'unit_concept_id' or 'unit_source_value'), the value NULL/NaN is considered
            a single unit.
        instantiate_tensor: Whether to instantiate the tensor within the `EHRData` object.
        sparse: Whether to store the tensor as a `sparse.COO` array instead of a :class:`numpy.ndarray`.
            Its fill value is that of the dense tensor: `NaN`, or 0 for `aggregation_strategy="count"` unless `data_field_to_keep='is_present'`.
            Requires numeric fields in `data_field_to_keep`, and `instantiate_tensor=True`.

    Returns:
        An :class:`~ehrdata.EHRData` object with populated `.X`/`.layers` and `.var` field.

    Examples:
        >>> import ehrdata as ed
        >>> import duckdb
        >>> con_gi = duckdb.connect(database=":memory:", read_only=False)
        >>> ed.dt.gibleed_omop(
        ...     con_gi,
        ... )
        >>> edata_gi = ed.io.omop.setup_obs(
        >>>     con_gi,
        >>>     observation_table="person_observation_period",
        >>> )
        >>> edata_gi = ed.io.omop.setup_variables(
        >>>     edata=edata_gi,
        >>>     backend_handle=con_gi,
        >>>     data_tables=["observation", "measurement"],
        >>>     data_field_to_keep={"observation": "observation_source_value", "measurement": "is_present"},
        >>>     interval_length_number=20,
        >>>     interval_length_unit="day",
        >>>     num_intervals=20,
        >>>     concept_ids="all",
        >>>     aggregation_strategy="last",
        >>>     enrich_var_with_feature_info=True,
        >>>     enrich_var_with_unit_info=True,
        >>> )
        >>> edata_gi
    """
    from ehrdata import EHRData

    _check_valid_edata(edata)
    _check_valid_backend_handle(backend_handle)
    data_tables = _check_valid_variable_data_tables(data_tables)
    data_field_to_keep = _check_valid_data_field_to_keep(data_field_to_keep, data_tables)
    _check_valid_interval_length_number(interval_length_number)
    _check_valid_interval_length_unit(interval_length_unit)
    _check_valid_time_precision(time_precision)
    _warn_time_precision_interval_mismatch(interval_length_unit, time_precision)
    _check_valid_num_intervals(num_intervals)
    concept_sets = None
    if isinstance(concept_ids, Mapping):
        concept_sets = _check_valid_concept_sets(concept_ids, data_tables)
        concept_ids = [concept_id for ids in concept_sets.values() for concept_id in ids]
    else:
        _check_valid_concept_ids(concept_ids)
    key_column = "variable" if concept_sets else "data_table_concept_id"
    value_range = _check_valid_value_range(value_range, concept_sets=concept_sets is not None)
    strategy_map = None
    if isinstance(aggregation_strategy, Mapping):
        strategy_map = _check_valid_aggregation_strategy_mapping(
            aggregation_strategy, concept_sets=concept_sets is not None
        )
    else:
        _check_valid_aggregation_strategy(aggregation_strategy)
    _check_valid_enrich_var_with_feature_info(enrich_var_with_feature_info)
    _check_valid_enrich_var_with_unit_info(enrich_var_with_unit_info)
    _check_valid_sparse(sparse, instantiate_tensor)
    _check_valid_clip_to_end_date(clip_to_end_date)
    if sparse and aggregation_strategy == "count":
        is_present_fields = {fields[0] == "is_present" for fields in data_field_to_keep.values()}
        if len(is_present_fields) > 1:
            msg = (
                "sparse=True needs one fill value for all data tables, but with aggregation_strategy='count' "
                "the empty cells are 0, and NaN for data_field_to_keep='is_present'. "
                "Set up these data tables in separate calls."
            )
            raise ValueError(msg)

    time_defining_table = edata.uns.get("omop_io_observation_table", None)
    if time_defining_table is None:
        msg = "The observation table must be set up first, use the `setup_obs` function."
        raise ValueError(msg)

    _check_valid_clip_to_end_date_for_observation_table(clip_to_end_date, time_defining_table)
    _write_obs_windows(
        backend_handle, edata.obs, time_defining_table, time_precision, clip_to_end_date=clip_to_end_date
    )
    n_obs = edata.n_obs

    table_strategies = variables_of_table = value_ranges = None
    if strategy_map is not None or value_range is not None:
        # the variables are the concepts with a concept id: with strategies or ranges per variable, rows without one belong to no variable
        variables_of_table = (
            {data_tables[0]: list(concept_sets)}
            if concept_sets
            else {
                data_table: sorted(
                    int(row[0])
                    for row in backend_handle.execute(
                        f"SELECT DISTINCT {DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id FROM {data_table} "
                        f"WHERE {_concept_ids_condition(data_table, concept_ids)} AND {DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id IS NOT NULL"
                    ).fetchall()
                )
                for data_table in data_tables
            }
        )
        requested = [] if concept_sets or concept_ids == "all" else [int(concept_id) for concept_id in concept_ids]
        numeric_fields = {
            data_table: _is_numeric_field(backend_handle, data_table, data_field_to_keep[data_table][0])
            for data_table in data_tables
        }
        if strategy_map is not None:
            table_strategies = _check_valid_aggregation_strategies(strategy_map, variables_of_table, requested)
            _check_valid_aggregation_fill_values(table_strategies, data_field_to_keep, numeric_fields, sparse=sparse)
        if value_range is not None:
            _check_valid_value_range_variables(
                value_range, variables_of_table, requested, data_field_to_keep, numeric_fields
            )
            # the ranges of the variables, resolved to their concepts
            ranges_of_concepts = [
                (concept_id, *bounds)
                for variable, bounds in value_range.items()
                for concept_id in (concept_sets[variable] if concept_sets else [variable])
            ]
            value_ranges = pd.DataFrame(
                {
                    "concept_id": pd.array([row[0] for row in ranges_of_concepts], dtype="Int64"),
                    "lower": pd.array([row[1] for row in ranges_of_concepts], dtype="Float64"),
                    "upper": pd.array([row[2] for row in ranges_of_concepts], dtype="Float64"),
                }
            )

    var_collector = {}
    r_collector = {}
    unit_report_collector = {}
    empty_table_counter = 0
    concept_mapping = False
    for data_table in data_tables:
        # dbms complains about our queries, which sometimes need a column to be of type e.g. datetime, when it can't infer types from data
        count = (
            backend_handle.execute(
                f"SELECT COUNT(*) as count FROM {data_table} WHERE {_concept_ids_condition(data_table, concept_ids)}"
            )
            .df()["count"]
            .item()
        )
        # concept sets are variables also without data points
        if count == 0 and not concept_sets:
            logging.warning(f"No data found in {data_table}. Returning edata without data of {data_table}.")
            empty_table_counter += 1
            continue

        unit_fields = _get_unit_fields(backend_handle, data_table)
        strategies = table_strategies[data_table] if table_strategies is not None else None
        variables_of_strategy: dict = {}
        for variable, strategy in (strategies or {}).items():
            variables_of_strategy.setdefault(strategy, []).append(variable)
        for strategy, variables in (variables_of_strategy or {aggregation_strategy: None}).items():
            # each variable is checked with its own strategy
            strategy_sets = (
                {name: concept_sets[name] for name in variables}
                if concept_sets and variables is not None
                else concept_sets
            )
            if variables is None:
                strategy_concept_ids = concept_ids
            elif concept_sets:
                strategy_concept_ids = [concept_id for name in variables for concept_id in concept_sets[name]]
            else:
                strategy_concept_ids = variables
            _check_one_unit_per_feature_for_aggregation(
                backend_handle,
                data_table,
                data_field_to_keep[data_table][0],
                strategy,
                unit_fields,
                concept_ids=strategy_concept_ids,
                concept_sets=strategy_sets,
                value_ranges=value_ranges,
            )

        _write_long_observed_table(
            backend_handle=backend_handle,
            data_table=data_table,
            time_precision=time_precision,
            data_field_to_keep=data_field_to_keep[data_table],
            unit_fields=unit_fields,
            interval_length_number=interval_length_number,
            interval_length_unit=interval_length_unit,
            num_intervals=num_intervals,
            aggregation_strategy="last" if strategies is not None else aggregation_strategy,
            concept_ids=concept_ids,
            link_visits=_links_rows_to_visits(backend_handle, time_defining_table, data_table),
            concept_sets=concept_sets,
            strategies=strategies,
            value_ranges=value_ranges,
        )
        _write_obs_names(backend_handle, data_table)

        _check_one_unit_per_feature(backend_handle, data_table, key_column=key_column)
        unit_report = _create_feature_unit_concept_id_report(
            backend_handle,
            data_table=data_table,
            key_column=key_column,
            report_key="variable" if concept_sets else "concept_id",
        )

        if concept_sets:
            var = pd.DataFrame(
                {
                    "variable": list(concept_sets),
                    "concept_ids": [",".join(str(concept_id) for concept_id in ids) for ids in concept_sets.values()],
                }
            )
        elif variables_of_table is not None:
            var = pd.DataFrame({"data_table_concept_id": variables_of_table[data_table]})
        else:
            var = backend_handle.execute(
                f"SELECT DISTINCT {DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id AS data_table_concept_id FROM {data_table} WHERE {_concept_ids_condition(data_table, concept_ids)}"
            ).df()
            var = var.sort_values("data_table_concept_id").reset_index(drop=True)

        if enrich_var_with_feature_info or enrich_var_with_unit_info:
            concepts = backend_handle.sql("SELECT * FROM concept").df()
            concepts.columns = concepts.columns.str.lower()

            if (
                enrich_var_with_feature_info
                and not concept_sets
                and any(elem not in set(concepts["concept_id"]) for elem in var["data_table_concept_id"])
            ):
                concept_mapping = True
                logging.warning("Concept_ids are only partially matching. Mapping concept_ids where applicable.")
                concepts_relationship = backend_handle.sql(
                    "SELECT concept_id_1, concept_id_2 FROM concept_relationship WHERE (relationship_id = 'Mapped from') AND (concept_id_1 <> concept_id_2)"
                ).df()
                concepts_dict = concepts_relationship.set_index("concept_id_1")["concept_id_2"].to_dict()
                concepts_mapped_idx = [x not in set(concepts["concept_id"]) for x in var["data_table_concept_id"]]
                var.loc[concepts_mapped_idx, "data_table_concept_id_mapped"] = var.loc[
                    concepts_mapped_idx, "data_table_concept_id"
                ].map(concepts_dict)
                var["data_table_concept_id_mapped"] = var["data_table_concept_id_mapped"].fillna(
                    var["data_table_concept_id"]
                )

        if enrich_var_with_feature_info and concept_sets:
            var = pd.concat([var, _concept_set_feature_info(backend_handle, concept_sets, concepts)], axis=1)
        elif enrich_var_with_feature_info:
            if concept_mapping:
                var = pd.merge(var, concepts, how="left", left_on="data_table_concept_id_mapped", right_on="concept_id")
            else:
                var = pd.merge(var, concepts, how="left", left_on="data_table_concept_id", right_on="concept_id")

        if enrich_var_with_unit_info:
            if unit_report["multiple_units"].sum() > 0:
                msg = "Multiple units per feature found. Enrichment with feature information not possible."
                raise ValueError(msg)
            else:
                var = pd.merge(
                    var,
                    unit_report,
                    how="left",
                    left_on=key_column,
                    right_on="variable" if concept_sets else "concept_id",
                    suffixes=("", "_unit"),
                )
                if concept_sets:
                    # the columns of the unit's concept, prefixed, so that they do not read as the concept of the variable
                    var = pd.merge(var, concepts.add_prefix("unit_"), how="left", on="unit_concept_id")
                else:
                    var = pd.merge(
                        var,
                        concepts,
                        how="left",
                        left_on="unit_concept_id",
                        right_on="concept_id",
                        suffixes=("", "_unit_concept"),
                    )

        if instantiate_tensor:
            r_collector[data_table] = _instantiate_long_observed_table(
                backend_handle,
                data_table,
                data_field_to_keep[data_table][0],
                variables=var[key_column].to_numpy(),
                n_obs=n_obs,
                num_intervals=num_intervals,
                aggregation_strategy=[strategies[variable] for variable in var[key_column]]
                if strategies is not None
                else aggregation_strategy,
                sparse=sparse,
                key_column=key_column,
            )

        var_collector[data_table] = var
        unit_report_collector[data_table] = unit_report

    if empty_table_counter == len(data_tables):
        logging.warning("No data found in any of the data tables. Returning edata without data.")
        return edata

    var = pd.concat(var_collector.values(), axis=0).reset_index(drop=True)

    if not instantiate_tensor:
        tem_layer = None
    elif sparse:
        tem_layer = sp.concatenate(list(r_collector.values()), axis=1)
    else:
        tem_layer = np.concatenate(list(r_collector.values()), axis=1)

    tem = pd.DataFrame({"interval_step": np.arange(num_intervals)})

    var.index = var.pop("variable").to_numpy() if concept_sets else var.index.astype(str)

    # AnnData does not allow to set a layer with value None
    if instantiate_tensor:
        edata = (
            EHRData(layers={layer: tem_layer}, obs=edata.obs, var=var, uns=edata.uns, tem=tem)
            if layer is not None
            else EHRData(X=tem_layer, obs=edata.obs, var=var, uns=edata.uns, tem=tem)
        )
    else:
        edata = EHRData(obs=edata.obs, var=var, uns=edata.uns, tem=tem)

    for data_table, unit_report in unit_report_collector.items():
        edata.uns[f"unit_report_{data_table}"] = unit_report

    return edata


def setup_interval_variables(
    edata,
    *,
    backend_handle: duckdb.DuckDBPyConnection,
    layer: str | None = None,
    data_tables: Sequence[
        Literal[
            "drug_exposure",
            "condition_occurrence",
            "procedure_occurrence",
            "device_exposure",
            "drug_era",
            "dose_era",
            "condition_era",
            "episode",
        ]
    ]
    | Literal[
        "drug_exposure",
        "condition_occurrence",
        "procedure_occurrence",
        "device_exposure",
        "drug_era",
        "dose_era",
        "condition_era",
        "episode",
    ],
    data_field_to_keep: str | Sequence[str] | dict[str, str | Sequence[str]],
    time_precision: Literal["date", "datetime"] = "date",
    clip_to_end_date: bool = True,
    interval_length_number: int,
    interval_length_unit: str,
    num_intervals: int,
    concept_ids: Literal["all"] | Sequence[int] = "all",
    aggregation_strategy: Literal[
        "last", "first", "mean", "median", "mode", "sum", "count", "min", "max", "std"
    ] = "last",
    enrich_var_with_feature_info: bool = False,
    keep_date: Literal["start", "end", "interval"] = "start",
    instantiate_tensor: bool = True,
):
    """Extracts selected tables of a time-span character from the OMOP CDM.

    The distinct `concept_id` s encountered in the selected tables form the variables in the EHRData object.
    The variables are sorted by the `concept_id` for each `data_table` in ascending order, and stacked together in the order that the `data_tables` are specified.
    The `data_field_to_keep` parameter specifies which Field in the selected table is to be used for the read-out of the value of a variable.

    The variables are set up for the rows of `edata.obs`, in their order: the time window of each row is read from the columns that :func:`~ehrdata.io.omop.setup_obs` writes to `.obs`.
    Hence `edata` can be filtered or reordered between :func:`~ehrdata.io.omop.setup_obs` and this function.
    With `observation_table="person_visit_occurrence"`, a data point whose `visit_occurrence_id` names a visit of its person in the `visit_occurrence` table belongs to that visit only, and to no visit if it lies outside that visit's window.
    Any other data point belongs to every visit of its person whose window holds it; the data tables without a `visit_occurrence_id`, such as `specimen`, have only such data points.
    Visit ids are compared as integers, as OMOP defines them.
    A row without a start date in `.obs`, or without an end date unless `clip_to_end_date=False`, gets no data; with `observation_table="person"`, every row needs a `birth_datetime`.

    In contrast to `setup_variables`, tables without unit information can be present here.
    Those tell no unit to verify or to carry along; the tables that do have a `unit_concept_id` are treated as in `setup_variables`.
    This function will not write a unit report. Should this be relevant for your work, please do open an issue on https://github.com/theislab/ehrdata.

    Stores a table(s) named `long_person_timestamp_feature_value_<data_table>` in long format in the RDBMS.
    Its `obs_index` is the position of the row in `edata.obs`; the table `long_person_timestamp_feature_value_<data_table>_obs` holds the `obs_names` of these rows.
    This table is instantiated in the `EHRData` object if `instantiate_tensor` is set to `True`;
    otherwise, the table is only stored in the RDBMS for later use.

    Args:
       edata: Data object to which the variables should be added.
       backend_handle: The backend handle to the database.
       layer: The layer to store the data in. If not specified, it uses `X`.
       data_tables: The tables to be used.
       data_field_to_keep: The CDM Field in the data tables to be kept. Can be e.g.
           'value_as_number' or 'value_as_concept_id'. Importantly, can be 'is_present'
           to have a one-hot encoding of the presence of the feature in a patient in an
           interval. Should be a dictionary to specify the data fields to keep per table
           if multiple data tables are used. For example, if data_tables=['measurement',
           'observation'], data_field_to_keep={'measurement': 'value_as_number',
           'observation': 'value_as_number'}.
       time_precision: The precision of the timestamp used in the table indicated in :func:`~ehrdata.io.omop.setup_obs`. If `"date"`, uses the `date` field (e.g. `visit_start_date` for `"person_visit_occurrence"`). If `"datetime"`, uses the `datetime` field (e.g. `visit_start_datetime` for `"person_visit_occurrence"`).
       clip_to_end_date: Whether the data points after the end date of an observation, such as the `visit_end_date`, are left out.
           If `False`, the data points are kept in all `num_intervals` intervals, also after the end date.
           Not available for `observation_table="person"`, whose observations have no end date.
       interval_length_number: Numeric value of the length of one interval.
       interval_length_unit: Unit of the interval length, needs to be a unit of :class:`pandas.Timedelta`.
       num_intervals: Number of intervals.
       concept_ids: Concept IDs to use from the data tables. If not specified, 'all' are used.
       aggregation_strategy: Strategy to use when aggregating multiple data points within one interval.
           For `'last'` and `'first'`, only data points that have a value in the data field of interest (the first field of `data_field_to_keep`) are considered.
           All kept fields are read from that very data point, so that e.g. the units describe the value that is kept.
           Every other strategy combines the values of several data points into one, which requires a single unit per feature.
           A `NotImplementedError` is raised for features with more than one `unit_concept_id` among their data points carrying a value (a missing unit is a unit of its own here).
           Exempt are `'count'`, `data_field_to_keep='is_present'`, and the data tables that have no `unit_concept_id` at all.
       enrich_var_with_feature_info: Whether to enrich the var table with feature
           information. If a concept_id is not found in the concept table, their respective alternate `concept_id` included in the concept_relationship table is retrieved to add the available feature information.
           Otherwise the feature information will be NaN.
       keep_date: Whether to keep the start or end date, or the interval span.
       instantiate_tensor: Whether to instantiate the tensor into the `EHRData` object.

    Returns:
        An `EHRData` object with fields.

    Examples:
        >>> import ehrdata as ed
        >>> import duckdb
        >>> con_gi = duckdb.connect(database=":memory:", read_only=False)
        >>> ed.dt.gibleed_omop(
        ...     con_gi,
        ... )
        >>> edata_gi = ed.io.omop.setup_obs(
        >>>     con_gi,
        >>>     observation_table="person_observation_period",
        >>> )
        >>> edata_gi = ed.io.omop.setup_interval_variables(
        >>>     edata=edata_gi,
        >>>     backend_handle=con_gi,
        >>>     data_tables=["drug_exposure", "condition_occurrence"],
        >>>     data_field_to_keep={"drug_exposure": "is_present", "condition_occurrence": "is_present"},
        >>>     interval_length_number=20,
        >>>     interval_length_unit="day",
        >>>     num_intervals=20,
        >>>     concept_ids="all",
        >>>     aggregation_strategy="last",
        >>>     enrich_var_with_feature_info=True,
        >>> )
        >>> edata_gi
    """
    from ehrdata import EHRData

    _check_valid_edata(edata)
    _check_valid_backend_handle(backend_handle)
    data_tables = _check_valid_interval_variable_data_tables(data_tables)
    data_field_to_keep = _check_valid_data_field_to_keep(data_field_to_keep, data_tables)
    _check_valid_interval_length_number(interval_length_number)
    _check_valid_interval_length_unit(interval_length_unit)
    _check_valid_time_precision(time_precision)
    _warn_time_precision_interval_mismatch(interval_length_unit, time_precision)
    _check_valid_num_intervals(num_intervals)
    _check_valid_concept_ids(concept_ids)
    _check_valid_aggregation_strategy(aggregation_strategy)
    _check_valid_enrich_var_with_feature_info(enrich_var_with_feature_info)
    _check_valid_clip_to_end_date(clip_to_end_date)
    _check_valid_keep_date(keep_date)

    time_defining_table = edata.uns.get("omop_io_observation_table", None)
    if time_defining_table is None:
        msg = "The observation table must be set up first, use the `setup_obs` function."
        raise ValueError(msg)

    _check_valid_clip_to_end_date_for_observation_table(clip_to_end_date, time_defining_table)
    _write_obs_windows(
        backend_handle, edata.obs, time_defining_table, time_precision, clip_to_end_date=clip_to_end_date
    )

    var_collector = {}
    r_collector = {}
    empty_table_counter = 0
    concept_mapping = False
    for data_table in data_tables:
        # dbms complains about our queries, which sometimes need a column to be of type e.g. datetime, when it can't infer types from data
        count = (
            backend_handle.execute(
                f"SELECT COUNT(*) as count FROM {data_table} WHERE {_concept_ids_condition(data_table, concept_ids)}"
            )
            .df()["count"]
            .item()
        )
        if count == 0:
            logging.warning(f"No data found in {data_table}. Returning edata without data of {data_table}.")
            empty_table_counter += 1
            continue

        unit_fields = _get_unit_fields(backend_handle, data_table)
        _check_one_unit_per_feature_for_aggregation(
            backend_handle,
            data_table,
            data_field_to_keep[data_table][0],
            aggregation_strategy,
            unit_fields,
            concept_ids=concept_ids,
        )

        _write_long_time_interval_table(
            backend_handle=backend_handle,
            data_table=data_table,
            time_precision=time_precision,
            data_field_to_keep=data_field_to_keep[data_table],
            unit_fields=unit_fields,
            interval_length_number=interval_length_number,
            interval_length_unit=interval_length_unit,
            num_intervals=num_intervals,
            aggregation_strategy=aggregation_strategy,
            concept_ids=concept_ids,
            keep_date=keep_date,
            link_visits=_links_rows_to_visits(backend_handle, time_defining_table, data_table),
        )
        _write_obs_names(backend_handle, data_table)

        var = backend_handle.execute(
            f"SELECT DISTINCT data_table_concept_id FROM long_person_timestamp_feature_value_{data_table}"
        ).df()

        var = var.sort_values("data_table_concept_id").reset_index(drop=True)

        if enrich_var_with_feature_info:
            concepts = backend_handle.sql("SELECT * FROM concept").df()
            concepts.columns = concepts.columns.str.lower()

            if any(elem not in set(concepts["concept_id"]) for elem in var["data_table_concept_id"]):
                concept_mapping = True
                logging.warning("Concept_ids are only partially matching. Mapping concept_ids where applicable.")
                concepts_relationship = backend_handle.sql(
                    "SELECT concept_id_1, concept_id_2 FROM concept_relationship WHERE (relationship_id = 'Mapped from') AND (concept_id_1 <> concept_id_2)"
                ).df()
                concepts_dict = concepts_relationship.set_index("concept_id_1")["concept_id_2"].to_dict()
                concepts_mapped_idx = [x not in set(concepts["concept_id"]) for x in var["data_table_concept_id"]]
                var.loc[concepts_mapped_idx, "data_table_concept_id_mapped"] = var.loc[
                    concepts_mapped_idx, "data_table_concept_id"
                ].map(concepts_dict)
                var["data_table_concept_id_mapped"] = var["data_table_concept_id_mapped"].fillna(
                    var["data_table_concept_id"]
                )

        if enrich_var_with_feature_info:
            if concept_mapping:
                var = pd.merge(var, concepts, how="left", left_on="data_table_concept_id_mapped", right_on="concept_id")
            else:
                var = pd.merge(var, concepts, how="left", left_on="data_table_concept_id", right_on="concept_id")

        if instantiate_tensor:
            ds = (
                (backend_handle.execute(f"SELECT * FROM long_person_timestamp_feature_value_{data_table}").df())
                .set_index(["obs_index", "data_table_concept_id", "interval_step"])
                .to_xarray()
            )
            ds = ds.sel(data_table_concept_id=var["data_table_concept_id"].values)

            r_collector[data_table] = ds[data_field_to_keep[data_table][0]].values

        else:
            ds = None

        var_collector[data_table] = var

    if empty_table_counter == len(data_tables):
        logging.warning("No data found in any of the data tables. Returning edata without data.")
        return edata

    var = pd.concat(var_collector.values(), axis=0).reset_index(drop=True)

    tem_layer = np.concatenate(list(r_collector.values()), axis=1) if instantiate_tensor else None

    tem = pd.DataFrame({"interval_step": np.arange(num_intervals)})

    var.index = var.index.astype(str)

    # AnnData does not allow to set a layer with value None
    if instantiate_tensor:
        edata = (
            EHRData(layers={layer: tem_layer}, obs=edata.obs, var=var, uns=edata.uns, tem=tem)
            if layer is not None
            else EHRData(X=tem_layer, obs=edata.obs, var=var, uns=edata.uns, tem=tem)
        )
    else:
        edata = EHRData(obs=edata.obs, var=var, uns=edata.uns, tem=tem)

    return edata
