from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ehrdata.io._omop_utils import get_table_catalog_dict
from ehrdata.io.omop._check_arguments import (
    VALID_OBSERVATION_TABLES_JOIN,
    VALID_OBSERVATION_TABLES_SINGLE,
    _check_valid_aggregation_strategy,
    _check_valid_backend_handle,
    _check_valid_concept_ids,
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
    _check_valid_variable_data_tables,
)
from ehrdata.io.omop._queries import _write_long_time_interval_table

if TYPE_CHECKING:
    from collections.abc import Sequence

    import duckdb
    from duckdb.duckdb import DuckDBPyConnection

DOWNLOAD_VERIFICATION_TAG = "download_verification_tag"


def _get_table_list() -> list:
    flat_table_list = [value for value_list in get_table_catalog_dict().values() for value in value_list]
    return flat_table_list


def _set_up_duckdb(path: Path, backend_handle: DuckDBPyConnection, prefix: str = "") -> None:
    """Create tables in the backend from the CSV files in the path from datasets in the OMOP Common Data model."""
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

            # read raw csv as temporary table
            temp_relation = backend_handle.read_csv(path / filename, dtype=dtype)  # noqa: F841
            backend_handle.execute("CREATE OR REPLACE TABLE temp_table AS SELECT * FROM temp_relation")

            # make query to create table with lowercase column names
            column_names = backend_handle.execute("DESCRIBE temp_table").df()["column_name"].values
            select_columns = ", ".join([f'"{col}" AS "{col.lower()}"' for col in column_names])
            create_table_with_lowercase_columns_query = (
                f"CREATE TABLE {regular_omop_table_name} AS SELECT {select_columns} FROM temp_table"
            )

            # write proper table
            existing_tables = backend_handle.execute("SHOW TABLES").df()["name"].values
            if regular_omop_table_name in existing_tables:
                logging.info(f"Table {regular_omop_table_name} already exists. Dropping and recreating...")
                backend_handle.execute(f"DROP TABLE {regular_omop_table_name}")

            backend_handle.execute(create_table_with_lowercase_columns_query)

            backend_handle.execute("DROP TABLE temp_table")

        elif filename_trunk != DOWNLOAD_VERIFICATION_TAG:
            unused_files.append(filename)

    missing_tables = [table for table in tables if table not in used_tables]

    logging.info(f"missing tables: {missing_tables}")
    logging.info(f"unused files: {unused_files}")


def _collect_units_per_feature(backend_handle, data_table, unit_key="unit_concept_id") -> dict:
    query = f"""
    SELECT DISTINCT data_table_concept_id, {unit_key} FROM long_person_timestamp_feature_value_{data_table}
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


def _check_one_unit_per_feature(backend_handle, data_table, unit_key="unit_concept_id") -> None:
    feature_units = _collect_units_per_feature(backend_handle, data_table, unit_key=unit_key)
    num_units = np.array([len(units) for _, units in feature_units.items()])

    # print(f"no units for features: {np.argwhere(num_units == 0)}")
    logging.warning(f"multiple units for features: {np.argwhere(num_units > 1)}")


def _create_feature_unit_concept_id_report(backend_handle, data_table) -> pd.DataFrame:
    feature_units_concept = _collect_units_per_feature(backend_handle, data_table, unit_key="unit_concept_id")
    feature_units_long_format = []
    for feature, units in feature_units_concept.items():
        if len(units) == 0:
            feature_units_long_format.append({"concept_id": feature, "no_units": True, "multiple_units": False})
        elif len(units) > 1:
            feature_units_long_format.extend(
                [
                    {
                        "concept_id": feature,
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
                    "concept_id": feature,
                    "unit_concept_id": units[0],
                    "no_units": False,
                    "multiple_units": False,
                }
            )

    df = pd.DataFrame(
        feature_units_long_format, columns=["concept_id", "unit_concept_id", "no_units", "multiple_units"]
    )
    df["unit_concept_id"] = df["unit_concept_id"].astype("Int64")

    return df


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


def setup_connection(path: Path | str, backend_handle: DuckDBPyConnection, prefix: str = "") -> None:
    """Setup a connection to the OMOP CDM database.

    This function sets up a connection to the OMOP CDM database.
    It checks the capitalization of the 'person' table, and assumes the same capitalization style is used for all other tables.


    Args:
        path: The path to the folder containing the CSV files.
        backend_handle: The backend handle to the database.
        prefix: The prefix to be removed from the CSV filenames.

    Returns:
        An EHRData object with populated .uns["omop_table_capitalization"] field.

    """
    _set_up_duckdb(Path(path), backend_handle, prefix)


def setup_obs(
    backend_handle: duckdb.duckdb.DuckDBPyConnection,
    observation_table: Literal["person", "person_cohort", "person_observation_period", "person_visit_occurrence"],
    *,
    death_table: bool = False,
):
    """Setup the observation table for :class:`~ehrdata.EHRData` object.

    For this, a table from the OMOP CDM which represents the "observed unit" via an id should be selected.
    A unit can be a person, or the data of a person together with either the information from cohort, observation_period, or visit_occurrence.
    Notice a single person can have multiple of the latter, and as such can appear multiple times.
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
        obs = get_table(backend_handle, observation_table)

    elif observation_table in VALID_OBSERVATION_TABLES_JOIN:
        if observation_table == "person_cohort":
            obs = _get_table_join(backend_handle, "person", "cohort", right_key="subject_id")
        elif observation_table == "person_observation_period":
            obs = _get_table_join(backend_handle, "person", "observation_period")
        elif observation_table == "person_visit_occurrence":
            obs = _get_table_join(backend_handle, "person", "visit_occurrence")

    if death_table:
        death = get_table(backend_handle, "death")
        obs = obs.merge(death, how="left", on="person_id")

    # AnnData will make this conversion and raise a Warning.
    # Since we have no reason for the RangeIndex, beyond it being introduced by the to_df function of the RDBMS, we can do it immediately and omit a confusing warning
    obs.index = obs.index.astype(str)

    return EHRData(obs=obs, uns={"omop_io_observation_table": observation_table.split("person_")[-1]})


def setup_variables(
    edata,
    *,
    backend_handle: duckdb.duckdb.DuckDBPyConnection,
    data_tables: Sequence[Literal["measurement", "observation", "specimen"]]
    | Literal["measurement", "observation", "specimen"],
    data_field_to_keep: str | Sequence[str] | dict[str, str | Sequence[str]],
    interval_length_number: int,
    interval_length_unit: str,
    num_intervals: int,
    concept_ids: Literal["all"] | Sequence[int] = "all",
    aggregation_strategy: Literal[
        "last", "first", "mean", "median", "mode", "sum", "count", "min", "max", "std"
    ] = "last",
    enrich_var_with_feature_info: bool = False,
    enrich_var_with_unit_info: bool = False,
    instantiate_tensor: bool = True,
):
    """Extracts selected tables of a data-point character from the OMOP CDM.

    The distinct `concept_id` is encountered in the selected tables form the variables in the EHRData object.
    The `data_field_to_keep` parameter specifies which Field in the selected table is to be used for the read-out of the value of a variable.

    It will fail if there is more than one `unit_concept_id` per variable.
    Writes a unit report of the features to `edata.uns['unit_report_<data_tables>']`.
    Writes the setup arguments into `edata.uns['omop_io_variable_setup']`.

    Stores a table(s) named `long_person_timestamp_feature_value_<data_table>` in long format in the RDBMS.
    This table is instantiated into `edata.r` if `instantiate_tensor` is set to `True`;
    otherwise, the table is only stored in the RDBMS for later use.

    Args:
        backend_handle: The backend handle to the database.
        edata: Data object to which the variables should be added.
        data_tables: The tables to be used.
        data_field_to_keep: The CDM Field in the data tables to be kept. Can be e.g.
            'value_as_number' or 'value_as_concept_id'. Importantly, can be 'is_present'
            to have a one-hot encoding of the presence of the feature in a patient in an
            interval. Should be a dictionary to specify the data fields to keep per table
            if multiple data tables are used. For example, if data_tables=['measurement',
            'observation'], data_field_to_keep={'measurement': 'value_as_number',
            'observation': 'value_as_number'}.
        interval_length_number: Numeric value of the length of one interval.
        interval_length_unit: Unit of the interval length, needs to be a unit of :class:`pandas.Timedelta`.
        num_intervals: Number of intervals.
        concept_ids: Concept IDs to use from the data tables. If not specified, 'all' are used.
        aggregation_strategy: Strategy to use when aggregating multiple data points within one interval.
        enrich_var_with_feature_info: Whether to enrich the var table with feature
            information. If a concept_id is not found in the concept table, the feature tinformation will be NaN.
        enrich_var_with_unit_info: Whether to enrich the var table with unit information.
            Raises an Error if multiple units per feature are found for at least one
            feature. For entire missing data points, the units are ignored. For observed
            data points with missing unit information (NULL in either 'unit_concept_id'
            or 'unit_source_value'), the value NULL/NaN is considered a single unit.
        instantiate_tensor: Whether to instantiate the tensor into the .r field of the EHRData object.

    Returns:
        An :class:`~ehrdata.EHRData` object with populated `.r` and `.var` field.

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
    _check_valid_num_intervals(num_intervals)
    _check_valid_concept_ids(concept_ids)
    _check_valid_aggregation_strategy(aggregation_strategy)
    _check_valid_enrich_var_with_feature_info(enrich_var_with_feature_info)
    _check_valid_enrich_var_with_unit_info(enrich_var_with_unit_info)

    time_defining_table = edata.uns.get("omop_io_observation_table", None)
    if time_defining_table is None:
        msg = "The observation table must be set up first, use the `setup_obs` function."
        raise ValueError(msg)

    data_field_to_keep = {k: [*list(v), "unit_concept_id", "unit_source_value"] for k, v in data_field_to_keep.items()}

    var_collector = {}
    r_collector = {}
    unit_report_collector = {}
    empty_table_counter = 0
    for data_table in data_tables:
        # dbms complains about our queries, which sometimes need a column to be of type e.g. datetime, when it can't infer types from data
        count = backend_handle.execute(f"SELECT COUNT(*) as count FROM {data_table}").df()["count"].item()
        if count == 0:
            logging.warning(f"No data found in {data_table}. Returning edata without data of {data_table}.")
            empty_table_counter += 1
            continue

        _write_long_time_interval_table(
            backend_handle=backend_handle,
            time_defining_table=time_defining_table,
            data_table=data_table,
            data_field_to_keep=data_field_to_keep[data_table],
            interval_length_number=interval_length_number,
            interval_length_unit=interval_length_unit,
            num_intervals=num_intervals,
            aggregation_strategy=aggregation_strategy,
        )

        _check_one_unit_per_feature(backend_handle, data_table)
        unit_report = _create_feature_unit_concept_id_report(backend_handle, data_table=data_table)

        var = backend_handle.execute(
            f"SELECT DISTINCT data_table_concept_id FROM long_person_timestamp_feature_value_{data_table}"
        ).df()

        if enrich_var_with_feature_info or enrich_var_with_unit_info:
            concepts = backend_handle.sql("SELECT * FROM concept").df()
            concepts.columns = concepts.columns.str.lower()

        if enrich_var_with_feature_info:
            var = pd.merge(var, concepts, how="left", left_index=True, right_on="concept_id")

        if enrich_var_with_unit_info:
            if unit_report["multiple_units"].sum() > 0:
                msg = "Multiple units per feature found. Enrichment with feature information not possible."
                raise ValueError(msg)
            else:
                var = pd.merge(
                    var,
                    unit_report,
                    how="left",
                    left_index=True,
                    right_on="unit_concept_id",
                    suffixes=("", "_unit"),
                )
                var = pd.merge(
                    var,
                    concepts,
                    how="left",
                    left_on="unit_concept_id",
                    right_on="concept_id",
                    suffixes=("", "_unit"),
                )

        if instantiate_tensor:
            ds = (
                (backend_handle.execute(f"SELECT * FROM long_person_timestamp_feature_value_{data_table}").df())
                .set_index(["person_id", "data_table_concept_id", "interval_step"])
                .to_xarray()
            )
            r_collector[data_table] = ds[data_field_to_keep[data_table][0]].values

        else:
            ds = None

        var_collector[data_table] = var
        unit_report_collector[data_table] = unit_report

    if empty_table_counter == len(data_tables):
        logging.warning("No data found in any of the data tables. Returning edata without data.")
        return edata

    var = pd.concat(var_collector.values(), axis=0)

    r = np.concatenate(list(r_collector.values()), axis=1) if instantiate_tensor else None

    tem = pd.DataFrame({"interval_step": np.arange(num_intervals)})

    var.index = var.index.astype(str)

    edata = EHRData(R=r, obs=edata.obs, var=var, uns=edata.uns, tem=tem)

    for data_table in data_tables:
        edata.uns[f"unit_report_{data_table}"] = unit_report_collector[data_table]

    return edata


def setup_interval_variables(
    edata,
    *,
    backend_handle: duckdb.duckdb.DuckDBPyConnection,
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
    The `data_field_to_keep` parameter specifies which Field in the selected table is to be used for the read-out of the value of a variable.

    In contrast to `setup_variables`, tables without unit unformation can be present here. Hence, this function will not verify that a single unit per feature (=`concept_id`) is used. Also, it will not write a unit report. Should this be relevant for your work, please do open an issue on https://github.com/theislab/ehrdata.

    Stores a table(s) named `long_person_timestamp_feature_value_<data_table>` in long format in the RDBMS.
    This table is instantiated into `edata.r` if `instantiate_tensor` is set to `True`;
    otherwise, the table is only stored in the RDBMS for later use.

    Args:
       backend_handle: The backend handle to the database.
       edata: Data object to which the variables should be added.
       data_tables: The tables to be used.
       data_field_to_keep: The CDM Field in the data tables to be kept. Can be e.g.
           'value_as_number' or 'value_as_concept_id'. Importantly, can be 'is_present'
           to have a one-hot encoding of the presence of the feature in a patient in an
           interval. Should be a dictionary to specify the data fields to keep per table
           if multiple data tables are used. For example, if data_tables=['measurement',
           'observation'], data_field_to_keep={'measurement': 'value_as_number',
           'observation': 'value_as_number'}.
       interval_length_number: Numeric value of the length of one interval.
       interval_length_unit: Unit of the interval length, needs to be a unit of :class:`pandas.Timedelta`.
       num_intervals: Number of intervals.
       concept_ids: Concept IDs to use from the data tables. If not specified, 'all' are used.
       aggregation_strategy: Strategy to use when aggregating multiple data points within one interval.
       enrich_var_with_feature_info: Whether to enrich the var table with feature
           information. If a concept_id is not found in the concept table, the feature information will be NaN.
       keep_date: Whether to keep the start or end date, or the interval span.
       instantiate_tensor: Whether to instantiate the tensor into the .r field of the EHRData object.

    Returns:
        An EHRData object with populated `.r` and `.var` field.

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
    _check_valid_num_intervals(num_intervals)
    _check_valid_concept_ids(concept_ids)
    _check_valid_aggregation_strategy(aggregation_strategy)
    _check_valid_enrich_var_with_feature_info(enrich_var_with_feature_info)
    _check_valid_keep_date(keep_date)

    time_defining_table = edata.uns.get("omop_io_observation_table", None)
    if time_defining_table is None:
        msg = "The observation table must be set up first, use the `setup_obs` function."
        raise ValueError(msg)

    var_collector = {}
    r_collector = {}
    empty_table_counter = 0
    for data_table in data_tables:
        # dbms complains about our queries, which sometimes need a column to be of type e.g. datetime, when it can't infer types from data
        count = backend_handle.execute(f"SELECT COUNT(*) as count FROM {data_table}").df()["count"].item()
        if count == 0:
            logging.warning(f"No data found in {data_table}. Returning edata without data of {data_table}.")
            empty_table_counter += 1
            continue

        _write_long_time_interval_table(
            backend_handle=backend_handle,
            time_defining_table=time_defining_table,
            data_table=data_table,
            data_field_to_keep=data_field_to_keep[data_table],
            interval_length_number=interval_length_number,
            interval_length_unit=interval_length_unit,
            num_intervals=num_intervals,
            aggregation_strategy=aggregation_strategy,
            keep_date=keep_date,
        )

        var = backend_handle.execute(
            f"SELECT DISTINCT data_table_concept_id FROM long_person_timestamp_feature_value_{data_table}"
        ).df()

        if enrich_var_with_feature_info:
            concepts = backend_handle.sql("SELECT * FROM concept").df()
            concepts.columns = concepts.columns.str.lower()

        if enrich_var_with_feature_info:
            var = pd.merge(var, concepts, how="left", left_index=True, right_on="concept_id")

        if instantiate_tensor:
            ds = (
                (backend_handle.execute(f"SELECT * FROM long_person_timestamp_feature_value_{data_table}").df())
                .set_index(["person_id", "data_table_concept_id", "interval_step"])
                .to_xarray()
            )
            r_collector[data_table] = ds[data_field_to_keep[data_table][0]].values

        else:
            ds = None

        var_collector[data_table] = var

    if empty_table_counter == len(data_tables):
        logging.warning("No data found in any of the data tables. Returning edata without data.")
        return edata

    var = pd.concat(var_collector.values(), axis=0)

    r = np.concatenate(list(r_collector.values()), axis=1) if instantiate_tensor else None

    tem = pd.DataFrame({"interval_step": np.arange(num_intervals)})

    var.index = var.index.astype(str)

    edata = EHRData(R=r, obs=edata.obs, var=var, uns=edata.uns, tem=tem)

    return edata


def get_table(duckdb_instance, table_name: str) -> pd.DataFrame:
    """Extract a table of an OMOP CDM Database."""
    return _lowercase_column_names(duckdb_instance.sql(f"SELECT * FROM {table_name}").df())


def _get_table_join(
    duckdb_instance, table1: str, table2: str, left_key: str = "person_id", right_key: str = "person_id"
) -> pd.DataFrame:
    """Extract a table of an OMOP CDM Database."""
    return _lowercase_column_names(
        duckdb_instance.sql(
            f"SELECT * \
        FROM {table1} as t1 \
        JOIN {table2} as t2 ON t1.{left_key} = t2.{right_key} \
        "
        ).df()
    )


def _lowercase_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all column names to lowercase."""
    df.columns = map(str.lower, df.columns)  # Convert all column names to lowercase
    return df
