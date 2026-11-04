"""The observed-rows path of :func:`~ehrdata.io.omop.setup_variables`.

The default path builds the full cross product of observations, features and intervals and fills it from the data table.
This path instead assigns each data row to its observation, variable and interval, and aggregates only the cells that have rows.
Its long table holds the observed cells alone, so its size follows the data rather than observations x features x intervals, and it instantiates as a :class:`sparse.COO` array or as a dense array.

It reads the observations and their time windows from `.obs`, and adds these options on top of the default path:

- concept sets: several concept ids pooled into one variable before the aggregation;
- an aggregation strategy per variable;
- a value range per variable, dropping rows outside it before the aggregation;
- converting the values of a variable from the units they were recorded in to one unit, before the aggregation;
- linking data rows to a visit observation by their `visit_occurrence_id`, instead of by `person_id` and time;
- keeping rows dated after the observation's end date (`cut_at_end=False`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ehrdata.io.omop._queries import (
    AGGREGATION_STRATEGY_KEY,
    DATA_TABLE_CONCEPT_ID_TRUNK,
    SINGLE_ROW_AGGREGATION_STRATEGIES,
    TIME_DEFINING_TABLE_ID_KEY,
    UNITLESS_AGGREGATION_STRATEGIES,
    _finer_datetime_keys,
    _generate_value_query,
    _get_datetime_key,
)

if TYPE_CHECKING:
    import duckdb

LINK_BY = ("person", "visit_occurrence")
VARIABLE_MAP_TABLE = "ehrdata_variable_map"
OBS_TABLE = "ehrdata_obs_windows"
UNIT_MAP_TABLE = "ehrdata_unit_map"
UNIT_CONVERSION_COLUMNS = ("variable", "unit_concept_id", "to_unit_concept_id", "multiplier")


def _obs_windows(obs: pd.DataFrame, time_defining_table: str, time_precision: str) -> pd.DataFrame:
    """The time window of every row of `.obs`, in `.obs` order: obs_index, obs_id, person_id, start_date, end_date.

    Taken from `.obs` itself rather than from the database, so that an `EHRData` whose observations were filtered after :func:`~ehrdata.io.omop.setup_obs` gets the variables of exactly its observations, in its order.
    """
    columns = {column.lower(): column for column in obs.columns}
    id_key = TIME_DEFINING_TABLE_ID_KEY[time_defining_table]
    start_key = _get_datetime_key("start", time_defining_table, time_precision)
    needed = [id_key, "person_id", start_key]
    if time_defining_table != "person":
        end_key = _get_datetime_key("end", time_defining_table, "date")
        needed.append(end_key)
    missing = [key for key in needed if key not in columns]
    if missing:
        msg = f"setup_variables needs the columns {missing} of .obs, as written by setup_obs, to place each observation's time window."
        raise ValueError(msg)
    windows = pd.DataFrame(
        {
            "obs_index": np.arange(len(obs), dtype=np.int64),
            "obs_id": obs[columns[id_key]].to_numpy(),
            "person_id": obs[columns["person_id"]].to_numpy(),
            "start_date": pd.to_datetime(obs[columns[start_key]]).to_numpy(),
            "end_date": (
                pd.to_datetime(obs[columns[end_key]]).dt.normalize().to_numpy()
                if time_defining_table != "person"
                else np.full(len(obs), np.datetime64("2999-12-31"), dtype="datetime64[ns]")
            ),
        }
    )
    return windows


def _variable_map(
    backend_handle: duckdb.DuckDBPyConnection,
    data_table: str,
    *,
    concept_ids: Literal["all"] | Sequence[int] | Mapping[str, Sequence[int]],
    aggregation_strategy: str | Mapping[str | int, str],
    value_range: Mapping[str | int, tuple[float | None, float | None]] | None,
    unit_conversions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (variable, concept_id), in variable order.

    `converted` marks the variables that `unit_conversions` converts.

    With a mapping of concept sets, the variables are its keys, in its order, including those without data.
    Otherwise the variables are the concept ids that have rows in the data table, sorted, as in the default path.
    """
    concept_key = f"{DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id"
    if isinstance(concept_ids, Mapping):
        rows = [
            (str(name), int(concept_id), position)
            for position, (name, ids) in enumerate(concept_ids.items())
            for concept_id in dict.fromkeys(ids)
        ]
        is_concept_set = True
    else:
        condition = (
            "TRUE"
            if isinstance(concept_ids, str)
            else f"{concept_key} IN ({', '.join(str(int(c)) for c in concept_ids)})"
        )
        present = sorted(
            int(row[0])
            for row in backend_handle.execute(
                f"SELECT DISTINCT {concept_key} FROM {data_table} WHERE {condition} AND {concept_key} IS NOT NULL"
            ).fetchall()
        )
        rows = [(str(concept_id), concept_id, position) for position, concept_id in enumerate(present)]
        is_concept_set = False

    variable_map = pd.DataFrame(rows, columns=["variable", "concept_id", "var_index"])
    variables = list(dict.fromkeys(variable_map["variable"]))

    def lookup(mapping, variable):
        if variable in mapping:
            return mapping[variable]
        if not is_concept_set and int(variable) in mapping:
            return mapping[int(variable)]
        return None

    if isinstance(aggregation_strategy, Mapping):
        missing = [v for v in variables if lookup(aggregation_strategy, v) is None]
        if missing:
            msg = f"aggregation_strategy has no strategy for the variable(s) {missing[:5]}."
            raise ValueError(msg)
        strategies = {v: lookup(aggregation_strategy, v) for v in variables}
    else:
        strategies = dict.fromkeys(variables, aggregation_strategy)
    variable_map["strategy"] = variable_map["variable"].map(strategies)

    bounds = {v: (lookup(value_range, v) or (None, None)) if value_range else (None, None) for v in variables}
    variable_map["lower"] = variable_map["variable"].map(lambda v: bounds[v][0]).astype("float64")
    variable_map["upper"] = variable_map["variable"].map(lambda v: bounds[v][1]).astype("float64")
    converted = set(_unit_map(unit_conversions)["variable"]) if unit_conversions is not None else set()
    variable_map["converted"] = variable_map["variable"].isin(converted)
    variable_map.attrs["is_concept_set"] = is_concept_set
    return variable_map


def _unit_map(unit_conversions: pd.DataFrame) -> pd.DataFrame:
    """`unit_conversions` with variable as text, offset defaulting to 0, one row per (variable, unit)."""
    missing = [c for c in UNIT_CONVERSION_COLUMNS if c not in unit_conversions.columns]
    if missing:
        msg = f"unit_conversions needs the columns {list(UNIT_CONVERSION_COLUMNS)} (and optionally 'offset'); missing {missing}."
        raise ValueError(msg)
    unit_map = pd.DataFrame(
        {
            "variable": unit_conversions["variable"].astype(str),
            "unit_concept_id": pd.to_numeric(unit_conversions["unit_concept_id"]).astype("Int64"),
            "to_unit_concept_id": pd.to_numeric(unit_conversions["to_unit_concept_id"]).astype("Int64"),
            "multiplier": unit_conversions["multiplier"].astype("float64"),
            "offset": (unit_conversions["offset"].astype("float64") if "offset" in unit_conversions.columns else 0.0),
        }
    )
    if unit_map.duplicated(["variable", "unit_concept_id"]).any():
        msg = "unit_conversions has more than one row for a (variable, unit_concept_id)."
        raise ValueError(msg)
    if unit_map.groupby("variable")["to_unit_concept_id"].nunique().gt(1).any():
        msg = "unit_conversions converts a variable to more than one unit."
        raise ValueError(msg)
    return unit_map


def _check_convertible(backend_handle: duckdb.DuckDBPyConnection, data_table: str, value_field: str) -> None:
    """Every row of a converted variable must be in a unit that `unit_conversions` lists for it."""
    concept_key = f"{DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id"
    unknown = backend_handle.execute(
        f"""
        SELECT vm.variable, TRY_CAST(d.unit_concept_id AS BIGINT) AS unit, count(*) AS n
        FROM {data_table} d JOIN {VARIABLE_MAP_TABLE} vm ON d.{concept_key} = vm.concept_id
        LEFT JOIN {UNIT_MAP_TABLE} um ON um.variable = vm.variable
             AND um.unit_concept_id IS NOT DISTINCT FROM TRY_CAST(d.unit_concept_id AS BIGINT)
        WHERE vm.converted AND d.{value_field} IS NOT NULL AND um.variable IS NULL
        GROUP BY ALL ORDER BY n DESC
        """
    ).fetchall()
    if unknown:
        listing = ", ".join(f"{variable}: unit {unit} ({n} rows)" for variable, unit, n in unknown[:5])
        msg = f"unit_conversions has no conversion for {len(unknown)} (variable, unit) pair(s) of '{data_table}': {listing}."
        raise ValueError(msg)


def _check_one_unit_per_variable(
    backend_handle: duckdb.DuckDBPyConnection, data_table: str, value_field: str, unit_fields: Sequence[str]
) -> None:
    """Like _check_one_unit_per_feature_for_aggregation, per variable: the rows of a concept set must share a unit to be combined."""
    if value_field == "is_present" or "unit_concept_id" not in unit_fields:
        return
    single = SINGLE_ROW_AGGREGATION_STRATEGIES + UNITLESS_AGGREGATION_STRATEGIES
    concept_key = f"{DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id"
    multiple = backend_handle.execute(
        f"""
        SELECT vm.variable, list(DISTINCT d.unit_concept_id ORDER BY d.unit_concept_id) AS units
        FROM {data_table} d JOIN {VARIABLE_MAP_TABLE} vm ON d.{concept_key} = vm.concept_id
        WHERE d.{value_field} IS NOT NULL AND NOT vm.converted
          AND vm.strategy NOT IN ({", ".join(f"'{s}'" for s in single)})
        GROUP BY vm.variable HAVING count(DISTINCT coalesce(CAST(d.unit_concept_id AS VARCHAR), chr(0))) > 1
        """
    ).fetchall()
    if multiple:
        listing = ", ".join(f"{variable}: {units}" for variable, units in multiple[:5])
        msg = (
            f"{len(multiple)} variable(s) of '{data_table}' combine rows with more than one unit_concept_id "
            f"({listing}), which their aggregation strategy cannot combine into one value. Use 'last' or "
            f"'first' for them, or harmonize their units in the database."
        )
        raise NotImplementedError(msg)


def _write_long_observed_table(
    backend_handle: duckdb.DuckDBPyConnection,
    *,
    time_defining_table: str,
    data_table: str,
    time_precision: Literal["date", "datetime"],
    interval_length_number: int,
    interval_length_unit: str,
    num_intervals: int,
    data_field_to_keep: Sequence[str],
    unit_fields: Sequence[str],
    variable_map: pd.DataFrame,
    link_by: Literal["person", "visit_occurrence"] = "person",
    convert_units: bool = False,
    cut_at_end: bool = True,
) -> None:
    """Write `long_person_timestamp_feature_value_<data_table>` with one row per observed (obs, variable, interval).

    Columns: obs_id, obs_index, person_id, variable, var_index, interval_step, is_present, the kept fields and the unit fields.
    The observations and their time windows are the rows of the table OBS_TABLE (see `_obs_windows`), and `obs_index` is the position in `.obs`.
    """
    if link_by == "visit_occurrence" and time_defining_table != "visit_occurrence":
        msg = "link_by='visit_occurrence' requires the observation table 'person_visit_occurrence'."
        raise ValueError(msg)
    interval_us = int(pd.Timedelta(interval_length_number, interval_length_unit) // pd.Timedelta(microseconds=1))
    window_us = interval_us * num_intervals
    datetime_col = _get_datetime_key("timepoint", data_table, time_precision)
    finer = _finer_datetime_keys("timepoint", data_table, time_precision)
    concept_key = f"{DATA_TABLE_CONCEPT_ID_TRUNK[data_table]}_concept_id"
    value_field = data_field_to_keep[0]

    obs_cte = f"SELECT obs_index, obs_id, person_id, start_date, CAST(end_date AS DATE) AS end_date FROM {OBS_TABLE}"
    link = "d.visit_occurrence_id = o.obs_id" if link_by == "visit_occurrence" else "d.person_id = o.person_id"

    # with unit conversions, a converted variable's value and unit are those after the conversion
    value_expr, unit_expr, unit_join = f"d.{value_field}", "d.unit_concept_id", ""
    if convert_units:
        unit_join = (
            f"LEFT JOIN {UNIT_MAP_TABLE} um ON um.variable = vm.variable "
            "AND um.unit_concept_id IS NOT DISTINCT FROM TRY_CAST(d.unit_concept_id AS BIGINT)"
        )
        value_expr = f"CASE WHEN vm.converted THEN d.{value_field} * um.multiplier + um.offset ELSE d.{value_field} END"
        unit_expr = "CASE WHEN vm.converted THEN um.to_unit_concept_id ELSE d.unit_concept_id END"

    def select(column):
        if column == value_field:
            return f"{value_expr} AS {value_field}"
        if column == "unit_concept_id":
            return f"{unit_expr} AS unit_concept_id"
        return f"d.{column}"

    kept = list(dict.fromkeys([*data_field_to_keep, *unit_fields, datetime_col, *finer]))
    kept_select = ", ".join(select(column) for column in kept if column != "is_present")
    numeric_value = value_field != "is_present"
    range_filter = (
        f"AND (vm.lower IS NULL OR {value_expr} IS NULL OR {value_expr} >= vm.lower) "
        f"AND (vm.upper IS NULL OR {value_expr} IS NULL OR {value_expr} <= vm.upper)"
        if numeric_value
        else ""
    )
    if convert_units:
        range_filter += " AND NOT (vm.converted AND um.variable IS NULL)"

    groups = []
    for strategy in dict.fromkeys(variable_map["strategy"]):
        value_query = _generate_value_query(
            "rows",
            list(data_field_to_keep),
            AGGREGATION_STRATEGY_KEY[strategy],
            datetime_col,
            unit_fields=unit_fields,
            finer_datetime_columns=finer,
        )
        groups.append(
            f"""
            SELECT obs_id, obs_index, person_id, variable, var_index, interval_step, {value_query}
            FROM rows WHERE strategy = '{strategy}'
            GROUP BY obs_id, obs_index, person_id, variable, var_index, interval_step"""
        )

    table = f"long_person_timestamp_feature_value_{data_table}"
    backend_handle.execute(f"DROP TABLE IF EXISTS {table}")
    backend_handle.execute(
        f"""
        CREATE TABLE {table} AS
        WITH obs_t AS ({obs_cte}),
        rows AS (
            SELECT o.obs_id, o.obs_index, o.person_id, vm.variable, vm.var_index, vm.strategy, 1 AS is_present,
                   CAST((epoch_us(CAST(d.{datetime_col} AS TIMESTAMP)) - epoch_us(o.start_date)) // {interval_us} AS INTEGER) AS interval_step,
                   {kept_select}
            FROM {data_table} d
            JOIN {VARIABLE_MAP_TABLE} vm ON d.{concept_key} = vm.concept_id
            JOIN obs_t o ON {link}
            {unit_join}
            WHERE CAST(d.{datetime_col} AS TIMESTAMP) >= o.start_date
              AND epoch_us(CAST(d.{datetime_col} AS TIMESTAMP)) - epoch_us(o.start_date) < {window_us}
              {"AND CAST(d." + datetime_col + " AS DATE) <= o.end_date" if cut_at_end else ""}
              {range_filter}
        ),
        aggregated AS ({" UNION ALL ".join(groups)})
        SELECT * FROM aggregated WHERE is_present IS NOT NULL
        """
    )


def _unit_report(
    backend_handle: duckdb.DuckDBPyConnection, data_table: str, unit_fields: Sequence[str]
) -> pd.DataFrame:
    """Units per variable among the observed cells, in the layout of the default path's unit report, keyed by variable."""
    table = f"long_person_timestamp_feature_value_{data_table}"
    if "unit_concept_id" not in unit_fields:
        return pd.DataFrame(columns=["variable", "unit_concept_id", "no_units", "multiple_units"])
    report = backend_handle.execute(
        f"""
        SELECT variable, unit_concept_id,
               unit_concept_id IS NULL AS no_units,
               count(*) OVER (PARTITION BY variable) > 1 AS multiple_units
        FROM (SELECT DISTINCT variable, unit_concept_id FROM {table})
        ORDER BY variable, unit_concept_id
        """
    ).df()
    report["unit_concept_id"] = report["unit_concept_id"].astype("Int64")
    return report


def _instantiate(
    backend_handle: duckdb.DuckDBPyConnection,
    data_table: str,
    value_field: str,
    *,
    n_obs: int,
    n_var: int,
    num_intervals: int,
    sparse: bool,
    fill_value: float = np.nan,
):
    """The (n_obs, n_var, num_intervals) array of the long table, as sparse.COO or dense.

    Cells without rows hold `fill_value`: NaN, or 0 when every variable counts data points, as in the default path.
    """
    import sparse as sp

    cells = backend_handle.execute(
        f"SELECT obs_index, var_index, interval_step, {value_field} AS value "
        f"FROM long_person_timestamp_feature_value_{data_table}"
    ).fetchnumpy()
    values = cells["value"]
    if hasattr(values, "filled"):  # masked array: NULL values
        values = values.astype("float64").filled(np.nan)
    try:
        values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = f"The observed-rows path stores numeric values only; '{value_field}' is not numeric."
        raise TypeError(msg) from error
    coords = np.stack([np.asarray(cells[key], dtype=np.int64) for key in ("obs_index", "var_index", "interval_step")])
    array = sp.COO(coords, values, shape=(n_obs, n_var, num_intervals), fill_value=fill_value, has_duplicates=False)
    return array if sparse else array.todense()


def _var_frame(variable_map: pd.DataFrame, data_table: str) -> pd.DataFrame:
    """The `.var` rows of one data table: one per variable, in variable order."""
    grouped = variable_map.groupby("var_index", sort=True)
    var = pd.DataFrame(
        {
            "variable": grouped["variable"].first(),
            "concept_ids": grouped["concept_id"].agg(lambda ids: ",".join(str(int(c)) for c in ids)),
            "aggregation_strategy": grouped["strategy"].first(),
            "data_table": data_table,
        }
    ).reset_index(drop=True)
    if not variable_map.attrs.get("is_concept_set", False):
        var.insert(0, "data_table_concept_id", grouped["concept_id"].first().to_numpy())
    return var
