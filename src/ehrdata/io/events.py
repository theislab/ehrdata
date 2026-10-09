from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import TYPE_CHECKING, Literal

import duckdb
import numpy as np
import pandas as pd
from sparse import COO

from ehrdata.core.constants import ANCHOR_TIME_KEY
from ehrdata.io.omop._queries import AGGREGATION_STRATEGY_KEY, _generate_timedeltas

if TYPE_CHECKING:
    from collections.abc import Collection

    import pyarrow as pa

    from ehrdata import EHRData

EVENT_AGGREGATION_STRATEGIES = ("last", "first", "mean", "median", "min", "max", "sum", "count")


def from_events(
    events: pd.DataFrame | pa.Table | duckdb.DuckDBPyRelation | str | PathLike | Sequence[str | PathLike],
    *,
    obs: pd.DataFrame | None = None,
    anchor: str | None = None,
    interval_length_number: int = 1,
    interval_length_unit: str = "h",
    num_intervals: int | None = None,
    codes: Collection[str] | None = None,
    aggregation_strategy: Literal["last", "first", "mean", "median", "min", "max", "sum", "count"] = "last",
    sparse: bool = False,
    layer: str | None = None,
    subject_id: str = "subject_id",
    time: str = "time",
    code: str = "code",
    numeric_value: str = "numeric_value",
) -> EHRData:
    """Bin a table of events into an :class:`~ehrdata.EHRData` object with a time axis.

    Every row of `events` is one event: a `code` recorded for a subject at a `time`, optionally with a `numeric_value`.
    This is the layout of the `Medical Event Data Standard (MEDS) <https://medical-event-data-standard.github.io>`__, whose column names are the defaults.
    The codes become the variables, and the time since each subject's anchor is divided into intervals of equal length.
    Interval `i` starts `i` interval lengths after the anchor and ends right before interval `i + 1` starts.
    Events without a time, before the anchor, or after the last interval are ignored.

    Codes whose events never carry a numeric value are presence codes, and their value in an interval is the number of events.
    For all other codes, events without a numeric value are ignored, and the numeric values within an interval are aggregated with `aggregation_strategy`.
    An interval without events of a code is missing (`NaN`).

    Args:
        events: The events, as a :class:`~pandas.DataFrame`, a :class:`pyarrow.Table`, a DuckDB relation, or the path(s) of parquet files, which may contain glob patterns.
        obs: Observation annotations with one row per subject, indexed by the subject id.
            Subjects without events are kept, and events of subjects not in `obs` are ignored.
            If not specified, the subjects are the ones in `events`, sorted by their id.
        anchor: Column of `obs` with the time at which each subject's first interval starts.
            If not specified, the first interval starts at the subject's first event, and this time is stored in `obs["anchor_time"]`.
            If `time` holds durations (:class:`~pandas.Timedelta`), these are the times since the anchor already, and `anchor` must not be specified.
        interval_length_number: Numeric value of the length of one interval.
        interval_length_unit: Unit of the interval length, needs to be a unit of :class:`pandas.Timedelta`.
        num_intervals: Number of intervals.
            If not specified, the intervals extend up to the latest event.
        codes: The codes to use as variables, in this order.
            If not specified, all codes of events with a time are used, sorted.
        aggregation_strategy: Strategy to use when aggregating the numeric values of a code within one interval.
            `"last"` and `"first"` keep the value of the latest and the earliest event.
        sparse: Whether to store the data as a `sparse.COO` array with `NaN` as fill value instead of a :class:`numpy.ndarray`.
        layer: The layer to store the data in.
            If not specified, uses `X`.
        subject_id: Column of `events` with the subject id.
        time: Column of `events` with the time of the event, either a timestamp or a duration.
        code: Column of `events` with the code.
        numeric_value: Column of `events` with the numeric value.
            If `events` has no such column, all codes are presence codes.

    Returns:
        An :class:`~ehrdata.EHRData` object of shape subjects × codes × intervals.
        `.var` holds the number of binned events per code in `n_events`, and `.tem` the start and end of each interval relative to the anchor.

    Examples:
        >>> import ehrdata as ed
        >>> import pandas as pd
        >>> events = pd.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1, 1, 2],
        ...         "time": pd.to_datetime(
        ...             [
        ...                 "2020-01-01 08:00",
        ...                 "2020-01-01 08:40",
        ...                 "2020-01-01 09:10",
        ...                 "2020-01-01 10:30",
        ...                 "2021-05-03 12:00",
        ...             ]
        ...         ),
        ...         "code": ["heart_rate", "heart_rate", "aspirin", "heart_rate", "heart_rate"],
        ...         "numeric_value": [80.0, 95.0, None, 72.0, 64.0],
        ...     }
        ... )
        >>> edata = ed.io.from_events(events, interval_length_number=1, interval_length_unit="h")
        >>> edata
        EHRData object with n_obs × n_vars × n_t = 2 × 2 × 3
            obs: 'anchor_time'
            var: 'n_events'
            tem: '0', '1', '2'
            shape of .X: (2, 2, 3)
        >>> edata.X[0]
        array([[nan,  1., nan],
               [95., nan, 72.]])
    """
    from ehrdata import EHRData

    if aggregation_strategy not in EVENT_AGGREGATION_STRATEGIES:
        msg = f"aggregation_strategy must be one of {EVENT_AGGREGATION_STRATEGIES}."
        raise ValueError(msg)
    # An exact integer, as DuckDB INTERVAL arithmetic would count 30 days or more as calendar months.
    interval_us = pd.Timedelta(interval_length_number, interval_length_unit) // pd.Timedelta(1, "us")
    if interval_us <= 0:
        msg = "The interval length must be positive."
        raise ValueError(msg)

    con = duckdb.connect()
    is_relative = _register_events(
        con, events, subject_id=subject_id, time=time, code=code, numeric_value=numeric_value
    )

    if obs is None:
        if anchor is not None:
            msg = "anchor is a column of obs, so obs must be specified."
            raise ValueError(msg)
        subject_ids = con.sql(
            f"SELECT DISTINCT {_quote(subject_id)} FROM raw_events WHERE {_quote(subject_id)} IS NOT NULL ORDER BY 1"
        ).fetchnumpy()[subject_id]
        obs = pd.DataFrame(index=pd.Index(subject_ids).astype(str))
    else:
        obs = obs.copy()
        obs.index = obs.index.astype(str)
        if not obs.index.is_unique:
            msg = "The index of obs must be unique."
            raise ValueError(msg)

    if is_relative:
        if anchor is not None:
            msg = "The times are durations since the anchor already, so anchor must not be specified."
            raise ValueError(msg)
        anchor_us = pd.Series(0, index=obs.index, dtype="Int64")
    elif anchor is not None:
        anchor_us = _to_epoch_us(obs[anchor])
    else:
        first_event_us = con.sql("SELECT subject_id, MIN(t_us) AS t_us FROM events GROUP BY subject_id").df()
        anchor_us = first_event_us.set_index("subject_id")["t_us"].astype("Int64").reindex(obs.index)
        anchor_time = pd.to_datetime(anchor_us, unit="us")
        obs[ANCHOR_TIME_KEY] = anchor_time.map(str).mask(anchor_time.isna())

    subjects = pd.DataFrame({"subject_id": obs.index, "obs_idx": np.arange(len(obs)), "anchor_us": anchor_us.array})
    con.register("subjects", subjects)

    code_kinds = con.sql(
        """
        SELECT
            code,
            COUNT(numeric_value) = 0 AS is_presence,
            bool_or(t_us IS NOT NULL AND subject_id IN (SELECT subject_id FROM subjects)) AS is_timed
        FROM events
        WHERE code IS NOT NULL
        GROUP BY code
        """
    ).df()
    if codes is None:
        var_names = pd.Index(np.sort(code_kinds.loc[code_kinds["is_timed"], "code"].to_numpy()), dtype=str)
    else:
        var_names = pd.Index(list(codes), dtype=str)
        if not var_names.is_unique:
            msg = "codes must be unique."
            raise ValueError(msg)
    is_presence = code_kinds.set_index("code")["is_presence"].reindex(var_names, fill_value=True)
    con.register(
        "variables",
        pd.DataFrame({"code": var_names, "var_idx": np.arange(len(var_names)), "is_presence": is_presence.to_numpy()}),
    )

    if aggregation_strategy == "last":
        value_aggregation = "arg_max(numeric_value, t_us)"
    elif aggregation_strategy == "first":
        value_aggregation = "arg_min(numeric_value, t_us)"
    else:
        value_aggregation = f"{AGGREGATION_STRATEGY_KEY[aggregation_strategy]}(numeric_value)"
    interval_filter = (
        "" if num_intervals is None else f"AND (e.t_us - s.anchor_us) // {interval_us} < {int(num_intervals)}"
    )
    cells = con.sql(
        f"""
        WITH binned AS (
            SELECT s.obs_idx, v.var_idx, v.is_presence, (e.t_us - s.anchor_us) // {interval_us} AS interval_idx, e.t_us, e.numeric_value
            FROM events e
            JOIN subjects s ON e.subject_id = s.subject_id
            JOIN variables v ON e.code = v.code
            WHERE e.t_us >= s.anchor_us AND (v.is_presence OR e.numeric_value IS NOT NULL) {interval_filter}
        )
        SELECT
            obs_idx,
            var_idx,
            interval_idx,
            CASE WHEN is_presence THEN COUNT(*) ELSE {value_aggregation} END::DOUBLE AS value,
            COUNT(*) AS n_events
        FROM binned
        GROUP BY obs_idx, var_idx, interval_idx, is_presence
        """
    ).fetchnumpy()

    if num_intervals is None:
        num_intervals = int(cells["interval_idx"].max()) + 1 if len(cells["interval_idx"]) else 0
    shape = (len(obs), len(var_names), num_intervals)
    coords = np.stack([cells["obs_idx"], cells["var_idx"], cells["interval_idx"]])
    if sparse:
        X = COO(coords, cells["value"], shape=shape, fill_value=np.nan, has_duplicates=False)
    else:
        X = np.full(shape, np.nan)
        X[tuple(coords)] = cells["value"]

    var = pd.DataFrame(
        {"n_events": np.bincount(cells["var_idx"], weights=cells["n_events"], minlength=len(var_names)).astype(int)},
        index=var_names,
    )
    tem = _generate_timedeltas(interval_length_number, interval_length_unit, num_intervals).set_index("interval_step")
    tem.index = tem.index.astype(str)
    tem = tem.astype(str)

    return (
        EHRData(layers={layer: X}, obs=obs, var=var, tem=tem)
        if layer is not None
        else EHRData(X=X, obs=obs, var=var, tem=tem)
    )


def _quote(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _register_events(
    con: duckdb.DuckDBPyConnection,
    events: pd.DataFrame | pa.Table | duckdb.DuckDBPyRelation | str | PathLike | Sequence[str | PathLike],
    *,
    subject_id: str,
    time: str,
    code: str,
    numeric_value: str,
) -> bool:
    """Register the events as view `events` with the columns subject_id, t_us, code and numeric_value in `con`.

    Returns whether the times are durations rather than timestamps.
    """
    if isinstance(events, str | PathLike):
        events = [events]
    if isinstance(events, Sequence):
        con.read_parquet([str(path) for path in events], union_by_name=True).create_view("raw_events")
    elif isinstance(events, duckdb.DuckDBPyRelation):
        con.register("raw_events", events.df())
    else:
        con.register("raw_events", events)

    columns = dict(zip(con.table("raw_events").columns, con.table("raw_events").types, strict=True))
    missing = [column for column in (subject_id, time, code) if column not in columns]
    if missing:
        msg = f"Columns {missing} not found in events."
        raise ValueError(msg)
    numeric_value_expression = f"{_quote(numeric_value)}::DOUBLE" if numeric_value in columns else "NULL::DOUBLE"
    con.execute(
        f"""
        CREATE VIEW events AS
        SELECT
            {_quote(subject_id)}::VARCHAR AS subject_id,
            epoch_us({_quote(time)}) AS t_us,
            {_quote(code)}::VARCHAR AS code,
            {numeric_value_expression} AS numeric_value
        FROM raw_events
        """
    )
    return str(columns[time]) == "INTERVAL"


def _to_epoch_us(times: pd.Series) -> pd.Series:
    times = pd.to_datetime(times)
    epoch_us = pd.Series(times.to_numpy().astype("datetime64[us]").view(np.int64), index=times.index, dtype="Int64")
    return epoch_us.mask(times.isna())
