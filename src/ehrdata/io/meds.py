from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import pandas as pd

from ehrdata.core.constants import ANCHOR_TIME_KEY
from ehrdata.io.events import from_events
from ehrdata.io.pandas import to_pandas

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from os import PathLike

    from ehrdata import EHRData

MEDS_VERSION = "0.4.1"
SPLIT_KEY = "split"
STATIC_CODES_KEY = "meds_static_codes"


def read_meds(
    root: str | PathLike,
    *,
    split: str | Sequence[str] | None = None,
    codes: Collection[str] | None = None,
    **binning: Any,
) -> EHRData:
    """Read a `Medical Event Data Standard (MEDS) <https://medical-event-data-standard.github.io>`__ dataset into an :class:`~ehrdata.EHRData` object with a time axis.

    The events are binned into intervals with :func:`~ehrdata.io.from_events`, starting at each subject's first event.
    To start the intervals at another time, such as an admission, pass the data files and an `obs` with this time to :func:`~ehrdata.io.from_events` instead.
    Static events, which have no time, become columns of `obs`.
    These hold the numeric value of the code, or whether the subject has the code if the code has no numeric values.
    Slashes in these codes become underscores in the column names, and `uns["meds_static_codes"]` maps the column names to the codes.
    The split of each subject is stored in `obs["split"]`, and the description of each code, if any, in `var["description"]`.

    Args:
        root: The root directory of the MEDS dataset.
        split: The split(s) of the subjects to read, such as `"train"`, `"tuning"`, or `"held_out"`.
            If not specified, all subjects are read.
        codes: The codes to use as variables, in this order.
            If not specified, all codes of events with a time are used, sorted.
        **binning: Passed to :func:`~ehrdata.io.from_events`, such as `interval_length_number`, `interval_length_unit`, `num_intervals`, `aggregation_strategy`, `sparse`, and `layer`.

    Returns:
        An :class:`~ehrdata.EHRData` object of shape subjects × codes × intervals.

    Examples:
        >>> import ehrdata as ed
        >>> import pandas as pd
        >>> events = pd.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 2],
        ...         "time": pd.to_datetime(["2020-01-01 08:00", "2020-01-01 09:30", "2021-05-03 12:00"]),
        ...         "code": ["heart_rate", "heart_rate", "heart_rate"],
        ...         "numeric_value": [80.0, 95.0, 64.0],
        ...     }
        ... )
        >>> edata = ed.io.from_events(events)
        >>> edata.obs["split"] = ["train", "held_out"]
        >>> ed.io.write_meds(edata, "meds_dataset")
        >>> edata = ed.io.read_meds("meds_dataset", split="train", interval_length_number=1, interval_length_unit="h")
        >>> edata
        EHRData object with n_obs × n_vars × n_t = 1 × 1 × 2
            obs: 'split', 'anchor_time'
            var: 'n_events'
            tem: '0', '1'
            shape of .X: (1, 1, 2)
    """
    root = Path(root)
    data = str(root / "data" / "**" / "*.parquet")
    con = duckdb.connect()
    con.read_parquet(data, union_by_name=True).create_view("data")

    subject_ids = con.sql("SELECT DISTINCT subject_id FROM data ORDER BY 1").fetchnumpy()["subject_id"]
    obs = pd.DataFrame(index=pd.Index(subject_ids).astype(str))
    splits_path = root / "metadata" / "subject_splits.parquet"
    if splits_path.exists():
        splits = con.read_parquet(str(splits_path)).df()
        obs[SPLIT_KEY] = splits.set_index(splits["subject_id"].astype(str))[SPLIT_KEY]
    if split is not None:
        if SPLIT_KEY not in obs.columns:
            msg = f"{splits_path} does not exist, so the subjects cannot be selected by split."
            raise ValueError(msg)
        obs = obs.loc[obs[SPLIT_KEY].isin([split] if isinstance(split, str) else split)].copy()

    static = con.sql("SELECT * REPLACE (INTERVAL 0 SECOND AS time) FROM data WHERE time IS NULL").df()
    static_codes = {}
    if len(static):
        static_edata = from_events(
            static, obs=obs, num_intervals=1, aggregation_strategy=binning.get("aggregation_strategy", "last")
        )
        numeric_codes = (
            set(static.loc[static["numeric_value"].notna(), "code"]) if "numeric_value" in static.columns else set()
        )
        for code, values in zip(static_edata.var_names, static_edata.X[:, :, 0].T, strict=True):
            column = code.replace("/", "_")
            if column in obs.columns:
                msg = f"The static code {code!r} would overwrite the column {column!r} of obs."
                raise ValueError(msg)
            static_codes[column] = code
            obs[column] = values if code in numeric_codes else ~np.isnan(values)

    edata = from_events(data, obs=obs, codes=codes, **binning)
    if static_codes:
        edata.uns[STATIC_CODES_KEY] = static_codes

    codes_path = root / "metadata" / "codes.parquet"
    if codes_path.exists():
        code_metadata = con.read_parquet(str(codes_path)).df().drop_duplicates("code").set_index("code")
        description = code_metadata.get("description", pd.Series()).reindex(edata.var_names)
        if description.notna().any():
            edata.var["description"] = description.to_numpy()

    return edata


def write_meds(
    edata: EHRData,
    root: str | PathLike,
    *,
    layer: str | None = None,
    anchor: str = ANCHOR_TIME_KEY,
) -> None:
    """Write an :class:`~ehrdata.EHRData` object with a time axis as a `Medical Event Data Standard (MEDS) <https://medical-event-data-standard.github.io>`__ dataset.

    Every value that is not missing becomes an event at the start of its interval, as given by `.tem["interval_start_offset"]`, which :func:`~ehrdata.io.from_events` creates.
    The time of the event is the subject's anchor time in `obs[anchor]` plus the start of the interval.
    If `obs` has no such column, the time is the start of the interval after 1970-01-01.
    Numeric and boolean columns of `obs` become static events, which have no time, and `obs["split"]` becomes the split of each subject.
    The codes of the static events are the column names, or the codes in `uns["meds_static_codes"]` for the columns that :func:`~ehrdata.io.read_meds` created.

    Args:
        edata: Central data object.
            Its `obs_names` must be integers, as MEDS identifies subjects by integers.
        root: The root directory of the MEDS dataset.
            It must not exist yet or be empty.
        layer: The layer to write.
            If not specified, uses `X`.
        anchor: Column of `obs` with the time at which each subject's first interval starts.

    Examples:
        >>> import ehrdata as ed
        >>> import pandas as pd
        >>> events = pd.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 2],
        ...         "time": pd.to_datetime(["2020-01-01 08:00", "2020-01-01 09:30", "2021-05-03 12:00"]),
        ...         "code": ["heart_rate", "heart_rate", "heart_rate"],
        ...         "numeric_value": [80.0, 95.0, 64.0],
        ...     }
        ... )
        >>> edata = ed.io.from_events(events)
        >>> ed.io.write_meds(edata, "heart_rate_meds")
    """
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        msg = f"{root} is not empty."
        raise FileExistsError(msg)
    try:
        subject_ids = edata.obs_names.astype(np.int64)
    except ValueError:
        msg = "MEDS identifies subjects by integers, but the obs_names are not integers."
        raise ValueError(msg) from None
    if "interval_start_offset" not in edata.tem.columns:
        msg = "The start of each interval must be in tem['interval_start_offset']."
        raise ValueError(msg)

    values = to_pandas(edata, layer=layer, format="long").dropna(subset=["value"])
    interval_start = pd.to_timedelta(edata.tem["interval_start_offset"]).reindex(values["time"]).to_numpy()
    if anchor in edata.obs.columns:
        anchor_time = pd.to_datetime(edata.obs[anchor]).reindex(values["observation_id"]).to_numpy()
        if pd.isna(anchor_time).any():
            msg = f"Subjects with values have no anchor time in obs[{anchor!r}]."
            raise ValueError(msg)
    else:
        anchor_time = np.datetime64(0, "us")
    events = [
        pd.DataFrame(
            {
                "subject_id": values["observation_id"].astype(np.int64).to_numpy(),
                "time": anchor_time + interval_start,
                "code": values["variable"].to_numpy(),
                "numeric_value": values["value"].to_numpy(),
            }
        )
    ]
    static_codes = []
    for column in edata.obs.columns.drop([anchor, SPLIT_KEY], errors="ignore"):
        code = edata.uns.get(STATIC_CODES_KEY, {}).get(column, column)
        if pd.api.types.is_bool_dtype(edata.obs[column]):
            has_event, numeric_value = edata.obs[column].to_numpy(), np.nan
        elif pd.api.types.is_numeric_dtype(edata.obs[column]):
            has_event = edata.obs[column].notna().to_numpy()
            numeric_value = edata.obs[column].to_numpy()[has_event]
        else:
            continue
        static_codes.append(code)
        events.append(
            pd.DataFrame(
                {"subject_id": subject_ids[has_event], "time": pd.NaT, "code": code, "numeric_value": numeric_value}
            )
        )

    (root / "data").mkdir(parents=True)
    (root / "metadata").mkdir()
    con = duckdb.connect()
    con.register("events", pd.concat(events, ignore_index=True))
    con.sql(
        """
        SELECT subject_id::BIGINT AS subject_id, time::TIMESTAMP AS time, code::VARCHAR AS code, numeric_value::FLOAT AS numeric_value
        FROM events
        ORDER BY subject_id, time NULLS FIRST, code
        """
    ).write_parquet(str(root / "data" / "0.parquet"))

    description = edata.var["description"].to_numpy() if "description" in edata.var.columns else None
    code_metadata = pd.concat(
        [
            pd.DataFrame({"code": edata.var_names, "description": description}),
            pd.DataFrame({"code": static_codes, "description": None}),
        ]
    ).drop_duplicates("code")
    con.register("code_metadata", code_metadata)
    con.sql(
        "SELECT code::VARCHAR AS code, description::VARCHAR AS description, NULL::VARCHAR[] AS parent_codes FROM code_metadata"
    ).write_parquet(str(root / "metadata" / "codes.parquet"))

    if SPLIT_KEY in edata.obs.columns:
        con.register("splits", pd.DataFrame({"subject_id": subject_ids, SPLIT_KEY: edata.obs[SPLIT_KEY].to_numpy()}))
        con.sql(
            f"SELECT subject_id::BIGINT AS subject_id, {SPLIT_KEY}::VARCHAR AS {SPLIT_KEY} FROM splits WHERE {SPLIT_KEY} IS NOT NULL"
        ).write_parquet(str(root / "metadata" / "subject_splits.parquet"))

    dataset_metadata = {
        "etl_name": "ehrdata",
        "etl_version": version("ehrdata"),
        "meds_version": MEDS_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (root / "metadata" / "dataset.json").write_text(json.dumps(dataset_metadata, indent=2))
