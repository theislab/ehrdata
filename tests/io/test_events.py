import duckdb
import numpy as np
import pandas as pd
import pytest
from sparse import COO

from ehrdata.core.constants import ANCHOR_TIME_KEY
from ehrdata.io import from_events, read_h5ed, write_h5ed

T0 = pd.Timestamp("2020-01-01")


def _events(rows, columns=("subject_id", "time", "code", "numeric_value")) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns))


@pytest.fixture
def events() -> pd.DataFrame:
    hour = pd.Timedelta(1, "h")
    return _events(
        [
            (1, T0, "a", 1.0),
            (1, T0 + 0.25 * hour, "dx", None),
            (1, T0 + 0.5 * hour, "a", 3.0),
            (1, T0 + 0.5 * hour, "dx", None),
            (1, T0 + 0.75 * hour, "a", 8.0),
            (1, T0 + 1.2 * hour, "b", 4.0),
            (1, T0 + 1.5 * hour, "b", None),
            (1, T0 + 2 * hour, "a", 5.0),
            (1, T0 + 2.5 * hour, "b", None),
            (2, T0 + 24 * hour, "b", 7.0),
        ]
    )


@pytest.mark.parametrize(
    ("aggregation_strategy", "a_interval_0", "a_interval_2"),
    [
        ("last", 8.0, 5.0),
        ("first", 1.0, 5.0),
        ("mean", 4.0, 5.0),
        ("median", 3.0, 5.0),
        ("min", 1.0, 5.0),
        ("max", 8.0, 5.0),
        ("sum", 12.0, 5.0),
        ("count", 3.0, 1.0),
    ],
)
def test_from_events_aggregation(events, aggregation_strategy, a_interval_0, a_interval_2):
    edata = from_events(events, aggregation_strategy=aggregation_strategy)

    b = 1.0 if aggregation_strategy == "count" else 4.0
    expected = np.array(
        [
            [[a_interval_0, np.nan, a_interval_2], [np.nan, b, np.nan], [2.0, np.nan, np.nan]],
            [[np.nan, np.nan, np.nan], [7.0 if aggregation_strategy != "count" else 1.0, np.nan, np.nan], [np.nan] * 3],
        ]
    )
    np.testing.assert_array_equal(edata.X, expected)
    assert list(edata.obs_names) == ["1", "2"]
    assert list(edata.var_names) == ["a", "b", "dx"]
    assert list(edata.var["n_events"]) == [4, 2, 2]
    assert list(edata.obs[ANCHOR_TIME_KEY]) == ["2020-01-01 00:00:00", "2020-01-02 00:00:00"]
    assert list(edata.tem.index) == ["0", "1", "2"]
    assert list(edata.tem["interval_start_offset"]) == [str(pd.Timedelta(i, "h")) for i in range(3)]
    assert list(edata.tem["interval_end_offset"]) == [str(pd.Timedelta(i, "h")) for i in range(1, 4)]


def test_from_events_anchor_from_obs(events):
    events = pd.concat([events, _events([(9, T0, "a", 1.0)])])
    obs = pd.DataFrame(
        {"admission": [T0 + pd.Timedelta(30, "min"), T0, T0], "age": [50, 60, 70]}, index=pd.Index([1, 2, 3])
    )

    edata = from_events(events, obs=obs, anchor="admission", codes=["a"])

    np.testing.assert_array_equal(edata.X[:, 0], [[8.0, 5.0], [np.nan] * 2, [np.nan] * 2])
    assert list(edata.obs_names) == ["1", "2", "3"]
    assert list(edata.obs.columns) == ["admission", "age"]
    assert edata.var.loc["a", "n_events"] == 3


def test_from_events_anchor_missing_for_subject(events):
    obs = pd.DataFrame({"admission": [T0, pd.NaT]}, index=["1", "2"])

    edata = from_events(events, obs=obs, anchor="admission")

    assert np.isnan(edata.X[1]).all()
    assert not np.isnan(edata.X[0]).all()


def test_from_events_relative_times(events):
    events["time"] = events["time"] - T0

    edata = from_events(events)

    assert edata.X.shape == (2, 3, 25)
    np.testing.assert_array_equal(edata.X[0, :, :3], from_events(events.assign(time=events["time"] + T0)).X[0])
    assert np.isnan(edata.X[0, :, 3:]).all()
    assert edata.X[1, 1, 24] == 7.0
    assert ANCHOR_TIME_KEY not in edata.obs.columns


def test_from_events_subjects_without_events(events):
    obs = pd.DataFrame(index=["3", "2", "1"])

    edata = from_events(events, obs=obs)

    assert list(edata.obs_names) == ["3", "2", "1"]
    assert np.isnan(edata.X[0]).all()
    assert pd.isna(edata.obs.loc["3", ANCHOR_TIME_KEY])
    np.testing.assert_array_equal(edata.X[2], from_events(events).X[0])


def test_from_events_codes(events):
    edata = from_events(events, codes=["dx", "a", "never_recorded"])

    assert list(edata.var_names) == ["dx", "a", "never_recorded"]
    assert list(edata.var["n_events"]) == [2, 4, 0]
    np.testing.assert_array_equal(edata.X[0, 0], [2.0, np.nan, np.nan])
    assert np.isnan(edata.X[:, 2]).all()


def test_from_events_without_numeric_value_column(events):
    edata = from_events(events.drop(columns="numeric_value"))

    np.testing.assert_array_equal(edata.X[0], [[3.0, np.nan, 1.0], [np.nan, 2.0, 1.0], [2.0, np.nan, np.nan]])


def test_from_events_ignores_events_without_time(events):
    events = pd.concat([events, _events([(1, pd.NaT, "static", 1.0), (3, pd.NaT, "a", 1.0)])])

    edata = from_events(events)

    assert "static" not in edata.var_names
    assert list(edata.obs_names) == ["1", "2", "3"]
    assert np.isnan(edata.X[2]).all()


@pytest.fixture(params=[False, True], ids=["absolute", "relative"])
def relative(request) -> bool:
    return request.param


@pytest.mark.parametrize(
    ("interval_length_number", "interval_length_unit", "times", "expected_interval"),
    [
        (1, "h", [T0 + pd.Timedelta(1, "h")], 1),
        (1, "h", [T0 + pd.Timedelta(1, "h") - pd.Timedelta(1, "us")], 0),
        (30, "D", [pd.Timestamp("2020-01-31")], 1),
        (30, "D", [pd.Timestamp("2020-01-30 23:59:59.999999")], 0),
        (30, "D", [pd.Timestamp("2020-03-01")], 2),
        (365, "D", [pd.Timestamp("2020-12-31")], 1),
        (365, "D", [pd.Timestamp("2020-12-30 23:59:59")], 0),
    ],
)
def test_from_events_interval_boundaries(
    interval_length_number, interval_length_unit, times, expected_interval, relative
):
    events = _events([(1, T0, "anchor", None)] + [(1, time, "a", 1.0) for time in times])
    if relative:
        events["time"] = events["time"] - T0

    edata = from_events(
        events, interval_length_number=interval_length_number, interval_length_unit=interval_length_unit, codes=["a"]
    )

    assert edata.n_t == expected_interval + 1
    assert edata.X[0, 0, expected_interval] == 1.0


def test_from_events_num_intervals(events):
    edata = from_events(events, num_intervals=2)
    np.testing.assert_array_equal(edata.X, from_events(events).X[:, :, :2])
    assert edata.var.loc["a", "n_events"] == 3

    edata = from_events(events, num_intervals=5)
    assert edata.n_t == 5
    assert len(edata.tem) == 5
    assert np.isnan(edata.X[:, :, 3:]).all()


def test_from_events_sparse(events):
    dense = from_events(events)
    edata = from_events(events, sparse=True, layer="binned")

    assert edata.X is None
    assert isinstance(edata.layers["binned"], COO)
    assert np.isnan(edata.layers["binned"].fill_value)
    np.testing.assert_array_equal(edata.layers["binned"].todense(), dense.X)


def test_from_events_input_types(events, tmp_path):
    expected = from_events(events).X
    duckdb.from_df(events.iloc[:5]).write_parquet(str(tmp_path / "part_0.parquet"))
    duckdb.from_df(events.iloc[5:]).write_parquet(str(tmp_path / "part_1.parquet"))

    for parquet in [tmp_path / "*.parquet", str(tmp_path / "*.parquet"), sorted(tmp_path.glob("*.parquet"))]:
        np.testing.assert_array_equal(from_events(parquet).X, expected)

    pa = pytest.importorskip("pyarrow")
    np.testing.assert_array_equal(from_events(pa.Table.from_pandas(events)).X, expected)
    np.testing.assert_array_equal(from_events(duckdb.connect().from_df(events)).X, expected)


def test_from_events_column_names(events):
    renamed = events.rename(
        columns={"subject_id": "person", "time": "charttime", "code": "item", "numeric_value": "value"}
    )

    edata = from_events(renamed, subject_id="person", time="charttime", code="item", numeric_value="value")

    np.testing.assert_array_equal(edata.X, from_events(events).X)


@pytest.mark.parametrize("sparse", [False, True])
def test_from_events_h5ed_roundtrip(events, tmp_path, sparse):
    edata = from_events(events, sparse=sparse)

    write_h5ed(edata, tmp_path / "events.h5ed")
    read = read_h5ed(tmp_path / "events.h5ed")

    np.testing.assert_array_equal(read.X.todense() if sparse else read.X, edata.X.todense() if sparse else edata.X)
    pd.testing.assert_frame_equal(read.var, edata.var)
    pd.testing.assert_frame_equal(read.tem, edata.tem, check_dtype=False, check_categorical=False)
    assert list(read.obs[ANCHOR_TIME_KEY].astype(str)) == list(edata.obs[ANCHOR_TIME_KEY])
    np.testing.assert_array_equal(from_events(events, obs=read.obs, anchor=ANCHOR_TIME_KEY).X, from_events(events).X)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"aggregation_strategy": "mode"}, "aggregation_strategy"),
        ({"interval_length_number": 0}, "positive"),
        ({"anchor": "admission"}, "obs must be specified"),
        ({"codes": ["a", "a"]}, "unique"),
        ({"time": "charttime"}, "charttime"),
    ],
)
def test_from_events_invalid_arguments(events, kwargs, match):
    with pytest.raises(ValueError, match=match):
        from_events(events, **kwargs)


def test_from_events_anchor_with_relative_times(events):
    events["time"] = events["time"] - T0
    obs = pd.DataFrame({"admission": [T0, T0]}, index=["1", "2"])

    with pytest.raises(ValueError, match="durations"):
        from_events(events, obs=obs, anchor="admission")


def test_from_events_obs_index_not_unique(events):
    with pytest.raises(ValueError, match="unique"):
        from_events(events, obs=pd.DataFrame(index=["1", "1"]))
