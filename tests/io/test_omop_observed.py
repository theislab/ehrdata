"""The observed-rows path of setup_variables: concept sets, per-variable aggregation, value ranges, unit conversions,
link_by, cut_at_end, sparse output, and observations read from .obs."""

import duckdb
import numpy as np
import pandas as pd
import pytest
import sparse

import ehrdata as ed


@pytest.mark.parametrize("observation_table", ["person", "person_observation_period", "person_visit_occurrence"])
@pytest.mark.parametrize("data_table", ["measurement", "observation"])
@pytest.mark.parametrize("aggregation_strategy", ["last", "first", "median", "max", "count"])
@pytest.mark.parametrize("time_precision", ["date", "datetime"])
def test_observed_path_equals_default_path(
    omop_connection_vanilla, observation_table, data_table, aggregation_strategy, time_precision
):
    """sparse=True only changes the storage: the values are those of the default path."""
    con = omop_connection_vanilla
    field = "value_as_number"
    kwargs = {
        "backend_handle": con,
        "data_tables": [data_table],
        "data_field_to_keep": {data_table: field},
        "interval_length_number": 1,
        "interval_length_unit": "day",
        "time_precision": time_precision,
        "num_intervals": 4,
        "aggregation_strategy": aggregation_strategy,
    }
    try:
        dense = ed.io.omop.setup_variables(ed.io.omop.setup_obs(con, observation_table), **kwargs)
    except (NotImplementedError, ValueError) as error:
        with pytest.raises(type(error)):
            ed.io.omop.setup_variables(ed.io.omop.setup_obs(con, observation_table), sparse=True, **kwargs)
        return
    observed = ed.io.omop.setup_variables(ed.io.omop.setup_obs(con, observation_table), sparse=True, **kwargs)
    assert isinstance(observed.X, sparse.COO)
    assert observed.shape == dense.shape
    np.testing.assert_array_equal(
        observed.var["data_table_concept_id"].to_numpy(), dense.var["data_table_concept_id"].to_numpy()
    )
    np.testing.assert_allclose(observed.X.todense(), dense.X.astype(float), equal_nan=True)


@pytest.fixture
def two_visits_one_person():
    """One person with two visits whose windows overlap, and a second person; values at known hours."""
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE person AS SELECT * FROM (VALUES
            (1, 1980, TIMESTAMP '1980-01-01'), (2, 1990, TIMESTAMP '1990-01-01'))
            t(person_id, year_of_birth, birth_datetime)""")
    con.execute("""
        CREATE TABLE visit_occurrence AS SELECT * FROM (VALUES
            (10, 1, DATE '2020-01-01', TIMESTAMP '2020-01-01 00:00', DATE '2020-01-03', TIMESTAMP '2020-01-03 00:00'),
            (11, 1, DATE '2020-01-01', TIMESTAMP '2020-01-01 02:00', DATE '2020-01-03', TIMESTAMP '2020-01-03 00:00'),
            (20, 2, DATE '2020-01-01', TIMESTAMP '2020-01-01 00:00', DATE '2020-01-03', TIMESTAMP '2020-01-03 00:00'))
            t(visit_occurrence_id, person_id, visit_start_date, visit_start_datetime, visit_end_date, visit_end_datetime)""")
    # concept 100 and 101 form one concept set ("crea"); 200 is urine, summed; 300 has an implausible value
    con.execute("""
        CREATE TABLE measurement AS SELECT * FROM (VALUES
            (1, 10, 100, TIMESTAMP '2020-01-01 00:10', 1.0, 8840),
            (1, 10, 101, TIMESTAMP '2020-01-01 00:40', 3.0, 8840),
            (1, 10, 100, TIMESTAMP '2020-01-01 00:50', 2.0, 8840),
            (1, 11, 100, TIMESTAMP '2020-01-01 02:30', 7.0, 8840),
            (1, 10, 200, TIMESTAMP '2020-01-01 01:05', 50.0, 8587),
            (1, 10, 200, TIMESTAMP '2020-01-01 01:55', 25.0, 8587),
            (2, 20, 300, TIMESTAMP '2020-01-01 00:05', 999.0, 8876),
            (2, 20, 300, TIMESTAMP '2020-01-01 00:15', 80.0, 8876),
            (2, NULL, 300, TIMESTAMP '2020-01-01 01:15', 90.0, 8876))
            t(person_id, visit_occurrence_id, measurement_concept_id, measurement_datetime, value_as_number, unit_concept_id)""")
    con.execute("ALTER TABLE measurement ADD COLUMN measurement_date DATE")
    con.execute("UPDATE measurement SET measurement_date = CAST(measurement_datetime AS DATE)")
    con.execute("ALTER TABLE measurement ADD COLUMN unit_source_value VARCHAR")
    return con


SETS = {"crea": [100, 101], "urine": [200], "map": [300], "absent": [999]}
STRATEGY = {"crea": "median", "urine": "sum", "map": "median", "absent": "median"}


def _setup(con, **kwargs):
    edata = ed.io.omop.setup_obs(con, "person_visit_occurrence")
    return ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep={"measurement": "value_as_number"},
        interval_length_number=1,
        interval_length_unit="h",
        time_precision="datetime",
        num_intervals=4,
        **kwargs,
    )


def test_concept_sets_strategies_and_ranges(two_visits_one_person):
    edata = _setup(
        two_visits_one_person,
        concept_ids=SETS,
        aggregation_strategy=STRATEGY,
        value_range={"map": (20, 200)},
        link_by="visit_occurrence",
        sparse=True,
    )
    assert list(edata.var_names) == list(SETS)
    assert edata.var.loc["crea", "concept_ids"] == "100,101"
    x = edata.X.todense()
    visits = edata.obs["visit_occurrence_id"].to_numpy()
    v10, v11, v20 = (int(np.flatnonzero(visits == v)[0]) for v in (10, 11, 20))
    crea, urine, map_, absent = range(4)
    # the rows of both concepts of the set are pooled: median(1, 3, 2) = 2
    assert x[v10, crea, 0] == 2.0
    assert x[v10, urine, 1] == 75.0
    # visit 11 starts at 02:00, so its 02:30 row is in its first hour; visit 10's rows never reach it
    assert x[v11, crea, 0] == 7.0
    assert np.isnan(x[v11, crea, 2:]).all()
    # 999 is outside the range and dropped, so it does not move the hour's median
    assert x[v20, map_, 0] == 80.0
    # a row without visit_occurrence_id belongs to no visit when linking by visit
    assert np.isnan(x[v20, map_, 1])
    assert np.isnan(x[:, absent]).all()


def test_link_by_person_pools_overlapping_visits(two_visits_one_person):
    edata = _setup(two_visits_one_person, concept_ids=SETS, aggregation_strategy=STRATEGY, sparse=True)
    x = edata.X.todense()
    visits = edata.obs["visit_occurrence_id"].to_numpy()
    v10 = int(np.flatnonzero(visits == 10)[0])
    v20 = int(np.flatnonzero(visits == 20)[0])
    # by person and time, visit 10 also takes visit 11's row at 02:30
    assert x[v10, 0, 2] == 7.0
    # and the visit-less row of person 2 joins visit 20 by its time
    assert x[v20, 2, 1] == 90.0


def test_dense_equals_sparse(two_visits_one_person):
    kwargs = {"concept_ids": SETS, "aggregation_strategy": STRATEGY, "link_by": "visit_occurrence"}
    dense = _setup(two_visits_one_person, **kwargs)
    observed = _setup(two_visits_one_person, sparse=True, **kwargs)
    assert isinstance(dense.X, np.ndarray)
    np.testing.assert_array_equal(dense.X, observed.X.todense())


def test_concept_set_with_mixed_units_raises(two_visits_one_person):
    with pytest.raises(NotImplementedError, match="more than one unit_concept_id"):
        _setup(two_visits_one_person, concept_ids={"mixed": [100, 200]}, aggregation_strategy="median")
    # a single-row strategy keeps one row's unit, so it may mix
    _setup(two_visits_one_person, concept_ids={"mixed": [100, 200]}, aggregation_strategy="last")


def test_strategy_mapping_must_cover_every_variable(two_visits_one_person):
    with pytest.raises(ValueError, match="no strategy"):
        _setup(two_visits_one_person, concept_ids=SETS, aggregation_strategy={"crea": "median"})


def test_link_by_visit_needs_visit_observations(two_visits_one_person):
    edata = ed.io.omop.setup_obs(two_visits_one_person, "person")
    with pytest.raises(ValueError, match="person_visit_occurrence"):
        ed.io.omop.setup_variables(
            edata,
            backend_handle=two_visits_one_person,
            data_tables=["measurement"],
            data_field_to_keep={"measurement": "value_as_number"},
            interval_length_number=1,
            interval_length_unit="h",
            num_intervals=4,
            link_by="visit_occurrence",
        )


def test_long_table_holds_observed_cells_only(two_visits_one_person):
    _setup(
        two_visits_one_person,
        concept_ids=SETS,
        aggregation_strategy=STRATEGY,
        link_by="visit_occurrence",
        instantiate_tensor=False,
    )
    long = two_visits_one_person.sql("SELECT * FROM long_person_timestamp_feature_value_measurement").df()
    # crea h0 and urine h1 of visit 10, crea h0 of visit 11, map h0 and h0 (999 kept without a range) of visit 20
    assert len(long) == 4
    assert set(long.columns) >= {"obs_id", "obs_index", "variable", "var_index", "interval_step", "value_as_number"}
    assert pd.api.types.is_integer_dtype(long["obs_index"])


def test_subset_obs_gets_its_own_rows(two_visits_one_person):
    """Observations filtered after setup_obs get the variables of exactly their rows, in their order."""
    con = two_visits_one_person
    full = _setup(con, concept_ids=SETS, aggregation_strategy=STRATEGY, link_by="visit_occurrence", sparse=True)
    edata = ed.io.omop.setup_obs(con, "person_visit_occurrence")
    subset = edata[edata.obs["visit_occurrence_id"].isin([20, 11]).to_numpy()][::-1].copy()
    subset = ed.io.omop.setup_variables(
        subset,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep={"measurement": "value_as_number"},
        interval_length_number=1,
        interval_length_unit="h",
        time_precision="datetime",
        num_intervals=4,
        concept_ids=SETS,
        aggregation_strategy=STRATEGY,
        link_by="visit_occurrence",
        sparse=True,
    )
    assert list(subset.obs["visit_occurrence_id"]) == [20, 11]
    position = {v: i for i, v in enumerate(full.obs["visit_occurrence_id"])}
    np.testing.assert_array_equal(subset.X.todense(), full.X.todense()[[position[20], position[11]]])


def test_unit_conversions(two_visits_one_person):
    """Rows are converted into the variable's unit before the range and the aggregation."""
    con = two_visits_one_person
    # visit 10's crea rows: 1.0 (unit 8840), 3.0 (8840), 2.0 (8840); recast one of them as umol/L
    con.execute(
        "UPDATE measurement SET value_as_number = 265.2, unit_concept_id = 8749 WHERE measurement_concept_id = 101"
    )
    conversions = pd.DataFrame(
        {
            "variable": ["crea", "crea"],
            "unit_concept_id": [8840, 8749],
            "to_unit_concept_id": [8840, 8840],
            "multiplier": [1.0, 1 / 88.4],
        }
    )
    edata = _setup(
        con,
        concept_ids=SETS,
        aggregation_strategy=STRATEGY,
        link_by="visit_occurrence",
        sparse=True,
        value_range={"crea": (0, 10)},
        unit_conversions=conversions,
    )
    x = edata.X.todense()
    v10 = int(np.flatnonzero(edata.obs["visit_occurrence_id"].to_numpy() == 10)[0])
    assert x[v10, 0, 0] == pytest.approx(2.0), "median(1, 265.2 / 88.4 = 3, 2)"
    report = edata.uns["unit_report_measurement"]
    assert report.loc[report["variable"] == "crea", "unit_concept_id"].tolist() == [8840]


def test_unit_conversions_offset_and_missing_unit(two_visits_one_person):
    con = two_visits_one_person
    # a made-up second unit 9289 with value * 2 + 1
    scaled = pd.DataFrame(
        {
            "variable": ["map", "map"],
            "unit_concept_id": [8876, 9289],
            "to_unit_concept_id": [8876, 8876],
            "multiplier": [1.0, 2.0],
            "offset": [0.0, 1.0],
        }
    )
    con.execute("UPDATE measurement SET unit_concept_id = 9289 WHERE value_as_number = 80")
    edata = _setup(
        con,
        concept_ids=SETS,
        aggregation_strategy=STRATEGY,
        link_by="visit_occurrence",
        sparse=True,
        value_range={"map": (20, 200)},
        unit_conversions=scaled,
    )
    v20 = int(np.flatnonzero(edata.obs["visit_occurrence_id"].to_numpy() == 20)[0])
    assert edata.X.todense()[v20, 2, 0] == 161.0, "80 * 2 + 1; 999 stays out of range"
    with pytest.raises(ValueError, match="no conversion"):
        _setup(con, concept_ids=SETS, aggregation_strategy=STRATEGY, sparse=True, unit_conversions=scaled.iloc[:1])


def test_cut_at_end(two_visits_one_person):
    """A row after the end date of its visit counts only with cut_at_end=False."""
    con = two_visits_one_person
    con.execute(
        "UPDATE visit_occurrence SET visit_end_date = DATE '2020-01-01', "
        "visit_end_datetime = TIMESTAMP '2020-01-01 01:00' WHERE visit_occurrence_id = 10"
    )
    con.execute(
        "INSERT INTO measurement VALUES (1, 10, 200, TIMESTAMP '2020-01-02 00:30', 10.0, 8587, DATE '2020-01-02', NULL)"
    )
    kwargs = {"concept_ids": SETS, "aggregation_strategy": STRATEGY, "link_by": "visit_occurrence", "sparse": True}
    edata = ed.io.omop.setup_obs(con, "person_visit_occurrence")
    common = {
        "backend_handle": con,
        "data_tables": ["measurement"],
        "data_field_to_keep": {"measurement": "value_as_number"},
        "interval_length_number": 1,
        "interval_length_unit": "h",
        "time_precision": "datetime",
        "num_intervals": 30,
    }
    cut = ed.io.omop.setup_variables(edata, **common, **kwargs)
    kept = ed.io.omop.setup_variables(edata, **common, **kwargs, cut_at_end=False)
    v10 = int(np.flatnonzero(edata.obs["visit_occurrence_id"].to_numpy() == 10)[0])
    assert np.isnan(cut.X.todense()[v10, 1, 24])
    assert kept.X.todense()[v10, 1, 24] == 10.0


def test_unit_conversions_null_unit(two_visits_one_person):
    """A row without a unit converts only if the table lists a NULL unit for its variable."""
    con = two_visits_one_person
    con.execute("UPDATE measurement SET unit_concept_id = NULL WHERE value_as_number = 80")
    conversions = pd.DataFrame(
        {"variable": ["map"], "unit_concept_id": [8876], "to_unit_concept_id": [8876], "multiplier": [1.0]}
    )
    with pytest.raises(ValueError, match="no conversion"):
        _setup(con, concept_ids=SETS, aggregation_strategy=STRATEGY, sparse=True, unit_conversions=conversions)
    with_null = pd.concat(
        [
            conversions,
            pd.DataFrame(
                {"variable": ["map"], "unit_concept_id": [None], "to_unit_concept_id": [8876], "multiplier": [1.0]}
            ),
        ]
    )
    edata = _setup(
        con,
        concept_ids=SETS,
        aggregation_strategy=STRATEGY,
        link_by="visit_occurrence",
        sparse=True,
        value_range={"map": (20, 200)},
        unit_conversions=with_null,
    )
    v20 = int(np.flatnonzero(edata.obs["visit_occurrence_id"].to_numpy() == 20)[0])
    assert edata.X.todense()[v20, 2, 0] == 80.0
