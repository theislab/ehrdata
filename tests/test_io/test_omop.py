import re

import pytest

import ehrdata as ed

# def test_register_omop_to_db_connection():
#     register_omop_to_db_connection(path="tests/data/toy_omop/vanilla", backend_handle=duckdb.connect(), source="csv")


# TODO: add test for death argument
@pytest.mark.parametrize(
    "observation_table, expected_length, expected_obs_num_columns",
    [
        ("person", 4, 18),
        ("person_cohort", 3, 22),
        ("person_observation_period", 3, 23),
        ("person_visit_occurrence", 3, 35),
    ],
)
def test_setup_obs(omop_connection_vanilla, observation_table, expected_length, expected_obs_num_columns):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table=observation_table)
    assert isinstance(edata, ed.EHRData)

    # 4 persons, only 3 are in cohort, or have observation period, or visit occurrence
    assert len(edata) == expected_length
    assert edata.obs.shape[1] == expected_obs_num_columns


def test_setup_obs_invalid_backend_handle_argument():
    with pytest.raises(ValueError, match="backend_handle must be a DuckDB connection."):
        ed.io.omop.setup_obs(backend_handle="not_a_con", observation_table="person")


def test_setup_obs_invalid_observation_table_argument(omop_connection_vanilla):
    con = omop_connection_vanilla
    with pytest.raises(
        ValueError,
        match=re.escape(
            "observation_table must be one of ['person', 'person_cohort', 'person_observation_period', 'person_visit_occurrence']."
        ),
    ):
        ed.io.omop.setup_obs(backend_handle=con, observation_table="perso")


def test_setup_variables_measurement_startdate_fixed(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person")
    ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        tables=["measurement"],
        start_time="2100-01-01",
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=31,
    )
    # check precise expected table
    assert edata.vars.shape[1] == 8


def test_setup_var_measurement_startdate_observation_period():
    # check precise expected table
    pass


def test_setup_var_observation_startdate_fixed():
    # check precise expected table
    pass


def test_setup_var_observation_startdate_observation_period():
    # check precise expected table
    pass
