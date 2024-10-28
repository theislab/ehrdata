import duckdb
import pytest

import ehrdata as ed
from ehrdata.io.omop import register_omop_to_db_connection


def test_register_omop_to_db_connection():
    register_omop_to_db_connection(path="tests/data/toy_omop/vanilla", backend_handle=duckdb.connect(), source="csv")


@pytest.mark.parametrize(
    "observation_table", ["person", "person_cohort", "person_observation_period", "person_visit_occurrence"]
)
def test_setup_obs(omop_connection_vanilla, observation_table):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table=observation_table)
    assert isinstance(edata, ed.EHRData)


@pytest.mark.parametrize("observation_table", ["perso"])
def test_setup_obs_unknown_observation_table_argument(omop_connection_vanilla, observation_table):
    con = omop_connection_vanilla
    with pytest.raises(ValueError):
        ed.io.omop.setup_obs(backend_handle=con, observation_table=observation_table)


def test_setup_obs_person():
    # check precise expected table
    con = duckdb.connect()
    register_omop_to_db_connection(path="../data/toy_omop/vanilla", backend_handle=con, source="csv")
    con.close()


def test_setup_var_measurement_startdate_fixed():
    # check precise expected table
    pass


def test_setup_var_measurement_startdate_observation_period():
    # check precise expected table
    pass


def test_setup_var_observation_startdate_fixed():
    # check precise expected table
    pass


def test_setup_var_observation_startdate_observation_period():
    # check precise expected table
    pass
