import re

import pytest

import ehrdata as ed


@pytest.mark.parametrize(
    "observation_table, death_table, expected_length, expected_obs_num_columns",
    [
        ("person", False, 4, 18),
        ("person", True, 4, 24),
        ("person_cohort", False, 3, 22),
        ("person_cohort", True, 3, 28),
        ("person_observation_period", False, 3, 23),
        ("person_observation_period", True, 3, 29),
        ("person_visit_occurrence", False, 3, 35),
        ("person_visit_occurrence", True, 3, 41),
    ],
)
def test_setup_obs(omop_connection_vanilla, observation_table, death_table, expected_length, expected_obs_num_columns):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table=observation_table, death_table=death_table)
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


@pytest.mark.parametrize(
    "observation_table",
    ["person_cohort", "person_observation_period", "person_visit_occurrence"],
)
@pytest.mark.parametrize(
    "data_tables",
    [["measurement"], ["observation"]],
)
@pytest.mark.parametrize(
    "data_field_to_keep",
    [["value_as_number"], ["value_as_concept_id"]],
)
def test_setup_variables(omop_connection_vanilla, observation_table, data_tables, data_field_to_keep):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table=observation_table)
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=data_tables,
        data_field_to_keep=data_field_to_keep,
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=30,
    )

    assert isinstance(edata, ed.EHRData)
    assert edata.n_obs == 3
    assert edata.n_vars == 2
    assert edata.r.shape[2] == 30
