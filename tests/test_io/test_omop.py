import re

import numpy as np
import pytest

import ehrdata as ed

# constants for toy_omop/vanilla
VANILLA_PERSONS_WITH_OBSERVATION_TABLE_ENTRY = {
    "person_cohort": 3,
    "person_observation_period": 3,
    "person_visit_occurrence": 3,
}
VANILLA_NUM_CONCEPTS = {
    "measurement": 2,
    "observation": 2,
    "specimen": 2,
    "drug_exposure": 2,
    "condition_occurrence": 2,
}

# constants for setup_variables
# only data_table_concept_id
VAR_DIM_BASE = 1
# number of columns in concept table
NUMBER_COLUMNS_CONCEPT_TABLE = 10
VAR_DIM_FEATURE_INFO = NUMBER_COLUMNS_CONCEPT_TABLE
# number of columns in concept table + number of columns
NUMBER_COLUMNS_FEATURE_REPORT = 4
VAR_DIM_UNIT_INFO = NUMBER_COLUMNS_CONCEPT_TABLE + NUMBER_COLUMNS_FEATURE_REPORT


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


@pytest.mark.parametrize(
    "backend_handle, observation_table, death_table, expected_error",
    [
        ("wrong_type", "person", False, "Expected backend_handle to be of type DuckDBPyConnection."),
        (None, 123, False, "Expected observation_table to be a string."),
        (None, "person", "wrong_type", "Expected death_table to be a boolean."),
    ],
)
def test_setup_obs_illegal_argument_types(
    omop_connection_vanilla,
    backend_handle,
    observation_table,
    death_table,
    expected_error,
):
    with pytest.raises(TypeError, match=expected_error):
        ed.io.omop.setup_obs(
            backend_handle=backend_handle or omop_connection_vanilla,
            observation_table=observation_table,
            death_table=death_table,
        )


def test_setup_obs_invalid_observation_table_value(omop_connection_vanilla):
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
# test 1 field from table, and is_present encoding
@pytest.mark.parametrize(
    "data_tables,data_field_to_keep,target_r",
    [
        (
            ["measurement"],
            ["value_as_number"],
            [
                [[np.nan, np.nan, np.nan, np.nan], [18.0, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [20.0, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [22.0, np.nan, np.nan, np.nan]],
            ],
        ),
        (
            ["measurement"],
            ["is_present"],
            [
                [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
                [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
                [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
            ],
        ),
        (
            ["observation"],
            ["value_as_number"],
            [
                [[np.nan, np.nan, np.nan, np.nan], [3, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [4, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [5, np.nan, np.nan, np.nan]],
            ],
        ),
        (
            ["observation"],
            ["is_present"],
            [
                [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
                [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
                [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
            ],
        ),
        (
            ["specimen"],
            ["quantity"],
            [
                [[0.5, np.nan, np.nan, np.nan], [1.5, np.nan, np.nan, np.nan]],
                [[0.5, np.nan, np.nan, np.nan], [1.5, np.nan, np.nan, np.nan]],
                [[0.5, np.nan, np.nan, np.nan], [1.5, np.nan, np.nan, np.nan]],
            ],
        ),
        (
            ["specimen"],
            ["is_present"],
            [
                [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
                [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
                [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "enrich_var_with_feature_info",
    [True, False],
)
@pytest.mark.parametrize(
    "enrich_var_with_unit_info",
    [True, False],
)
def test_setup_variables(
    omop_connection_vanilla,
    observation_table,
    data_tables,
    data_field_to_keep,
    enrich_var_with_feature_info,
    enrich_var_with_unit_info,
    target_r,
):
    num_intervals = 4
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table=observation_table)
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=data_tables,
        data_field_to_keep=data_field_to_keep,
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=num_intervals,
        enrich_var_with_feature_info=enrich_var_with_feature_info,
        enrich_var_with_unit_info=enrich_var_with_unit_info,
    )

    assert isinstance(edata, ed.EHRData)
    assert edata.n_obs == VANILLA_PERSONS_WITH_OBSERVATION_TABLE_ENTRY[observation_table]
    assert edata.n_vars == VANILLA_NUM_CONCEPTS[data_tables[0]]
    assert edata.r.shape[2] == num_intervals
    assert edata.var.shape[1] == VAR_DIM_BASE + (VAR_DIM_FEATURE_INFO if enrich_var_with_feature_info else 0) + (
        VAR_DIM_UNIT_INFO if enrich_var_with_unit_info else 0
    )

    assert np.allclose(edata.r, np.array(target_r), equal_nan=True)


@pytest.mark.parametrize(
    "observation_table",
    ["person_cohort", "person_observation_period", "person_visit_occurrence"],
)
# test 1 field from table, and is_present encoding, with start, end, and interval
@pytest.mark.parametrize(
    "data_tables,data_field_to_keep,keep_date,target_r",
    [
        (
            ["drug_exposure"],
            ["days_supply"],
            "start",
            [
                [[31.0, np.nan, np.nan, np.nan], [31.0, np.nan, np.nan, np.nan]],
                [[31.0, np.nan, np.nan, np.nan], [31.0, np.nan, np.nan, np.nan]],
                [[31.0, np.nan, np.nan, np.nan], [31.0, np.nan, np.nan, np.nan]],
            ],
        ),
        (
            ["drug_exposure"],
            ["days_supply"],
            "end",
            [
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
            ],
        ),
        (
            ["drug_exposure"],
            ["days_supply"],
            "interval",
            [
                [[31.0, 31.0, 31.0, 31.0], [31.0, 31.0, 31.0, 31.0]],
                [[31.0, 31.0, 31.0, 31.0], [31.0, 31.0, 31.0, 31.0]],
                [[31.0, 31.0, 31.0, 31.0], [31.0, 31.0, 31.0, 31.0]],
            ],
        ),
        (
            ["drug_exposure"],
            ["is_present"],
            "start",
            [
                [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
                [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
                [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
            ],
        ),
        (
            ["drug_exposure"],
            ["is_present"],
            "end",
            [
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
            ],
        ),
        (
            ["drug_exposure"],
            ["is_present"],
            "interval",
            [
                [[1, 1, 1, 1], [1, 1, 1, 1]],
                [[1, 1, 1, 1], [1, 1, 1, 1]],
                [[1, 1, 1, 1], [1, 1, 1, 1]],
            ],
        ),
        (
            ["condition_occurrence"],
            ["condition_source_value"],
            "start",
            [
                [[15, np.nan, np.nan, np.nan], [10, np.nan, np.nan, np.nan]],
                [[15, np.nan, np.nan, np.nan], [10, np.nan, np.nan, np.nan]],
                [[15, np.nan, np.nan, np.nan], [10, np.nan, np.nan, np.nan]],
            ],
        ),
        (
            ["condition_occurrence"],
            ["condition_source_value"],
            "end",
            [
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
            ],
        ),
        (
            ["condition_occurrence"],
            ["condition_source_value"],
            "interval",
            [
                [[15, 15, 15, 15], [10, 10, 10, 10]],
                [[15, 15, 15, 15], [10, 10, 10, 10]],
                [[15, 15, 15, 15], [10, 10, 10, 10]],
            ],
        ),
        (
            ["condition_occurrence"],
            ["is_present"],
            "start",
            [
                [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
                [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
                [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
            ],
        ),
        (
            ["condition_occurrence"],
            ["is_present"],
            "end",
            [
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
            ],
        ),
        (
            ["condition_occurrence"],
            ["is_present"],
            "interval",
            [
                [[1, 1, 1, 1], [1, 1, 1, 1]],
                [[1, 1, 1, 1], [1, 1, 1, 1]],
                [[1, 1, 1, 1], [1, 1, 1, 1]],
            ],
        ),
        # (["procedure_occurrence"], ["is_present"]), # TODO: write test file
        # (["device_exposure"], ["is_present"]), # TODO: write test file
        # (["note"], ["is_present"]),
    ],
)
@pytest.mark.parametrize(
    "enrich_var_with_feature_info",
    [False, True],
)
def test_setup_interval_type_variables(
    omop_connection_vanilla,
    observation_table,
    data_tables,
    data_field_to_keep,
    target_r,
    enrich_var_with_feature_info,
    keep_date,
):
    num_intervals = 4
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table=observation_table)
    edata = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        data_tables=data_tables,
        data_field_to_keep=data_field_to_keep,
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=num_intervals,
        enrich_var_with_feature_info=enrich_var_with_feature_info,
        keep_date=keep_date,
    )

    assert isinstance(edata, ed.EHRData)
    assert edata.n_obs == VANILLA_PERSONS_WITH_OBSERVATION_TABLE_ENTRY[observation_table]
    assert edata.n_vars == VANILLA_NUM_CONCEPTS[data_tables[0]]
    assert edata.r.shape[2] == num_intervals
    assert edata.var.shape[1] == VAR_DIM_BASE + (VAR_DIM_FEATURE_INFO if enrich_var_with_feature_info else 0)

    assert np.allclose(edata.r, np.array(target_r), equal_nan=True)


@pytest.mark.parametrize(
    "edata, backend_handle, data_tables, data_field_to_keep, interval_length_number, interval_length_unit, num_intervals, enrich_var_with_feature_info, enrich_var_with_unit_info, expected_error",
    [
        (
            "wrong_type",
            None,
            ["measurement"],
            ["value_as_number"],
            1,
            "day",
            4,
            False,
            False,
            "Expected edata to be of type EHRData.",
        ),
        (
            None,
            "wrong_type",
            ["measurement"],
            ["value_as_number"],
            1,
            "day",
            4,
            False,
            False,
            "Expected backend_handle to be of type DuckDBPyConnection.",
        ),
        (
            None,
            None,
            123,
            ["value_as_number"],
            1,
            "day",
            4,
            False,
            False,
            "Expected data_tables to be a string or Sequence.",
        ),
        (
            None,
            None,
            ["measurement"],
            123,
            1,
            "day",
            4,
            False,
            False,
            "Expected data_field_to_keep to be a string or Sequence.",
        ),
        (
            None,
            None,
            ["measurement"],
            ["value_as_number"],
            "wrong_type",
            "day",
            4,
            False,
            False,
            "Expected interval_length_number to be an integer.",
        ),
        (
            None,
            None,
            ["measurement"],
            ["value_as_number"],
            1,
            123,
            4,
            False,
            False,
            "Expected interval_length_unit to be a string.",
        ),
        (
            None,
            None,
            ["measurement"],
            ["value_as_number"],
            1,
            "day",
            "wrong_type",
            False,
            False,
            "Expected num_intervals to be an integer.",
        ),
        (
            None,
            None,
            ["measurement"],
            ["value_as_number"],
            1,
            "day",
            123,
            "wrong_type",
            False,
            "Expected enrich_var_with_feature_info to be a boolean.",
        ),
        (
            None,
            None,
            ["measurement"],
            ["value_as_number"],
            1,
            "day",
            123,
            False,
            "wrong_type",
            "Expected enrich_var_with_unit_info to be a boolean.",
        ),
    ],
)
def test_setup_variables_illegal_argument_types(
    omop_connection_vanilla,
    edata,
    backend_handle,
    data_tables,
    data_field_to_keep,
    interval_length_number,
    interval_length_unit,
    num_intervals,
    enrich_var_with_feature_info,
    enrich_var_with_unit_info,
    expected_error,
):
    con = omop_connection_vanilla
    with pytest.raises(TypeError, match=expected_error):
        ed.io.omop.setup_variables(
            edata or ed.io.omop.setup_obs(backend_handle=omop_connection_vanilla, observation_table="person_cohort"),
            backend_handle=backend_handle or con,
            data_tables=data_tables,
            data_field_to_keep=data_field_to_keep,
            interval_length_number=interval_length_number,
            interval_length_unit=interval_length_unit,
            num_intervals=num_intervals,
            enrich_var_with_feature_info=enrich_var_with_feature_info,
            enrich_var_with_unit_info=enrich_var_with_unit_info,
        )


def test_capital_letters(omop_connection_capital_letters):
    # test capital letters both in table names and column names
    con = omop_connection_capital_letters
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=1,
        enrich_var_with_feature_info=False,
        enrich_var_with_unit_info=False,
    )

    assert edata.r[0, 0, 0] == 18

    tables = con.execute("SHOW TABLES").df()["name"].values
    assert "measurement" in tables
    assert "MEASUREMENT" not in tables

    measurement_columns = con.execute("SELECT * FROM measurement").df().columns
    assert "measurement_id" in measurement_columns
    assert "MEASUREMENT_ID" not in measurement_columns


def test_empty_observation(omop_connection_empty_observation, caplog):
    con = omop_connection_empty_observation
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["observation"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=1,
        enrich_var_with_feature_info=False,
        enrich_var_with_unit_info=False,
    )
    assert edata.shape == (1, 0)
    assert "No data found in observation. Returning edata without additional variables." in caplog.text
