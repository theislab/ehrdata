import re

import numpy as np
import pandas as pd
import pytest

import ehrdata as ed
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

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
    "procedure_occurrence": 2,
    "device_exposure": 2,
    "drug_era": 2,
    "dose_era": 2,
    "condition_era": 2,
    "episode": 2,
}

VANILLA_IS_PRESENT_START = [
    [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
    [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
    [[1, np.nan, np.nan, np.nan], [1, np.nan, np.nan, np.nan]],
]

VANILLA_IS_PRESENT_END = [
    [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
]

VANILLA_IS_PRESENT_INTERVAL = [
    [[1, 1, 1, 1], [1, 1, 1, 1]],
    [[1, 1, 1, 1], [1, 1, 1, 1]],
    [[1, 1, 1, 1], [1, 1, 1, 1]],
]

MEASUREMENT_VAR = pd.DataFrame({"data_table_concept_id": [3022318, 3031147]}, index=["0", "1"])
OBSERVATION_VAR = pd.DataFrame({"data_table_concept_id": [3001062, 3034263]}, index=["0", "1"])
SPECIMEN_VAR = pd.DataFrame({"data_table_concept_id": [4001225, 4121345]}, index=["0", "1"])
DRUG_EXPOSURE_VAR = pd.DataFrame({"data_table_concept_id": [19019979, 19073183]}, index=["0", "1"])
CONDITION_OCCURRENCE_VAR = pd.DataFrame({"data_table_concept_id": [4112343, 43530622]}, index=["0", "1"])
PROCEDURE_OCCURRENCE_VAR = pd.DataFrame({"data_table_concept_id": [4107731, 4326177]}, index=["0", "1"])
DEVICE_EXPOSURE_VAR = pd.DataFrame({"data_table_concept_id": [4217646, 45768171]}, index=["0", "1"])
DRUG_ERA_VAR = pd.DataFrame({"data_table_concept_id": [1124957, 1368671]}, index=["0", "1"])
DOSE_ERA_VAR = pd.DataFrame({"data_table_concept_id": [714785, 902427]}, index=["0", "1"])
CONDITION_ERA_VAR = pd.DataFrame({"data_table_concept_id": [434610, 4140598]}, index=["0", "1"])
EPISODE_VAR = pd.DataFrame({"data_table_concept_id": [32531, 32941]}, index=["0", "1"])

# constants for setup_variables
# only data_table_concept_id
VAR_DIM_BASE = 1
# number of columns in concept table
NUMBER_COLUMNS_CONCEPT_TABLE = 10
VAR_DIM_FEATURE_INFO = NUMBER_COLUMNS_CONCEPT_TABLE
# number of columns in concept table + number of columns
NUMBER_COLUMNS_FEATURE_REPORT = 4
VAR_DIM_UNIT_INFO = NUMBER_COLUMNS_CONCEPT_TABLE + NUMBER_COLUMNS_FEATURE_REPORT
# array of ids in concept table
VAR_MAPPING_INFO = [2000030004, 2000001003]


@pytest.mark.parametrize(
    ("observation_table", "death_table", "expected_length", "expected_obs_num_columns"),
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
    ("backend_handle", "observation_table", "death_table", "expected_error"),
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
    ("data_tables", "data_field_to_keep", "target_R", "target_var"),
    [
        (
            ["measurement"],
            ["value_as_number"],
            [
                [[np.nan, np.nan, np.nan, np.nan], [18.0, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [20.0, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [22.0, np.nan, np.nan, np.nan]],
            ],
            MEASUREMENT_VAR,
        ),
        (
            ["measurement"],
            ["is_present"],
            VANILLA_IS_PRESENT_START,
            MEASUREMENT_VAR,
        ),
        (
            ["observation"],
            ["value_as_number"],
            [
                [[np.nan, np.nan, np.nan, np.nan], [3, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [4, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [5, np.nan, np.nan, np.nan]],
            ],
            OBSERVATION_VAR,
        ),
        (
            ["observation"],
            ["is_present"],
            VANILLA_IS_PRESENT_START,
            OBSERVATION_VAR,
        ),
        (
            ["specimen"],
            ["quantity"],
            [
                [[0.5, np.nan, np.nan, np.nan], [1.5, np.nan, np.nan, np.nan]],
                [[0.5, np.nan, np.nan, np.nan], [1.5, np.nan, np.nan, np.nan]],
                [[0.5, np.nan, np.nan, np.nan], [1.5, np.nan, np.nan, np.nan]],
            ],
            SPECIMEN_VAR,
        ),
        (
            ["specimen"],
            ["is_present"],
            VANILLA_IS_PRESENT_START,
            SPECIMEN_VAR,
        ),
        (
            ["measurement", "observation", "specimen"],
            {
                "measurement": "value_as_number",
                "observation": "is_present",
                "specimen": "quantity",
            },
            [
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [18.0, np.nan, np.nan, np.nan],
                    [1.0, np.nan, np.nan, np.nan],
                    [1.0, np.nan, np.nan, np.nan],
                    [0.5, np.nan, np.nan, np.nan],
                    [1.5, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [20.0, np.nan, np.nan, np.nan],
                    [1.0, np.nan, np.nan, np.nan],
                    [1.0, np.nan, np.nan, np.nan],
                    [0.5, np.nan, np.nan, np.nan],
                    [1.5, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [22.0, np.nan, np.nan, np.nan],
                    [1.0, np.nan, np.nan, np.nan],
                    [1.0, np.nan, np.nan, np.nan],
                    [0.5, np.nan, np.nan, np.nan],
                    [1.5, np.nan, np.nan, np.nan],
                ],
            ],
            pd.concat([MEASUREMENT_VAR, OBSERVATION_VAR, SPECIMEN_VAR]).set_index(pd.Index(map(str, range(6)))),
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
    target_R,
    target_var,
):
    num_intervals = 4
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table=observation_table)
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
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
    assert edata.n_vars == sum(VANILLA_NUM_CONCEPTS[data_table] for data_table in data_tables)
    assert edata.layers[DEFAULT_TEM_LAYER_NAME].shape[2] == num_intervals
    assert edata.var.shape[1] == VAR_DIM_BASE + (VAR_DIM_FEATURE_INFO if enrich_var_with_feature_info else 0) + (
        VAR_DIM_UNIT_INFO if enrich_var_with_unit_info else 0
    ) + (
        (1 if any(elem not in VAR_MAPPING_INFO for elem in edata.var["data_table_concept_id"]) else 0)
        if enrich_var_with_feature_info
        else 0
    )
    pd.testing.assert_frame_equal(edata.var[["data_table_concept_id"]], target_var)

    assert np.allclose(edata.layers[DEFAULT_TEM_LAYER_NAME], np.array(target_R), equal_nan=True)


@pytest.mark.parametrize(
    "observation_table",
    ["person_cohort", "person_observation_period", "person_visit_occurrence"],
)
# test 1 field from table, and is_present encoding, with start, end, and interval
@pytest.mark.parametrize(
    ("data_tables", "data_field_to_keep", "keep_date", "target_R", "target_var"),
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
            DRUG_EXPOSURE_VAR,
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
            DRUG_EXPOSURE_VAR,
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
            DRUG_EXPOSURE_VAR,
        ),
        (
            ["drug_exposure"],
            ["is_present"],
            "start",
            VANILLA_IS_PRESENT_START,
            DRUG_EXPOSURE_VAR,
        ),
        (
            ["drug_exposure"],
            ["is_present"],
            "end",
            VANILLA_IS_PRESENT_END,
            DRUG_EXPOSURE_VAR,
        ),
        (
            ["drug_exposure"],
            ["is_present"],
            "interval",
            VANILLA_IS_PRESENT_INTERVAL,
            DRUG_EXPOSURE_VAR,
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
            CONDITION_OCCURRENCE_VAR,
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
            CONDITION_OCCURRENCE_VAR,
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
            CONDITION_OCCURRENCE_VAR,
        ),
        (
            ["condition_occurrence"],
            ["is_present"],
            "start",
            VANILLA_IS_PRESENT_START,
            CONDITION_OCCURRENCE_VAR,
        ),
        (
            ["condition_occurrence"],
            ["is_present"],
            "end",
            VANILLA_IS_PRESENT_END,
            CONDITION_OCCURRENCE_VAR,
        ),
        (
            ["condition_occurrence"],
            ["is_present"],
            "interval",
            VANILLA_IS_PRESENT_INTERVAL,
            CONDITION_OCCURRENCE_VAR,
        ),
        (
            ["procedure_occurrence"],
            ["procedure_source_value"],
            "start",
            [
                [[180256009, np.nan, np.nan, np.nan], [430193006, np.nan, np.nan, np.nan]],
                [[180256009, np.nan, np.nan, np.nan], [430193006, np.nan, np.nan, np.nan]],
                [[180256009, np.nan, np.nan, np.nan], [430193006, np.nan, np.nan, np.nan]],
            ],
            PROCEDURE_OCCURRENCE_VAR,
        ),
        (
            ["procedure_occurrence"],
            ["procedure_source_value"],
            "end",
            [
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
            ],
            PROCEDURE_OCCURRENCE_VAR,
        ),
        (
            ["procedure_occurrence"],
            ["procedure_source_value"],
            "interval",
            [
                [[180256009, 180256009, 180256009, 180256009], [430193006, 430193006, 430193006, 430193006]],
                [[180256009, 180256009, 180256009, 180256009], [430193006, 430193006, 430193006, 430193006]],
                [[180256009, 180256009, 180256009, 180256009], [430193006, 430193006, 430193006, 430193006]],
            ],
            PROCEDURE_OCCURRENCE_VAR,
        ),
        (
            ["procedure_occurrence"],
            ["is_present"],
            "start",
            VANILLA_IS_PRESENT_START,
            PROCEDURE_OCCURRENCE_VAR,
        ),
        (
            ["procedure_occurrence"],
            ["is_present"],
            "end",
            VANILLA_IS_PRESENT_END,
            PROCEDURE_OCCURRENCE_VAR,
        ),
        (
            ["procedure_occurrence"],
            ["is_present"],
            "interval",
            VANILLA_IS_PRESENT_INTERVAL,
            PROCEDURE_OCCURRENCE_VAR,
        ),
        (
            ["device_exposure"],
            ["device_source_value"],
            "start",
            [
                [[72506001, np.nan, np.nan, np.nan], [224087, np.nan, np.nan, np.nan]],
                [[72506001, np.nan, np.nan, np.nan], [224087, np.nan, np.nan, np.nan]],
                [[72506001, np.nan, np.nan, np.nan], [224087, np.nan, np.nan, np.nan]],
            ],
            DEVICE_EXPOSURE_VAR,
        ),
        (
            ["device_exposure"],
            ["device_source_value"],
            "end",
            [
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
            ],
            DEVICE_EXPOSURE_VAR,
        ),
        (
            ["device_exposure"],
            ["device_source_value"],
            "interval",
            [
                [[72506001, 72506001, 72506001, 72506001], [224087, 224087, 224087, 224087]],
                [[72506001, 72506001, 72506001, 72506001], [224087, 224087, 224087, 224087]],
                [[72506001, 72506001, 72506001, 72506001], [224087, 224087, 224087, 224087]],
            ],
            DEVICE_EXPOSURE_VAR,
        ),
        (
            ["device_exposure"],
            ["is_present"],
            "start",
            VANILLA_IS_PRESENT_START,
            DEVICE_EXPOSURE_VAR,
        ),
        (
            ["device_exposure"],
            ["is_present"],
            "end",
            VANILLA_IS_PRESENT_END,
            DEVICE_EXPOSURE_VAR,
        ),
        (
            ["device_exposure"],
            ["is_present"],
            "interval",
            VANILLA_IS_PRESENT_INTERVAL,
            DEVICE_EXPOSURE_VAR,
        ),
        (
            ["drug_era"],
            ["drug_exposure_count"],
            "start",
            [
                [[2, np.nan, np.nan, np.nan], [4, np.nan, np.nan, np.nan]],
                [[2, np.nan, np.nan, np.nan], [4, np.nan, np.nan, np.nan]],
                [[2, np.nan, np.nan, np.nan], [4, np.nan, np.nan, np.nan]],
            ],
            DRUG_ERA_VAR,
        ),
        (
            ["drug_era"],
            ["drug_exposure_count"],
            "end",
            [
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
            ],
            DRUG_ERA_VAR,
        ),
        (
            ["drug_era"],
            ["drug_exposure_count"],
            "interval",
            [
                [[2, 2, 2, 2], [4, 4, 4, 4]],
                [[2, 2, 2, 2], [4, 4, 4, 4]],
                [[2, 2, 2, 2], [4, 4, 4, 4]],
            ],
            DRUG_ERA_VAR,
        ),
        (
            ["drug_era"],
            ["is_present"],
            "start",
            VANILLA_IS_PRESENT_START,
            DRUG_ERA_VAR,
        ),
        (
            ["drug_era"],
            ["is_present"],
            "end",
            VANILLA_IS_PRESENT_END,
            DRUG_ERA_VAR,
        ),
        (
            ["drug_era"],
            ["is_present"],
            "interval",
            VANILLA_IS_PRESENT_INTERVAL,
            DRUG_ERA_VAR,
        ),
        (
            ["dose_era"],
            ["dose_value"],
            "start",
            [
                [[2.5, np.nan, np.nan, np.nan], [10, np.nan, np.nan, np.nan]],
                [[2.5, np.nan, np.nan, np.nan], [10, np.nan, np.nan, np.nan]],
                [[2.5, np.nan, np.nan, np.nan], [10, np.nan, np.nan, np.nan]],
            ],
            DOSE_ERA_VAR,
        ),
        (
            ["dose_era"],
            ["dose_value"],
            "end",
            [
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
            ],
            DOSE_ERA_VAR,
        ),
        (
            ["dose_era"],
            ["dose_value"],
            "interval",
            [
                [[2.5, 2.5, 2.5, 2.5], [10, 10, 10, 10]],
                [[2.5, 2.5, 2.5, 2.5], [10, 10, 10, 10]],
                [[2.5, 2.5, 2.5, 2.5], [10, 10, 10, 10]],
            ],
            DOSE_ERA_VAR,
        ),
        (
            ["dose_era"],
            ["is_present"],
            "start",
            VANILLA_IS_PRESENT_START,
            DOSE_ERA_VAR,
        ),
        (
            ["dose_era"],
            ["is_present"],
            "end",
            VANILLA_IS_PRESENT_END,
            DOSE_ERA_VAR,
        ),
        (
            ["dose_era"],
            ["is_present"],
            "interval",
            VANILLA_IS_PRESENT_INTERVAL,
            DOSE_ERA_VAR,
        ),
        (
            ["condition_era"],
            ["condition_occurrence_count"],
            "start",
            [
                [[1, np.nan, np.nan, np.nan], [256, np.nan, np.nan, np.nan]],
                [[1, np.nan, np.nan, np.nan], [256, np.nan, np.nan, np.nan]],
                [[1, np.nan, np.nan, np.nan], [256, np.nan, np.nan, np.nan]],
            ],
            CONDITION_ERA_VAR,
        ),
        (
            ["condition_era"],
            ["condition_occurrence_count"],
            "end",
            [
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
            ],
            CONDITION_ERA_VAR,
        ),
        (
            ["condition_era"],
            ["condition_occurrence_count"],
            "interval",
            [
                [[1, 1, 1, 1], [256, 256, 256, 256]],
                [[1, 1, 1, 1], [256, 256, 256, 256]],
                [[1, 1, 1, 1], [256, 256, 256, 256]],
            ],
            CONDITION_ERA_VAR,
        ),
        (
            ["condition_era"],
            ["is_present"],
            "start",
            VANILLA_IS_PRESENT_START,
            CONDITION_ERA_VAR,
        ),
        (
            ["condition_era"],
            ["is_present"],
            "end",
            VANILLA_IS_PRESENT_END,
            CONDITION_ERA_VAR,
        ),
        (
            ["condition_era"],
            ["is_present"],
            "interval",
            VANILLA_IS_PRESENT_INTERVAL,
            CONDITION_ERA_VAR,
        ),
        (
            ["episode"],
            ["episode_source_value"],
            "start",
            [
                [[5, np.nan, np.nan, np.nan], [10, np.nan, np.nan, np.nan]],
                [[5, np.nan, np.nan, np.nan], [10, np.nan, np.nan, np.nan]],
                [[5, np.nan, np.nan, np.nan], [10, np.nan, np.nan, np.nan]],
            ],
            EPISODE_VAR,
        ),
        (
            ["episode"],
            ["episode_source_value"],
            "end",
            [
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
            ],
            EPISODE_VAR,
        ),
        (
            ["episode"],
            ["episode_source_value"],
            "interval",
            [
                [[5, 5, 5, 5], [10, 10, 10, 10]],
                [[5, 5, 5, 5], [10, 10, 10, 10]],
                [[5, 5, 5, 5], [10, 10, 10, 10]],
            ],
            EPISODE_VAR,
        ),
        (
            ["episode"],
            ["is_present"],
            "start",
            VANILLA_IS_PRESENT_START,
            EPISODE_VAR,
        ),
        (
            ["episode"],
            ["is_present"],
            "end",
            VANILLA_IS_PRESENT_END,
            EPISODE_VAR,
        ),
        (
            ["episode"],
            ["is_present"],
            "interval",
            VANILLA_IS_PRESENT_INTERVAL,
            EPISODE_VAR,
        ),
        (
            ["condition_era", "episode"],
            {"condition_era": "is_present", "episode": "episode_source_value"},
            "interval",
            [
                [[1, 1, 1, 1], [1, 1, 1, 1], [5, 5, 5, 5], [10, 10, 10, 10]],
                [[1, 1, 1, 1], [1, 1, 1, 1], [5, 5, 5, 5], [10, 10, 10, 10]],
                [[1, 1, 1, 1], [1, 1, 1, 1], [5, 5, 5, 5], [10, 10, 10, 10]],
            ],
            pd.concat([CONDITION_ERA_VAR, EPISODE_VAR]).set_index(pd.Index(map(str, range(4)))),
        ),
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
    target_R,
    enrich_var_with_feature_info,
    keep_date,
    target_var,
):
    num_intervals = 4
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table=observation_table)
    edata = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
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
    assert edata.n_vars == sum(VANILLA_NUM_CONCEPTS[data_table] for data_table in data_tables)
    assert edata.layers[DEFAULT_TEM_LAYER_NAME].shape[2] == num_intervals
    assert edata.var.shape[1] == VAR_DIM_BASE + (VAR_DIM_FEATURE_INFO if enrich_var_with_feature_info else 0) + (
        (1 if any(elem not in VAR_MAPPING_INFO for elem in edata.var["data_table_concept_id"]) else 0)
        if enrich_var_with_feature_info
        else 0
    )

    assert np.allclose(edata.layers[DEFAULT_TEM_LAYER_NAME], np.array(target_R), equal_nan=True)


@pytest.mark.parametrize(
    (
        "edata",
        "backend_handle",
        "data_tables",
        "data_field_to_keep",
        "interval_length_number",
        "interval_length_unit",
        "num_intervals",
        "enrich_var_with_feature_info",
        "enrich_var_with_unit_info",
        "expected_error",
    ),
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
            "Expected data_field_to_keep to be a string, Sequence, or dict, but is <class 'int'>",
        ),
        (
            None,
            None,
            ["measurement", "observation"],
            ["value_as_number"],
            1,
            "day",
            4,
            False,
            False,
            "data_field_to_keep must be a dictionary if more than one data table is used.",
        ),
        (
            None,
            None,
            ["measurement"],
            {"measurement": 123},
            1,
            "day",
            4,
            False,
            False,
            "data_field_to_keep values must be a string or Sequence.",
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
            layer=DEFAULT_TEM_LAYER_NAME,
            data_tables=data_tables,
            data_field_to_keep=data_field_to_keep,
            interval_length_number=interval_length_number,
            interval_length_unit=interval_length_unit,
            num_intervals=num_intervals,
            enrich_var_with_feature_info=enrich_var_with_feature_info,
            enrich_var_with_unit_info=enrich_var_with_unit_info,
        )


@pytest.mark.parametrize(
    (
        "edata",
        "backend_handle",
        "data_tables",
        "data_field_to_keep",
        "interval_length_number",
        "interval_length_unit",
        "num_intervals",
        "enrich_var_with_feature_info",
        "expected_error",
    ),
    [
        (
            "wrong_type",
            None,
            ["drug_exposure"],
            ["is_present"],
            1,
            "day",
            4,
            False,
            "Expected edata to be of type EHRData.",
        ),
        (
            None,
            "wrong_type",
            ["drug_exposure"],
            ["is_present"],
            1,
            "day",
            4,
            False,
            "Expected backend_handle to be of type DuckDBPyConnection.",
        ),
        (
            None,
            None,
            123,
            ["is_present"],
            1,
            "day",
            4,
            False,
            "Expected data_tables to be a string or Sequence.",
        ),
        (
            None,
            None,
            ["drug_exposure"],
            123,
            1,
            "day",
            4,
            False,
            "Expected data_field_to_keep to be a string, Sequence, or dict, but is <class 'int'>",
        ),
        (
            None,
            None,
            ["drug_exposure", "condition_occurrence"],
            ["is_present"],
            1,
            "day",
            4,
            False,
            "data_field_to_keep must be a dictionary if more than one data table is used.",
        ),
        (
            None,
            None,
            ["drug_exposure"],
            {"drug_exposure": 123},
            1,
            "day",
            4,
            False,
            "data_field_to_keep values must be a string or Sequence.",
        ),
        (
            None,
            None,
            ["drug_exposure"],
            ["is_present"],
            "wrong_type",
            "day",
            4,
            False,
            "Expected interval_length_number to be an integer.",
        ),
        (
            None,
            None,
            ["drug_exposure"],
            ["value_as_number"],
            1,
            123,
            4,
            False,
            "Expected interval_length_unit to be a string.",
        ),
        (
            None,
            None,
            ["drug_exposure"],
            ["is_present"],
            1,
            "day",
            "wrong_type",
            False,
            "Expected num_intervals to be an integer.",
        ),
        (
            None,
            None,
            ["drug_exposure"],
            ["is_present"],
            1,
            "day",
            123,
            "wrong_type",
            "Expected enrich_var_with_feature_info to be a boolean.",
        ),
    ],
)
def test_setup_interval_variables_illegal_argument_types(
    omop_connection_vanilla,
    edata,
    backend_handle,
    data_tables,
    data_field_to_keep,
    interval_length_number,
    interval_length_unit,
    num_intervals,
    enrich_var_with_feature_info,
    expected_error,
):
    con = omop_connection_vanilla
    with pytest.raises(TypeError, match=expected_error):
        ed.io.omop.setup_interval_variables(
            edata or ed.io.omop.setup_obs(backend_handle=omop_connection_vanilla, observation_table="person_cohort"),
            backend_handle=backend_handle or con,
            layer=DEFAULT_TEM_LAYER_NAME,
            data_tables=data_tables,
            data_field_to_keep=data_field_to_keep,
            interval_length_number=interval_length_number,
            interval_length_unit=interval_length_unit,
            num_intervals=num_intervals,
            enrich_var_with_feature_info=enrich_var_with_feature_info,
        )


@pytest.mark.parametrize(
    (
        "edata",
        "backend_handle",
        "data_tables",
        "data_field_to_keep",
        "interval_length_number",
        "interval_length_unit",
        "num_intervals",
        "enrich_var_with_feature_info",
        "enrich_var_with_unit_info",
        "expected_error",
    ),
    [
        (
            None,
            None,
            ["measurementt"],
            ["value_as_number"],
            1,
            "day",
            4,
            False,
            False,
            re.escape("data_tables must be a subset of ['measurement', 'observation', 'specimen']."),
        ),
        (
            None,
            None,
            ["measurement", "observation"],
            {"measurement": "value_as_number"},
            1,
            "day",
            4,
            False,
            False,
            "data_field_to_keep keys must be equal to data_tables.",
        ),
    ],
)
def test_setup_variables_illegal_argument_values(
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
    with pytest.raises(ValueError, match=expected_error):
        ed.io.omop.setup_variables(
            edata or ed.io.omop.setup_obs(backend_handle=omop_connection_vanilla, observation_table="person_cohort"),
            backend_handle=backend_handle or con,
            layer=DEFAULT_TEM_LAYER_NAME,
            data_tables=data_tables,
            data_field_to_keep=data_field_to_keep,
            interval_length_number=interval_length_number,
            interval_length_unit=interval_length_unit,
            num_intervals=num_intervals,
            enrich_var_with_feature_info=enrich_var_with_feature_info,
            enrich_var_with_unit_info=enrich_var_with_unit_info,
        )


@pytest.mark.parametrize(
    (
        "edata",
        "backend_handle",
        "data_tables",
        "data_field_to_keep",
        "interval_length_number",
        "interval_length_unit",
        "num_intervals",
        "enrich_var_with_feature_info",
        "expected_error",
    ),
    [
        (
            None,
            None,
            ["drug_exposuree"],
            ["is_present"],
            1,
            "day",
            4,
            False,
            re.escape(
                "data_tables must be a subset of ['drug_exposure', 'condition_occurrence', 'procedure_occurrence', 'device_exposure', 'drug_era', 'dose_era', 'condition_era', 'episode']."
            ),
        ),
        (
            None,
            None,
            ["drug_exposure", "condition_occurrence"],
            {"drug_exposure": "is_present"},
            1,
            "day",
            4,
            False,
            "data_field_to_keep keys must be equal to data_tables.",
        ),
    ],
)
def test_setup_interval_variables_illegal_argument_values(
    omop_connection_vanilla,
    edata,
    backend_handle,
    data_tables,
    data_field_to_keep,
    interval_length_number,
    interval_length_unit,
    num_intervals,
    enrich_var_with_feature_info,
    expected_error,
):
    con = omop_connection_vanilla
    with pytest.raises(ValueError, match=expected_error):
        ed.io.omop.setup_interval_variables(
            edata or ed.io.omop.setup_obs(backend_handle=omop_connection_vanilla, observation_table="person_cohort"),
            backend_handle=backend_handle or con,
            layer=DEFAULT_TEM_LAYER_NAME,
            data_tables=data_tables,
            data_field_to_keep=data_field_to_keep,
            interval_length_number=interval_length_number,
            interval_length_unit=interval_length_unit,
            num_intervals=num_intervals,
            enrich_var_with_feature_info=enrich_var_with_feature_info,
        )


def test_capital_letters(omop_connection_capital_letters):
    # test capital letters both in table names and column names
    con = omop_connection_capital_letters
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=1,
        enrich_var_with_feature_info=False,
        enrich_var_with_unit_info=False,
    )

    assert edata.layers[DEFAULT_TEM_LAYER_NAME][0, 0, 0] == 18

    tables = con.execute("SHOW TABLES").df()["name"].values
    assert "measurement" in tables
    assert "MEASUREMENT" not in tables

    measurement_columns = con.execute("SELECT * FROM measurement").df().columns
    assert "measurement_id" in measurement_columns
    assert "MEASUREMENT_ID" not in measurement_columns


def test_setup_variables_empty_observation(omop_connection_empty_observation, caplog):
    con = omop_connection_empty_observation
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=["observation"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=1,
        enrich_var_with_feature_info=False,
        enrich_var_with_unit_info=False,
    )
    assert edata.shape == (1, 0, 1)
    assert "No data found in observation. Returning edata without data of observation." in caplog.text
    assert "No data found in any of the data tables. Returning edata without data." in caplog.text


def test_setup_interval_variables_empty_observation(omop_connection_empty_observation, caplog):
    con = omop_connection_empty_observation
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person")
    edata = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=["drug_exposure"],
        data_field_to_keep=["is_present"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=1,
        enrich_var_with_feature_info=False,
    )
    assert edata.shape == (1, 0, 1)
    assert "No data found in drug_exposure. Returning edata without data of drug_exposure." in caplog.text
    assert "No data found in any of the data tables. Returning edata without data." in caplog.text


def test_multiple_units(omop_connection_multiple_units, caplog):
    con = omop_connection_multiple_units
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=["observation"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=2,
        enrich_var_with_feature_info=False,
        enrich_var_with_unit_info=False,
    )
    # assert edata.shape == (1, 0)
    assert "multiple units for features: [[0]\n [1]]\n" in caplog.text
