import re

import numpy as np
import pandas as pd
import pytest
import sparse

import ehrdata as ed
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME
from ehrdata.io.omop._queries import _get_unit_fields

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
    (
        "observation_table",
        "death_table",
        "expected_length",
        "expected_obs_num_columns",
        "expected_row_identifier_key",
        "expected_row_identifier_values",
    ),
    [
        ("person", False, 4, 18, "person_id", [1, 2, 3, 4]),
        ("person", True, 4, 24, "person_id", [1, 2, 3, 4]),
        ("person_cohort", False, 3, 22, "subject_id", [1, 2, 3]),
        ("person_cohort", True, 3, 28, "subject_id", [1, 2, 3]),
        ("person_observation_period", False, 3, 23, "observation_period_id", [1, 2, 3]),
        ("person_observation_period", True, 3, 29, "observation_period_id", [1, 2, 3]),
        ("person_visit_occurrence", False, 3, 35, "visit_occurrence_id", [1, 2, 3]),
        ("person_visit_occurrence", True, 3, 41, "visit_occurrence_id", [1, 2, 3]),
    ],
)
def test_setup_obs(
    omop_connection_vanilla,
    observation_table,
    death_table,
    expected_length,
    expected_obs_num_columns,
    expected_row_identifier_key,
    expected_row_identifier_values,
):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table=observation_table, death_table=death_table)
    assert isinstance(edata, ed.EHRData)

    # 4 persons, only 3 are in cohort, or have observation period, or visit occurrence
    assert len(edata) == expected_length
    assert edata.obs.shape[1] == expected_obs_num_columns

    assert np.array_equal(
        edata.obs[expected_row_identifier_key].values,
        np.array(expected_row_identifier_values, dtype=np.int64),
    )


# person_id: 1, 2, 3, 4
# observation_period_id [1, 2, 3]


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
                [[np.nan, np.nan, np.nan, np.nan], [19.0, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [21.0, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan], [23.0, np.nan, np.nan, np.nan]],
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
                    [19.0, np.nan, np.nan, np.nan],
                    [1.0, np.nan, np.nan, np.nan],
                    [1.0, np.nan, np.nan, np.nan],
                    [0.5, np.nan, np.nan, np.nan],
                    [1.5, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [21.0, np.nan, np.nan, np.nan],
                    [1.0, np.nan, np.nan, np.nan],
                    [1.0, np.nan, np.nan, np.nan],
                    [0.5, np.nan, np.nan, np.nan],
                    [1.5, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [23.0, np.nan, np.nan, np.nan],
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
@pytest.mark.parametrize(
    "time_precision",
    ["date", "datetime"],
)
def test_setup_variables(
    omop_connection_vanilla,
    observation_table,
    data_tables,
    data_field_to_keep,
    enrich_var_with_feature_info,
    enrich_var_with_unit_info,
    time_precision,
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
        time_precision=time_precision,
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
@pytest.mark.parametrize(
    "time_precision",
    ["date", "datetime"],
)
def test_setup_interval_type_variables(
    omop_connection_vanilla,
    observation_table,
    data_tables,
    data_field_to_keep,
    time_precision,
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
        time_precision=time_precision,
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
        "time_precision",
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
            "date",
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
            "date",
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
            "date",
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
            "date",
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
            "date",
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
            "date",
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
            "date",
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
            "date",
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
            123,
            4,
            False,
            False,
            "Expected time_precision to be a string.",
        ),
        (
            None,
            None,
            ["measurement"],
            ["value_as_number"],
            1,
            "day",
            "date",
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
            "date",
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
            "date",
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
    time_precision,
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
            time_precision=time_precision,
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
        "time_precision",
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
            "date",
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
            "date",
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
            "date",
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
            "date",
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
            "date",
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
            "date",
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
            "date",
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
            "date",
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
            123,
            4,
            False,
            "Expected time_precision to be a string.",
        ),
        (
            None,
            None,
            ["drug_exposure"],
            ["is_present"],
            1,
            "day",
            "date",
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
            "date",
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
    time_precision,
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
            time_precision=time_precision,
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
        "time_precision",
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
            "date",
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
            "date",
            4,
            False,
            False,
            "data_field_to_keep keys must be equal to data_tables.",
        ),
        (
            None,
            None,
            ["measurement"],
            ["value_as_number"],
            1,
            "day",
            "invalid",
            4,
            False,
            False,
            re.escape("time_precision must be one of ['date', 'datetime']."),
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
    time_precision,
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
            time_precision=time_precision,
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
        "time_precision",
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
            "date",
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
            "date",
            4,
            False,
            "data_field_to_keep keys must be equal to data_tables.",
        ),
        (
            None,
            None,
            ["drug_exposure"],
            ["is_present"],
            1,
            "day",
            "invalid",
            4,
            False,
            re.escape("time_precision must be one of ['date', 'datetime']."),
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
    time_precision,
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
            time_precision=time_precision,
            num_intervals=num_intervals,
            enrich_var_with_feature_info=enrich_var_with_feature_info,
        )


@pytest.mark.parametrize(
    ("interval_length_unit", "time_precision", "should_warn"),
    [
        ("hour", "date", True),
        ("minute", "date", True),
        ("second", "date", True),
        ("ms", "date", True),
        ("day", "date", False),
        ("hour", "datetime", False),
    ],
)
def test_time_precision_interval_mismatch_warning(
    omop_connection_vanilla, interval_length_unit, time_precision, should_warn, caplog
):
    """Test that warnings are logged for fine-grained intervals with date precision."""
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_cohort")

    # Test setup_variables
    caplog.clear()
    ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit=interval_length_unit,
        time_precision=time_precision,
        num_intervals=2,
    )

    if should_warn:
        assert "Using interval_length_unit" in caplog.text
    else:
        assert "Using interval_length_unit" not in caplog.text

    # Test setup_interval_variables
    caplog.clear()
    ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=["drug_exposure"],
        data_field_to_keep=["is_present"],
        interval_length_number=1,
        interval_length_unit=interval_length_unit,
        time_precision=time_precision,
        num_intervals=2,
    )

    if should_warn:
        assert "Using interval_length_unit" in caplog.text
    else:
        assert "Using interval_length_unit" not in caplog.text


@pytest.mark.parametrize(
    ("data_table", "time_precision", "should_warn"),
    [
        ("drug_era", "datetime", True),
        ("dose_era", "datetime", True),
        ("condition_era", "datetime", True),
        ("drug_era", "date", False),
        ("drug_exposure", "datetime", False),
    ],
)
def test_datetime_precision_fallback_warning(omop_connection_vanilla, data_table, time_precision, should_warn, caplog):
    """Test that warnings are logged when datetime precision is requested but not available."""
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_cohort")

    caplog.clear()
    ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=[data_table],
        data_field_to_keep=["is_present"],
        interval_length_number=1,
        interval_length_unit="day",
        time_precision=time_precision,
        num_intervals=2,
    )

    if should_warn:
        assert "Time precision datetime not available" in caplog.text
        assert "Using '...date' and midnight" in caplog.text
    else:
        assert "Time precision" not in caplog.text or "not found" not in caplog.text


def test_person_table_requires_valid_birthdates(omop_connection_vanilla):
    """Test that using person observation table requires all persons to have valid birthdates."""
    con = omop_connection_vanilla

    # Set one person's birth_datetime to NULL
    con.execute("UPDATE person SET birth_datetime = NULL WHERE person_id = 1")

    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person")

    # Should raise ValueError when trying to setup_variables
    with pytest.raises(ValueError, match=re.escape("with NULL birth_datetime")):
        ed.io.omop.setup_variables(
            edata,
            backend_handle=con,
            layer=DEFAULT_TEM_LAYER_NAME,
            data_tables=["measurement"],
            data_field_to_keep=["value_as_number"],
            interval_length_number=1,
            interval_length_unit="day",
            num_intervals=2,
        )

    # Restore the birthdate for other tests
    con.execute("UPDATE person SET birth_datetime = '1970-01-01 00:00:00' WHERE person_id = 1")


def test_person_table_with_valid_birthdates(omop_connection_vanilla):
    """Test that person observation table works when all persons have valid birthdates."""
    con = omop_connection_vanilla

    # Ensure all persons have birthdates
    con.execute("UPDATE person SET birth_datetime = '1970-01-01 00:00:00' WHERE birth_datetime IS NULL")

    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person")

    # Should work without errors
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        time_precision="datetime",
        num_intervals=2,
    )

    # Check that data was extracted
    assert edata.n_obs > 0
    assert edata.n_vars > 0


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
    # only feature 3034263 is flagged: the value_as_number of feature 3001062 is NULL in all its rows,
    # so none of its rows contributes a value - and hence no unit - to the extraction
    assert "multiple units for features: [[0]]\n" in caplog.text


@pytest.mark.parametrize("aggregation_strategy", ["last", "first"])
def test_setup_variables_aggregation_ignores_rows_without_value(omop_connection_vanilla, aggregation_strategy):
    """A row without a value must not shadow an observed value of the same interval, see issue #298."""
    con = omop_connection_vanilla

    # person 1 has value_as_number 18 (12:00) and 19 (13:00) of concept 3031147, in unit 9557/mEq/L;
    # surround them by rows without a value_as_number, carrying a deviating unit
    con.execute(
        """INSERT INTO measurement
            (measurement_id, person_id, measurement_concept_id, measurement_date, measurement_datetime,
             value_as_number, unit_concept_id, unit_source_value)
        VALUES
            (10, 1, 3031147, '2100-01-01', '2100-01-01 11:00:00', NULL, 8888, 'deviating_unit'),
            (11, 1, 3031147, '2100-01-01', '2100-01-01 14:00:00', NULL, 8888, 'deviating_unit')"""
    )

    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        time_precision="datetime",
        num_intervals=2,
        aggregation_strategy=aggregation_strategy,
    )

    # obs 0 is visit_occurrence 1 of person 1, var 1 is concept 3031147, interval 0 is 2100-01-01
    assert edata.layers[DEFAULT_TEM_LAYER_NAME][0, 1, 0] == (19 if aggregation_strategy == "last" else 18)

    # the unit is the one of the value that is kept, not the one of a row without a value
    long_format_row = con.execute(
        """SELECT * FROM long_person_timestamp_feature_value_measurement
        WHERE obs_id = 1 AND data_table_concept_id = 3031147 AND interval_step = 0"""
    ).df()
    assert long_format_row["unit_concept_id"].item() == 9557
    assert long_format_row["unit_source_value"].item() == "mEq/L"


@pytest.mark.parametrize("aggregation_strategy", ["last", "first"])
def test_setup_variables_single_row_aggregation_breaks_ties_deterministically(
    omop_connection_vanilla, aggregation_strategy
):
    """Timestamp ties are broken by value, independent of duckdb's scheduling."""
    con = omop_connection_vanilla
    con.execute("SET threads = 4")

    # tie the existing values 18 (12:00) and 19 (13:00) of person 1, concept 3031147
    con.execute(
        """INSERT INTO measurement
            (measurement_id, person_id, measurement_concept_id, measurement_date, measurement_datetime,
             value_as_number, unit_concept_id, unit_source_value)
        VALUES
            (10, 1, 3031147, '2100-01-01', '2100-01-01 12:00:00', 30, 9557, 'mEq/L'),
            (11, 1, 3031147, '2100-01-01', '2100-01-01 13:00:00', 7, 9557, 'mEq/L')"""
    )

    for _ in range(5):
        edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
        edata = ed.io.omop.setup_variables(
            edata,
            backend_handle=con,
            layer=DEFAULT_TEM_LAYER_NAME,
            data_tables=["measurement"],
            data_field_to_keep=["value_as_number"],
            interval_length_number=1,
            interval_length_unit="day",
            time_precision="datetime",
            num_intervals=2,
            aggregation_strategy=aggregation_strategy,
        )
        # last: max(19, 7); first: min(18, 30)
        assert edata.layers[DEFAULT_TEM_LAYER_NAME][0, 1, 0] == (19 if aggregation_strategy == "last" else 18)


@pytest.mark.parametrize("aggregation_strategy", ["last", "first"])
def test_setup_interval_variables_aggregation_ignores_rows_without_value(omop_connection_vanilla, aggregation_strategy):
    """A row without a value must not shadow an observed value of the same interval, see issue #298."""
    con = omop_connection_vanilla

    # person 1 has days_supply 31 (12:00) of concept 19073183 starting on 2100-01-01;
    # surround it by rows without a days_supply
    con.execute(
        """INSERT INTO drug_exposure
            (drug_exposure_id, person_id, drug_concept_id, drug_exposure_start_date, drug_exposure_start_datetime,
             drug_exposure_end_date, drug_exposure_end_datetime, days_supply)
        VALUES
            (10, 1, 19073183, '2100-01-01', '2100-01-01 11:00:00', '2100-01-31', '2100-01-31 00:00:00', NULL),
            (11, 1, 19073183, '2100-01-01', '2100-01-01 13:00:00', '2100-01-31', '2100-01-31 00:00:00', NULL)"""
    )

    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    edata = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=["drug_exposure"],
        data_field_to_keep=["days_supply"],
        interval_length_number=1,
        interval_length_unit="day",
        time_precision="datetime",
        num_intervals=2,
        aggregation_strategy=aggregation_strategy,
        keep_date="start",
    )

    # obs 0 is visit_occurrence 1 of person 1, var 1 is concept 19073183, interval 0 is 2100-01-01
    assert edata.layers[DEFAULT_TEM_LAYER_NAME][0, 1, 0] == 31


@pytest.mark.parametrize(
    ("aggregation_strategy", "expected_value"),
    [
        ("last", 23),
        ("first", 22),
        ("mean", pytest.approx(68 / 3)),
        ("median", 23),
        ("mode", 23),
        ("sum", 68),
        ("count", 3),
        ("min", 22),
        ("max", 23),
        ("std", pytest.approx(0.5773502691896258)),
    ],
)
def test_setup_variables_aggregation_strategies_carry_the_unit(
    omop_connection_vanilla, aggregation_strategy, expected_value
):
    """Every aggregation strategy works, and the units are carried along instead of being aggregated, see issue #300."""
    con = omop_connection_vanilla

    # person 3 has value_as_number 22 (12:00) and 23 (13:00) of concept 3031147, in unit 9557/mEq/L
    # a third value of the same unit makes the aggregates of the interval free of ties
    con.execute(
        """INSERT INTO measurement
            (measurement_id, person_id, measurement_concept_id, measurement_date, measurement_datetime,
             value_as_number, unit_concept_id, unit_source_value)
        VALUES (10, 3, 3031147, '2100-01-01', '2100-01-01 14:00:00', 23, 9557, 'mEq/L')"""
    )

    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        time_precision="datetime",
        num_intervals=2,
        aggregation_strategy=aggregation_strategy,
    )

    # obs 2 is visit_occurrence 3 of person 3, var 1 is concept 3031147, interval 0 is 2100-01-01
    assert edata.layers[DEFAULT_TEM_LAYER_NAME][2, 1, 0] == expected_value

    long_format_row = con.execute(
        """SELECT * FROM long_person_timestamp_feature_value_measurement
        WHERE obs_id = 3 AND data_table_concept_id = 3031147 AND interval_step = 0"""
    ).df()
    assert long_format_row["unit_concept_id"].item() == 9557
    assert long_format_row["unit_source_value"].item() == "mEq/L"


@pytest.mark.parametrize("aggregation_strategy", ["mean", "std"])
def test_setup_variables_combining_aggregation_raises_on_multiple_units(
    omop_connection_multiple_units, aggregation_strategy
):
    """Values of different units must not be combined into a single value, see issue #300."""
    con = omop_connection_multiple_units
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")

    # concept 3034263 has a value_as_number in unit 8587/mL and one in unit 9665/uL
    with pytest.raises(NotImplementedError, match="requires a single unit per feature"):
        ed.io.omop.setup_variables(
            edata,
            backend_handle=con,
            layer=DEFAULT_TEM_LAYER_NAME,
            data_tables=["observation"],
            data_field_to_keep=["value_as_number"],
            interval_length_number=1,
            interval_length_unit="day",
            num_intervals=2,
            aggregation_strategy=aggregation_strategy,
        )


@pytest.mark.parametrize(
    ("aggregation_strategy", "data_field_to_keep"),
    [("count", "value_as_number"), ("last", "value_as_number"), ("mean", "is_present")],
)
def test_setup_variables_unitless_aggregation_allows_multiple_units(
    omop_connection_multiple_units, aggregation_strategy, data_field_to_keep
):
    """Numbers of data points carry no unit, and a single kept data point mixes none, see issue #300."""
    con = omop_connection_multiple_units
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")

    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=["observation"],
        data_field_to_keep=[data_field_to_keep],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=2,
        aggregation_strategy=aggregation_strategy,
    )
    assert edata.shape == (1, 2, 2)


@pytest.mark.parametrize(
    ("aggregation_strategy", "expected_value"),
    [
        ("last", 20),
        ("first", 10),
        ("mean", 15),
        ("median", 15),
        ("sum", 30),
        ("count", 2),
        ("min", 10),
        ("max", 20),
        ("std", pytest.approx(7.071067811865476)),
    ],
)
def test_setup_interval_variables_aggregation_strategies_carry_the_unit(
    omop_connection_vanilla, aggregation_strategy, expected_value
):
    """The units are carried along in setup_interval_variables, too, see issue #300."""
    con = omop_connection_vanilla

    # person 1 has dose_value 10 of drug 902427 in unit 8576 in two dose_eras, starting 2100-01-01 and 2100-02-01
    # make the two differ, so that the aggregates of the interval holding both are free of ties
    con.execute("UPDATE dose_era SET dose_value = 20 WHERE dose_era_id = 2")
    # and extend visit 1 of person 1, which ends 2100-01-31, so that it covers both
    con.execute("UPDATE visit_occurrence SET visit_end_date = '2100-02-28' WHERE visit_occurrence_id = 1")

    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    edata = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=["dose_era"],
        data_field_to_keep=["dose_value"],
        interval_length_number=60,
        interval_length_unit="day",
        num_intervals=1,
        aggregation_strategy=aggregation_strategy,
        keep_date="start",
    )

    # obs 0 is visit_occurrence 1 of person 1, var 1 is drug 902427, interval 0 holds both dose_eras
    assert edata.layers[DEFAULT_TEM_LAYER_NAME][0, 1, 0] == expected_value

    # dose_era has a unit_concept_id, but no unit_source_value
    long_format_row = con.execute(
        """SELECT * FROM long_person_timestamp_feature_value_dose_era
        WHERE obs_id = 1 AND data_table_concept_id = 902427 AND interval_step = 0"""
    ).df()
    assert long_format_row["unit_concept_id"].item() == 8576
    assert "unit_source_value" not in long_format_row.columns


@pytest.mark.parametrize("aggregation_strategy", ["mean", "std"])
def test_setup_interval_variables_combining_aggregation_raises_on_multiple_units(
    omop_connection_vanilla, aggregation_strategy
):
    """Values of different units must not be combined into a single value here either, see issue #300."""
    con = omop_connection_vanilla

    # give the two dose_eras of drug 902427 of person 1 a deviating unit each
    con.execute("UPDATE dose_era SET unit_concept_id = 8587 WHERE dose_era_id = 2")

    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    with pytest.raises(NotImplementedError, match="requires a single unit per feature"):
        ed.io.omop.setup_interval_variables(
            edata,
            backend_handle=con,
            layer=DEFAULT_TEM_LAYER_NAME,
            data_tables=["dose_era"],
            data_field_to_keep=["dose_value"],
            interval_length_number=60,
            interval_length_unit="day",
            num_intervals=1,
            aggregation_strategy=aggregation_strategy,
            keep_date="start",
            # the second dose_era starts after the end of the visit
            clip_to_end_date=False,
        )


@pytest.mark.parametrize(
    ("data_table", "data_field_to_keep"),
    [("drug_exposure", "days_supply"), ("drug_era", "drug_exposure_count")],
)
def test_setup_interval_variables_without_unit_concept_id_allows_combining_aggregation(
    omop_connection_vanilla, data_table, data_field_to_keep
):
    """A data table that tells no unit_concept_id has no unit to disagree on, see issue #300."""
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")

    edata = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=[data_table],
        data_field_to_keep=[data_field_to_keep],
        interval_length_number=60,
        interval_length_unit="day",
        num_intervals=1,
        aggregation_strategy="mean",
        keep_date="start",
    )

    long_format_columns = con.execute(f"DESCRIBE long_person_timestamp_feature_value_{data_table}").df()["column_name"]
    assert not any("unit" in column for column in long_format_columns)


def test_get_unit_fields_adapts_to_the_data_table(omop_connection_vanilla):
    """The unit fields of a data table depend on the table and on the OMOP CDM version, see issue #300."""
    con = omop_connection_vanilla

    assert _get_unit_fields(con, "measurement") == ("unit_concept_id", "unit_source_value")
    assert _get_unit_fields(con, "device_exposure") == ("unit_concept_id", "unit_source_value")
    assert _get_unit_fields(con, "dose_era") == ("unit_concept_id",)
    assert _get_unit_fields(con, "drug_exposure") == ()

    # device_exposure has the unit fields in OMOP CDM 5.4, but not in 5.3
    con.execute("ALTER TABLE device_exposure DROP COLUMN unit_concept_id")
    con.execute("ALTER TABLE device_exposure DROP COLUMN unit_source_value")
    assert _get_unit_fields(con, "device_exposure") == ()


def test_multiple_visit_occurrences_for_single_patient(omop_connection_multiple_visit_occurrences):
    """Test that multiple visits for a single patient are handled correctly.

    Person 1 has 2 visits (visit_occurrence_id 1 and 4).
    The obs should have 4 rows total (one per visit occurrence).
    """
    con = omop_connection_multiple_visit_occurrences
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")

    assert edata.n_obs == 4

    assert np.array_equal(edata.obs["visit_occurrence_id"].values, np.array([1, 2, 3, 4], dtype=np.int64)), (
        f"visit_occurrence_id not ordered correctly: {edata.obs['visit_occurrence_id'].values}"
    )

    assert np.array_equal(edata.obs["person_id"].values, np.array([1, 2, 3, 1], dtype=np.int64))

    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=2,
    )

    assert np.allclose(
        np.array(edata[edata.obs["person_id"] == 3, edata.var["data_table_concept_id"] == 3031147].layers["tem_data"]),
        np.array([23.0, np.nan]),
        equal_nan=True,
    )

    assert np.allclose(
        np.array(
            edata[edata.obs["visit_occurrence_id"] == 4, edata.var["data_table_concept_id"] == 3031147].layers[
                "tem_data"
            ]
        ),
        np.array([[[29.0, np.nan]]]),
        equal_nan=True,
    )


def test_multiple_visit_occurrences_for_single_patient_datetime(omop_connection_multiple_visit_occurrences):
    """Test that multiple visits for a single patient are handled correctly.

    Person 1 has 2 visits (visit_occurrence_id 1 and 4).
    The obs should have 4 rows total (one per visit occurrence).
    """
    con = omop_connection_multiple_visit_occurrences
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")

    assert edata.n_obs == 4

    assert np.array_equal(edata.obs["visit_occurrence_id"].values, np.array([1, 2, 3, 4], dtype=np.int64)), (
        f"visit_occurrence_id not ordered correctly: {edata.obs['visit_occurrence_id'].values}"
    )

    assert np.array_equal(edata.obs["person_id"].values, np.array([1, 2, 3, 1], dtype=np.int64))

    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        time_precision="date",
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=2,
    )

    assert np.allclose(
        np.array(edata[edata.obs["person_id"] == 3, edata.var["data_table_concept_id"] == 3031147].layers["tem_data"]),
        np.array([23.0, np.nan]),
        equal_nan=True,
    )

    assert np.allclose(
        np.array(
            edata[edata.obs["visit_occurrence_id"] == 4, edata.var["data_table_concept_id"] == 3031147].layers[
                "tem_data"
            ]
        ),
        np.array([[[29.0, np.nan]]]),
        equal_nan=True,
    )


def test_multiple_visit_occurrences_for_single_patient_datetime_specific(omop_connection_multiple_visit_occurrences):
    """Test that multiple visits for a single patient are handled correctly.

    Person 1 has 2 visits (visit_occurrence_id 1 and 4).
    The obs should have 4 rows total (one per visit occurrence).
    """
    con = omop_connection_multiple_visit_occurrences
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")

    assert edata.n_obs == 4

    assert np.array_equal(edata.obs["visit_occurrence_id"].values, np.array([1, 2, 3, 4], dtype=np.int64)), (
        f"visit_occurrence_id not ordered correctly: {edata.obs['visit_occurrence_id'].values}"
    )

    assert np.array_equal(edata.obs["person_id"].values, np.array([1, 2, 3, 1], dtype=np.int64))

    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        time_precision="datetime",
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="h",
        num_intervals=24,
    )
    assert np.allclose(
        edata[0, 1, :].layers["tem_data"],
        np.array(
            [
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    18.0,
                    19.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            ]
        ),
        equal_nan=True,
    )


# Shares the default mimic-iv download path with test_mimic_iv_omop (tests/dt/test_dt.py); same xdist group so the
# two don't download/extract concurrently on a cold cache.
@pytest.mark.xdist_group(name="dataset_mimic_iv_omop")
def test_mimic_iv_omop_visit_measurement_validation(omop_connection_mimic_iv):
    """Validation test using real MIMIC-IV OMOP data.

    This test validates the complete pipeline from OMOP database to EHRData tensor
    using a concrete example from the MIMIC-IV demo dataset:
    - Visit -9149771978458038515 (Patient 4239478333578644568)
    - Creatinine measurement (concept_id 3016723) with value 1.1
    - Measured at 2177-03-12 12:47:00 (5.5 hours after visit start at 07:15:00)
    - Should appear at interval 5 (the 5-6 hour time bin)
    """
    con = omop_connection_mimic_iv

    edata = ed.io.omop.setup_obs(
        backend_handle=con,
        observation_table="person_visit_occurrence",
        death_table=True,
    )

    edata = ed.io.omop.setup_variables(
        edata=edata,
        layer="tem_data",
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="h",
        num_intervals=24,
        concept_ids="all",
        aggregation_strategy="last",
        time_precision="datetime",
    )

    # Validate the dataset structure
    assert edata.n_obs == 852, f"Expected 852 visits, got {edata.n_obs}"
    assert edata.shape[2] == 24, f"Expected 24 time intervals, got {edata.shape[2]}"

    # Validate the specific example
    visit_occurrence_id = -9149771978458038515
    variable_concept_id = 3016723  # Creatinine [Mass/volume] in Serum or Plasma

    # Create boolean masks to select the visit and variable
    visit_index = edata.obs["visit_occurrence_id"] == visit_occurrence_id
    variable_index = edata.var["data_table_concept_id"] == variable_concept_id

    # Extract the time series
    time_series = edata[visit_index, variable_index, :].layers["tem_data"]

    # Expected: value 1.1 at interval 5, NaN everywhere else
    expected = np.full((1, 1, 24), np.nan)
    expected[0, 0, 5] = 1.1

    assert np.allclose(time_series, expected, equal_nan=True), (
        f"Expected value 1.1 at interval 5. Got: {time_series[0, 0, :]}"
    )

    # Verify visit metadata
    patient_id = edata.obs[visit_index]["person_id"].item()
    assert patient_id == 4239478333578644568, f"Expected patient 4239478333578644568, got {patient_id}"


def test_setup_obs_parquet(omop_connection_vanilla_parquet):
    """Test that setup_obs works with Parquet files."""
    con = omop_connection_vanilla_parquet
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")

    assert isinstance(edata, ed.EHRData)
    assert len(edata) == 3
    assert edata.obs.shape[1] == 23


def test_setup_variables_parquet(omop_connection_vanilla_parquet):
    """Test that setup_variables works with Parquet files."""
    num_intervals = 4
    con = omop_connection_vanilla_parquet
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        time_precision="date",
        num_intervals=num_intervals,
        enrich_var_with_feature_info=False,
        enrich_var_with_unit_info=False,
    )

    assert isinstance(edata, ed.EHRData)
    assert edata.n_obs == 3
    assert edata.n_vars == 2
    assert edata.layers[DEFAULT_TEM_LAYER_NAME].shape[2] == num_intervals

    # Verify the data matches expected values (same as CSV)
    expected_data = [
        [[np.nan, np.nan, np.nan, np.nan], [19.0, np.nan, np.nan, np.nan]],
        [[np.nan, np.nan, np.nan, np.nan], [21.0, np.nan, np.nan, np.nan]],
        [[np.nan, np.nan, np.nan, np.nan], [23.0, np.nan, np.nan, np.nan]],
    ]
    assert np.allclose(edata.layers[DEFAULT_TEM_LAYER_NAME], np.array(expected_data), equal_nan=True)


@pytest.mark.parametrize(
    ("setup_function", "data_tables", "data_field_to_keep", "concept_id", "expected_R"),
    [
        (
            ed.io.omop.setup_variables,
            ["measurement", "observation"],
            "value_as_number",
            3031147,
            [[[18.5, np.nan]], [[20.5, np.nan]], [[22.5, np.nan]]],
        ),
        (
            ed.io.omop.setup_interval_variables,
            ["drug_exposure", "condition_occurrence"],
            "is_present",
            19073183,
            [[[1, np.nan]]] * 3,
        ),
    ],
)
def test_concept_ids_select_the_variables(
    omop_connection_vanilla, setup_function, data_tables, data_field_to_keep, concept_id, expected_R
):
    """Only the selected concept_ids become variables, also when a data table has none of them."""
    con = omop_connection_vanilla

    # give the concept that is not selected values in several units, which only the selected ones must agree on
    con.execute(
        "UPDATE measurement SET value_as_number = 1, unit_concept_id = measurement_id WHERE measurement_concept_id = 3022318"
    )

    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    setup_kwargs = {
        "backend_handle": con,
        "data_tables": data_tables,
        "data_field_to_keep": dict.fromkeys(data_tables, data_field_to_keep),
        "interval_length_number": 1,
        "interval_length_unit": "day",
        "num_intervals": 2,
        "aggregation_strategy": "mean",
    }
    edata_selected = setup_function(edata, concept_ids=[concept_id], **setup_kwargs)

    assert edata_selected.var["data_table_concept_id"].tolist() == [concept_id]
    assert np.allclose(edata_selected.X, np.array(expected_R), equal_nan=True)

    with pytest.raises(ValueError, match="concept_ids must not be empty"):
        setup_function(edata, concept_ids=[], **setup_kwargs)


def test_setup_variables_enrich_var_with_unit_info_matches_the_concept_id(omop_connection_vanilla):
    """The unit report is merged into var on the concept id, not on the position."""
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=1,
        enrich_var_with_unit_info=True,
    )

    # concept 3022318 has no value_as_number, and hence no unit; concept 3031147 is in unit 9557
    assert edata.var["data_table_concept_id"].tolist() == [3022318, 3031147]
    assert edata.var["unit_concept_id"].isna().tolist() == [True, False]
    assert edata.var["unit_concept_id"].iloc[1] == 9557


@pytest.mark.parametrize("time_precision", ["date", "datetime"])
def test_setup_variables_and_interval_variables_end_with_the_observation(omop_connection_vanilla, time_precision):
    """Data points after the last day of the observation are not binned."""
    con = omop_connection_vanilla

    # the observation_period of person 1 ends 2100-01-31: add a value of concept 3031147 on that day, and one after it
    con.execute(
        """INSERT INTO measurement
            (measurement_id, person_id, measurement_concept_id, measurement_date, measurement_datetime,
             value_as_number, unit_concept_id, unit_source_value)
        VALUES
            (10, 1, 3031147, '2100-01-31', '2100-01-31 14:00:00', 30, 9557, 'mEq/L'),
            (11, 1, 3031147, '2100-02-02', '2100-02-02 10:00:00', 40, 9557, 'mEq/L')"""
    )
    # drug 19073183 of person 1 has a second drug_exposure starting after that day; let the one of drug 19019979 outlast it
    con.execute(
        """UPDATE drug_exposure SET drug_exposure_end_date = '2100-03-31', drug_exposure_end_datetime = '2100-03-31 00:00:00'
        WHERE drug_exposure_id = 3"""
    )

    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    intervals = {"interval_length_number": 7, "interval_length_unit": "day", "num_intervals": 6}

    edata_variables = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        time_precision=time_precision,
        **intervals,
    )
    # obs 0 is person 1, var 1 is concept 3031147, interval 4 holds both values and interval 5 starts after 2100-01-31
    assert np.allclose(edata_variables.X[0, 1, 4:], [30, np.nan], equal_nan=True)

    edata_interval_variables = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        data_tables=["drug_exposure"],
        data_field_to_keep=["is_present"],
        time_precision=time_precision,
        keep_date="interval",
        **intervals,
    )
    assert np.allclose(edata_interval_variables.X[0], [[1, 1, 1, 1, 1, np.nan]] * 2, equal_nan=True)


def test_setup_connection_replaces_the_tables(omop_connection_vanilla):
    """Setting up a connection again replaces its tables rather than failing on them."""
    con = omop_connection_vanilla
    con.execute("DELETE FROM measurement")

    ed.io.omop.setup_connection(path="tests/data/toy_omop/vanilla", backend_handle=con)

    assert con.execute("SELECT COUNT(*) FROM measurement").fetchall()[0][0] == 9


@pytest.mark.parametrize(
    ("interval_length_number", "interval_length_unit"),
    [(30, "day"), (365, "day"), (720, "h")],
)
def test_setup_variables_intervals_have_the_exact_length(
    omop_connection_vanilla, interval_length_number, interval_length_unit
):
    """Intervals of 30 days or more are exactly as long as asked for, rather than calendar months."""
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=interval_length_number,
        interval_length_unit=interval_length_unit,
        num_intervals=3,
    )

    # the observation_period of person 1 starts 2100-01-01; the long format table holds its observed intervals
    intervals = con.execute(
        """SELECT interval_step, interval_start, interval_end FROM long_person_timestamp_feature_value_measurement
        WHERE obs_id = 1 AND data_table_concept_id = 3031147 ORDER BY interval_step"""
    ).df()
    boundaries = pd.date_range("2100-01-01", periods=4, freq=pd.Timedelta(interval_length_number, interval_length_unit))
    assert len(intervals) > 0
    assert intervals["interval_start"].tolist() == boundaries[intervals["interval_step"]].tolist()
    assert intervals["interval_end"].tolist() == boundaries[intervals["interval_step"] + 1].tolist()


@pytest.mark.parametrize("observation_table", VANILLA_PERSONS_WITH_OBSERVATION_TABLE_ENTRY)
@pytest.mark.parametrize(
    ("write", "read", "suffix"),
    [(ed.io.write_h5ed, ed.io.read_h5ed, "h5ed"), (ed.io.write_zarr, ed.io.read_zarr, "zarr")],
)
def test_write_read_of_omop_setup(omop_connection_vanilla, observation_table, write, read, suffix, tmp_path):
    # https://github.com/theislab/ehrdata/issues/303
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table=observation_table)
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        layer=DEFAULT_TEM_LAYER_NAME,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=4,
        enrich_var_with_feature_info=True,
        enrich_var_with_unit_info=True,
    )
    path = tmp_path / f"{observation_table}.{suffix}"

    write(edata, path)
    edata_read = read(path)

    assert edata_read.shape == edata.shape
    assert list(edata_read.obs.columns) == list(edata.obs.columns)
    assert list(edata_read.var.columns) == list(edata.var.columns)
    assert np.allclose(edata_read.layers[DEFAULT_TEM_LAYER_NAME], edata.layers[DEFAULT_TEM_LAYER_NAME], equal_nan=True)
    # datetimes read back as ISO 8601 strings of the same times
    for df, df_read in [(edata.obs, edata_read.obs), (edata.var, edata_read.var)]:
        datetime_columns = [column for column in df.columns if pd.api.types.is_datetime64_any_dtype(df[column])]
        assert datetime_columns
        for column in datetime_columns:
            assert pd.to_datetime(df_read[column].astype(object), format="ISO8601").tolist() == df[column].tolist()


@pytest.mark.parametrize(
    ("setup_function", "data_tables", "kwargs"),
    [
        (ed.io.omop.setup_variables, ["measurement", "observation"], {}),
        (ed.io.omop.setup_interval_variables, ["drug_exposure", "condition_occurrence"], {"keep_date": "start"}),
    ],
)
def test_setup_functions_leave_data_field_to_keep_unchanged(
    omop_connection_vanilla, setup_function, data_tables, kwargs
):
    con = omop_connection_vanilla
    data_field_to_keep = dict.fromkeys(data_tables, "is_present")
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    setup_function(
        edata,
        backend_handle=con,
        data_tables=data_tables,
        data_field_to_keep=data_field_to_keep,
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=4,
        **kwargs,
    )

    assert data_field_to_keep == dict.fromkeys(data_tables, "is_present")


@pytest.mark.parametrize("observation_table", VANILLA_PERSONS_WITH_OBSERVATION_TABLE_ENTRY)
@pytest.mark.parametrize("aggregation_strategy", ["last", "mean", "count"])
@pytest.mark.parametrize(
    ("data_tables", "data_field_to_keep"),
    [
        (["measurement"], {"measurement": "value_as_number"}),
        (
            ["measurement", "observation", "specimen"],
            dict.fromkeys(["measurement", "observation", "specimen"], "is_present"),
        ),
        (["measurement", "specimen"], {"measurement": "value_as_number", "specimen": "quantity"}),
    ],
)
def test_setup_variables_sparse_matches_dense(
    omop_connection_vanilla, observation_table, aggregation_strategy, data_tables, data_field_to_keep
):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table=observation_table)
    kwargs = {
        "backend_handle": con,
        "data_tables": data_tables,
        "data_field_to_keep": data_field_to_keep,
        "interval_length_number": 1,
        "interval_length_unit": "day",
        "num_intervals": 4,
        "aggregation_strategy": aggregation_strategy,
    }
    edata_dense = ed.io.omop.setup_variables(edata, **kwargs)
    edata_sparse = ed.io.omop.setup_variables(edata, sparse=True, **kwargs)

    assert isinstance(edata_dense.X, np.ndarray)
    assert isinstance(edata_sparse.X, sparse.COO)
    # is_present is 1 or missing whatever the aggregation_strategy
    is_present = "is_present" in data_field_to_keep.values()
    expected_fill_value = 0 if aggregation_strategy == "count" and not is_present else np.nan
    np.testing.assert_equal(edata_sparse.X.fill_value, expected_fill_value)
    np.testing.assert_array_equal(edata_sparse.X.todense(), edata_dense.X)
    pd.testing.assert_frame_equal(edata_sparse.var, edata_dense.var)


def test_setup_variables_long_table_holds_observed_cells_only(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=100,
    )

    num_rows = con.execute("SELECT COUNT(*) FROM long_person_timestamp_feature_value_measurement").fetchone()[0]
    assert num_rows == np.count_nonzero(~np.isnan(edata.X))
    assert num_rows < edata.X.size


def test_setup_variables_sparse_requires_numeric_field(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    with pytest.raises(TypeError, match="sparse=True requires a numeric data field"):
        ed.io.omop.setup_variables(
            edata,
            backend_handle=con,
            data_tables=["measurement"],
            data_field_to_keep=["unit_source_value"],
            interval_length_number=1,
            interval_length_unit="day",
            num_intervals=4,
            sparse=True,
        )


@pytest.mark.parametrize(
    ("sparse_argument", "instantiate_tensor", "data_field_to_keep", "error", "match"),
    [
        ("yes", True, ["value_as_number"] * 2, TypeError, "Expected sparse to be a boolean"),
        (True, False, ["value_as_number"] * 2, ValueError, "instantiate_tensor=False"),
        (True, True, ["value_as_number", "is_present"], ValueError, "one fill value for all data tables"),
    ],
)
def test_setup_variables_illegal_sparse(
    omop_connection_vanilla, sparse_argument, instantiate_tensor, data_field_to_keep, error, match
):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    with pytest.raises(error, match=match):
        ed.io.omop.setup_variables(
            edata,
            backend_handle=con,
            data_tables=["measurement", "observation"],
            data_field_to_keep=dict(zip(["measurement", "observation"], data_field_to_keep, strict=True)),
            interval_length_number=1,
            interval_length_unit="day",
            num_intervals=4,
            aggregation_strategy="count",
            instantiate_tensor=instantiate_tensor,
            sparse=sparse_argument,
        )


def test_setup_variables_mode_breaks_ties_by_smallest_value(omop_connection_vanilla):
    con = omop_connection_vanilla
    con.execute("SET threads = 4")

    # person 1 has value_as_number 18 and 19 of concept 3031147 on 2100-01-01; add a second 19 and two 7s
    con.execute(
        """INSERT INTO measurement
            (measurement_id, person_id, measurement_concept_id, measurement_date, measurement_datetime,
             value_as_number, unit_concept_id, unit_source_value)
        VALUES
            (10, 1, 3031147, '2100-01-01', '2100-01-01 14:00:00', 19, 9557, 'mEq/L'),
            (11, 1, 3031147, '2100-01-01', '2100-01-01 15:00:00', 7, 9557, 'mEq/L'),
            (12, 1, 3031147, '2100-01-01', '2100-01-01 16:00:00', 7, 9557, 'mEq/L')"""
    )

    for _ in range(5):
        edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
        edata = ed.io.omop.setup_variables(
            edata,
            backend_handle=con,
            data_tables=["measurement"],
            data_field_to_keep=["value_as_number"],
            interval_length_number=1,
            interval_length_unit="day",
            time_precision="datetime",
            num_intervals=2,
            aggregation_strategy="mode",
        )
        # obs 0 is visit_occurrence 1 of person 1, var 1 is concept 3031147
        assert edata.X[0, 1, 0] == 7


def test_setup_variables_datetime_field_keeps_its_dtype(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["measurement_datetime"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=4,
    )

    assert edata.X.dtype.kind == "M"
    assert np.isnat(edata.X).any()
    assert not np.isnat(edata.X).all()


SETUP_FUNCTIONS = {
    "setup_variables": (
        ed.io.omop.setup_variables,
        {"data_tables": ["measurement"], "data_field_to_keep": ["value_as_number"]},
    ),
    "setup_interval_variables": (
        ed.io.omop.setup_interval_variables,
        {"data_tables": ["drug_exposure"], "data_field_to_keep": ["days_supply"], "keep_date": "start"},
    ),
}


@pytest.mark.parametrize("observation_table", VANILLA_PERSONS_WITH_OBSERVATION_TABLE_ENTRY)
@pytest.mark.parametrize("setup_function", SETUP_FUNCTIONS)
@pytest.mark.parametrize("order", [[2, 1, 0], [2, 0], [1]])
@pytest.mark.parametrize("time_precision", ["date", "datetime"])
def test_setup_functions_follow_the_rows_of_obs(
    omop_connection_vanilla, observation_table, setup_function, order, time_precision
):
    """The variables are set up for the rows of .obs, in their order, also after .obs was filtered or reordered."""
    function, kwargs = SETUP_FUNCTIONS[setup_function]
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table=observation_table)
    kwargs = {
        "time_precision": time_precision,
        "backend_handle": con,
        "interval_length_number": 1,
        "interval_length_unit": "day",
        "num_intervals": 4,
        **kwargs,
    }

    edata_full = function(edata, **kwargs)
    edata_subset = function(edata[order], **kwargs)

    assert edata_subset.n_obs == len(order)
    assert edata_subset.obs_names.tolist() == edata.obs_names[order].tolist()
    np.testing.assert_array_equal(edata_subset.X, edata_full.X[order])


@pytest.mark.parametrize("setup_function", SETUP_FUNCTIONS)
def test_setup_functions_keep_repeated_cohort_subjects_apart(omop_connection_vanilla, setup_function):
    """A subject in a cohort twice has two rows, each with the time window of its own cohort entry."""
    function, kwargs = SETUP_FUNCTIONS[setup_function]
    con = omop_connection_vanilla
    # subject 1 enters a second cohort a day earlier, so that its data of 2100-01-01 fall into the second interval
    con.execute("INSERT INTO cohort VALUES (2, 1, '2099-12-31', '2100-01-31')")
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_cohort")
    edata = function(
        edata, backend_handle=con, interval_length_number=1, interval_length_unit="day", num_intervals=4, **kwargs
    )

    assert edata.n_obs == 4
    for obs_index, cohort_start in enumerate(pd.to_datetime(edata.obs["cohort_start_date"])):
        observed_intervals = np.flatnonzero(~np.isnan(edata.X[obs_index]).all(axis=0))
        assert observed_intervals.tolist() == [(pd.Timestamp("2100-01-01") - cohort_start).days]


def test_setup_variables_reads_dates_of_obs_written_as_text(omop_connection_vanilla):
    """Dates in .obs may be text, for instance after writing and reading the EHRData."""
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    kwargs = {
        "backend_handle": con,
        "data_tables": ["measurement"],
        "data_field_to_keep": ["value_as_number"],
        "interval_length_number": 1,
        "interval_length_unit": "day",
        "num_intervals": 4,
    }
    edata_text = edata.copy()
    for column in ["observation_period_start_date", "observation_period_end_date"]:
        edata_text.obs[column] = edata_text.obs[column].astype(str)

    np.testing.assert_array_equal(
        ed.io.omop.setup_variables(edata_text, **kwargs).X, ed.io.omop.setup_variables(edata, **kwargs).X
    )


def test_setup_variables_requires_the_time_window_in_obs(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata.obs = edata.obs.drop(columns=["observation_period_end_date"])
    with pytest.raises(ValueError, match=r"lacks the column\(s\) \['observation_period_end_date'\]"):
        ed.io.omop.setup_variables(
            edata,
            backend_handle=con,
            data_tables=["measurement"],
            data_field_to_keep=["value_as_number"],
            interval_length_number=1,
            interval_length_unit="day",
            num_intervals=4,
        )


def test_setup_variables_on_persons_with_a_birth_datetime(omop_connection_vanilla):
    """With observation_table="person", .obs can be filtered to the persons that have a birth_datetime."""
    con = omop_connection_vanilla
    con.execute("UPDATE person SET birth_datetime = '1970-01-01 00:00:00' WHERE person_id = 1")
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person")
    kwargs = {
        "backend_handle": con,
        "data_tables": ["measurement"],
        "data_field_to_keep": ["value_as_number"],
        "interval_length_number": 365,
        "interval_length_unit": "day",
        "num_intervals": 200,
    }
    with pytest.raises(ValueError, match="with NULL birth_datetime"):
        ed.io.omop.setup_variables(edata, **kwargs)

    edata = edata[edata.obs["birth_datetime"].notna().to_numpy()].copy()
    edata = ed.io.omop.setup_variables(edata, **kwargs)
    assert edata.obs["person_id"].tolist() == [1]
    assert not np.isnan(edata.X).all()


def test_setup_variables_row_without_window_gets_no_data(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata.obs["observation_period_end_date"] = edata.obs["observation_period_end_date"].astype(object)
    edata.obs.iloc[0, edata.obs.columns.get_loc("observation_period_end_date")] = None
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=4,
    )

    assert np.isnan(edata.X[0]).all()
    assert not np.isnan(edata.X[1]).all()


def test_setup_variables_with_a_database_table_named_like_the_windows(omop_connection_vanilla):
    con = omop_connection_vanilla
    con.execute("CREATE TABLE windows (x INTEGER)")
    con.execute("CREATE TABLE _ehrdata_obs_windows_frame (x INTEGER)")
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=4,
    )

    assert not np.isnan(edata.X).all()


@pytest.mark.parametrize("setup_function", SETUP_FUNCTIONS)
@pytest.mark.parametrize("clip_to_end_date", [True, False])
def test_setup_functions_clip_to_end_date(omop_connection_vanilla, setup_function, clip_to_end_date):
    """Data points after the end date of the cohort entry are left out, unless clip_to_end_date=False."""
    function, kwargs = SETUP_FUNCTIONS[setup_function]
    con = omop_connection_vanilla
    # the cohort entries end 2100-01-31; add data points of person 1 a few days later
    con.execute(
        """INSERT INTO measurement (measurement_id, person_id, measurement_concept_id, measurement_date,
            measurement_datetime, value_as_number, unit_concept_id, unit_source_value)
        VALUES (10, 1, 3031147, '2100-02-05', '2100-02-05 12:00:00', 99, 9557, 'mEq/L')"""
    )
    con.execute(
        """INSERT INTO drug_exposure (drug_exposure_id, person_id, drug_concept_id, drug_exposure_start_date,
            drug_exposure_start_datetime, drug_exposure_end_date, drug_exposure_end_datetime, days_supply)
        VALUES (10, 1, 19073183, '2100-02-05', '2100-02-05 12:00:00', '2100-02-06', '2100-02-06 00:00:00', 99)"""
    )
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_cohort")
    edata = function(
        edata,
        backend_handle=con,
        interval_length_number=30,
        interval_length_unit="day",
        num_intervals=3,
        clip_to_end_date=clip_to_end_date,
        **kwargs,
    )

    # obs 0 is person 1, var 1 the concept of the added data point, interval 1 starts 2100-01-31
    if clip_to_end_date:
        # nothing after the end date of any cohort entry
        assert np.isnan(edata.X[:, :, 1:]).all()
    else:
        assert edata.X[0, 1, 1] == 99


def test_setup_variables_clip_to_end_date_false_needs_no_end_date_in_obs(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_cohort")
    kwargs = {
        "backend_handle": con,
        "data_tables": ["measurement"],
        "data_field_to_keep": ["value_as_number"],
        "interval_length_number": 1,
        "interval_length_unit": "day",
        "num_intervals": 4,
        "clip_to_end_date": False,
    }
    expected = ed.io.omop.setup_variables(edata, **kwargs).X
    edata.obs = edata.obs.drop(columns=["cohort_end_date"])

    np.testing.assert_array_equal(ed.io.omop.setup_variables(edata, **kwargs).X, expected)


@pytest.mark.parametrize("setup_function", SETUP_FUNCTIONS)
@pytest.mark.parametrize(
    ("observation_table", "clip_to_end_date", "error", "match"),
    [
        ("person_cohort", "no", TypeError, "Expected clip_to_end_date to be a boolean"),
        ("person", False, ValueError, "not available for observation_table='person'"),
    ],
)
def test_setup_functions_illegal_clip_to_end_date(
    omop_connection_vanilla, setup_function, observation_table, clip_to_end_date, error, match
):
    function, kwargs = SETUP_FUNCTIONS[setup_function]
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table=observation_table)
    with pytest.raises(error, match=match):
        function(
            edata,
            backend_handle=con,
            interval_length_number=1,
            interval_length_unit="day",
            num_intervals=4,
            clip_to_end_date=clip_to_end_date,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("clip_to_end_date", "expected"),
    [(True, [31, 31, 31, 31, 31, np.nan, np.nan, np.nan]), (False, [31, 31, 31, 31, 28, 28, 28, 28])],
)
def test_setup_interval_variables_clip_to_end_date_with_intervals(omop_connection_vanilla, clip_to_end_date, expected):
    """With keep_date="interval", the intervals after the end date get the exposures of these intervals, unless clipped."""
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    edata = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        data_tables=["drug_exposure"],
        data_field_to_keep=["days_supply"],
        interval_length_number=7,
        interval_length_unit="day",
        num_intervals=8,
        keep_date="interval",
        clip_to_end_date=clip_to_end_date,
    )

    # obs 0 is the visit of person 1, ending 2100-01-31; var 1 is concept 19073183, exposed from 2100-01-01 and 2100-02-01
    np.testing.assert_array_equal(edata.X[0, 1], expected)


@pytest.mark.parametrize("keep_date", ["start", "end", "interval"])
def test_setup_interval_variables_long_table_holds_observed_cells_only(omop_connection_vanilla, keep_date):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        data_tables=["drug_exposure"],
        data_field_to_keep=["days_supply"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=100,
        keep_date=keep_date,
    )

    num_rows = con.execute("SELECT COUNT(*) FROM long_person_timestamp_feature_value_drug_exposure").fetchone()[0]
    assert num_rows == np.count_nonzero(~np.isnan(edata.X))
    assert 0 < num_rows < edata.X.size


@pytest.mark.parametrize(
    ("keep_date", "aggregation_strategy", "expected"),
    [
        ("start", "last", [31, np.nan, np.nan, np.nan, np.nan, np.nan]),
        ("end", "last", [np.nan, np.nan, np.nan, np.nan, 31, np.nan]),
        ("interval", "last", [31, 31, 31, 31, 31, np.nan]),
        ("start", "count", [1, 0, 0, 0, 0, 0]),
        ("end", "count", [0, 0, 0, 0, 1, 0]),
        ("interval", "count", [1, 1, 1, 1, 1, 0]),
    ],
)
def test_setup_interval_variables_bins_by_keep_date(omop_connection_vanilla, keep_date, aggregation_strategy, expected):
    """Each person takes both drugs from 2100-01-01 to 2100-01-31, its observation period, and one of them again in February."""
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        data_tables=["drug_exposure"],
        data_field_to_keep=["days_supply"],
        interval_length_number=7,
        interval_length_unit="day",
        num_intervals=6,
        aggregation_strategy=aggregation_strategy,
        keep_date=keep_date,
    )

    # the intervals start on 2100-01-01, 08, 15, 22, 29 and 2100-02-05; the exposures in February start after the end date
    assert edata.X.dtype == (np.int64 if aggregation_strategy == "count" else np.float64)
    np.testing.assert_array_equal(edata.X, np.broadcast_to(expected, (3, 2, 6)))
    pd.testing.assert_frame_equal(edata.var, DRUG_EXPOSURE_VAR)


def test_setup_interval_variables_interval_spans(omop_connection_vanilla):
    """With keep_date="interval", an exposure belongs to the intervals that hold its start or its end, and to those it covers."""
    con = omop_connection_vanilla
    con.execute("DELETE FROM drug_exposure WHERE person_id = 1")
    con.execute(
        """INSERT INTO drug_exposure (drug_exposure_id, person_id, drug_concept_id, drug_exposure_start_date,
            drug_exposure_start_datetime, drug_exposure_end_date, drug_exposure_end_datetime, days_supply)
        VALUES (10, 1, 19073183, '2100-01-03', '2100-01-03 00:00:00', '2100-01-17', '2100-01-17 00:00:00', 10),
               (11, 1, 19073183, '2100-01-24', '2100-01-24 00:00:00', NULL, NULL, 20),
               (12, 1, 19019979, '2099-12-20', '2099-12-20 00:00:00', '2100-01-10', '2100-01-10 00:00:00', 30),
               (13, 1, 19019979, '2100-01-20', '2100-01-20 00:00:00', '2100-01-05', '2100-01-05 00:00:00', 40)"""
    )
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")

    def setup(keep_date, aggregation_strategy):
        return ed.io.omop.setup_interval_variables(
            edata,
            backend_handle=con,
            data_tables=["drug_exposure"],
            data_field_to_keep=["days_supply"],
            interval_length_number=7,
            interval_length_unit="day",
            num_intervals=5,
            aggregation_strategy=aggregation_strategy,
            keep_date=keep_date,
        ).X[0]

    # the intervals start on 2100-01-01, 08, 15, 22 and 29; var 0 is concept 19019979, var 1 concept 19073183.
    # 10 spans the first three intervals, and 20, without an end, belongs to the interval of its start.
    # 30 starts before the window and ends in its second interval; 40 ends before it starts and belongs to the intervals of both.
    np.testing.assert_array_equal(setup("interval", "count"), [[2, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    np.testing.assert_array_equal(setup("interval", "max"), [[40, 30, 40, np.nan, np.nan], [10, 10, 10, 20, np.nan]])
    np.testing.assert_array_equal(setup("start", "count"), [[0, 0, 1, 0, 0], [1, 0, 0, 1, 0]])
    np.testing.assert_array_equal(setup("end", "count"), [[1, 1, 0, 0, 0], [0, 0, 1, 0, 0]])


def test_setup_interval_variables_rows_without_concept_id(omop_connection_vanilla):
    con = omop_connection_vanilla
    # the February exposure of person 1 to concept 19073183
    con.execute("UPDATE drug_exposure SET drug_concept_id = NULL WHERE drug_exposure_id = 2")
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        data_tables=["drug_exposure"],
        data_field_to_keep=["days_supply"],
        interval_length_number=7,
        interval_length_unit="day",
        num_intervals=6,
        aggregation_strategy="count",
        keep_date="start",
        clip_to_end_date=False,
    )

    assert edata.var["data_table_concept_id"].iloc[:2].tolist() == [19019979, 19073183]
    assert pd.isna(edata.var["data_table_concept_id"].iloc[2])
    # the exposure without a concept ID starts on 2100-02-01, in the fifth interval
    np.testing.assert_array_equal(edata.X[:, 2], [[0, 0, 0, 0, 1, 0], [0] * 6, [0] * 6])
    np.testing.assert_array_equal(edata.X[1:, 1], [[1, 0, 0, 0, 1, 0]] * 2)


def _add_overlapping_visit(con):
    """Give person 1 a second visit, 5, overlapping its visit 1 of 2100-01-01 to 2100-01-31, and a third, 6, in March."""
    con.execute(
        """INSERT INTO visit_occurrence (visit_occurrence_id, person_id, visit_concept_id, visit_start_date,
            visit_start_datetime, visit_end_date, visit_end_datetime)
        VALUES (5, 1, 8870, '2100-01-01', '2100-01-01 00:00:00', '2100-01-31', '2100-01-31 00:00:00'),
               (6, 1, 8870, '2100-03-01', '2100-03-01 00:00:00', '2100-03-31', '2100-03-31 00:00:00')"""
    )


VISIT_ROWS = {
    # measurement concept 3022318 of person 1, on the second day: (visit_occurrence_id, date, value)
    "measurement": """INSERT INTO measurement (measurement_id, person_id, measurement_concept_id, measurement_date,
            measurement_datetime, value_as_number, unit_concept_id, unit_source_value, visit_occurrence_id)
        VALUES (10, 1, 3022318, '2100-01-02', '2100-01-02 12:00:00', 51, 9557, 'mEq/L', 1),
               (11, 1, 3022318, '2100-01-02', '2100-01-02 13:00:00', 55, 9557, 'mEq/L', 5),
               (12, 1, 3022318, '2100-01-02', '2100-01-02 14:00:00', 40, 9557, 'mEq/L', NULL),
               (13, 1, 3022318, '2100-01-03', '2100-01-03 12:00:00', 80, 9557, 'mEq/L', 999),
               (15, 1, 3022318, '2100-01-03', '2100-01-03 13:00:00', 33, 9557, 'mEq/L', 2),
               (14, 1, 3022318, '2100-03-02', '2100-03-02 12:00:00', 90, 9557, 'mEq/L', 1)""",
    "drug_exposure": """INSERT INTO drug_exposure (drug_exposure_id, person_id, drug_concept_id, drug_exposure_start_date,
            drug_exposure_start_datetime, drug_exposure_end_date, drug_exposure_end_datetime, days_supply, visit_occurrence_id)
        VALUES (10, 1, 19073183, '2100-01-02', '2100-01-02 12:00:00', '2100-01-03', '2100-01-03 00:00:00', 51, 1),
               (11, 1, 19073183, '2100-01-02', '2100-01-02 13:00:00', '2100-01-03', '2100-01-03 00:00:00', 55, 5),
               (12, 1, 19073183, '2100-01-02', '2100-01-02 14:00:00', '2100-01-03', '2100-01-03 00:00:00', 40, NULL),
               (13, 1, 19073183, '2100-01-03', '2100-01-03 12:00:00', '2100-01-04', '2100-01-04 00:00:00', 80, 999),
               (15, 1, 19073183, '2100-01-03', '2100-01-03 13:00:00', '2100-01-04', '2100-01-04 00:00:00', 33, 2),
               (14, 1, 19073183, '2100-03-02', '2100-03-02 12:00:00', '2100-03-03', '2100-03-03 00:00:00', 90, 1)""",
}


@pytest.mark.parametrize(
    ("setup_function", "data_table", "kwargs"),
    [
        (ed.io.omop.setup_variables, "measurement", {}),
        (ed.io.omop.setup_interval_variables, "drug_exposure", {"keep_date": "start"}),
        (ed.io.omop.setup_interval_variables, "drug_exposure", {"keep_date": "interval"}),
    ],
)
def test_setup_functions_attribute_rows_to_the_visit_they_name(
    omop_connection_vanilla, setup_function, data_table, kwargs
):
    """A row that names a visit of its person belongs to that visit only; any other row to every visit whose window holds it."""
    con = omop_connection_vanilla
    _add_overlapping_visit(con)
    con.execute(VISIT_ROWS[data_table])
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    visit = {visit_id: i for i, visit_id in enumerate(edata.obs["visit_occurrence_id"])}
    var = 0 if data_table == "measurement" else 1  # concept 3022318, or 19073183

    def setup(aggregation_strategy):
        return setup_function(
            edata,
            backend_handle=con,
            data_tables=[data_table],
            data_field_to_keep=["value_as_number" if data_table == "measurement" else "days_supply"],
            interval_length_number=1,
            interval_length_unit="day",
            num_intervals=4,
            aggregation_strategy=aggregation_strategy,
            **kwargs,
        ).X[:, var]

    largest, count = setup("max"), setup("count")
    # the second day: each overlapping visit takes its own row, 51 or 55, and the row without a visit, 40;
    # with keep_date="interval" visit 1 also the exposure that lasts the whole month, which names visit 1
    month_long = int(kwargs.get("keep_date") == "interval")
    assert (largest[visit[1], 1], count[visit[1], 1]) == (51, 2 + month_long)
    assert (largest[visit[5], 1], count[visit[5], 1]) == (55, 2)
    # the third day: a visit id that is no visit, 80, or a visit of another person, 33, falls back to person and time,
    # so both overlapping visits take these data points, and the visit of the other person does not
    assert largest[visit[1], 2] == largest[visit[5], 2] == 80
    if kwargs.get("keep_date") != "interval":  # with intervals, the month-long exposures cover the third day, too
        assert count[visit[1], 2] == count[visit[5], 2] == 2
        assert count[visit[2], 2] == 0
    # the row of March names visit 1, outside of whose window it lies, so visit 6 does not take it
    assert (count[visit[6]] == 0).all()


def test_setup_variables_attributes_rows_by_visit_also_after_filtering(omop_connection_vanilla):
    con = omop_connection_vanilla
    _add_overlapping_visit(con)
    con.execute(VISIT_ROWS["measurement"])
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    kwargs = {
        "backend_handle": con,
        "data_tables": ["measurement"],
        "data_field_to_keep": ["value_as_number"],
        "interval_length_number": 1,
        "interval_length_unit": "day",
        "num_intervals": 4,
        "aggregation_strategy": "count",
    }
    full = ed.io.omop.setup_variables(edata, **kwargs)
    only_visit_5 = ed.io.omop.setup_variables(edata[(edata.obs["visit_occurrence_id"] == 5).to_numpy()], **kwargs)

    # on the second day visit 5 has its own row and the one without a visit, not the one of visit 1,
    # whether visit 1 is set up along or not
    assert only_visit_5.X[0, 0, 1] == 2
    np.testing.assert_array_equal(only_visit_5.X[0], full.X[(full.obs["visit_occurrence_id"] == 5).to_numpy()][0])


def test_setup_interval_variables_attributes_rows_by_visit_at_the_end_date(omop_connection_vanilla):
    """keep_date="end" bins by the end date, with the same attribution."""
    con = omop_connection_vanilla
    _add_overlapping_visit(con)
    con.execute(VISIT_ROWS["drug_exposure"])
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    largest = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        data_tables=["drug_exposure"],
        data_field_to_keep=["days_supply"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=4,
        aggregation_strategy="max",
        keep_date="end",
    ).X[:, 1]
    visit = {visit_id: i for i, visit_id in enumerate(edata.obs["visit_occurrence_id"])}
    # the exposures of the second day end on the third
    assert largest[visit[1], 2] == 51
    assert largest[visit[5], 2] == 55


def test_setup_variables_keeps_the_next_visit_out_of_a_longer_window(omop_connection_multiple_visit_occurrences):
    """With clip_to_end_date=False, the window of a visit can reach the person's next visit, whose data points it does not take."""
    con = omop_connection_multiple_visit_occurrences
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=100,
        clip_to_end_date=False,
    )
    visit = {visit_id: i for i, visit_id in enumerate(edata.obs["visit_occurrence_id"])}

    # visit 1 of person 1 starts 2100-01-01, its visit 4 2100-03-01, 59 days later
    assert not np.isnan(edata.X[visit[4]]).all()
    assert np.isnan(edata.X[visit[1], :, 31:]).all()


def test_setup_variables_specimen_goes_to_every_overlapping_visit(omop_connection_vanilla):
    """specimen has no visit_occurrence_id, so its data points belong to every visit whose window holds them."""
    con = omop_connection_vanilla
    _add_overlapping_visit(con)
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["specimen"],
        data_field_to_keep=["quantity"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=4,
    )
    visit = {visit_id: i for i, visit_id in enumerate(edata.obs["visit_occurrence_id"])}

    assert not np.isnan(edata.X[visit[1]]).all()
    np.testing.assert_array_equal(edata.X[visit[5]], edata.X[visit[1]])


def test_setup_variables_without_a_visit_table_attributes_by_person_and_time(omop_connection_vanilla):
    """Without a visit_occurrence table, the visit ids cannot be checked, and the data points go by person and time."""
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    kwargs = {
        "backend_handle": con,
        "data_tables": ["measurement"],
        "data_field_to_keep": ["value_as_number"],
        "interval_length_number": 1,
        "interval_length_unit": "day",
        "num_intervals": 4,
    }
    expected = ed.io.omop.setup_variables(edata, **kwargs).X
    con.execute("DROP TABLE visit_occurrence")

    np.testing.assert_array_equal(ed.io.omop.setup_variables(edata, **kwargs).X, expected)


def _setup_measurement(con, concept_ids, **kwargs):
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    kwargs = {
        "data_tables": ["measurement"],
        "data_field_to_keep": ["value_as_number"],
        "interval_length_number": 12,
        "interval_length_unit": "h",
        "num_intervals": 4,
        **kwargs,
    }
    return ed.io.omop.setup_variables(edata, backend_handle=con, concept_ids=concept_ids, **kwargs)


def _add_second_concept_values(con, unit_concept_id=9557, unit_source_value="mEq/L"):
    """Give concept 3022318, whose toy rows have no value, a value per person on 2100-01-01 15:00."""
    con.execute(
        f"""INSERT INTO measurement (measurement_id, person_id, measurement_concept_id, measurement_date,
            measurement_datetime, value_as_number, unit_concept_id, unit_source_value)
        SELECT 100 + person_id, person_id, 3022318, DATE '2100-01-01', TIMESTAMP '2100-01-01 15:00:00',
            50 + person_id, {unit_concept_id}, '{unit_source_value}' FROM person"""
    )


@pytest.mark.parametrize("aggregation_strategy", ["count", "max", "min", "sum", "last", "first"])
def test_setup_variables_concept_sets_combine_their_concepts(omop_connection_vanilla, aggregation_strategy):
    con = omop_connection_vanilla
    _add_second_concept_values(con)
    single = _setup_measurement(con, [3031147, 3022318], aggregation_strategy=aggregation_strategy)
    concept_set = _setup_measurement(
        con, {"both": [3031147, 3022318], "absent": [123456789]}, aggregation_strategy=aggregation_strategy
    )

    assert list(concept_set.var_names) == ["both", "absent"]
    assert concept_set.var["concept_ids"].tolist() == ["3031147,3022318", "123456789"]
    first, second = (single.X[:, single.var["data_table_concept_id"].tolist().index(c)] for c in (3031147, 3022318))
    combine = {
        "count": np.add,
        "sum": lambda a, b: np.where(np.isnan(a), b, np.where(np.isnan(b), a, a + b)),
        "max": np.fmax,
        "min": np.fmin,
        # the rows of 3022318 come after those of 3031147, at 15:00
        "last": lambda a, b: np.where(np.isnan(b), a, b),
        "first": lambda a, b: np.where(np.isnan(a), b, a),
    }[aggregation_strategy]
    np.testing.assert_array_equal(concept_set.X[:, 0], combine(first, second))
    expected_empty = 0 if aggregation_strategy == "count" else np.nan
    np.testing.assert_array_equal(concept_set.X[:, 1], np.full_like(concept_set.X[:, 1], expected_empty))


def test_setup_variables_concept_set_of_one_concept_equals_the_concept(omop_connection_vanilla):
    con = omop_connection_vanilla
    single = _setup_measurement(con, [3031147])
    concept_set = _setup_measurement(con, {"bicarbonate": [3031147]})

    np.testing.assert_array_equal(concept_set.X, single.X)
    unit_report = concept_set.uns["unit_report_measurement"]
    assert unit_report["variable"].tolist() == ["bicarbonate"]
    assert unit_report["unit_concept_id"].tolist() == single.uns["unit_report_measurement"]["unit_concept_id"].tolist()


def test_setup_variables_concept_sets_check_the_units_per_set(omop_connection_vanilla):
    """Each concept has one unit, but the set combines two, which a strategy combining values cannot."""
    con = omop_connection_vanilla
    _add_second_concept_values(con, unit_concept_id=8753, unit_source_value="mmol/L")
    _setup_measurement(con, [3031147, 3022318], aggregation_strategy="mean")

    with pytest.raises(
        NotImplementedError, match=r"1 feature\(s\) have more than one unit_concept_id .*\(both: \[8753, 9557\]\)"
    ):
        _setup_measurement(con, {"both": [3031147, 3022318]}, aggregation_strategy="mean")
    # hourly, so that the rows of both units are kept in cells of their own
    edata = _setup_measurement(
        con,
        {"both": [3031147, 3022318]},
        aggregation_strategy="last",
        interval_length_number=1,
        interval_length_unit="h",
        num_intervals=24,
        time_precision="datetime",
    )
    unit_report = edata.uns["unit_report_measurement"]
    assert sorted(unit_report["unit_concept_id"].tolist()) == [8753, 9557]
    assert unit_report["multiple_units"].all()


def test_setup_variables_concept_sets_enrich_var_with_unit_info(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = _setup_measurement(con, {"bicarbonate": [3031147]}, enrich_var_with_unit_info=True)

    assert edata.var.loc["bicarbonate", "unit_concept_id"] == 9557
    # the unit's concept is in prefixed columns, apart from the concept ids of the set
    assert "concept_id" not in edata.var.columns
    assert {"unit_concept_name", "unit_vocabulary_id"} <= set(edata.var.columns)


def test_setup_variables_concept_sets_sparse_and_dataset(omop_connection_vanilla):
    con = omop_connection_vanilla
    _add_second_concept_values(con)
    concept_sets = {"both": [3031147, 3022318], "second": [123456789]}
    dense = _setup_measurement(con, concept_sets)
    sparse_edata = _setup_measurement(con, concept_sets, sparse=True)
    np.testing.assert_array_equal(sparse_edata.X.todense(), dense.X)

    torch = pytest.importorskip("torch")
    edata = _setup_measurement(con, concept_sets, instantiate_tensor=False)
    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"], datetime=False)
    for index in range(edata.n_obs):
        torch.testing.assert_close(
            ehr_dataset[index][0], torch.tensor(dense.X[index], dtype=torch.float32), equal_nan=True
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (
            {
                "data_tables": ["measurement", "observation"],
                "data_field_to_keep": dict.fromkeys(["measurement", "observation"], "value_as_number"),
            },
            ValueError,
            "Concept sets take one data table",
        ),
    ],
)
def test_setup_variables_concept_sets_reject_other_arguments(omop_connection_vanilla, kwargs, error, match):
    with pytest.raises(error, match=match):
        _setup_measurement(omop_connection_vanilla, {"bicarbonate": [3031147]}, **kwargs)


@pytest.mark.parametrize(
    ("concept_ids", "error", "match"),
    [
        ({}, ValueError, "must not be empty"),
        ({"a": []}, TypeError, "non-empty sequence of integers"),
        ({"a": "3031147"}, TypeError, "non-empty sequence of integers"),
        ({"a": [3031147.0]}, TypeError, "integer concept ids"),
        ({"a": [True]}, TypeError, "integer concept ids"),
        ({1: [3031147]}, TypeError, "non-empty strings"),
        ({"a": [3031147], "b": [3031147, 3022318]}, ValueError, "may belong to one only"),
    ],
)
def test_setup_variables_illegal_concept_sets(omop_connection_vanilla, concept_ids, error, match):
    with pytest.raises(error, match=match):
        _setup_measurement(omop_connection_vanilla, concept_ids)


def test_setup_interval_variables_rejects_concept_sets(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    with pytest.raises(TypeError, match="concept_ids must be a sequence of integers or 'all'"):
        ed.io.omop.setup_interval_variables(
            edata,
            backend_handle=con,
            data_tables=["drug_exposure"],
            data_field_to_keep=["days_supply"],
            interval_length_number=1,
            interval_length_unit="day",
            num_intervals=4,
            concept_ids={"a": [19073183]},
        )


@pytest.mark.parametrize(
    ("aggregation_strategy", "expected"),
    [("mean", (18 + 19 + 51) / 3), ("median", 19), ("mode", 18), ("std", np.std([18, 19, 51], ddof=1))],
)
def test_setup_variables_concept_sets_combine_values_across_concepts(
    omop_connection_vanilla, aggregation_strategy, expected
):
    """Person 1 has 18 and 19 of concept 3031147, and 51 of concept 3022318, in the 12 h from noon on 2100-01-01."""
    con = omop_connection_vanilla
    _add_second_concept_values(con)
    edata = _setup_measurement(
        con, {"both": [3031147, 3022318]}, aggregation_strategy=aggregation_strategy, time_precision="datetime"
    )

    assert edata.X[0, 0, 1] == pytest.approx(expected)


def test_setup_variables_concept_sets_is_present(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = _setup_measurement(
        con, {"both": [3031147, 3022318]}, data_field_to_keep=["is_present"], aggregation_strategy="count"
    )

    assert set(np.unique(edata.X[~np.isnan(edata.X)])) == {1}


@pytest.mark.parametrize("aggregation_strategy", ["last", "count"])
def test_setup_variables_concept_sets_without_any_data(omop_connection_vanilla, aggregation_strategy):
    """Sets without data points are variables, also if no set has any."""
    con = omop_connection_vanilla
    edata = _setup_measurement(con, {"a": [123456789], "b": [987654321]}, aggregation_strategy=aggregation_strategy)

    assert list(edata.var_names) == ["a", "b"]
    np.testing.assert_array_equal(edata.X, np.full(edata.X.shape, 0 if aggregation_strategy == "count" else np.nan))


def test_setup_variables_concept_sets_on_a_text_concept_column(omop_connection_vanilla):
    con = omop_connection_vanilla
    _add_second_concept_values(con)
    expected = _setup_measurement(con, {"both": [3031147, 3022318]}, aggregation_strategy="mean").X
    con.execute(
        "CREATE OR REPLACE TABLE measurement AS "
        "SELECT * REPLACE (CAST(measurement_concept_id AS VARCHAR) AS measurement_concept_id) FROM measurement"
    )

    np.testing.assert_array_equal(
        _setup_measurement(con, {"both": [3031147, 3022318]}, aggregation_strategy="mean").X, expected
    )


@pytest.mark.parametrize("ids", [np.array([3031147, 3022318]), pd.Series([3031147, 3022318]), (3031147, 3022318)])
def test_setup_variables_concept_sets_of_arrays(omop_connection_vanilla, ids):
    con = omop_connection_vanilla
    expected = _setup_measurement(con, {"both": [3031147, 3022318]}).X

    np.testing.assert_array_equal(_setup_measurement(con, {"both": ids}).X, expected)


def test_ehrdataset_rejects_a_long_table_of_another_setup(omop_connection_vanilla):
    pytest.importorskip("torch")
    con = omop_connection_vanilla
    edata = _setup_measurement(con, {"both": [3031147, 3022318]}, instantiate_tensor=False)
    _setup_measurement(con, [3031147, 3022318], instantiate_tensor=False)

    with pytest.raises(ValueError, match="was set up by another call of setup_variables"):
        ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"])


def test_setup_variables_concept_sets_enrich_var_with_feature_info(omop_connection_vanilla):
    """Each field of the concept table lists the values of the concepts of a set, in the order of the set.

    The concept table of the toy CDM holds the source concepts, which the measurement concepts are mapped from.
    """
    con = omop_connection_vanilla
    edata = _setup_measurement(
        con, {"combined": [3022318, 3031147], "absent": [123456789]}, enrich_var_with_feature_info=True
    )

    assert edata.var["concept_name"].tolist() == ["Heart Rhythm; Calculated Total CO2|Blood|Blood Gas", ""]
    assert edata.var.loc["combined", "concept_code"] == "220048; 50804"
    assert edata.var.loc["combined", "vocabulary_id"] == "mimiciv_meas_chart; mimiciv_meas_lab_loinc"
    assert edata.var["concept_ids"].tolist() == ["3022318,3031147", "123456789"]

    reversed_order = _setup_measurement(con, {"combined": [3031147, 3022318]}, enrich_var_with_feature_info=True)
    assert reversed_order.var.loc["combined", "concept_name"] == "Calculated Total CO2|Blood|Blood Gas; Heart Rhythm"


@pytest.mark.parametrize(
    "strategies",
    [
        {3031147: "sum", 3022318: "last"},
        {3031147: "median", 3022318: "count"},
        {3031147: "count", 3022318: "max"},
        {3031147: "mode", 3022318: "std"},
    ],
)
def test_setup_variables_aggregation_strategy_per_variable(omop_connection_vanilla, strategies):
    """A strategy per variable gives each variable what its strategy gives it alone."""
    con = omop_connection_vanilla
    _add_second_concept_values(con)
    edata = _setup_measurement(con, [3031147, 3022318], aggregation_strategy=strategies)

    concepts = edata.var["data_table_concept_id"].tolist()
    for concept_id, strategy in strategies.items():
        alone = _setup_measurement(con, [3031147, 3022318], aggregation_strategy=strategy)
        index = concepts.index(concept_id)
        np.testing.assert_array_equal(edata.X[:, index], alone.X[:, index])


def test_setup_variables_aggregation_strategy_per_concept_set(omop_connection_vanilla):
    con = omop_connection_vanilla
    _add_second_concept_values(con)
    concept_sets = {"first": [3031147], "second": [3022318], "absent": [123456789]}
    strategies = {"first": "sum", "second": "max", "absent": "count"}
    edata = _setup_measurement(con, concept_sets, aggregation_strategy=strategies)

    for index, (name, ids) in enumerate(concept_sets.items()):
        alone = _setup_measurement(con, {name: ids}, aggregation_strategy=strategies[name])
        np.testing.assert_array_equal(edata.X[:, index], alone.X[:, 0])
    np.testing.assert_array_equal(edata.X[:, 2], np.zeros_like(edata.X[:, 2]))


def test_setup_variables_aggregation_strategy_checks_the_units_per_variable(omop_connection_vanilla):
    """Concept 3022318 has two units, which 'mean' cannot combine but 'last' can, whatever the other variable takes."""
    con = omop_connection_vanilla
    _add_second_concept_values(con)
    _add_second_concept_values(con, unit_concept_id=8753, unit_source_value="mmol/L")
    con.execute("UPDATE measurement SET measurement_id = measurement_id + 1000 WHERE unit_concept_id = 8753")

    with pytest.raises(NotImplementedError, match="3022318"):
        _setup_measurement(con, [3031147, 3022318], aggregation_strategy={3031147: "last", 3022318: "mean"})
    _setup_measurement(con, [3031147, 3022318], aggregation_strategy={3031147: "mean", 3022318: "last"})


def test_setup_variables_aggregation_strategy_sparse(omop_connection_vanilla):
    con = omop_connection_vanilla
    strategies = {3031147: "sum", 3022318: "max"}
    dense = _setup_measurement(con, [3031147, 3022318], aggregation_strategy=strategies)
    sparse_edata = _setup_measurement(con, [3031147, 3022318], aggregation_strategy=strategies, sparse=True)
    np.testing.assert_array_equal(sparse_edata.X.todense(), dense.X)

    with pytest.raises(ValueError, match="sparse=True needs one fill value"):
        _setup_measurement(
            con, [3031147, 3022318], aggregation_strategy={3031147: "sum", 3022318: "count"}, sparse=True
        )


def test_setup_variables_aggregation_strategy_for_a_requested_concept_without_data(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = _setup_measurement(con, [3031147, 123456789], aggregation_strategy={3031147: "sum", 123456789: "max"})

    assert edata.var["data_table_concept_id"].tolist() == [3031147]


@pytest.mark.parametrize(
    ("concept_ids", "aggregation_strategy", "error", "match"),
    [
        ("all", {3031147: "sum"}, ValueError, r"no strategy for the variable\(s\) \[3022318\]"),
        ([3031147], {3031147: "sum", 3022318: "max"}, ValueError, "which are no variables"),
        ({"a": [3031147]}, {"a": "sum", "b": "max"}, ValueError, "which are no variables"),
        ({"a": [3031147]}, {"a": "average"}, TypeError, "aggregation_strategy must be one of"),
    ],
)
def test_setup_variables_illegal_aggregation_strategy_per_variable(
    omop_connection_vanilla, concept_ids, aggregation_strategy, error, match
):
    with pytest.raises(error, match=match):
        _setup_measurement(omop_connection_vanilla, concept_ids, aggregation_strategy=aggregation_strategy)


def test_setup_interval_variables_rejects_aggregation_strategy_per_variable(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    with pytest.raises(TypeError, match="aggregation_strategy must be one of"):
        ed.io.omop.setup_interval_variables(
            edata,
            backend_handle=con,
            data_tables=["drug_exposure"],
            data_field_to_keep=["days_supply"],
            interval_length_number=1,
            interval_length_unit="day",
            num_intervals=4,
            aggregation_strategy={19073183: "sum"},
        )


def test_setup_variables_aggregation_strategy_mapping_of_one_strategy_equals_the_strategy(omop_connection_vanilla):
    con = omop_connection_vanilla
    for strategy in ["count", "mean"]:
        alone = _setup_measurement(con, "all", aggregation_strategy=strategy)
        mapping = _setup_measurement(con, "all", aggregation_strategy={3031147: strategy, np.int64(3022318): strategy})
        assert mapping.X.dtype == alone.X.dtype
        np.testing.assert_array_equal(mapping.X, alone.X)


def test_setup_variables_aggregation_strategy_over_several_data_tables(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    kwargs = {
        "backend_handle": con,
        "data_tables": ["measurement", "observation"],
        "data_field_to_keep": {"measurement": "value_as_number", "observation": "value_as_number"},
        "interval_length_number": 1,
        "interval_length_unit": "day",
        "num_intervals": 4,
    }
    concepts = ed.io.omop.setup_variables(edata, **kwargs).var["data_table_concept_id"].tolist()
    # all of measurement count, observation is aggregated otherwise: int and float tables are concatenated
    strategies = dict.fromkeys(concepts[:2], "count") | dict.fromkeys(concepts[2:], "max")
    mixed = ed.io.omop.setup_variables(edata, aggregation_strategy=strategies, **kwargs)

    counted = ed.io.omop.setup_variables(edata, aggregation_strategy="count", **kwargs)
    largest = ed.io.omop.setup_variables(edata, aggregation_strategy="max", **kwargs)
    np.testing.assert_array_equal(mixed.X[:, :2], counted.X[:, :2])
    np.testing.assert_array_equal(mixed.X[:, 2:], largest.X[:, 2:])


def test_setup_variables_aggregation_strategy_without_concept_id(omop_connection_vanilla):
    """With a strategy per variable, rows without a concept id belong to no variable."""
    con = omop_connection_vanilla
    con.execute("UPDATE measurement SET measurement_concept_id = NULL WHERE measurement_id = 1")
    edata = _setup_measurement(con, "all", aggregation_strategy={3031147: "sum", 3022318: "count"})

    assert edata.var["data_table_concept_id"].tolist() == [3022318, 3031147]


def test_setup_variables_aggregation_strategy_is_present_sparse(omop_connection_vanilla):
    """data_field_to_keep='is_present' fills NaN whatever the strategy, so count next to others can be sparse."""
    con = omop_connection_vanilla
    kwargs = {"data_field_to_keep": ["is_present"], "aggregation_strategy": {3031147: "count", 3022318: "mean"}}
    dense = _setup_measurement(con, [3031147, 3022318], **kwargs)
    np.testing.assert_array_equal(
        _setup_measurement(con, [3031147, 3022318], sparse=True, **kwargs).X.todense(), dense.X
    )


def test_setup_variables_aggregation_strategy_quoted_set_name(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = _setup_measurement(
        con, {"it's": [3031147], "other": [3022318]}, aggregation_strategy={"it's": "sum", "other": "count"}
    )

    np.testing.assert_array_equal(edata.X[:, 0], _setup_measurement(con, [3031147], aggregation_strategy="sum").X[:, 0])


@pytest.mark.parametrize(
    ("concept_ids", "aggregation_strategy", "data_field_to_keep", "error", "match"),
    [
        (
            [3031147, 3022318],
            {"3031147": "sum", "3022318": "max"},
            ["value_as_number"],
            TypeError,
            "integer concept ids",
        ),
        ([3031147, 3022318], {True: "sum", 3022318: "max"}, ["value_as_number"], TypeError, "integer concept ids"),
        ({"a": [3031147]}, {3031147: "sum"}, ["value_as_number"], TypeError, "names of the sets"),
        (
            [3031147, 3022318],
            {3031147: "last", 3022318: "count"},
            ["unit_source_value"],
            TypeError,
            "needs a numeric data field",
        ),
    ],
)
def test_setup_variables_illegal_aggregation_strategy_keys_and_fields(
    omop_connection_vanilla, concept_ids, aggregation_strategy, data_field_to_keep, error, match
):
    with pytest.raises(error, match=match):
        _setup_measurement(
            omop_connection_vanilla,
            concept_ids,
            aggregation_strategy=aggregation_strategy,
            data_field_to_keep=data_field_to_keep,
        )


@pytest.mark.parametrize("aggregation_strategy", ["last", "first", "mean", "count", "sum", "max"])
@pytest.mark.parametrize("bounds", [(None, 18.5), (20, 22), (19, None), (100, 200)])
def test_setup_variables_value_range_equals_dropping_the_rows(omop_connection_vanilla, aggregation_strategy, bounds):
    """The rows of concept 3031147 (persons 1 to 3: 18 and 19, 20 and 21, 22 and 23) outside the range are left out.

    As rows without a value: the variable stays, also if the range leaves none of its rows.
    """
    con = omop_connection_vanilla
    ranged = _setup_measurement(con, "all", value_range={3031147: bounds}, aggregation_strategy=aggregation_strategy)

    lower, upper = bounds
    con.execute(
        f"""UPDATE measurement SET value_as_number = NULL WHERE measurement_concept_id = 3031147
        AND (value_as_number < {lower if lower is not None else "-'infinity'::DOUBLE"}
             OR value_as_number > {upper if upper is not None else "'infinity'::DOUBLE"})"""
    )
    dropped = _setup_measurement(con, "all", aggregation_strategy=aggregation_strategy)

    np.testing.assert_array_equal(ranged.X, dropped.X)
    pd.testing.assert_frame_equal(ranged.var, dropped.var)


def test_setup_variables_value_range_of_a_concept_set(omop_connection_vanilla):
    con = omop_connection_vanilla
    _add_second_concept_values(con)
    # 51 to 53 of concept 3022318 lie above the range, 18 to 23 of 3031147 in it
    edata = _setup_measurement(
        con, {"both": [3031147, 3022318]}, value_range={"both": (0, 30)}, aggregation_strategy="count"
    )
    alone = _setup_measurement(con, {"both": [3031147]}, aggregation_strategy="count")

    np.testing.assert_array_equal(edata.X, alone.X)


def test_setup_variables_value_range_leaves_out_the_units_of_dropped_rows(omop_connection_vanilla):
    """Implausible values in another unit neither block a combining strategy nor show in the unit report."""
    con = omop_connection_vanilla
    con.execute(
        """INSERT INTO measurement (measurement_id, person_id, measurement_concept_id, measurement_date,
            measurement_datetime, value_as_number, unit_concept_id, unit_source_value)
        VALUES (10, 1, 3031147, '2100-01-01', '2100-01-01 15:00:00', 18000, 8840, 'mg/dL')"""
    )
    with pytest.raises(NotImplementedError):
        _setup_measurement(con, [3031147], aggregation_strategy="mean")

    edata = _setup_measurement(con, [3031147], aggregation_strategy="mean", value_range={3031147: (0, 100)})
    assert edata.uns["unit_report_measurement"]["unit_concept_id"].tolist() == [9557]
    # with time_precision='date', the rows of 2100-01-01 fall into the first interval
    assert edata.X[0, 0, 0] == pytest.approx(18.5)


def test_setup_variables_value_range_with_strategies_per_variable(omop_connection_vanilla):
    con = omop_connection_vanilla
    _add_second_concept_values(con)
    edata = _setup_measurement(
        con,
        [3031147, 3022318],
        aggregation_strategy={3031147: "sum", 3022318: "count"},
        value_range={3031147: (19, 22), 3022318: (None, 52)},
    )
    concepts = edata.var["data_table_concept_id"].tolist()

    # sums of 19, of 20 and 21, and of 22; the 53 of person 3 lies above 52
    np.testing.assert_array_equal(edata.X[:, concepts.index(3031147), 0], [19, 41, 22])
    np.testing.assert_array_equal(edata.X[:, concepts.index(3022318), 0], [1, 1, 0])


def test_setup_variables_value_range_for_a_requested_concept_without_data(omop_connection_vanilla):
    edata = _setup_measurement(omop_connection_vanilla, [3031147, 123456789], value_range={123456789: (0, 1)})

    assert edata.var["data_table_concept_id"].tolist() == [3031147]


@pytest.mark.parametrize(
    ("concept_ids", "value_range", "kwargs", "error", "match"),
    [
        ("all", {999: (0, 1)}, {}, ValueError, "which are no variables"),
        ({"a": [3031147]}, {"b": (0, 1)}, {}, ValueError, "which are no variables"),
        ("all", {3031147: (5, 1)}, {}, ValueError, "lower bound above its upper bound"),
        ("all", {3031147: ("0", 1)}, {}, TypeError, "must be numbers or None"),
        ("all", {3031147: (float("nan"), 1)}, {}, TypeError, "must be numbers or None"),
        ("all", {3031147: (True, 1)}, {}, TypeError, "must be numbers or None"),
        ("all", {3031147: (0, 1, 2)}, {}, TypeError, "must be a pair"),
        ("all", {"3031147": (0, 1)}, {}, TypeError, "integer concept ids"),
        ({"a": [3031147]}, {3031147: (0, 1)}, {}, TypeError, "names of the sets"),
        ("all", [(0, 1)], {}, TypeError, "must be a mapping"),
        ("all", {3031147: (0, 1)}, {"data_field_to_keep": ["unit_source_value"]}, TypeError, "numeric data field"),
        ("all", {3031147: (0, 1)}, {"data_field_to_keep": ["is_present"]}, TypeError, "numeric data field"),
    ],
)
def test_setup_variables_illegal_value_range(omop_connection_vanilla, concept_ids, value_range, kwargs, error, match):
    with pytest.raises(error, match=match):
        _setup_measurement(omop_connection_vanilla, concept_ids, value_range=value_range, **kwargs)


@pytest.mark.parametrize("bounds", [(0, None), (None, 100), (0, np.inf)])
def test_setup_variables_value_range_leaves_out_nan(omop_connection_vanilla, bounds):
    """NaN lies in no range, whichever side is open."""
    con = omop_connection_vanilla
    con.execute(
        "CREATE OR REPLACE TABLE measurement AS SELECT * REPLACE (CAST(value_as_number AS DOUBLE) AS value_as_number) FROM measurement"
    )
    con.execute("UPDATE measurement SET value_as_number = 'NaN' WHERE measurement_id = 1")
    edata = _setup_measurement(con, [3031147], value_range={3031147: bounds}, aggregation_strategy="count")

    np.testing.assert_array_equal(edata.X[:, 0, 0], [1, 2, 2])


@pytest.mark.parametrize("value_type", ["REAL", "DOUBLE", "DECIMAL(18, 3)"])
def test_setup_variables_value_range_includes_its_bounds(omop_connection_vanilla, value_type):
    con = omop_connection_vanilla
    con.execute(
        f"CREATE OR REPLACE TABLE measurement AS SELECT * REPLACE (CAST(value_as_number + 0.1 AS {value_type}) AS value_as_number) FROM measurement"
    )
    edata = _setup_measurement(con, [3031147], value_range={3031147: (18.1, 23.1)}, aggregation_strategy="count")

    np.testing.assert_array_equal(edata.X[:, 0, 0], [2, 2, 2])


def test_setup_variables_empty_value_range_changes_nothing(omop_connection_vanilla):
    con = omop_connection_vanilla
    con.execute("UPDATE measurement SET measurement_concept_id = NULL WHERE measurement_id = 1")
    expected = _setup_measurement(con, "all")
    edata = _setup_measurement(con, "all", value_range={})

    pd.testing.assert_frame_equal(edata.var, expected.var)
    np.testing.assert_array_equal(edata.X, expected.X)


def test_setup_variables_value_range_rows_without_concept_id_are_no_variable(omop_connection_vanilla):
    """Rows without a concept id, in two units, do not block 'mean', as they belong to no variable."""
    con = omop_connection_vanilla
    con.execute(
        """INSERT INTO measurement (measurement_id, person_id, measurement_concept_id, measurement_date,
            measurement_datetime, value_as_number, unit_concept_id, unit_source_value)
        VALUES (10, 1, NULL, '2100-01-01', '2100-01-01 15:00:00', 1, 1, 'a'),
               (11, 1, NULL, '2100-01-01', '2100-01-01 16:00:00', 2, 2, 'b')"""
    )
    edata = _setup_measurement(con, "all", aggregation_strategy="mean", value_range={3031147: (0, 100)})

    assert edata.var["data_table_concept_id"].tolist() == [3022318, 3031147]
    assert edata.uns["unit_report_measurement"]["concept_id"].notna().all()


def test_setup_variables_value_range_in_every_data_table_of_a_concept(omop_connection_vanilla):
    """The range of a concept filters it in every data table that has it."""
    con = omop_connection_vanilla
    con.execute(
        """INSERT INTO observation (observation_id, person_id, observation_concept_id, observation_date,
            observation_datetime, value_as_number)
        VALUES (10, 1, 3031147, '2100-01-01', '2100-01-01 12:00:00', 1000),
               (11, 1, 3031147, '2100-01-01', '2100-01-01 13:00:00', 20)"""
    )
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement", "observation"],
        data_field_to_keep={"measurement": "value_as_number", "observation": "value_as_number"},
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=4,
        aggregation_strategy="max",
        value_range={3031147: (0, 100)},
    )
    # the measurement concepts come first, then those of observation
    observation_concept = edata.var["data_table_concept_id"].tolist().index(3031147, 2)

    assert edata.X[0, observation_concept, 0] == 20


def test_setup_variables_value_range_sparse(omop_connection_vanilla):
    con = omop_connection_vanilla
    kwargs = {"value_range": {3031147: (19, 22)}, "aggregation_strategy": "mean"}
    dense = _setup_measurement(con, [3031147], **kwargs)

    np.testing.assert_array_equal(_setup_measurement(con, [3031147], sparse=True, **kwargs).X.todense(), dense.X)


def test_setup_variables_value_range_infinite_bounds(omop_connection_vanilla):
    """Infinite bounds, also as integers too large for a float, leave their side open."""
    con = omop_connection_vanilla
    expected = _setup_measurement(con, [3031147], value_range={3031147: (19, None)})

    for upper in [np.inf, 10**400]:
        edata = _setup_measurement(con, [3031147], value_range={3031147: (19, upper)})
        np.testing.assert_array_equal(edata.X, expected.X)


def _second_unit_for_person_2(con):
    """Record the values of concept 3031147 of person 2 in umol/L (unit concept 8842), 1000-fold."""
    con.execute(
        """UPDATE measurement SET value_as_number = value_as_number * 1000, unit_concept_id = 8842, unit_source_value = 'umol/L'
        WHERE person_id = 2 AND measurement_concept_id = 3031147"""
    )


BICARBONATE_CONVERSIONS = pd.DataFrame(
    {
        "concept_id": [3031147, 3031147],
        "unit_concept_id": [9557, 8842],
        "to_unit_concept_id": [9557, 9557],
        "multiplier": [1, 0.001],
        "to_unit_source_value": ["mEq/L", "mEq/L"],
    }
)


@pytest.mark.parametrize("aggregation_strategy", ["mean", "last", "sum", "max"])
def test_setup_variables_unit_conversions_equal_converting_the_rows(omop_connection_vanilla, aggregation_strategy):
    con = omop_connection_vanilla
    expected = _setup_measurement(con, [3031147], aggregation_strategy=aggregation_strategy)
    expected_long = con.execute(
        "SELECT * FROM long_person_timestamp_feature_value_measurement ORDER BY obs_index, interval_step"
    ).df()
    _second_unit_for_person_2(con)

    converted = _setup_measurement(
        con, [3031147], aggregation_strategy=aggregation_strategy, unit_conversions=BICARBONATE_CONVERSIONS
    )
    converted_long = con.execute(
        "SELECT * FROM long_person_timestamp_feature_value_measurement ORDER BY obs_index, interval_step"
    ).df()

    np.testing.assert_allclose(converted.X, expected.X)
    pd.testing.assert_frame_equal(
        converted.uns["unit_report_measurement"], expected.uns["unit_report_measurement"], check_dtype=False
    )
    # the converted rows carry the target unit in both unit fields
    assert converted_long["unit_concept_id"].tolist() == expected_long["unit_concept_id"].tolist()
    assert converted_long["unit_source_value"].tolist() == expected_long["unit_source_value"].tolist()


def test_setup_variables_unit_conversions_need_every_unit(omop_connection_vanilla):
    con = omop_connection_vanilla
    _second_unit_for_person_2(con)
    with pytest.raises(NotImplementedError, match="more than one unit_concept_id"):
        _setup_measurement(con, [3031147], aggregation_strategy="mean")
    with pytest.raises(ValueError, match=r"concept 3031147 in unit 8842 \(2 data points\)"):
        _setup_measurement(con, [3031147], unit_conversions=BICARBONATE_CONVERSIONS.iloc[:1])


def test_setup_variables_unit_conversions_with_offset_and_without_unit(omop_connection_vanilla):
    """An offset, as from degrees Fahrenheit, and a conversion of the data points without a unit."""
    con = omop_connection_vanilla
    con.execute("UPDATE measurement SET unit_concept_id = NULL, unit_source_value = NULL WHERE person_id = 3")
    conversions = pd.DataFrame(
        {
            "concept_id": [3031147, 3031147],
            "unit_concept_id": [9557, None],
            "to_unit_concept_id": [9557, 9557],
            "multiplier": [2.0, 0.5],
            "offset": [-32.0, 1.0],
        }
    )
    edata = _setup_measurement(con, [3031147], aggregation_strategy="max", unit_conversions=conversions)

    # persons 1 and 2: 2 * 19 - 32, 2 * 21 - 32; person 3: 0.5 * 23 + 1
    np.testing.assert_allclose(edata.X[:, 0, 0], [6, 10, 12.5])
    assert edata.uns["unit_report_measurement"]["unit_concept_id"].tolist() == [9557]


def test_setup_variables_unit_conversions_before_the_value_range(omop_connection_vanilla):
    con = omop_connection_vanilla
    _second_unit_for_person_2(con)
    edata = _setup_measurement(
        con,
        [3031147],
        aggregation_strategy="count",
        unit_conversions=BICARBONATE_CONVERSIONS,
        value_range={3031147: (19, 21)},
    )

    # 19 of person 1, and 20 and 21 of person 2 after their conversion from 20000 and 21000
    np.testing.assert_array_equal(edata.X[:, 0, 0], [1, 2, 0])


def test_setup_variables_unit_conversions_of_a_concept_set(omop_connection_vanilla):
    con = omop_connection_vanilla
    _second_unit_for_person_2(con)
    conversions = BICARBONATE_CONVERSIONS.drop(columns="concept_id").assign(variable="bicarbonate")
    edata = _setup_measurement(
        con, {"bicarbonate": [3031147]}, aggregation_strategy="mean", unit_conversions=conversions
    )

    np.testing.assert_allclose(edata.X[:, 0, 0], [18.5, 20.5, 22.5])


def test_setup_variables_unit_conversions_take_the_code_of_the_target_unit(omop_connection_vanilla):
    con = omop_connection_vanilla
    _second_unit_for_person_2(con)
    conversions = BICARBONATE_CONVERSIONS.drop(columns="to_unit_source_value")
    con.execute(
        "CREATE OR REPLACE TABLE concept AS SELECT * REPLACE (CAST(concept_code AS VARCHAR) AS concept_code) FROM concept"
    )
    con.execute(
        "INSERT INTO concept (concept_id, concept_name, domain_id, vocabulary_id, concept_class_id, concept_code) "
        "VALUES (9557, 'milliequivalent per liter', 'Unit', 'UCUM', 'Unit', 'meq/L')"
    )
    _setup_measurement(con, [3031147], unit_conversions=conversions)

    unit_source_values = con.execute(
        "SELECT DISTINCT unit_source_value FROM long_person_timestamp_feature_value_measurement"
    ).fetchall()
    assert unit_source_values == [("meq/L",)]


@pytest.mark.parametrize(
    ("concept_ids", "unit_conversions", "kwargs", "error", "match"),
    [
        ([3031147], BICARBONATE_CONVERSIONS.to_dict(), {}, TypeError, "must be a pandas DataFrame"),
        ([3031147], BICARBONATE_CONVERSIONS.assign(variable="a"), {}, ValueError, "exactly one of the columns"),
        ([3031147], BICARBONATE_CONVERSIONS.drop(columns="concept_id"), {}, ValueError, "exactly one of the columns"),
        (
            [3031147],
            BICARBONATE_CONVERSIONS.drop(columns="concept_id").assign(variable="a"),
            {},
            ValueError,
            "needs concept sets",
        ),
        (
            {"a": [3031147]},
            BICARBONATE_CONVERSIONS.drop(columns="concept_id").assign(variable="b"),
            {},
            ValueError,
            "which are no concept sets",
        ),
        (
            [3031147],
            BICARBONATE_CONVERSIONS.drop(columns="multiplier"),
            {},
            ValueError,
            r"lacks the column\(s\) \['multiplier'\]",
        ),
        (
            [3031147],
            BICARBONATE_CONVERSIONS.assign(unit_concept_id=[9557.5, 8842]),
            {},
            TypeError,
            "integer concept ids",
        ),
        ([3031147], BICARBONATE_CONVERSIONS.assign(multiplier=[1, np.inf]), {}, TypeError, "finite numbers"),
        ([3031147], BICARBONATE_CONVERSIONS.assign(unit_concept_id=[9557, 9557]), {}, ValueError, "more than once"),
        (
            [3031147],
            BICARBONATE_CONVERSIONS.assign(to_unit_concept_id=[9557, 8842]),
            {},
            ValueError,
            "to more than one unit",
        ),
        (
            [3031147],
            BICARBONATE_CONVERSIONS.assign(to_unit_concept_id=[True, True]),
            {},
            TypeError,
            "integer concept ids",
        ),
        ([3031147], BICARBONATE_CONVERSIONS.assign(offset=["a", 0]), {}, TypeError, "finite numbers"),
        ([3031147], BICARBONATE_CONVERSIONS, {"data_field_to_keep": ["is_present"]}, TypeError, "numeric data field"),
        (
            [3031147],
            BICARBONATE_CONVERSIONS,
            {"data_field_to_keep": ["unit_source_value"]},
            TypeError,
            "numeric data field",
        ),
    ],
)
def test_setup_variables_illegal_unit_conversions(
    omop_connection_vanilla, concept_ids, unit_conversions, kwargs, error, match
):
    with pytest.raises(error, match=match):
        _setup_measurement(omop_connection_vanilla, concept_ids, unit_conversions=unit_conversions, **kwargs)


def test_setup_variables_unit_conversions_of_other_concepts_are_not_used(omop_connection_vanilla):
    """One conversion table can serve several calls: rows of concepts that are no variables of a call are left aside."""
    con = omop_connection_vanilla
    _second_unit_for_person_2(con)
    con.execute("UPDATE measurement SET unit_concept_id = 8842 WHERE measurement_concept_id = 3022318")
    other = pd.DataFrame({"concept_id": [999], "unit_concept_id": [1], "to_unit_concept_id": [1], "multiplier": [1]})
    conversions = pd.concat([BICARBONATE_CONVERSIONS, other])
    edata = _setup_measurement(con, [3031147], aggregation_strategy="mean", unit_conversions=conversions)

    np.testing.assert_allclose(edata.X[:, 0, 0], [18.5, 20.5, 22.5])


def test_setup_variables_unit_conversions_leave_other_data_tables_as_they_are(omop_connection_vanilla):
    """A data table without converted concepts keeps its fields, also a text field or is_present."""
    con = omop_connection_vanilla
    _second_unit_for_person_2(con)
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    for observation_field in ["value_as_string", "is_present"]:
        kwargs = {
            "backend_handle": con,
            "data_tables": ["measurement", "observation"],
            "data_field_to_keep": {"measurement": "value_as_number", "observation": observation_field},
            "interval_length_number": 1,
            "interval_length_unit": "day",
            "num_intervals": 4,
            "concept_ids": [3031147, 3001062],
        }
        converted = ed.io.omop.setup_variables(edata, unit_conversions=BICARBONATE_CONVERSIONS, **kwargs)
        plain = ed.io.omop.setup_variables(edata, **kwargs)
        converted_values, plain_values = converted.X[:, 1:], plain.X[:, 1:]
        np.testing.assert_array_equal(pd.isna(converted_values), pd.isna(plain_values))
        assert converted_values[~pd.isna(converted_values)].tolist() == plain_values[~pd.isna(plain_values)].tolist()


def test_setup_variables_unit_conversions_of_a_concept_set_with_an_absent_concept(omop_connection_vanilla):
    con = omop_connection_vanilla
    _second_unit_for_person_2(con)
    conversions = BICARBONATE_CONVERSIONS.drop(columns="concept_id").assign(variable="bicarbonate")
    edata = _setup_measurement(
        con, {"bicarbonate": [3031147, 123456789]}, aggregation_strategy="mean", unit_conversions=conversions
    )

    np.testing.assert_allclose(edata.X[:, 0, 0], [18.5, 20.5, 22.5])


def test_setup_variables_unit_conversions_and_value_range_on_single_precision(omop_connection_vanilla):
    """A converted value at a bound of the range is kept, also if the value field is of single precision."""
    con = omop_connection_vanilla
    _second_unit_for_person_2(con)
    con.execute(
        "CREATE OR REPLACE TABLE measurement AS "
        "SELECT * REPLACE (CAST(value_as_number AS REAL) AS value_as_number) FROM measurement"
    )
    edata = _setup_measurement(
        con,
        [3031147],
        aggregation_strategy="count",
        unit_conversions=BICARBONATE_CONVERSIONS,
        value_range={3031147: (19, 20)},
    )

    # 19 of person 1, and 20 of person 2, converted from 20000
    np.testing.assert_array_equal(edata.X[:, 0, 0], [1, 1, 0])


def test_setup_variables_unit_conversions_unit_zero_and_missing_offsets(omop_connection_vanilla):
    """unit_concept_id 0 is a unit of its own, and a missing offset is 0."""
    con = omop_connection_vanilla
    con.execute("UPDATE measurement SET unit_concept_id = 0 WHERE person_id = 3")
    conversions = pd.DataFrame(
        {
            "concept_id": [3031147, 3031147],
            "unit_concept_id": [9557, 0],
            "to_unit_concept_id": [9557, 9557],
            "multiplier": [1, 2],
            "offset": [np.nan, 1],
        }
    )
    edata = _setup_measurement(con, [3031147], aggregation_strategy="max", unit_conversions=conversions)

    np.testing.assert_allclose(edata.X[:, 0, 0], [19, 21, 47])
    with pytest.raises(ValueError, match="in unit 0"):
        _setup_measurement(con, [3031147], unit_conversions=conversions.iloc[:1])


def test_setup_variables_unit_conversions_take_the_code_of_a_text_concept_table(omop_connection_vanilla):
    con = omop_connection_vanilla
    con.execute(
        "CREATE OR REPLACE TABLE concept AS SELECT * REPLACE (CAST(concept_id AS VARCHAR) AS concept_id, "
        "CAST(concept_code AS VARCHAR) AS concept_code) FROM concept"
    )
    con.execute(
        "INSERT INTO concept (concept_id, concept_name, domain_id, vocabulary_id, concept_class_id, concept_code) "
        "VALUES ('9557', 'milliequivalent per liter', 'Unit', 'UCUM', 'Unit', 'meq/L')"
    )
    conversions = BICARBONATE_CONVERSIONS.drop(columns="to_unit_source_value").iloc[:1]
    _setup_measurement(con, [3031147], unit_conversions=conversions)

    unit_source_values = con.execute(
        "SELECT DISTINCT unit_source_value FROM long_person_timestamp_feature_value_measurement"
    ).fetchall()
    assert unit_source_values == [("meq/L",)]


def test_setup_interval_variables_interval_bounds_and_missing_dates(omop_connection_vanilla):
    """With 6 h intervals: an exposure from one interval bound to the next, one without a start, and one without an end."""
    con = omop_connection_vanilla
    con.execute("DELETE FROM drug_exposure WHERE person_id = 1")
    con.execute(
        """INSERT INTO drug_exposure (drug_exposure_id, person_id, drug_concept_id, drug_exposure_start_date,
            drug_exposure_start_datetime, drug_exposure_end_date, drug_exposure_end_datetime, days_supply)
        VALUES (10, 1, 19073183, '2100-01-01', '2100-01-01 06:00:00', '2100-01-01', '2100-01-01 12:00:00', 10),
               (11, 1, 19073183, NULL, NULL, '2100-01-01', '2100-01-01 03:00:00', 20),
               (12, 1, 19073183, '2100-01-02', '2100-01-02 01:00:00', NULL, NULL, 30)"""
    )
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        data_tables=["drug_exposure"],
        data_field_to_keep=["days_supply"],
        interval_length_number=6,
        interval_length_unit="h",
        time_precision="datetime",
        num_intervals=8,
        aggregation_strategy="max",
        keep_date="interval",
    )

    # 10 starts on the bound of the second interval and ends on that of the third, which it then belongs to as well;
    # 20 has no start and belongs to no interval; 30 has no end and belongs to the interval of its start, the fifth
    np.testing.assert_array_equal(edata.X[0, 1], [np.nan, 10, 10, np.nan, 30, np.nan, np.nan, np.nan])


@pytest.mark.parametrize("clip_to_end_date", [True, False])
def test_setup_interval_variables_interval_after_the_end_date(omop_connection_vanilla, clip_to_end_date):
    """An exposure that starts after the end date of the observation, 2100-01-31, belongs to no interval unless clip_to_end_date=False."""
    con = omop_connection_vanilla
    con.execute("DELETE FROM drug_exposure WHERE person_id = 1")
    con.execute(
        """INSERT INTO drug_exposure (drug_exposure_id, person_id, drug_concept_id, drug_exposure_start_date,
            drug_exposure_start_datetime, drug_exposure_end_date, drug_exposure_end_datetime, days_supply)
        VALUES (10, 1, 19073183, '2100-02-02', '2100-02-02 00:00:00', '2100-02-03', '2100-02-03 00:00:00', 10)"""
    )
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        data_tables=["drug_exposure"],
        data_field_to_keep=["days_supply"],
        interval_length_number=7,
        interval_length_unit="day",
        num_intervals=6,
        aggregation_strategy="count",
        keep_date="interval",
        clip_to_end_date=clip_to_end_date,
    )

    # the fifth interval runs from 2100-01-29 to 2100-02-05
    np.testing.assert_array_equal(edata.X[0, 1], [0, 0, 0, 0, 0 if clip_to_end_date else 1, 0])


LOOK_BACK_ROWS = {
    # person 1, whose visit 1 starts on 2100-01-01 00:00: (time, value); the first lies before the two hours of the look-back
    "measurement": """INSERT INTO measurement (measurement_id, person_id, measurement_concept_id, measurement_date,
            measurement_datetime, value_as_number, unit_concept_id, unit_source_value, visit_occurrence_id)
        VALUES (20, 1, 3022318, '2099-12-31', '2099-12-31 21:30:00', 1, 9557, 'mEq/L', 1),
               (21, 1, 3022318, '2099-12-31', '2099-12-31 22:00:00', 2, 9557, 'mEq/L', 1),
               (22, 1, 3022318, '2099-12-31', '2099-12-31 23:59:59', 3, 9557, 'mEq/L', 1),
               (23, 1, 3022318, '2100-01-01', '2100-01-01 00:30:00', 4, 9557, 'mEq/L', 1)""",
}


@pytest.mark.parametrize("sparse", [False, True])
def test_setup_variables_num_intervals_before(omop_connection_vanilla, sparse):
    """The intervals before the start of an observation hold its data points there, at negative interval steps."""
    con = omop_connection_vanilla
    con.execute("DELETE FROM measurement WHERE person_id = 1")
    con.execute(LOOK_BACK_ROWS["measurement"])
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="h",
        time_precision="datetime",
        num_intervals=2,
        num_intervals_before=2,
        concept_ids=[3022318],
        aggregation_strategy="last",
        sparse=sparse,
    )
    row = list(edata.obs["visit_occurrence_id"]).index(1)
    x = edata.X.todense() if sparse else edata.X
    assert list(edata.tem["interval_step"]) == [-2, -1, 0, 1]
    np.testing.assert_array_equal(x[row, 0], [2, 3, 4, np.nan])
    steps = con.execute(
        "SELECT interval_step, interval_start FROM long_person_timestamp_feature_value_measurement "
        f"WHERE obs_index = {row} ORDER BY interval_step"
    ).fetchall()
    assert [step for step, _ in steps] == [-2, -1, 0]
    assert str(steps[0][1]) == "2099-12-31 22:00:00"


def test_setup_variables_num_intervals_before_attributes_rows_to_the_visit_they_name(omop_connection_vanilla):
    """In the intervals before a visit, a data point naming another visit of the person belongs to that one only."""
    con = omop_connection_vanilla
    con.execute("DELETE FROM measurement WHERE person_id = 1")
    con.execute(
        """INSERT INTO visit_occurrence (visit_occurrence_id, person_id, visit_concept_id, visit_start_date,
            visit_start_datetime, visit_end_date, visit_end_datetime)
        VALUES (10, 1, 9201, '2099-12-31', '2099-12-31 20:00:00', '2099-12-31', '2099-12-31 23:00:00')"""
    )
    con.execute(
        """INSERT INTO measurement (measurement_id, person_id, measurement_concept_id, measurement_date,
            measurement_datetime, value_as_number, unit_concept_id, unit_source_value, visit_occurrence_id)
        VALUES (20, 1, 3022318, '2099-12-31', '2099-12-31 22:30:00', 1, 9557, 'mEq/L', 10),
               (21, 1, 3022318, '2099-12-31', '2099-12-31 23:30:00', 2, 9557, 'mEq/L', NULL)"""
    )
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="h",
        time_precision="datetime",
        num_intervals=1,
        num_intervals_before=2,
        concept_ids=[3022318],
        aggregation_strategy="last",
    )
    visits = list(edata.obs["visit_occurrence_id"])
    # visit 1 starts on 2100-01-01 00:00: the data point naming visit 10 is not in its look-back, the one naming no visit is
    np.testing.assert_array_equal(edata.X[visits.index(1), 0], [np.nan, 2, np.nan])
    np.testing.assert_array_equal(edata.X[visits.index(10), 0], [np.nan, np.nan, np.nan])


def test_setup_variables_num_intervals_before_at_date_precision(omop_connection_vanilla):
    con = omop_connection_vanilla
    con.execute("DELETE FROM measurement WHERE person_id = 1")
    con.execute(LOOK_BACK_ROWS["measurement"])
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=1,
        num_intervals_before=2,
        concept_ids=[3022318],
        aggregation_strategy="count",
    )
    row = list(edata.obs["person_id"]).index(1)
    np.testing.assert_array_equal(edata.X[row, 0], [0, 3, 1])


@pytest.mark.parametrize("setup_function", [ed.io.omop.setup_variables, ed.io.omop.setup_interval_variables])
def test_setup_functions_num_intervals_before_leave_the_later_intervals_unchanged(
    omop_connection_vanilla, setup_function
):
    """The intervals from the start of the observation on are those without a look-back."""
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    data_table = "measurement" if setup_function is ed.io.omop.setup_variables else "drug_exposure"
    kwargs = {
        "backend_handle": con,
        "data_tables": [data_table],
        "data_field_to_keep": ["value_as_number" if data_table == "measurement" else "days_supply"],
        "interval_length_number": 3,
        "interval_length_unit": "day",
        "num_intervals": 12,
        "aggregation_strategy": "max",
    }
    plain = setup_function(edata, **kwargs)
    look_back = setup_function(edata, num_intervals_before=5, **kwargs)
    assert look_back.X.shape[2] == 17
    np.testing.assert_array_equal(look_back.X[:, :, 5:], plain.X)


def test_setup_interval_variables_num_intervals_before_with_intervals(omop_connection_vanilla):
    """An exposure that starts before the observation belongs to the intervals before it that it covers."""
    con = omop_connection_vanilla
    con.execute("DELETE FROM drug_exposure WHERE person_id = 1")
    con.execute(
        """INSERT INTO drug_exposure (drug_exposure_id, person_id, drug_concept_id, drug_exposure_start_date,
            drug_exposure_start_datetime, drug_exposure_end_date, drug_exposure_end_datetime, days_supply)
        VALUES (10, 1, 19073183, '2099-12-25', '2099-12-25 12:00:00', '2100-01-02', '2100-01-02 00:00:00', 10)"""
    )
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        data_tables=["drug_exposure"],
        data_field_to_keep=["days_supply"],
        interval_length_number=7,
        interval_length_unit="day",
        num_intervals=2,
        num_intervals_before=2,
        aggregation_strategy="count",
        keep_date="interval",
    )

    # the intervals start on 2099-12-18, 2099-12-25, 2100-01-01 and 2100-01-08
    np.testing.assert_array_equal(edata.X[0, 1], [0, 1, 1, 0])


@pytest.mark.parametrize(
    ("num_intervals_before", "error"), [(-1, ValueError), (1.0, TypeError), (True, TypeError), ("1", TypeError)]
)
def test_setup_variables_illegal_num_intervals_before(omop_connection_vanilla, num_intervals_before, error):
    with pytest.raises(error, match="num_intervals_before"):
        _setup_measurement(omop_connection_vanilla, "all", num_intervals_before=num_intervals_before)


def _add_measurement_rows(con, rows):
    """Add measurement rows (measurement_id, person_id, concept, datetime, value, unit_concept_id)."""
    for measurement_id, person_id, concept_id, datetime, value, unit in rows:
        con.execute(
            f"""INSERT INTO measurement (measurement_id, person_id, measurement_concept_id, measurement_date,
                measurement_datetime, value_as_number, unit_concept_id, unit_source_value)
            VALUES ({measurement_id}, {person_id}, {concept_id}, CAST(TIMESTAMP '{datetime}' AS DATE),
                TIMESTAMP '{datetime}', {value}, {unit}, NULL)"""
        )


# a row of concept 3022318 in a second unit: of a person left out of .obs, after the observation window, and in it
ROW_OF_ANOTHER_PERSON = (40, 3, 3022318, "2100-01-01 15:00:00", 5, 8753)
ROW_AFTER_THE_WINDOW = (41, 1, 3022318, "2100-03-01 15:00:00", 5, 8753)
ROW_IN_THE_WINDOW = (42, 1, 3022318, "2100-01-01 18:00:00", 5, 8753)


def _setup_measurement_of_persons(con, person_ids, **kwargs):
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    edata = edata[edata.obs["person_id"].isin(person_ids).to_numpy()].copy()
    kwargs = {
        "data_tables": ["measurement"],
        "data_field_to_keep": ["value_as_number"],
        "interval_length_number": 1,
        "interval_length_unit": "day",
        "num_intervals": 2,
        "concept_ids": [3022318],
        **kwargs,
    }
    return ed.io.omop.setup_variables(edata, backend_handle=con, **kwargs)


@pytest.mark.parametrize("row", [ROW_OF_ANOTHER_PERSON, ROW_AFTER_THE_WINDOW])
def test_setup_variables_unit_check_ignores_rows_outside_the_observations(omop_connection_vanilla, row):
    """Only the rows that enter the aggregation need to share a unit."""
    con = omop_connection_vanilla
    _add_second_concept_values(con)
    _add_measurement_rows(con, [row])
    edata = _setup_measurement_of_persons(con, [1, 2], aggregation_strategy="mean")
    assert edata.uns["unit_report_measurement"]["unit_concept_id"].tolist() == [9557]

    _add_measurement_rows(con, [ROW_IN_THE_WINDOW])
    with pytest.raises(NotImplementedError, match="more than one unit_concept_id"):
        _setup_measurement_of_persons(con, [1, 2], aggregation_strategy="mean")


@pytest.mark.parametrize("row", [ROW_OF_ANOTHER_PERSON, ROW_AFTER_THE_WINDOW])
def test_setup_variables_unit_conversions_ignore_unlisted_units_outside_the_observations(omop_connection_vanilla, row):
    """Only the rows that enter the aggregation need a listed unit."""
    con = omop_connection_vanilla
    _add_second_concept_values(con)
    _add_measurement_rows(con, [row])
    conversions = pd.DataFrame(
        {
            "concept_id": [3022318],
            "unit_concept_id": [9557],
            "to_unit_concept_id": [9557],
            "multiplier": [1.0],
            "offset": [0.0],
        }
    )
    edata = _setup_measurement_of_persons(con, [1, 2], aggregation_strategy="mean", unit_conversions=conversions)
    assert edata.uns["unit_report_measurement"]["unit_concept_id"].tolist() == [9557]

    _add_measurement_rows(con, [ROW_IN_THE_WINDOW])
    with pytest.raises(ValueError, match=r"concept 3022318 in unit 8753 \(1 data points\)"):
        _setup_measurement_of_persons(con, [1, 2], aggregation_strategy="mean", unit_conversions=conversions)


def test_setup_interval_variables_unit_check_ignores_rows_after_the_window(omop_connection_vanilla):
    """The second dose_era of person 1, in another unit, starts after the end of its visit."""
    con = omop_connection_vanilla
    con.execute("UPDATE dose_era SET unit_concept_id = 8587 WHERE dose_era_id = 2")
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    edata = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        data_tables=["dose_era"],
        data_field_to_keep=["dose_value"],
        interval_length_number=60,
        interval_length_unit="day",
        num_intervals=1,
        aggregation_strategy="mean",
        keep_date="start",
    )
    assert edata.n_vars > 0


def test_setup_variables_raising_unit_check_keeps_the_earlier_long_table(omop_connection_vanilla):
    con = omop_connection_vanilla
    con.execute("CREATE TABLE _ehrdata_observed_rows AS SELECT 1 AS user_column")
    _add_second_concept_values(con)
    earlier = _setup_measurement_of_persons(con, [1, 2], aggregation_strategy="mean", instantiate_tensor=False)
    cells = con.execute("SELECT count(*) FROM long_person_timestamp_feature_value_measurement").fetchone()[0]

    _add_measurement_rows(con, [ROW_IN_THE_WINDOW])
    with pytest.raises(NotImplementedError, match="more than one unit_concept_id"):
        _setup_measurement_of_persons(con, [1, 2], aggregation_strategy="mean")

    assert con.execute("SELECT count(*) FROM long_person_timestamp_feature_value_measurement").fetchone()[0] == cells
    assert con.execute("SELECT user_column FROM main._ehrdata_observed_rows").fetchall() == [(1,)]
    temp_tables = con.execute("SELECT table_name FROM duckdb_tables() WHERE temporary").fetchall()
    assert ("_ehrdata_observed_rows",) not in temp_tables
    assert earlier.n_vars == 1


def _hourly_measurement_of_visits(con, **kwargs):
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    return ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="h",
        time_precision="datetime",
        num_intervals=24,
        concept_ids=[3031147],
        aggregation_strategy="count",
        **kwargs,
    )


def test_setup_variables_datetime_precision_reads_a_row_without_datetime_at_midnight(omop_connection_vanilla):
    """OMOP requires the date of a data point but not its datetime."""
    con = omop_connection_vanilla
    with_datetime = _hourly_measurement_of_visits(con)
    con.execute("UPDATE measurement SET measurement_datetime = NULL WHERE measurement_id = 1")

    edata = _hourly_measurement_of_visits(con)

    # the data point of 12:00 moves to midnight of its date, the first interval of the visit
    expected = with_datetime.X.copy()
    expected[0, 0, 12] -= 1
    expected[0, 0, 0] += 1
    np.testing.assert_array_equal(edata.X, expected)


def test_setup_variables_datetime_precision_starts_a_visit_without_start_datetime_at_midnight(
    omop_connection_vanilla,
):
    """Visit 1 starts at 10:00 on 2100-01-01; without its start datetime it starts at midnight."""
    con = omop_connection_vanilla
    con.execute(
        "UPDATE visit_occurrence SET visit_start_datetime = TIMESTAMP '2100-01-01 10:00:00' WHERE visit_occurrence_id = 1"
    )
    con.execute(
        "UPDATE measurement SET measurement_datetime = TIMESTAMP '2100-01-01 05:00:00' WHERE measurement_id = 1"
    )
    with_datetime = _hourly_measurement_of_visits(con)
    con.execute("UPDATE visit_occurrence SET visit_start_datetime = NULL WHERE visit_occurrence_id = 1")

    edata = _hourly_measurement_of_visits(con)

    # the data point of 05:00 is before the start at 10:00, and in hour 5 of a start at midnight
    assert with_datetime.X[0, 0, 5] == 0
    assert edata.X[0, 0, 5] == 1
    assert pd.isna(edata.obs["visit_start_datetime"].iloc[0])


def test_setup_variables_datetime_precision_handles_a_tz_aware_start_without_datetime(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    starts = pd.to_datetime(edata.obs["visit_start_datetime"]).dt.tz_localize("UTC")
    starts.iloc[0] = pd.NaT
    edata.obs["visit_start_datetime"] = starts
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="h",
        time_precision="datetime",
        num_intervals=24,
        concept_ids=[3031147],
        aggregation_strategy="count",
    )
    assert edata.X[0].sum() > 0


def test_setup_variables_datetime_precision_keeps_a_missing_datetime_as_value(omop_connection_vanilla):
    """The date fills in the time of a data point, not the datetime column kept as its value."""
    con = omop_connection_vanilla
    con.execute("UPDATE measurement SET measurement_datetime = NULL WHERE measurement_id = 1")
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=["measurement"],
        data_field_to_keep=["measurement_datetime"],
        interval_length_number=1,
        interval_length_unit="h",
        time_precision="datetime",
        num_intervals=24,
        concept_ids=[3031147],
        aggregation_strategy="last",
    )
    # a data point without a datetime carries no value here, rather than one made up from its date
    assert pd.isna(edata.X[0, 0, 0])


def test_setup_interval_variables_datetime_precision_ends_an_interval_without_end_datetime_at_its_end_date(
    omop_connection_vanilla,
):
    con = omop_connection_vanilla
    kwargs = {
        "backend_handle": con,
        "data_tables": ["drug_exposure"],
        "data_field_to_keep": ["days_supply"],
        "interval_length_number": 1,
        "interval_length_unit": "day",
        "time_precision": "datetime",
        "num_intervals": 31,
        "aggregation_strategy": "count",
        "keep_date": "interval",
        "concept_ids": [19073183],
    }
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_observation_period")
    with_datetime = ed.io.omop.setup_interval_variables(edata, **kwargs)
    con.execute("UPDATE drug_exposure SET drug_exposure_end_datetime = NULL WHERE drug_exposure_id = 1")

    result = ed.io.omop.setup_interval_variables(edata, **kwargs)

    np.testing.assert_array_equal(result.X, with_datetime.X)
    assert result.X[0, 0, 20] == 1


@pytest.mark.parametrize(("end_date", "expected_hours"), [("2100-01-01", range(12, 24)), ("2100-01-02", range(12, 48))])
def test_setup_interval_variables_datetime_precision_ends_a_date_only_end_at_the_end_of_the_day(
    omop_connection_vanilla, end_date, expected_hours
):
    """OMOP end dates are inclusive, so an end without datetime lies at the end of its day, not at midnight before the start."""
    con = omop_connection_vanilla
    con.execute(
        f"""UPDATE drug_exposure SET drug_exposure_end_date = DATE '{end_date}', drug_exposure_end_datetime = NULL
        WHERE drug_exposure_id = 1"""
    )
    con.execute("DELETE FROM drug_exposure WHERE person_id = 1 AND drug_exposure_id <> 1")
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence")
    edata = ed.io.omop.setup_interval_variables(
        edata,
        backend_handle=con,
        data_tables=["drug_exposure"],
        data_field_to_keep=["days_supply"],
        interval_length_number=1,
        interval_length_unit="h",
        time_precision="datetime",
        num_intervals=48,
        aggregation_strategy="count",
        keep_date="interval",
        concept_ids=[19073183],
    )
    row = list(edata.obs["person_id"]).index(1)
    assert list(np.flatnonzero(edata.X[row, 0])) == list(expected_hours)
