import re

import numpy as np
import pandas as pd
import pytest

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
