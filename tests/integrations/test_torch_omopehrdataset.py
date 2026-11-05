import pytest

pytest.importorskip("torch")

import torch

import ehrdata as ed


def test_ehrdataset_vanilla(omop_connection_vanilla):
    num_intervals = 3
    batch_size = 2

    con = omop_connection_vanilla

    edata = ed.io.omop.setup_obs(con, observation_table="person_observation_period", death_table=True)
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables="measurement",
        data_field_to_keep="value_as_number",
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=num_intervals,
        enrich_var_with_feature_info=False,
        enrich_var_with_unit_info=False,
        instantiate_tensor=False,
    )

    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(
        con, edata, data_tables=["measurement"], datetime=False, idxs=None
    )
    assert isinstance(ehr_dataset, torch.utils.data.Dataset)
    single_item = next(iter(ehr_dataset))
    assert single_item[0].shape == (2, num_intervals)
    assert len(single_item[1]) == 1

    loader = torch.utils.data.DataLoader(ehr_dataset, batch_size=batch_size)
    batch = next(iter(loader))
    assert batch[0].shape == (batch_size, 2, num_intervals)
    assert len(batch[1]) == batch_size


def test_ehrdataset_items_match_the_tensor(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(con, observation_table="person_observation_period", death_table=True)
    kwargs = {
        "backend_handle": con,
        "data_tables": "measurement",
        "data_field_to_keep": "value_as_number",
        "interval_length_number": 1,
        "interval_length_unit": "day",
        "num_intervals": 3,
    }
    edata_tensor = ed.io.omop.setup_variables(edata, **kwargs)
    edata = ed.io.omop.setup_variables(edata, instantiate_tensor=False, **kwargs)

    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"], datetime=False)
    assert len(ehr_dataset) == edata.n_obs
    for obs_index in range(edata.n_obs):
        expected = torch.tensor(edata_tensor.X[obs_index], dtype=torch.float32)
        torch.testing.assert_close(ehr_dataset[obs_index][0], expected, equal_nan=True)


def _setup(con, data_tables, data_field_to_keep, **kwargs):
    edata = ed.io.omop.setup_obs(con, observation_table="person_observation_period", death_table=True)
    return ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables=data_tables,
        data_field_to_keep=data_field_to_keep,
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=3,
        **kwargs,
    )


def test_ehrdataset_follows_edata_obs_and_idxs(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata_tensor = _setup(con, "measurement", "value_as_number")
    edata = _setup(con, "measurement", "value_as_number", instantiate_tensor=False)

    # a reordered subset of the observations, and positions within it
    order = [2, 0]
    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata[order], data_tables=["measurement"], datetime=False)
    assert len(ehr_dataset) == len(order)
    for index, obs_index in enumerate(order):
        expected = torch.tensor(edata_tensor.X[obs_index], dtype=torch.float32)
        torch.testing.assert_close(ehr_dataset[index][0], expected, equal_nan=True)

    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"], idxs=[1])
    assert len(ehr_dataset) == 1
    torch.testing.assert_close(ehr_dataset[0][0], torch.tensor(edata_tensor.X[1], dtype=torch.float32), equal_nan=True)
    with pytest.raises(IndexError):
        ehr_dataset[1]


def test_ehrdataset_uses_the_variables_of_its_data_table(omop_connection_vanilla):
    con = omop_connection_vanilla
    data_field_to_keep = {"measurement": "value_as_number", "observation": "value_as_number"}
    edata_tensor = _setup(con, ["measurement", "observation"], data_field_to_keep)
    edata = _setup(con, ["measurement", "observation"], data_field_to_keep, instantiate_tensor=False)

    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"], datetime=False)
    # the measurement concepts come first in .var
    n_measurement = ehr_dataset.n_variables
    assert n_measurement == 2
    torch.testing.assert_close(
        ehr_dataset[0][0], torch.tensor(edata_tensor.X[0, :n_measurement], dtype=torch.float32), equal_nan=True
    )


def test_ehrdataset_requires_the_variables(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = ed.io.omop.setup_obs(con, observation_table="person_observation_period", death_table=True)
    with pytest.raises(ValueError, match="needs the EHRData returned by setup_variables"):
        ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"])


def test_ehrdataset_keeps_repeated_cohort_subjects_apart(omop_connection_vanilla):
    con = omop_connection_vanilla
    # subject 1 enters a second cohort a day earlier, so that its data fall into another interval
    con.execute("INSERT INTO cohort VALUES (2, 1, '2099-12-31', '2100-01-31')")
    edata = ed.io.omop.setup_obs(con, observation_table="person_cohort", death_table=True)
    kwargs = {
        "backend_handle": con,
        "data_tables": "measurement",
        "data_field_to_keep": "value_as_number",
        "interval_length_number": 1,
        "interval_length_unit": "day",
        "num_intervals": 3,
    }
    edata_tensor = ed.io.omop.setup_variables(edata, **kwargs)
    edata = ed.io.omop.setup_variables(edata, instantiate_tensor=False, **kwargs)

    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"], datetime=False)
    assert len(ehr_dataset) == 4
    for index in range(4):
        item, target = ehr_dataset[index]
        torch.testing.assert_close(item, torch.tensor(edata_tensor.X[index], dtype=torch.float32), equal_nan=True)
        assert target.shape == (1, 1)


def test_ehrdataset_rejects_rows_the_variables_were_not_set_up_for(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = _setup(con, "measurement", "value_as_number", instantiate_tensor=False)
    edata.obs_names = [f"other_{name}" for name in edata.obs_names]
    with pytest.raises(ValueError, match="were not set up for the rows"):
        ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"])


def test_ehrdataset_items_match_the_tensor_with_intervals_before_the_start(omop_connection_vanilla):
    con = omop_connection_vanilla
    con.execute(
        """INSERT INTO measurement (measurement_id, person_id, measurement_concept_id, measurement_date,
            measurement_datetime, value_as_number, unit_concept_id, unit_source_value)
        SELECT 30 + person_id, person_id, 3022318, DATE '2099-12-30', TIMESTAMP '2099-12-30 12:00:00',
            person_id, 9557, 'mEq/L' FROM person"""
    )
    edata = ed.io.omop.setup_obs(con, observation_table="person_observation_period", death_table=True)
    kwargs = {
        "backend_handle": con,
        "data_tables": "measurement",
        "data_field_to_keep": "value_as_number",
        "interval_length_number": 1,
        "interval_length_unit": "day",
        "num_intervals": 3,
        "num_intervals_before": 4,
    }
    edata_tensor = ed.io.omop.setup_variables(edata, **kwargs)
    edata = ed.io.omop.setup_variables(edata, instantiate_tensor=False, **kwargs)

    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"], datetime=False)
    assert ehr_dataset[0][0].shape == (edata.n_vars, 7)
    assert torch.isfinite(torch.stack([ehr_dataset[i][0][:, :4] for i in range(edata.n_obs)])).any()
    for obs_index in range(edata.n_obs):
        expected = torch.tensor(edata_tensor.X[obs_index], dtype=torch.float32)
        torch.testing.assert_close(ehr_dataset[obs_index][0], expected, equal_nan=True)

    # without .tem["interval_step"], the intervals are those of the setup
    edata.tem = edata.tem.drop(columns="interval_step")
    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"], datetime=False)
    for obs_index in range(edata.n_obs):
        expected = torch.tensor(edata_tensor.X[obs_index], dtype=torch.float32)
        torch.testing.assert_close(ehr_dataset[obs_index][0], expected, equal_nan=True)


def test_ehrdataset_items_match_the_tensor_of_a_subset_of_the_intervals(omop_connection_vanilla):
    """An edata whose time axis is a subset of the intervals set up holds only these intervals."""
    con = omop_connection_vanilla
    con.execute(
        """INSERT INTO measurement (measurement_id, person_id, measurement_concept_id, measurement_date,
            measurement_datetime, value_as_number, unit_concept_id, unit_source_value)
        SELECT 30 + person_id, person_id, 3022318, DATE '2099-12-30', TIMESTAMP '2099-12-30 12:00:00',
            person_id, 9557, 'mEq/L' FROM person"""
    )
    edata = ed.io.omop.setup_obs(con, observation_table="person_observation_period", death_table=True)
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables="measurement",
        data_field_to_keep="value_as_number",
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=3,
        num_intervals_before=4,
    )
    subset = edata[:, :, 4:]

    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, subset, data_tables=["measurement"], datetime=False)
    for obs_index in range(subset.n_obs):
        expected = torch.tensor(subset.X[obs_index], dtype=torch.float32)
        torch.testing.assert_close(ehr_dataset[obs_index][0], expected, equal_nan=True)


def test_ehrdataset_rejects_a_long_table_of_a_later_call(omop_connection_vanilla):
    """A later setup_variables call replaces the long format table that an earlier edata reads."""
    con = omop_connection_vanilla
    edata = _setup(con, "measurement", "value_as_number", aggregation_strategy="mean")
    _setup(con, "measurement", "value_as_number", aggregation_strategy="max")
    with pytest.raises(ValueError, match="written by another call"):
        ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"])


def test_ehrdataset_raises_for_items_after_a_later_call(omop_connection_vanilla):
    """A later setup_variables call can replace the long format table after the dataset was created."""
    con = omop_connection_vanilla
    edata = _setup(con, "measurement", "value_as_number")
    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"])
    ehr_dataset[0]
    _setup(con, "measurement", "value_as_number")
    with pytest.raises(ValueError, match="replaced by a later call"):
        ehr_dataset[0]


def test_ehrdataset_needs_the_long_table(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = _setup(con, "measurement", "value_as_number")
    con.execute("DROP TABLE long_person_timestamp_feature_value_measurement")
    with pytest.raises(ValueError, match="no long format table"):
        ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"])


def test_ehrdataset_needs_the_setup_of_setup_variables(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = _setup(con, "measurement", "value_as_number")
    del edata.uns["omop_io_variable_setup"]
    with pytest.raises(ValueError, match="omop_io_variable_setup"):
        ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"])


@pytest.mark.parametrize(
    ("data_field_to_keep", "aggregation_strategy"),
    [
        ("value_as_number", "count"),
        ("value_as_number", {3031147: "count", 3022318: "max"}),
        ("is_present", "count"),
        ("value_as_concept_id", "last"),
    ],
)
def test_ehrdataset_items_match_the_tensor_of_the_setup(
    omop_connection_vanilla, data_field_to_keep, aggregation_strategy
):
    """The items hold the field of the setup, and 0 in the empty cells of counted variables, as its tensor."""
    con = omop_connection_vanilla
    edata = _setup(con, "measurement", data_field_to_keep, aggregation_strategy=aggregation_strategy)

    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"], datetime=False)
    for obs_index in range(edata.n_obs):
        expected = torch.tensor(edata.X[obs_index].astype(float), dtype=torch.float32)
        torch.testing.assert_close(ehr_dataset[obs_index][0], expected, equal_nan=True)


def test_ehrdataset_needs_a_numeric_field(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = _setup(con, "measurement", "unit_source_value", aggregation_strategy="last")
    with pytest.raises(TypeError, match="numeric data field"):
        ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"])


@pytest.mark.parametrize(
    ("death_date", "death_datetime", "datetime", "expected"),
    [
        # the observation periods run from 2100-01-01 to 2100-01-31
        ("2100-01-31", "2100-01-31 18:00:00", True, 1),
        ("2100-01-31", None, True, 1),
        ("2100-01-01", None, True, 1),
        ("2100-02-01", "2100-02-01 00:00:00", True, 0),
        ("2099-12-31", "2099-12-31 23:00:00", False, 0),
        (None, "2100-01-15 12:00:00", True, 1),
    ],
)
def test_ehrdataset_mortality_in_the_window(omop_connection_vanilla, death_date, death_datetime, datetime, expected):
    con = omop_connection_vanilla
    con.execute(
        "UPDATE death SET death_date = CAST(? AS DATE), death_datetime = CAST(? AS TIMESTAMP) WHERE person_id = 1",
        [death_date, death_datetime],
    )
    edata = _setup(con, "measurement", "value_as_number")
    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"], datetime=datetime)
    row = list(edata.obs["person_id"]).index(1)
    assert ehr_dataset[row][1].item() == expected


def test_ehrdataset_mortality_of_persons(omop_connection_vanilla):
    """The observations of the person table have no end date, so any death after the birth counts."""
    con = omop_connection_vanilla
    con.execute("UPDATE person SET birth_datetime = TIMESTAMP '2099-01-01 00:00:00'")
    edata = ed.io.omop.setup_obs(con, observation_table="person")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables="measurement",
        data_field_to_keep="value_as_number",
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=3,
    )
    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"])
    targets = {int(person): ehr_dataset[i][1].item() for i, person in enumerate(edata.obs["person_id"])}
    assert targets == {1: 1, 2: 1, 3: 0, 4: 0}


def test_ehrdataset_mortality_needs_a_death_table(omop_connection_vanilla):
    con = omop_connection_vanilla
    edata = _setup(con, "measurement", "value_as_number")
    con.execute("DROP TABLE death")
    with pytest.raises(ValueError, match="'death' table"):
        ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"])


def test_ehrdataset_of_concept_sets_with_count_read_back_from_disk(omop_connection_vanilla, tmp_path):
    con = omop_connection_vanilla
    edata = _setup(
        con,
        "measurement",
        "value_as_number",
        concept_ids={"co2/blood": [3031147], "rhythm": [3022318]},
        aggregation_strategy={"co2/blood": "max", "rhythm": "count"},
    )
    ed.io.write_zarr(edata, tmp_path / "edata.zarr")
    read_back = ed.io.read_zarr(tmp_path / "edata.zarr")

    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, read_back, data_tables=["measurement"], datetime=False)
    for obs_index in range(edata.n_obs):
        expected = torch.tensor(edata.X[obs_index].astype(float), dtype=torch.float32)
        torch.testing.assert_close(ehr_dataset[obs_index][0], expected, equal_nan=True)


def test_ehrdataset_mortality_without_death_datetime_column(omop_connection_vanilla):
    con = omop_connection_vanilla
    con.execute("UPDATE death SET death_date = DATE '2100-01-31' WHERE person_id = 1")
    con.execute("ALTER TABLE death DROP COLUMN death_datetime")
    edata = _setup(con, "measurement", "value_as_number")
    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"])
    assert ehr_dataset[list(edata.obs["person_id"]).index(1)][1].item() == 1


@pytest.mark.parametrize(("death_datetime", "expected"), [("2100-01-01 09:00:00", 0), ("2100-01-01 11:00:00", 1)])
def test_ehrdataset_mortality_of_a_visit_from_its_start_datetime(omop_connection_vanilla, death_datetime, expected):
    con = omop_connection_vanilla
    con.execute(
        "UPDATE visit_occurrence SET visit_start_datetime = TIMESTAMP '2100-01-01 10:00:00' WHERE person_id = 1"
    )
    con.execute(
        "UPDATE death SET death_date = DATE '2100-01-01', death_datetime = CAST(? AS TIMESTAMP) WHERE person_id = 1",
        [death_datetime],
    )
    edata = ed.io.omop.setup_obs(con, observation_table="person_visit_occurrence")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables="measurement",
        data_field_to_keep="value_as_number",
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=3,
    )
    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"], datetime=True)
    assert ehr_dataset[list(edata.obs["person_id"]).index(1)][1].item() == expected


@pytest.mark.parametrize(("time_precision", "expected"), [("date", 1), ("datetime", 0)])
def test_ehrdataset_mortality_compares_at_the_precision_of_the_setup(omop_connection_vanilla, time_precision, expected):
    """By default, a death at 09:00 on the day of a visit that starts at 10:00 counts at date precision only."""
    con = omop_connection_vanilla
    con.execute(
        "UPDATE visit_occurrence SET visit_start_datetime = TIMESTAMP '2100-01-01 10:00:00' WHERE person_id = 1"
    )
    con.execute(
        "UPDATE death SET death_date = DATE '2100-01-01', death_datetime = TIMESTAMP '2100-01-01 09:00:00' WHERE person_id = 1"
    )
    edata = ed.io.omop.setup_obs(con, observation_table="person_visit_occurrence")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables="measurement",
        data_field_to_keep="value_as_number",
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=3,
        time_precision=time_precision,
    )
    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"])
    assert ehr_dataset[list(edata.obs["person_id"]).index(1)][1].item() == expected


def test_ehrdataset_mortality_of_a_visit_without_end_date(omop_connection_vanilla):
    """An observation without an end date has no death in its window, unless it is a person, which has no end."""
    con = omop_connection_vanilla
    con.execute("UPDATE death SET death_date = DATE '2100-01-02', death_datetime = NULL WHERE person_id = 1")
    edata = ed.io.omop.setup_obs(con, observation_table="person_visit_occurrence")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=con,
        data_tables="measurement",
        data_field_to_keep="value_as_number",
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=3,
        clip_to_end_date=False,
    )
    row = list(edata.obs["person_id"]).index(1)
    ehr_dataset = ed.integrations.torch.OMOPEHRDataset(con, edata, data_tables=["measurement"])
    assert ehr_dataset[row][1].item() == 1
    end_column = next(column for column in edata.obs.columns if column.lower() == "visit_end_date")
    edata.obs[end_column] = edata.obs[end_column].astype(object)
    edata.obs.iloc[row, edata.obs.columns.get_loc(end_column)] = None
    assert ehr_dataset[row][1].item() == 0
