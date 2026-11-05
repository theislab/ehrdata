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
