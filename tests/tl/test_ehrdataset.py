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

    ehr_dataset = ed.tl.omop.EHRDataset(con, edata, data_tables=["measurement"], datetime=False, idxs=None)
    assert isinstance(ehr_dataset, torch.utils.data.Dataset)
    single_item = next(iter(ehr_dataset))
    assert single_item[0].shape == (2, num_intervals)
    assert len(single_item[1]) == 1

    loader = torch.utils.data.DataLoader(ehr_dataset, batch_size=batch_size)
    batch = next(iter(loader))
    assert batch[0].shape == (batch_size, 2, num_intervals)
    assert len(batch[1]) == batch_size
