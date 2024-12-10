from collections.abc import Sequence

import torch
from duckdb.duckdb import DuckDBPyConnection
from torch.utils.data import Dataset


class EHRDataSet(Dataset):
    def __init__(
        self,
        con: DuckDBPyConnection,
        n_variables: int,
        n_timesteps: int,
        batch_size: int = 10,
        idxs: Sequence[int] | None = None,
    ):
        super().__init__()
        self.con = con
        self.batch_size = batch_size
        self.idxs = idxs

        # TODO: get from database or EHRData?
        self.n_timesteps = n_timesteps
        self.n_variables = n_variables

    def __len__(self):
        if self.idxs:
            where_clause = f"WHERE person_id IN ({','.join(str(_) for _ in self.idxs)})"
        else:
            where_clause = ""
        query = f"""
            SELECT COUNT(DISTINCT person_id)
            FROM long_person_timestamp_feature_value
            {where_clause}
        """
        return self.con.execute(query).fetchone()[0]  # .item()

    def __getitem__(self, person_id):
        # if isinstance(person_ids, int):
        #     person_ids = [person_ids]  # Make it a list for consistent handling
        # elif isinstance(person_ids, slice):
        #     person_ids = range(person_ids.start or 0, person_ids.stop, person_ids.step or 1)

        where_clause = f"WHERE person_index = {person_id}"

        if self.idxs:
            where_clause += f" AND person_index IN ({','.join(str(_) for _ in self.idxs)})"
        # else:
        #     where_clause = ""

        query = f"""
            SELECT person_index, data_table_concept_id, interval_step, COALESCE(CAST(value_as_number AS DOUBLE), 'NaN') AS value_as_number
            FROM long_person_timestamp_feature_value
            {where_clause}
        """
        # AND data_table_concept_id = {feature_id}
        # AND interval_step = {timestep}
        # data is fetched in long format
        long_format_data = torch.tensor(self.con.execute(query).fetchall(), dtype=torch.float32)

        # convert long format to 3D tensor
        # sample_ids, sample_idx = torch.unique(long_format_data[:, 0], return_inverse=True)
        feature_ids, feature_idx = torch.unique(long_format_data[:, 1], return_inverse=True)
        step_ids, step_idx = torch.unique(long_format_data[:, 2], return_inverse=True)

        result = torch.zeros(len(feature_ids), len(step_ids))
        values = long_format_data[:, 3]
        result.index_put_((feature_idx, step_idx), values)
        return result
