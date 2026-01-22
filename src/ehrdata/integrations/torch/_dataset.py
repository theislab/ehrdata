from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from duckdb import DuckDBPyConnection

from ehrdata._compat import lazy_import_torch
from ehrdata.core import EHRData
from ehrdata.io.omop._queries import DATA_TABLE_DATE_KEYS, TIME_DEFINING_TABLE_ID_KEY

if TYPE_CHECKING:
    import torch

torch = lazy_import_torch()


class OMOPEHRDataset(torch.utils.data.Dataset):
    """A :class:`~torch.utils.data.Dataset` built from an OMOP CDM database.

    This class is a :class:`~torch.utils.data.Dataset` from an OMOP CDM database.
    It is a Dataset structure for the tensor in ehrdata.R, in a suitable format for :class:`~torch.utils.data.DataLoader`.
    This allows to stream the data in batches from the RDBMS, not requiring to load the entire dataset in memory.

    Note: Each item in the dataset represents an observation unit (e.g., a visit, observation period, or person),
    not necessarily a unique patient. A single patient can have multiple observation units.

    Args:
        con: The connection to the database.
        edata: Central data object.
        data_tables: The OMOP data tables to extract.
        target: The target variable to be used.
        datetime: If True, use datetime, if False, use date.
        idxs: The indices of the observation units to be used, can be used to include only a
            subset of the data, for e.g. train-test splits.
    """

    # TODO: data tables should also accept interval-style tables
    # TODO: implement for multiple data tables
    # TODO: test for multiple data tables
    def __init__(
        self,
        con: DuckDBPyConnection,
        edata: EHRData,
        *,
        data_tables: Sequence[Literal["measurement", "observation", "specimen"]],
        target: Literal["mortality"] = "mortality",
        datetime: bool = True,
        idxs: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.con = con
        self.edata = edata
        self.data_tables = data_tables
        self.target = target
        self.datetime = datetime
        self.idxs = idxs

        self.n_timesteps = con.execute(
            f"SELECT COUNT(DISTINCT interval_step) FROM long_person_timestamp_feature_value_{self.data_tables[0]}"
        ).fetchone()[0]
        self.n_variables = con.execute(
            f"SELECT COUNT(DISTINCT data_table_concept_id) FROM long_person_timestamp_feature_value_{self.data_tables[0]}"
        ).fetchone()[0]

    def __len__(self):
        where_clause = f"WHERE obs_id IN ({','.join(str(_) for _ in self.idxs)})" if self.idxs else ""
        query = f"""
            SELECT COUNT(DISTINCT obs_id)
            FROM long_person_timestamp_feature_value_{self.data_tables[0]}
            {where_clause}
        """
        return self.con.execute(query).fetchone()[0]

    def __getitem__(self, obs_index):
        obs_id_query = f"SELECT DISTINCT obs_id FROM long_person_timestamp_feature_value_{self.data_tables[0]} WHERE obs_index = {obs_index}"
        obs_id = self.con.execute(obs_id_query).fetchone()[0]
        where_clause = f"WHERE obs_index = {obs_index}"

        if self.idxs:
            where_clause += f" AND obs_index IN ({','.join(str(_) for _ in self.idxs)})"

        query = f"""
            SELECT obs_index, data_table_concept_id, interval_step, COALESCE(CAST(value_as_number AS DOUBLE), 'NaN') AS value_as_number
            FROM long_person_timestamp_feature_value_{self.data_tables[0]}
            {where_clause}
        """

        long_format_data = torch.tensor(self.con.execute(query).fetchall(), dtype=torch.float32)

        # convert long format to 3D tensor
        feature_ids, feature_idx = torch.unique(long_format_data[:, 1], return_inverse=True)
        step_ids, step_idx = torch.unique(long_format_data[:, 2], return_inverse=True)

        result = torch.zeros(len(feature_ids), len(step_ids))
        values = long_format_data[:, 3]
        result.index_put_((feature_idx, step_idx), values)

        if self.target != "mortality":
            msg = f"Target {self.target} is not implemented"
            raise NotImplementedError(msg)

        # If person has an entry in the death table that is within the observation period, report 1, else 0:
        # Left join ensures that for every observation unit, 0 or 1 is obtained
        omop_io_observation_table = self.edata.uns["omop_io_observation_table"]

        # Get the actual table ID column name and value
        obs_id_column = TIME_DEFINING_TABLE_ID_KEY[omop_io_observation_table]
        obs_table_id = self.edata.obs[self.edata.obs[obs_id_column] == obs_id][obs_id_column].item()

        # Get the appropriate date/datetime column names based on precision
        time_precision = "datetime" if self.datetime else "date"
        start_date_col = DATA_TABLE_DATE_KEYS["start"][omop_io_observation_table].get(
            time_precision, DATA_TABLE_DATE_KEYS["start"][omop_io_observation_table]["date"]
        )
        end_date_col = DATA_TABLE_DATE_KEYS["end"][omop_io_observation_table].get(
            time_precision, DATA_TABLE_DATE_KEYS["end"][omop_io_observation_table]["date"]
        )

        target_query = f"""
        SELECT
            CASE
                WHEN death_datetime BETWEEN {start_date_col} AND {end_date_col} THEN 1
                ELSE 0
            END AS mortality
        FROM {self.edata.uns["omop_io_observation_table"]}
        LEFT JOIN death USING (person_id)
        WHERE {obs_id_column} = {obs_table_id}
        """

        targets = torch.tensor(self.con.execute(target_query).fetchall(), dtype=torch.float32)

        return result, targets
