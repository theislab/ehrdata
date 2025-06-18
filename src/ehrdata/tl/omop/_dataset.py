from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from duckdb.duckdb import DuckDBPyConnection

from ehrdata import EHRData
from ehrdata._compat import lazy_import_torch
from ehrdata.io.omop._queries import DATA_TABLE_DATE_KEYS

if TYPE_CHECKING:
    import torch

torch = lazy_import_torch()


class EHRDataset(torch.utils.data.Dataset):
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
        """:class:`~torch.utils.data.Dataset` for a :class:`~ehrdata.EHRData` object.

        This function builds a :class:`~torch.utils.data.Dataset` for a :class:`~ehrdata.EHRData` object. The :class:`~ehrdata.EHRData` object is assumed to be in the OMOP CDM format.
        It is a Dataset structure for the tensor in ehrdata.R, in a suitable format for :class:`~pytorch.utils.data.DataLoader`.
        This allows to stream the data in batches from the RDBMS, not requiring to load the entire dataset in memory.

        Args:
            con: The connection to the database.
            edata: Data object.
            data_tables: The OMOP data tables to extract.
            target: The target variable to be used.
            datetime: If True, use datetime, if False, use date.
            idxs: The indices of the patients to be used, can be used to include only a
                subset of the data, for e.g. train-test splits.

        Returns:
            A :class:`torch.utils.data.Dataset` object of the :class:`~ehrdata.EHRData` object.
        """
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
        where_clause = f"WHERE person_id IN ({','.join(str(_) for _ in self.idxs)})" if self.idxs else ""
        query = f"""
            SELECT COUNT(DISTINCT person_id)
            FROM long_person_timestamp_feature_value_{self.data_tables[0]}
            {where_clause}
        """
        return self.con.execute(query).fetchone()[0]

    def __getitem__(self, person_index):
        person_id_query = f"SELECT DISTINCT person_id FROM long_person_timestamp_feature_value_{self.data_tables[0]} WHERE person_index = {person_index}"
        person_id = self.con.execute(person_id_query).fetchone()[0]
        where_clause = f"WHERE person_index = {person_index}"

        if self.idxs:
            where_clause += f" AND person_index IN ({','.join(str(_) for _ in self.idxs)})"

        query = f"""
            SELECT person_index, data_table_concept_id, interval_step, COALESCE(CAST(value_as_number AS DOUBLE), 'NaN') AS value_as_number
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

        # If person has an entry in the death table that is within visit_start_datetime and visit_end_datetime of the visit_occurrence table, report 1, else 0:
        # Left join ensures that for every patient, 0 or 1 is obtained
        omop_io_observation_table = self.edata.uns["omop_io_observation_table"]
        time_postfix = "time" if self.datetime else ""
        target_query = f"""
        SELECT
            CASE
                WHEN death_datetime BETWEEN {DATA_TABLE_DATE_KEYS["start"][omop_io_observation_table]}{time_postfix} AND {DATA_TABLE_DATE_KEYS["end"][omop_io_observation_table]}{time_postfix} THEN 1
                ELSE 0
            END AS mortality
        FROM {self.edata.uns["omop_io_observation_table"]}
        LEFT JOIN death USING (person_id)
        WHERE person_id = {person_id} AND {omop_io_observation_table}_id = {self.edata.obs[self.edata.obs["person_id"] == person_id][f"{omop_io_observation_table}_id"].item()}
        """

        targets = torch.tensor(self.con.execute(target_query).fetchall(), dtype=torch.float32)

        return result, targets
