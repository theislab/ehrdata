from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from duckdb import DuckDBPyConnection

from ehrdata._compat import lazy_import_torch
from ehrdata.core import EHRData
from ehrdata.io.omop._queries import DATA_TABLE_CONCEPT_ID_TRUNK, DATA_TABLE_DATE_KEYS, TIME_DEFINING_TABLE_ID_KEY

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
        edata: Central data object, as returned by :func:`~ehrdata.io.omop.setup_variables`.
            Its `.obs` defines the items, and its `.var` and `.tem` the variables and intervals of each item.
        data_tables: The OMOP data tables to extract.
        target: The target variable to be used.
        datetime: If True, use datetime, if False, use date.
        idxs: The positions in `edata.obs` of the observation units to be used, can be used to include only a
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

        if "data_table_concept_id" not in edata.var.columns:
            msg = "OMOPEHRDataset needs the EHRData returned by setup_variables, whose .var lists the concepts."
            raise ValueError(msg)

        # The long format table holds only the observed cells, so the variables and intervals are taken from edata:
        # the concepts of edata.var that occur in the data table, in the order of edata.var.
        concept_key = f"{DATA_TABLE_CONCEPT_ID_TRUNK[data_tables[0]]}_concept_id"
        table_concepts = {
            row[0] for row in con.execute(f"SELECT DISTINCT {concept_key} FROM {data_tables[0]}").fetchall()
        }
        concepts = [
            concept for concept in dict.fromkeys(edata.var["data_table_concept_id"]) if concept in table_concepts
        ]
        self.variable_index = {concept: i for i, concept in enumerate(concepts)}
        self.n_variables = len(concepts)
        self.n_timesteps = len(edata.tem)

        obs_ids = edata.obs[TIME_DEFINING_TABLE_ID_KEY[edata.uns["omop_io_observation_table"]]]
        self.obs_ids = obs_ids if idxs is None else obs_ids.iloc[list(idxs)]

    def __len__(self):
        return len(self.obs_ids)

    def __getitem__(self, index):
        obs_id = int(self.obs_ids.iloc[index])

        query = f"""
            SELECT data_table_concept_id, interval_step, COALESCE(CAST(value_as_number AS DOUBLE), 'NaN') AS value_as_number
            FROM long_person_timestamp_feature_value_{self.data_tables[0]}
            WHERE obs_id = ?
        """
        long_format_data = [
            row for row in self.con.execute(query, [obs_id]).fetchall() if row[0] in self.variable_index
        ]

        # convert long format to a variables x intervals tensor, with NaN in the cells without data
        result = torch.full((self.n_variables, self.n_timesteps), float("nan"))
        if long_format_data:
            feature_idx = torch.tensor([self.variable_index[row[0]] for row in long_format_data])
            step_idx = torch.tensor([row[1] for row in long_format_data])
            values = torch.tensor([row[2] for row in long_format_data], dtype=torch.float32)
            result.index_put_((feature_idx, step_idx), values)

        if self.target != "mortality":
            msg = f"Target {self.target} is not implemented"
            raise NotImplementedError(msg)

        # If person has an entry in the death table that is within the observation period, report 1, else 0:
        # Left join ensures that for every observation unit, 0 or 1 is obtained
        omop_io_observation_table = self.edata.uns["omop_io_observation_table"]

        # Get the actual table ID column name and value
        obs_id_column = TIME_DEFINING_TABLE_ID_KEY[omop_io_observation_table]

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
        WHERE {obs_id_column} = ?
        """

        targets = torch.tensor(self.con.execute(target_query, [obs_id]).fetchall(), dtype=torch.float32)

        return result, targets
