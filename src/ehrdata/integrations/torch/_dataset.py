from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import pandas as pd
from duckdb import DuckDBPyConnection

from ehrdata._compat import lazy_import_torch
from ehrdata.core import EHRData
from ehrdata.io.omop._queries import DATA_TABLE_CONCEPT_ID_TRUNK, DATA_TABLE_DATE_KEYS

NUMERIC_TYPES = (
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "BOOLEAN",
)

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
        edata: Central data object, as returned by :func:`~ehrdata.io.omop.setup_variables`, or a subset of it.
            Its `.obs` defines the items: item `i` is the row `i` of the tensor that :func:`~ehrdata.io.omop.setup_variables` sets up for this `.obs`.
            Its `.var` and `.tem` define the variables and intervals of each item.
        data_tables: The OMOP data tables to extract.
            The values are those of the last call of :func:`~ehrdata.io.omop.setup_variables` that returned `edata`, as recorded in `edata.uns['omop_io_variable_setup']`:
            its field of `data_field_to_keep`, and in the cells without data 0 for the variables aggregated with `"count"` and NaN otherwise, as in its tensor.
            If a later call has replaced the long format table of the data table in the database, creating the dataset or reading an item raises a `ValueError`.
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
        setup = edata.uns.get("omop_io_variable_setup")
        if not isinstance(setup, Mapping) or data_tables[0] not in setup.get("data_tables", []):
            msg = f"OMOPEHRDataset needs the EHRData returned by setup_variables for '{data_tables[0]}', which records its setup in .uns['omop_io_variable_setup']."
            raise ValueError(msg)
        tables = {row[0].lower() for row in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        long_table = f"long_person_timestamp_feature_value_{data_tables[0]}"
        if long_table not in tables or f"{long_table}_obs" not in tables:
            msg = (
                f"The database has no long format table of '{data_tables[0]}' for edata: setup_variables found no data in it, "
                "or the table was removed. Run setup_variables again for this edata."
            )
            raise ValueError(msg)
        self.con = con
        self.edata = edata
        self.data_tables = data_tables
        self.target = target
        self.datetime = datetime
        self.idxs = idxs

        # The long format table holds only the observed cells, so the variables and intervals are taken from edata.
        if "data_table_concept_id" in edata.var.columns:
            # the concepts of edata.var that occur in the data table, in the order of edata.var
            self.key_column = "data_table_concept_id"
            concept_key = f"{DATA_TABLE_CONCEPT_ID_TRUNK[data_tables[0]]}_concept_id"
            table_concepts = {
                row[0] for row in con.execute(f"SELECT DISTINCT {concept_key} FROM {data_tables[0]}").fetchall()
            }
            variables = [
                concept for concept in dict.fromkeys(edata.var["data_table_concept_id"]) if concept in table_concepts
            ]
        elif "concept_ids" in edata.var.columns:
            # concept sets, named by edata.var_names
            self.key_column = "variable"
            variables = list(edata.var_names)
        else:
            msg = "OMOPEHRDataset needs the EHRData returned by setup_variables, whose .var lists the concepts."
            raise ValueError(msg)
        long_table_columns = {
            row[0] for row in con.execute(f"DESCRIBE long_person_timestamp_feature_value_{data_tables[0]}").fetchall()
        }
        if self.key_column not in long_table_columns:
            msg = (
                f"The long format table of '{data_tables[0]}' was set up by another call of setup_variables than edata, "
                f"{'without' if self.key_column == 'variable' else 'with'} concept sets."
            )
            raise ValueError(msg)
        obs_columns = {
            row[0]
            for row in con.execute(f"DESCRIBE long_person_timestamp_feature_value_{data_tables[0]}_obs").fetchall()
        }
        setup_ids = (
            con.execute(
                f"SELECT DISTINCT setup_id FROM long_person_timestamp_feature_value_{data_tables[0]}_obs"
            ).fetchall()
            if "setup_id" in obs_columns
            else []
        )
        self.setup_id = setup["setup_id"]
        if setup_ids != [(self.setup_id,)]:
            msg = (
                f"The long format table of '{data_tables[0]}' in the database was written by another call of setup_variables "
                "or setup_interval_variables than the one that returned edata. Run setup_variables again for this edata."
            )
            raise ValueError(msg)
        self.value_field = setup["data_field_to_keep"][data_tables[0]][0]
        value_type = {
            row[0]: row[1]
            for row in con.execute(f"DESCRIBE long_person_timestamp_feature_value_{data_tables[0]}").fetchall()
        }.get(self.value_field, "")
        if self.value_field != "is_present" and not value_type.startswith(NUMERIC_TYPES):
            msg = f"OMOPEHRDataset needs a numeric data field, but '{self.value_field}' of '{data_tables[0]}' is of type {value_type}."
            raise TypeError(msg)
        self.variable_index = {variable: i for i, variable in enumerate(variables)}
        # as in the tensor of setup_variables, the cells without data of counted variables hold 0
        strategy = setup["aggregation_strategy"]
        strategy_of = (
            {} if isinstance(strategy, str) else dict(zip(strategy["variables"], strategy["strategies"], strict=True))
        )
        self.counted = [
            i
            for variable, i in self.variable_index.items()
            if (strategy if isinstance(strategy, str) else strategy_of.get(str(variable))) == "count"
            and self.value_field != "is_present"
        ]
        self.n_variables = len(variables)
        self.n_timesteps = len(edata.tem)
        # the interval of each interval_step of the long format table, which counts from the start of the observation;
        # the intervals of edata may begin before the start, or be a subset of those set up
        before = setup.get("num_intervals_before", 0)
        steps = (
            edata.tem["interval_step"]
            if "interval_step" in edata.tem.columns
            else range(-before, self.n_timesteps - before)
        )
        self.step_index = {int(step): i for i, step in enumerate(steps)}

        # The obs_index of the long format table is the position in the .obs it was set up for; map the rows of edata to it by obs_names.
        setup_obs_names = con.execute(
            f"SELECT obs_name, obs_index FROM long_person_timestamp_feature_value_{data_tables[0]}_obs"
        ).fetchall()
        obs_index_of = dict(setup_obs_names)
        if len(obs_index_of) < len(setup_obs_names):
            msg = "The .obs that the variables were set up for has duplicate obs_names, which cannot be mapped to its rows."
            raise ValueError(msg)
        missing = [name for name in edata.obs_names if name not in obs_index_of]
        if missing:
            msg = f"The variables in '{data_tables[0]}' were not set up for the rows {missing[:5]} of edata.obs; run setup_variables on this edata."
            raise ValueError(msg)
        self.rows = list(range(edata.n_obs)) if idxs is None else [int(row) for row in idxs]
        self.obs_names = [str(edata.obs_names[row]) for row in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        # the position of the row in the .obs the variables were set up for, checked against the setup of edata at every item,
        # as a later call can replace the long format table after the dataset was created
        long_table = f"long_person_timestamp_feature_value_{self.data_tables[0]}"
        obs_index = self.con.execute(
            f"SELECT obs_index FROM {long_table}_obs WHERE obs_name = ? AND setup_id = ?",
            [self.obs_names[index], self.setup_id],
        ).fetchall()
        if not obs_index:
            msg = (
                f"The long format table of '{self.data_tables[0]}' in the database was replaced by a later call of setup_variables "
                "or setup_interval_variables. Run setup_variables again for this edata."
            )
            raise ValueError(msg)
        query = f"""
            SELECT {self.key_column}, interval_step, COALESCE(CAST({self.value_field} AS DOUBLE), 'NaN') AS value
            FROM {long_table}
            WHERE obs_index = ?
        """
        long_format_data = [
            cell
            for cell in self.con.execute(query, [obs_index[0][0]]).fetchall()
            if cell[0] in self.variable_index and cell[1] in self.step_index
        ]

        # convert long format to a variables x intervals tensor, with NaN in the cells without data
        result = torch.full((self.n_variables, self.n_timesteps), float("nan"))
        result[self.counted] = 0
        if long_format_data:
            feature_idx = torch.tensor([self.variable_index[cell[0]] for cell in long_format_data])
            step_idx = torch.tensor([self.step_index[cell[1]] for cell in long_format_data])
            values = torch.tensor([cell[2] for cell in long_format_data], dtype=torch.float32)
            result.index_put_((feature_idx, step_idx), values)

        if self.target != "mortality":
            msg = f"Target {self.target} is not implemented"
            raise NotImplementedError(msg)

        # If the person has an entry in the death table within the observation's window, report 1, else 0.
        # The window is that of the row of edata.obs, so that rows of the same person, such as repeated cohort entries, keep their own.
        omop_io_observation_table = self.edata.uns["omop_io_observation_table"]

        # Get the appropriate date/datetime column names based on precision
        time_precision = "datetime" if self.datetime else "date"
        start_date_col = DATA_TABLE_DATE_KEYS["start"][omop_io_observation_table].get(
            time_precision, DATA_TABLE_DATE_KEYS["start"][omop_io_observation_table]["date"]
        )
        end_date_col = DATA_TABLE_DATE_KEYS["end"][omop_io_observation_table].get(
            time_precision, DATA_TABLE_DATE_KEYS["end"][omop_io_observation_table]["date"]
        )

        obs = self.edata.obs.iloc[row]
        columns = {column.lower(): column for column in self.edata.obs.columns}
        start, end = (pd.to_datetime(obs[columns[column]]) for column in (start_date_col, end_date_col))
        target_query = """
        SELECT COALESCE(MAX(CASE WHEN death_datetime BETWEEN ? AND ? THEN 1 ELSE 0 END), 0) AS mortality
        FROM death
        WHERE person_id = ?
        """
        window = [None if pd.isna(date) else date.to_pydatetime() for date in (start, end)]
        person_id = int(obs[columns["person_id"]])
        targets = torch.tensor(self.con.execute(target_query, [*window, person_id]).fetchall(), dtype=torch.float32)

        return result, targets
