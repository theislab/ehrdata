from collections.abc import Iterable
from typing import Literal

from anndata import AnnData


# TODO: check against aggregate_timeseries_in_bins
# TODO: here just some draft arguments to spark ideas
def time_interval_table(
    self,
    duckdb_instance,
    start_time: str = "patient_hospital_entry",
    observation_duration: int = 48,
    interval_length: float = 60,
    concept_ids: str | Iterable = "all",
    interval_unit="minutes",
) -> None:
    """Takes as input extracted time series; returns person x feature x timestep table"""
    pass


# TODO should be able to also extract data from .obsm
# TODO: allow for some ideas of get_feature_statistics, but break down
def get_feature_statistics(
    adata: AnnData,
    features: str | int | list[str | int] = None,
    level: Literal["stay_level", "patient_level"] = "stay_level",
    value_col: str = None,
    aggregation_methods: Literal["min", "max", "mean", "std", "count"]
    | list[Literal["min", "max", "mean", "std", "count"]]
    | None = None,
    add_aggregation_to: Literal["obs", "X", "return"] = "return",
    verbose: bool = False,
    use_dask: bool = None,
) -> AnnData:
    pass


# TODO: study previous functionality from thesis and what to include


def last_observation_carried_forward(
    adata: AnnData,
    features: str | int | list[str | int] = None,
    # level: Literal["stay_level", "patient_level"] = "stay_level",
    value_col: str = None,
    verbose: bool = False,
    use_dask: bool = None,
) -> AnnData:
    """Impute missing values with the last observation carried forward (LOCF) method."""
    pass


# TODO: more imputation methods to work on time series data tensor
