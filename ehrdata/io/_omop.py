from __future__ import annotations

from pathlib import Path
from typing import Literal

import duckdb


def _check_sanity_of_folder(folder_path: str | Path):
    pass


def _check_sanity_of_database(backend_handle: duckdb.DuckDB):
    pass


def load(
    backend_handle: Literal[str, duckdb, Path],
    # folder_path: str,
    # delimiter: str = ",",
    # make_filename_lowercase: bool = True,
    # use_dask: bool = False,
    # level: Literal["stay_level", "patient_level"] = "stay_level",
    # load_tables: str | list[str] | tuple[str] | Literal["auto"] | None = None,
    # remove_empty_column: bool = True,
) -> None:
    """Initialize a connection to the OMOP CDM Database"""
    if isinstance(backend_handle, str) or isinstance(backend_handle, Path):
        _check_sanity_of_folder(backend_handle)
    elif isinstance(backend_handle, duckdb.DuckDB):
        _check_sanity_of_database(backend_handle)
    else:
        raise NotImplementedError(f"Backend {backend_handle} not supported. Choose a valid backend.")


def extract_person():
    pass


def extract_measurement():
    pass


def extract_observation():
    pass


def extract_procedure_occurrence():
    pass


def extract_specimen():
    pass


def extract_device_exposure():
    pass


def extract_drug_exposure():
    pass


def extract_condition_occurrence():
    pass


def extract_note():
    pass
