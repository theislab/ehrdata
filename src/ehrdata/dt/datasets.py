from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import requests
from duckdb.duckdb import DuckDBPyConnection

from ehrdata.utils._omop_utils import get_table_catalog_dict


def _get_table_list() -> list:
    flat_table_list = []
    for _, value_list in get_table_catalog_dict().items():
        for value in value_list:
            flat_table_list.append(value)
    return flat_table_list


def _set_up_duckdb(path: Path, backend_handle: DuckDBPyConnection) -> None:
    tables = _get_table_list()

    missing_tables = []
    for table in tables:
        # if path exists lowercse, uppercase, capitalized:
        table_path = f"{path}/{table}.csv"
        if os.path.exists(table_path):
            if table == "measurement":
                backend_handle.register(
                    table, backend_handle.read_csv(f"{path}/{table}.csv", dtype={"measurement_source_value": str})
                )
            else:
                backend_handle.register(table, backend_handle.read_csv(f"{path}/{table}.csv"))
        else:
            missing_tables.append([table])
    print("missing tables: ", missing_tables)


def mimic_iv_omop(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the MIMIC-IV demo data in the OMOP Common Data model.

    More details: https://physionet.org/content/mimic-iv-demo-omop/0.9/#files-panel.

    Parameters
    ----------
    backend_handle
        A handle to the backend which shall be used. Only duckdb connection supported at the moment.
    data_path
        Path to the tables. If the path exists, the data is loaded from there. Else, the data is downloaded.

    Returns
    -------
    Returns nothing, but adds the tables to the backend via the handle.

    Examples
    --------
        >>> import ehrapy as ep
        >>> import ehrdata as ed
        >>> import duckdb
        >>> con = duckdb.connect()
        >>> ed.dt.mimic_iv_omop(backend_handle=con)
        >>> con.execute("SHOW TABLES;").fetchall()
    """
    if data_path is None:
        data_path = "ehrapy_data/mimic-iv-demo-data-in-the-omop-common-data-model-0.9"

    if os.path.exists(data_path):
        print(f"Path to data exists, load tables from there: {data_path}")
    else:
        print("Downloading data...")
        URL = "https://physionet.org/static/published-projects/mimic-iv-demo-omop/mimic-iv-demo-data-in-the-omop-common-data-model-0.9.zip"
        response = requests.get(URL)

        if response.status_code == 200:
            # Step 2: Use zipfile and io to open the ZIP file in memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Extract all contents of the ZIP file
                z.extractall("ehrapy_data")  # Specify the folder where files will be extracted
                print(f"Download successful. ZIP file downloaded and extracted successfully to {data_path}.")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
            return

    return _set_up_duckdb(data_path + "/1_omop_data_csv", backend_handle)


def gibleed_omop(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the GIBleed dataset.

    More details: https://github.com/OHDSI/EunomiaDatasets.

    Parameters
    ----------
    backend_handle
        A handle to the backend which shall be used. Only duckdb connection supported at the moment.
    data_path
        Path to the tables. If the path exists, the data is loaded from there. Else, the data is downloaded.

    Returns
    -------
    Returns nothing, but adds the tables to the backend via the handle.

    Examples
    --------
        >>> import ehrapy as ep
        >>> import ehrdata as ed
        >>> import duckdb
        >>> con = duckdb.connect()
        >>> ed.dt.gibleed_omop(backend_handle=con)
        >>> con.execute("SHOW TABLES;").fetchall()
    """
    # TODO:
    # https://github.com/darwin-eu/EunomiaDatasets/tree/main/datasets/GiBleed
    raise NotImplementedError()


def synthea27nj_omop(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the Synthe27Nj dataset.

    Parameters
    ----------
    backend_handle
        A handle to the backend which shall be used. Only duckdb connection supported at the moment.
    data_path
        Path to the tables. If the path exists, the data is loaded from there. Else, the data is downloaded.

    Returns
    -------
    Returns nothing, but adds the tables to the backend via the handle.

    Examples
    --------
        >>> import ehrapy as ep
        >>> import ehrdata as ed
        >>> import duckdb
        >>> con = duckdb.connect()
        >>> ed.dt.synthea27nj_omop(backend_handle=con)
        >>> con.execute("SHOW TABLES;").fetchall()
    """
    # TODO
    # https://github.com/darwin-eu/EunomiaDatasets/tree/main/datasets/Synthea27Nj
    raise NotImplementedError()


def mimic_ii(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the MIMIC2 dataset"""
    # TODO: replace mimic_ii as is in ehrapy with its dict-of-table return time - map variables to OMOP?
    raise NotImplementedError()
