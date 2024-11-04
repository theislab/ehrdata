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


def _set_up_duckdb(path: Path, backend_handle: DuckDBPyConnection, prefix: str = "") -> None:
    tables = _get_table_list()

    used_tables = []
    missing_tables = []
    unused_files = []
    for file_name in os.listdir(path):
        file_name_trunk = file_name.split(".")[0].lower()

        if file_name_trunk in tables or file_name_trunk.replace(prefix, "") in tables:
            used_tables.append(file_name_trunk.replace(prefix, ""))

            if file_name_trunk == "measurement":
                dtype = {"measurement_source_value": str}
            else:
                dtype = None

            backend_handle.register(
                file_name_trunk.replace(prefix, ""),
                backend_handle.read_csv(f"{path}/{file_name_trunk}.csv", dtype=dtype),
            )
        else:
            unused_files.append(file_name)

    for table in tables:
        if table not in used_tables:
            missing_tables.append(table)

    print("missing tables: ", missing_tables)
    print("unused files: ", unused_files)


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
        data_path = Path("ehrapy_data/mimic-iv-demo-data-in-the-omop-common-data-model-0.9")

    if os.path.exists(data_path):
        print(f"Path to data exists, load tables from there: {data_path}")
    else:
        print("Downloading data...")
        URL = "https://physionet.org/static/published-projects/mimic-iv-demo-omop/mimic-iv-demo-data-in-the-omop-common-data-model-0.9.zip"
        response = requests.get(URL)

        if response.status_code == 200:
            # Use zipfile and io to open the ZIP file in memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Extract all contents of the ZIP file
                z.extractall("ehrapy_data")  # Specify the folder where files will be extracted
                print(f"Download successful. ZIP file downloaded and extracted successfully to {data_path}.")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
            return

    return _set_up_duckdb(data_path / "1_omop_data_csv", backend_handle, prefix="2b_")


def gibleed_omop(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the GIBleed dataset in the OMOP Common Data model.

    More details: https://github.com/OHDSI/EunomiaDatasets/tree/main/datasets/GiBleed.

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
    if data_path is None:
        data_path = Path("ehrapy_data/GIBleed_dataset")

    if data_path.exists():
        print(f"Path to data exists, load tables from there: {data_path}")
    else:
        print("Downloading data...")
        URL = "https://github.com/OHDSI/EunomiaDatasets/raw/main/datasets/GiBleed/GiBleed_5.3.zip"
        response = requests.get(URL)

        if response.status_code == 200:
            # Use zipfile and io to open the ZIP file in memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Extract all contents of the ZIP file into the correct subdirectory
                z.extractall(data_path)  # Extracting to 'extract_path'
                print(f"Download successful. ZIP file downloaded and extracted successfully to {data_path}.")

        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
            return

    return _set_up_duckdb(data_path / "GiBleed_5.3", backend_handle)


def synthea27nj_omop(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the Synthea27NJ dataset in the OMOP Common Data model.

    More details: https://github.com/OHDSI/EunomiaDatasets/tree/main/datasets/Synthea27Nj.

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
    if data_path is None:
        data_path = Path("ehrapy_data/Synthea27Nj")

    if data_path.exists():
        print(f"Path to data exists, load tables from there: {data_path}")
    else:
        print("Downloading data...")
        URL = "https://github.com/OHDSI/EunomiaDatasets/raw/main/datasets/Synthea27Nj/Synthea27Nj_5.4.zip"
        response = requests.get(URL)

        if response.status_code == 200:
            # Use zipfile and io to open the ZIP file in memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Extract all contents of the ZIP file into the correct subdirectory
                z.extractall(data_path)  # Extracting to 'extract_path'
                print(f"Download successful. ZIP file downloaded and extracted successfully to {data_path}.")

        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
            return

    return _set_up_duckdb(data_path, backend_handle)


def mimic_ii(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the MIMIC2 dataset"""
    # TODO: replace mimic_ii as is in ehrapy with its dict-of-table return time - map variables to OMOP?
    raise NotImplementedError()


# TODO: physionet2012, physionet2019
