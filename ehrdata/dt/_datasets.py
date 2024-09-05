# maybe download csv for all of them;
# and then can do ep.io.extract_omop(...)
# or duckdb.connect_or_so_to_csvs_in_directory(paths)

# decide on api; connect vs load vs extract..

import io
import os
import zipfile

import requests

from ehrdata.utils._omop_utils import get_table_catalog_dict


def _get_table_list():
    flat_table_list = []
    for _, value_list in get_table_catalog_dict().items():
        for value in value_list:
            flat_table_list.append(value)
    return flat_table_list


def _set_up_duckdb(path, backend_handle):
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


def mimic_iv_omop(backend="duckdb", backend_handle=None) -> None:
    """Loads the MIMIC-IV demo data in the OMOP Common Data model.

    More details: https://physionet.org/content/mimic-iv-demo-omop/0.9/#files-panel

    Args:
        backend: What backend shall be used to load and handle the tables.
        Currently, only "duckdb" is supported.
        backend_handle: A handle to the backend which shall be used. Should be a duckdb connection.

    Returns
    -------
        None. Maybe a report in the future

    Examples
    --------
        >>> import ehrapy as ep
        >>> import ehrdata as ed
        >>> import duckdb
        >>> conn = duckdb.connect()
        >>> ed.dt.mimic_iv_omop(backend_handle=conn)
        >>> con.execute("SHOW TABLES;").fetchall()
    """
    DATA_PATH = "ehrapy_data/mimic-iv-demo-data-in-the-omop-common-data-model-0.9"
    if os.path.exists(DATA_PATH):
        print(f"Load downloaded tables from {DATA_PATH}")
    else:
        URL = "https://physionet.org/static/published-projects/mimic-iv-demo-omop/mimic-iv-demo-data-in-the-omop-common-data-model-0.9.zip"
        response = requests.get(URL)

        if response.status_code == 200:
            # Step 2: Use zipfile and io to open the ZIP file in memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Extract all contents of the ZIP file
                z.extractall("ehrapy_data")  # Specify the folder where files will be extracted
                print(f"ZIP file downloaded and extracted successfully to {DATA_PATH}.")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
            return

    return _set_up_duckdb(DATA_PATH + "/1_omop_data_csv", backend_handle)


def gibleed_omop(backend="duckdb"):
    # TODO:
    # https://github.com/darwin-eu/EunomiaDatasets/tree/main/datasets/GiBleed
    raise NotImplementedError()


def synthea27Nj_omop(backend="duckdb"):
    # TODO
    # https://github.com/darwin-eu/EunomiaDatasets/tree/main/datasets/Synthea27Nj
    raise NotImplementedError()


def mimic_2(backend="duckdb"):
    # TODO: replace mimic_2 as is in ehrapy with its dict-of-table return time - map variables to OMOP?
    raise NotImplementedError()
