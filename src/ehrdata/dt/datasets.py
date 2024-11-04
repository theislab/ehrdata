from __future__ import annotations

import io
import os
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from duckdb.duckdb import DuckDBPyConnection

if TYPE_CHECKING:
    from ehrdata import EHRData
from ehrdata.utils._omop_utils import get_table_catalog_dict


def _get_table_list() -> list:
    flat_table_list = []
    for _, value_list in get_table_catalog_dict().items():
        for value in value_list:
            flat_table_list.append(value)
    return flat_table_list


def _set_up_duckdb(path: Path, backend_handle: DuckDBPyConnection, prefix: str = "") -> None:
    """Create tables in the backend from the CSV files in the path from datasets in the OMOP Common Data model."""
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
                backend_handle.read_csv(f"{path}/{file_name}", dtype=dtype),
            )
        else:
            unused_files.append(file_name)

    for table in tables:
        if table not in used_tables:
            missing_tables.append(table)

    print("missing tables: ", missing_tables)
    print("unused files: ", unused_files)


def _setup_eunomia_datasets(
    backend_handle: DuckDBPyConnection,
    data_path: Path | None = None,
    URL: str = None,
    dataset_postfix: str = "",
    dataset_prefix: str = "",
) -> None:
    """Loads the Eunomia datasets in the OMOP Common Data model."""
    if os.path.exists(data_path):
        print(f"Path to data exists, load tables from there: {data_path}")
    else:
        print("Downloading data...")
        response = requests.get(URL)

        if response.status_code == 200:
            # Use zipfile and io to open the ZIP file in memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Extract all contents of the ZIP file
                z.extractall(data_path)  # Specify the folder where files will be extracted
                print(
                    f"Download successful. ZIP file downloaded and extracted successfully to {data_path/dataset_postfix}."
                )
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
            return

    return _set_up_duckdb(data_path / dataset_postfix, backend_handle, prefix=dataset_prefix)


def mimic_iv_omop(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the MIMIC-IV demo data in the OMOP Common Data model.

    This function loads the MIMIC-IV demo dataset from its `physionet repository <https://physionet.org/content/mimic-iv-demo-omop/0.9/#files-panel>_` .
    See also this link for more details.

    DOI https://doi.org/10.13026/2d25-8g07.

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

    return _setup_eunomia_datasets(
        backend_handle,
        data_path,
        URL="https://physionet.org/static/published-projects/mimic-iv-demo-omop/mimic-iv-demo-data-in-the-omop-common-data-model-0.9.zip",
        dataset_postfix="mimic-iv-demo-data-in-the-omop-common-data-model-0.9/1_omop_data_csv",
        dataset_prefix="2b_",
    )


def gibleed_omop(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the GIBleed dataset in the OMOP Common Data model.

    This function loads the GIBleed dataset from the `EunomiaDatasets repository <https://github.com/OHDSI/EunomiaDatasets>_`.
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

    return _setup_eunomia_datasets(
        backend_handle,
        data_path,
        URL="https://github.com/OHDSI/EunomiaDatasets/raw/main/datasets/GiBleed/GiBleed_5.3.zip",
        dataset_postfix="GiBleed_5.3",
    )


def synthea27nj_omop(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the Synthea27NJ dataset in the OMOP Common Data model.

    This function loads the Synthea27NJ dataset from the `EunomiaDatasets repository <https://github.com/OHDSI/EunomiaDatasets>_`.
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

    return _setup_eunomia_datasets(
        backend_handle,
        data_path,
        URL="https://github.com/OHDSI/EunomiaDatasets/raw/main/datasets/Synthea27Nj/Synthea27Nj_5.4.zip",
    )


def physionet2012(
    data_path: Path | None = None,
    interval_length_number: int = 1,
    interval_length_unit: str = "day",
    num_intervals: int = 48,
    aggregation_strategy: str = "last",
    drop_samples: Sequence[str] = [
        147514,
        142731,
        145611,
        140501,
        155655,
        143656,
        156254,
        150309,
        140936,
        141264,
        150649,
        142998,
    ],
) -> EHRData:
    """Loads the dataset of the `PhysioNet challenge 2012 (v1.0.0) <https://physionet.org/content/challenge-2012/1.0.0/>_`.

    If interval_length_number is 1, interval_length_unit is "day", and num_intervals is 48, this is equivalent to the SAITS preprocessing (insert paper/link/citation).
    Truncated if a sample has more num_intervals steps; Padded if a sample has less than num_intervals steps.
    Further, by default the following 12 samples are dropped since they have no time series information at all: 147514, 142731, 145611, 140501, 155655, 143656, 156254, 150309,
    140936, 141264, 150649, 142998.

    Taken the defaults of interval_length_number, interval_length_unit, num_intervals, and drop_samples, the tensor stored in .r of edata is the same as when doing the PyPOTS <insert citation/link/reference> preprocessing.
    A simple deviation is that the tensor in ehrdata is of shape n_obs x n_vars x n_intervals (with defaults, 3000x37x48) while the tensor in PyPOTS is of shape n_obs x n_intervals x n_vars (3000x48x37).
    The tensor stored in .r is hence also fully compatible with the PyPOTS package, as the .r tensor of EHRData objects generally is.

    data_path
        Path to the raw data. If the path exists, the data is loaded from there. Else, the data is downloaded.
    interval_length_number
        Numeric value of the length of one interval.
    interval_length_unit
        Unit belonging to the interval length.
    num_intervals
        Number of intervals.
    aggregation_strategy
        Aggregation strategy for the time series data.
    drop_samples
        Samples to drop from the dataset (indicate their RecordID).

    Returns
    -------
    Returns a the processed physionet2012 dataset in an EHRData object. The raw data is also downloaded, stored and available under the data_path.

    Examples
    --------
        >>> import ehrapy as ep
        >>> import ehrdata as ed
        >>> edata = ed.dt.physionet_2012()
        >>> edata
    """
    if data_path is None:
        data_path = Path("ehrapy_data/physionet2012")

    pass
    # download data
    # load data
    # put a/b/c in obs
    # put outcomes in obs
    # put record id in obs
    # put units to var
    # put featurenames to var
    # put time to t


def physionet2019():
    """Loads the dataset of the `PhysioNet challenge 2019 <https://physionet.org/content/challenge-2019/1.0.0/>_`."""
    raise NotImplementedError()


def mimic_ii(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the MIMIC2 dataset."""
    # TODO: replace mimic_ii as is in ehrapy with its dict-of-table return time - map variables to OMOP?
    raise NotImplementedError()
