from __future__ import annotations

import io
import os
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import requests
from duckdb.duckdb import DuckDBPyConnection

from ehrdata.utils._omop_utils import get_table_catalog_dict

if TYPE_CHECKING:
    from ehrdata import EHRData
import numpy as np

from ehrdata.dt.dataloader import download
from ehrdata.io.omop._queries import _generate_timedeltas


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
    # TODO: capitalization, and lowercase, and containing the name
    return _set_up_duckdb(data_path + "/1_omop_data_csv", backend_handle, prefix="2b_")


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
            # extract_path = data_path / "gibleed_data_csv"
            # extract_path.mkdir(parents=True, exist_ok=True)

            # Use zipfile and io to open the ZIP file in memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Extract all contents of the ZIP file into the correct subdirectory
                z.extractall(data_path)  # Extracting to 'extract_path'
                print(f"Download successful. ZIP file downloaded and extracted successfully to {data_path}.")

        else:
            print(f"Failed to download the file. Status code: {response.status_code}")

    # extracted_folder = next(data_path.iterdir(), data_path)
    # extracted_folder = next((folder for folder in data_path.iterdir() if folder.is_dir() and "_csv" in folder.name and "__MACOSX" not in folder.name), data_path)
    return _set_up_duckdb(data_path / "GiBleed_5.3", backend_handle)


def synthea27nj_omop(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the Synthea27NJ dataset in the OMOP Common Data model.

    More details: https://github.com/darwin-eu/EunomiaDatasets/tree/main/datasets/Synthea27Nj.

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
            extract_path = data_path / "synthea27nj_omop_csv"
            extract_path.mkdir(parents=True, exist_ok=True)

            # Use zipfile and io to open the ZIP file in memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Extract all contents of the ZIP file into the correct subdirectory
                z.extractall(extract_path)  # Extracting to 'extract_path'
                print(f"Download successful. ZIP file downloaded and extracted successfully to {extract_path}.")

        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
            return

    extracted_folder = next(
        (
            folder
            for folder in data_path.iterdir()
            if folder.is_dir() and "_csv" in folder.name and "__MACOSX" not in folder.name
        ),
        data_path,
    )
    return _set_up_duckdb(extracted_folder, backend_handle)


def mimic_ii(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the MIMIC2 dataset"""
    # TODO: replace mimic_ii as is in ehrapy with its dict-of-table return time - map variables to OMOP?
    raise NotImplementedError()


def physionet2012(
    data_path: Path | None = None,
    interval_length_number: int = 1,
    interval_length_unit: str = "h",
    num_intervals: int = 48,
    aggregation_strategy: str = "last",
    drop_samples: Sequence[str] = [
        "147514",
        "142731",
        "145611",
        "140501",
        "155655",
        "143656",
        "156254",
        "150309",
        "140936",
        "141264",
        "150649",
        "142998",
    ],
) -> EHRData:
    """Loads the dataset of the `PhysioNet challenge 2012 (v1.0.0) <https://physionet.org/content/challenge-2012/1.0.0/>_`.

    If interval_length_number is 1, interval_length_unit is "h" (hour), and num_intervals is 48, this is equivalent to the SAITS preprocessing (insert paper/link/citation).
    Truncated if a sample has more num_intervals steps; Padded if a sample has less than num_intervals steps.
    Further, by default the following 12 samples are dropped since they have no time series information at all: 147514, 142731, 145611, 140501, 155655, 143656, 156254, 150309,
    140936, 141264, 150649, 142998.
    Taken the defaults of interval_length_number, interval_length_unit, num_intervals, and drop_samples, the tensor stored in .r of edata is the same as when doing the PyPOTS <insert citation/link/reference> preprocessing.
    A simple deviation is that the tensor in ehrdata is of shape n_obs x n_vars x n_intervals (with defaults, 3000x37x48) while the tensor in PyPOTS is of shape n_obs x n_intervals x n_vars (3000x48x37).
    The tensor stored in .r is hence also fully compatible with the PyPOTS package, as the .r tensor of EHRData objects generally is.

    Parameters
    ----------
    data_path
        Path to the raw data. If the path exists, the data is loaded from there. Else, the data is downloaded.
    interval_length_number
        Numeric value of the length of one interval.
    interval_length_unit
        Unit belonging to the interval length.
    num_intervals
        Number of intervals.
    aggregation_strategy
        Aggregation strategy for the time series data when multiple measurements for a person's parameter within a time interval is available. Available are 'first' and 'last', as used in `pd.DataFrame.drop_duplicates <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html>`_.
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
    from ehrdata import EHRData

    if data_path is None:
        data_path = Path("ehrapy_data/physionet2012")

    # alternative in future?
    # config_parser = tsdb.utils.config.read_configs()
    # tsdb.utils.config.write_configs(
    #     config_parser=config_parser, key_value_set={"path": {"data_home": "ehrapy_data/physionet2012"}}
    # )

    # tsdb.load(dataset_name="physionet_2012")

    # # returns a dictionary
    # tsdb.data_processing.load_physionet2012(data_path)

    outcome_file_names = ["Outcomes-a.txt", "Outcomes-b.txt", "Outcomes-c.txt"]
    temp_data_set_names = ["set-a", "set-b", "set-c"]

    for file_name in temp_data_set_names:
        download(
            url=f"https://physionet.org/files/challenge-2012/1.0.0/{file_name}.tar.gz?download",
            output_path=data_path,
            output_file_name=file_name + ".tar.gz",
            archive_format="gztar",
        )

    for file_name in outcome_file_names:
        download(
            url=f"https://physionet.org/files/challenge-2012/1.0.0/{file_name}?download",
            output_path=data_path,
            output_file_name=file_name,
        )

    static_features = ["Age", "Gender", "ICUType", "Height"]

    person_long_across_set_collector = []
    for data_subset_dir in temp_data_set_names:
        person_long_within_set_collector = []

        # each txt file is the data of a person, in long format
        # the columns in the txt files are: Time, Parameter, Value
        for txt_file in (data_path / data_subset_dir).glob("*.txt"):
            person_long = pd.read_csv(txt_file)

            # add RecordID (=person id in this dataset) to all data points of this person
            person_long["RecordID"] = int(txt_file.stem)
            person_long_within_set_collector.append(person_long)

        person_long_within_set_df = pd.concat(person_long_within_set_collector)

        person_long_within_set_df["set"] = data_subset_dir
        person_long_across_set_collector.append(person_long_within_set_df)

    person_long_across_set_df = pd.concat(person_long_across_set_collector)

    person_outcome_collector = []
    for outcome_file_name in outcome_file_names:
        outcome_df = pd.read_csv(data_path / outcome_file_name)
        # outcome_df["set"] = "set-" + outcome_file_name.split("-")[1].split(".")[0]
        person_outcome_collector.append(outcome_df)

    person_outcome_df = pd.concat(person_outcome_collector)

    # gather the static_features together with RecordID and set for each person into the obs table
    obs = (
        person_long_across_set_df[person_long_across_set_df["Parameter"].isin(static_features)]
        .pivot(index=["RecordID", "set"], columns=["Parameter"], values=["Value"])
        .reset_index(level="set", col_level=1)
    )
    obs.columns = obs.columns.droplevel(0)

    obs = obs.merge(person_outcome_df, how="left", left_on="RecordID", right_on="RecordID")
    obs.set_index("RecordID", inplace=True)
    # obs.index = obsobs["RecordID"] = obs.index

    # consider only time series features from now
    df_long = person_long_across_set_df[~person_long_across_set_df["Parameter"].isin(static_features)]

    interval_df = _generate_timedeltas(
        interval_length_number=interval_length_number,
        interval_length_unit=interval_length_unit,
        num_intervals=48,
    )

    df_long_time_seconds = np.array(pd.to_timedelta(df_long["Time"] + ":00").dt.total_seconds())
    interval_df_interval_end_offset_seconds = np.array(interval_df["interval_end_offset"].dt.total_seconds())
    df_long_interval_step = np.argmax(df_long_time_seconds[:, None] <= interval_df_interval_end_offset_seconds, axis=1)
    df_long["interval_step"] = df_long_interval_step

    # if one person for one feature (=Parameter) within one interval_step has multiple measurements, decide which one to keep
    df_long = df_long.drop_duplicates(subset=["RecordID", "Parameter", "interval_step"], keep=aggregation_strategy)

    xa = df_long.set_index(["RecordID", "Parameter", "interval_step"]).to_xarray()

    var = xa["Parameter"].to_dataframe()
    t = xa["interval_step"].to_dataframe()
    r = xa["Value"].values

    edata = EHRData(r=r, obs=obs, var=var, t=t)

    return edata[~edata.obs.index.isin(drop_samples)]


def physionet2019():
    """Loads the dataset of the `PhysioNet challenge 2019 <https://physionet.org/content/challenge-2019/1.0.0/>_`."""
    raise NotImplementedError()
