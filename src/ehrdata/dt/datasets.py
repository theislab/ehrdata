from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from duckdb.duckdb import DuckDBPyConnection

from ehrdata.dt.dataloader import download
from ehrdata.io.omop import setup_connection
from ehrdata.io.omop._queries import _generate_timedeltas

if TYPE_CHECKING:
    from ehrdata import EHRData


def _setup_eunomia_datasets(
    data_url: str,
    backend_handle: DuckDBPyConnection,
    data_path: Path | None = None,
    nested_omop_tables_folder: str = None,
    dataset_prefix: str = "",
) -> None:
    """Loads the Eunomia datasets in the OMOP Common Data model."""
    download(
        data_url,
        saving_path=data_path,
    )

    if nested_omop_tables_folder:
        for file_path in (data_path / nested_omop_tables_folder).glob("*.csv"):
            shutil.move(file_path, data_path)

    setup_connection(
        data_path,
        backend_handle,
        prefix=dataset_prefix,
    )


def mimic_iv_omop(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the MIMIC-IV demo data in the OMOP Common Data model.

    This function loads the MIMIC-IV demo dataset from its `physionet repository <https://physionet.org/content/mimic-iv-demo-omop/0.9/#files-panel>`_.
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
    Returns nothing, adds the tables to the backend via the handle.

    Examples
    --------
        >>> import ehrapy as ep
        >>> import ehrdata as ed
        >>> import duckdb
        >>> con = duckdb.connect()
        >>> ed.dt.mimic_iv_omop(backend_handle=con)
        >>> con.execute("SHOW TABLES;").fetchall()
    """
    data_url = "https://physionet.org/static/published-projects/mimic-iv-demo-omop/mimic-iv-demo-data-in-the-omop-common-data-model-0.9.zip"
    if data_path is None:
        data_path = Path("ehrapy_data/mimic-iv-demo-data-in-the-omop-common-data-model-0.9")

    _setup_eunomia_datasets(
        data_url=data_url,
        backend_handle=backend_handle,
        data_path=data_path,
        nested_omop_tables_folder="mimic-iv-demo-data-in-the-omop-common-data-model-0.9/1_omop_data_csv",
        dataset_prefix="2b_",
    )


def gibleed_omop(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the GiBleed dataset in the OMOP Common Data model.

    This function loads the GIBleed dataset from the `EunomiaDatasets repository <https://github.com/OHDSI/EunomiaDatasets>`_.
    More details: https://github.com/OHDSI/EunomiaDatasets/tree/main/datasets/GiBleed.

    Parameters
    ----------
    backend_handle
        A handle to the backend which shall be used. Only duckdb connection supported at the moment.
    data_path
        Path to the tables. If the path exists, the data is loaded from there. Else, the data is downloaded.

    Returns
    -------
    Returns nothing, adds the tables to the backend via the handle.

    Examples
    --------
        >>> import ehrapy as ep
        >>> import ehrdata as ed
        >>> import duckdb
        >>> con = duckdb.connect()
        >>> ed.dt.gibleed_omop(backend_handle=con)
        >>> con.execute("SHOW TABLES;").fetchall()
    """
    data_url = "https://github.com/OHDSI/EunomiaDatasets/raw/main/datasets/GiBleed/GiBleed_5.3.zip"

    if data_path is None:
        data_path = Path("ehrapy_data/GiBleed_5.3")

    _setup_eunomia_datasets(
        data_url=data_url,
        backend_handle=backend_handle,
        data_path=data_path,
        nested_omop_tables_folder="GiBleed_5.3",
    )


def synthea27nj_omop(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the Synthea27Nj dataset in the OMOP Common Data model.

    This function loads the Synthea27Nj dataset from the `EunomiaDatasets repository <https://github.com/OHDSI/EunomiaDatasets>`_.
    More details: https://github.com/OHDSI/EunomiaDatasets/tree/main/datasets/Synthea27Nj.

    Parameters
    ----------
    backend_handle
        A handle to the backend which shall be used. Only duckdb connection supported at the moment.
    data_path
        Path to the tables. If the path exists, the data is loaded from there. Else, the data is downloaded.

    Returns
    -------
    Returns nothing, adds the tables to the backend via the handle.

    Examples
    --------
        >>> import ehrapy as ep
        >>> import ehrdata as ed
        >>> import duckdb
        >>> con = duckdb.connect()
        >>> ed.dt.synthea27nj_omop(backend_handle=con)
        >>> con.execute("SHOW TABLES;").fetchall()
    """
    data_url = "https://github.com/OHDSI/EunomiaDatasets/raw/main/datasets/Synthea27Nj/Synthea27Nj_5.4.zip"

    if data_path is None:
        data_path = Path("ehrapy_data/Synthea27Nj_5.4")

    _setup_eunomia_datasets(
        data_url=data_url,
        backend_handle=backend_handle,
        data_path=data_path,
    )


def mimic_ii(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the MIMIC2 dataset."""
    # TODO: replace mimic_ii as is in ehrapy with its dict-of-table return time - map variables to OMOP?
    raise NotImplementedError()


def physionet2012(
    data_path: Path | None = None,
    interval_length_number: int = 1,
    interval_length_unit: str = "h",
    num_intervals: int = 48,
    aggregation_strategy: str = "last",
    drop_samples: Sequence[str] | None = [
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
    """Loads the dataset of the `PhysioNet challenge 2012 (v1.0.0) <https://physionet.org/content/challenge-2012/1.0.0/>`_.

    If interval_length_number is 1, interval_length_unit is "h" (hour), and num_intervals is 48, this is equivalent to the `SAITS <https://arxiv.org/pdf/2202.08516>`_ preprocessing.
    Truncated if a sample has more num_intervals steps; Padded if a sample has less than num_intervals steps.
    Further, by default the following 12 samples are dropped since they have no time series information at all: 147514, 142731, 145611, 140501, 155655, 143656, 156254, 150309,
    140936, 141264, 150649, 142998.
    Taken the defaults of interval_length_number, interval_length_unit, num_intervals, and drop_samples, the tensor stored in .r of edata is the same as when doing the `PyPOTS <https://github.com/WenjieDu/PyPOTS>`_ preprocessing.
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
            saving_path=data_path,
        )

    for file_name in outcome_file_names:
        download(
            url=f"https://physionet.org/files/challenge-2012/1.0.0/{file_name}?download",
            saving_path=data_path,
        )

    static_features = ["Age", "Gender", "ICUType", "Height"]

    person_long_across_set_collector = []
    for data_subset_dir in temp_data_set_names:
        person_long_within_set_collector = []

        # each txt file is the data of a person, in long format
        # the columns in the txt files are: Time, Parameter, Value
        for txt_file in (data_path / data_subset_dir).glob("*.txt"):
            person_long = pd.read_csv(txt_file)
            # drop the first row, which has the RecordID
            person_long = person_long.iloc[1:]

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

    # consider only time series features from now
    df_long = person_long_across_set_df[~person_long_across_set_df["Parameter"].isin(static_features)]

    interval_df = _generate_timedeltas(
        interval_length_number=interval_length_number,
        interval_length_unit=interval_length_unit,
        num_intervals=num_intervals,
    )

    df_long_time_seconds = np.array(pd.to_timedelta(df_long["Time"] + ":00").dt.total_seconds())
    interval_df_interval_end_offset_seconds = np.array(interval_df["interval_end_offset"].dt.total_seconds())
    df_long_interval_step = np.argmax(df_long_time_seconds[:, None] <= interval_df_interval_end_offset_seconds, axis=1)
    df_long.loc[:, ["interval_step"]] = df_long_interval_step

    # if one person for one feature (=Parameter) within one interval_step has multiple measurements, decide which one to keep
    df_long = df_long.drop_duplicates(subset=["RecordID", "Parameter", "interval_step"], keep=aggregation_strategy)

    xa = df_long.set_index(["RecordID", "Parameter", "interval_step"]).to_xarray()

    var = xa["Parameter"].to_dataframe()
    t = xa["interval_step"].to_dataframe()
    r = xa["Value"].values

    edata = EHRData(r=r, obs=obs, var=var, t=t)

    return edata[~edata.obs.index.isin(drop_samples or [])]


def physionet2019():
    """Loads the dataset of the `PhysioNet challenge 2019 <https://physionet.org/content/challenge-2019/1.0.0/>_`."""
    raise NotImplementedError()
