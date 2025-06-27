from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from lamin_utils import logger

from ehrdata.core.constants import DEFAULT_DATA_PATH
from ehrdata.dt._dataloader import _download
from ehrdata.io import read_csv, read_h5ad
from ehrdata.io.omop import setup_connection
from ehrdata.io.omop._queries import _generate_timedeltas

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from duckdb.duckdb import DuckDBPyConnection

    from ehrdata import EHRData


from scipy.sparse import csr_matrix
from sparse import COO


def ehrdata_blobs(
    *,
    n_variables: int = 11,
    n_centers: int = 5,
    cluster_std: float = 1.0,
    n_observations: int = 1000,
    base_timepoints: int = 100,
    random_state: int | np.random.Generator = 0,
    sparse: bool = False,
    sparsity: float = 0.9,
    variable_length: bool = False,
    time_shifts: bool = False,
    seasonality: bool = False,
    irregular_sampling: bool = False,
    missing_values: float = 0.0,
) -> EHRData:
    """Generates time series example dataset suited for alignment tasks.

    Args:
        n_variables: Dimension of feature space.
        n_centers: Number of cluster centers.
        cluster_std: Standard deviation of clusters.
        n_observations: Number of observations.
        base_timepoints: Base number of time points (actual may vary per observation).
        random_state: Determines random number generation for dataset creation.
        sparse: Whether to use sparse matrices.
        sparsity: Target sparsity level when sparse=True.
        variable_length: Whether observations have different time series lengths.
        time_shifts: Whether to add time shifts between similar observations.
        seasonality: Whether to add seasonal patterns to time series.
        irregular_sampling: Whether sampling intervals vary between observations.
        missing_values: Fraction of random missing values in time series.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.ehrdata_blobs(
        ...     variable_length=True, time_shifts=True, seasonality=True, irregular_sampling=True
        ... )

    results in a dataset like:

    .. image:: /_static/tutorial_images/ehrdata_blobs.png
       :alt: EHR data blobs visualization
    """
    rng = np.random.default_rng(random_state if isinstance(random_state, int) else None)

    # Generate cluster centers and assignments
    centers = rng.normal(0, 5, size=(n_centers, n_variables))
    y = rng.integers(0, n_centers, size=n_observations)

    # Generate base feature values (X)
    X = np.zeros((n_observations, n_variables))
    X = centers[y] + rng.normal(0, cluster_std, size=(n_observations, n_variables))

    # Determine time series lengths for each observation
    if variable_length:
        # Vary length from 50% to 150% of base_timepoints
        lengths = rng.integers(max(10, int(base_timepoints * 0.5)), int(base_timepoints * 1.5), size=n_observations)
    else:
        lengths = np.full(n_observations, base_timepoints)

    max_length = int(lengths.max())

    # Create time points for each observation (potentially irregular)
    all_timepoints = []
    for i in range(n_observations):
        length = lengths[i]

        if irregular_sampling:
            # Create non-uniform time spacing
            if time_shifts and i > 0:
                # Add random shift for similar clusters
                shift = rng.uniform(0, 0.3 * base_timepoints)
                start = shift if y[i] == y[i - 1] else 0
            else:
                start = 0

            # Irregular intervals with increasing spacing
            intervals = rng.exponential(scale=1.0, size=length)
            intervals = intervals / intervals.sum() * (max_length - start)
            timepoints = np.cumsum(intervals) + start
        else:
            # Regular intervals
            if time_shifts and i > 0:  # noqa: SIM108
                # Add cluster-based shifts
                shift = rng.uniform(0, 0.3 * base_timepoints) if y[i] == y[i - 1] else 0
            else:
                shift = 0

            timepoints = np.linspace(shift, max_length + shift, length)

        all_timepoints.append(timepoints)

    # Create time index - use all unique timepoints from all observations
    all_unique_times = np.unique(np.concatenate(all_timepoints))
    all_unique_times.sort()

    n_total_timepoints = len(all_unique_times)
    t_index = pd.Index([str(i) for i in range(n_total_timepoints)])

    # Create time DataFrame with actual time values
    t_df = pd.DataFrame(
        {
            "timepoint": range(n_total_timepoints),
            "time_value": all_unique_times,
        },
        index=t_index,
    )

    # Prepare 3D array to store all time series
    R = np.zeros((n_observations, n_variables, n_total_timepoints))
    R.fill(np.nan)

    # Generate time series for each observation
    for i in range(n_observations):
        # Map this observation's time points to the global time index
        obs_timepoints = all_timepoints[i]

        # Find indices of these timepoints in the global time array
        time_indices = np.searchsorted(all_unique_times, obs_timepoints)

        # Generate patterns for this observation
        for v in range(n_variables):
            base_value = X[i, v]

            # Time series with different patterns
            time_series = np.zeros(len(time_indices))

            # Add trend component (linear increase based on variable value)
            trend = np.linspace(0, base_value * 0.5, len(time_indices))
            time_series += trend

            # Add seasonality if enabled
            if seasonality:
                freq = rng.uniform(3, 15)

                # Phase shift based on cluster
                phase = y[i] * np.pi / n_centers

                # Amplitude based on variable value
                amplitude = np.abs(base_value) * 0.3

                # Add seasonal component
                seasonal = amplitude * np.sin(freq * np.pi * np.arange(len(time_indices)) / len(time_indices) + phase)
                time_series += seasonal

            # Add noise increasing with time
            for t_idx, t in enumerate(time_indices):
                noise_scale = cluster_std / 2 * (0.5 + t / n_total_timepoints)
                time_series[t_idx] += rng.normal(0, noise_scale)

            time_series += base_value

            R[i, v, time_indices] = time_series

    # Add random missing values if requested
    if missing_values > 0:
        # Create a mask for random missing values (ignoring already missing values)
        missing_mask = rng.random(R.shape) < missing_values
        not_nan_mask = ~np.isnan(R)
        R[missing_mask & not_nan_mask] = np.nan

    # Ensure that X contains a snapshot of R at a common time index
    # Use the middle timepoint if available for each observation
    for i in range(n_observations):
        valid_times = ~np.isnan(R[i, 0, :])
        if np.any(valid_times):
            # Find middle timepoint for this observation
            valid_indices = np.where(valid_times)[0]
            mid_idx = valid_indices[len(valid_indices) // 2]

            # Update X to contain this snapshot
            for v in range(n_variables):
                X[i, v] = R[i, v, mid_idx]

    if sparse:
        mask_x = rng.random(X.shape) > sparsity
        data_x = X.copy()
        data_x[~mask_x] = 0
        X = csr_matrix(data_x)

        # For R, handle both NaN and sparsity
        # First replace NaN with 0 where we're keeping values
        mask_r = rng.random(R.shape) > sparsity
        R_copy = R.copy()
        R_copy[np.isnan(R)] = 0
        R_copy[~mask_r] = 0

        # Get coordinates and values for non-zero entries
        coords = np.where(R_copy != 0)
        values = R_copy[coords]

        R = COO(coords, values, shape=R.shape)

    from ehrdata import EHRData

    return EHRData(
        X=X,
        obs=pd.DataFrame({"cluster": y.astype(str)}, index=pd.Index([str(i) for i in range(n_observations)])),
        var=pd.DataFrame(index=pd.Index([f"feature_{i}" for i in range(n_variables)])),
        R=R,
        tem=t_df,
    )


def _setup_eunomia_datasets(
    data_url: str,
    backend_handle: DuckDBPyConnection,
    data_path: Path,
    nested_omop_tables_folder: str | None = None,
    dataset_prefix: str = "",
) -> None:
    """Loads the Eunomia datasets in the OMOP Common Data model."""
    _download(
        data_url,
        output_path=data_path,
    )

    if nested_omop_tables_folder:
        if len(list((data_path / nested_omop_tables_folder).glob("*.csv"))) > 0:
            logger.info(f"Moving files from {data_path / nested_omop_tables_folder} to {data_path}")
        for file_path in (data_path / nested_omop_tables_folder).glob("*.csv"):
            shutil.move(file_path, data_path)

    setup_connection(
        data_path,
        backend_handle,
        prefix=dataset_prefix,
    )


def mimic_iv_omop(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the MIMIC-IV demo data in the OMOP Common Data model.

    Loads the MIMIC-IV demo dataset from its `physionet repository <https://physionet.org/content/mimic-iv-demo-omop/0.9/#files-panel>`_ :cite:`kallfelz2021mimic`.

    Args:
        backend_handle: A handle to the backend which shall be used. Only duckdb connection supported at the moment.
        data_path: Path to the tables. If the path exists, the data is loaded from there. Else, the data is downloaded.

    Returns:
        Nothing. Adds the tables to the backend via the handle.

    Examples:
        >>> import ehrdata as ed
        >>> import duckdb
        >>> con = duckdb.connect()
        >>> ed.dt.mimic_iv_omop(backend_handle=con)
        >>> con.execute("SHOW TABLES;").fetchall()
    """
    data_url = "https://physionet.org/static/published-projects/mimic-iv-demo-omop/mimic-iv-demo-data-in-the-omop-common-data-model-0.9.zip"
    if data_path is None:
        data_path = DEFAULT_DATA_PATH / "ehrapy_data/mimic-iv-demo-data-in-the-omop-common-data-model-0.9"

    _setup_eunomia_datasets(
        data_url=data_url,
        backend_handle=backend_handle,
        data_path=data_path,
        nested_omop_tables_folder="mimic-iv-demo-data-in-the-omop-common-data-model-0.9/1_omop_data_csv",
        dataset_prefix="2b_",
    )


def gibleed_omop(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the GiBleed dataset in the OMOP Common Data model.

    Loads the GIBleed dataset from the `EunomiaDatasets repository <https://github.com/OHDSI/EunomiaDatasets>`_.
    More details: https://github.com/OHDSI/EunomiaDatasets/tree/main/datasets/GiBleed.

    Args:
        backend_handle: A handle to the backend which shall be used. Only duckdb connection supported at the moment.
        data_path: Path to the tables. If the path exists, the data is loaded from there. Else, the data is downloaded.

    Returns:
        Nothing. Adds the tables to the backend via the handle.

    Examples:
        >>> import ehrdata as ed
        >>> import duckdb
        >>> con = duckdb.connect()
        >>> ed.dt.gibleed_omop(backend_handle=con)
        >>> con.execute("SHOW TABLES;").fetchall()
    """
    data_url = "https://github.com/OHDSI/EunomiaDatasets/raw/main/datasets/GiBleed/GiBleed_5.3.zip"

    if data_path is None:
        data_path = DEFAULT_DATA_PATH / "GiBleed_5.3"

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

    Args:
        backend_handle: A handle to the backend which shall be used. Only duckdb connection supported at the moment.
        data_path: Path to the tables. If the path exists, the data is loaded from there. Else, the data is downloaded.

    Returns:
        Nothing. Adds the tables to the backend via the handle.

    Examples:
        >>> import ehrdata as ed
        >>> import duckdb
        >>> con = duckdb.connect()
        >>> ed.dt.synthea27nj_omop(backend_handle=con)
        >>> con.execute("SHOW TABLES;").fetchall()
    """
    data_url = "https://github.com/OHDSI/EunomiaDatasets/raw/main/datasets/Synthea27Nj/Synthea27Nj_5.4.zip"

    if data_path is None:
        data_path = DEFAULT_DATA_PATH / "Synthea27Nj_5.4"

    _setup_eunomia_datasets(
        data_url=data_url,
        backend_handle=backend_handle,
        data_path=data_path,
    )


def physionet2012(
    data_path: Path | None = None,
    *,
    interval_length_number: int = 1,
    interval_length_unit: str = "h",
    num_intervals: int = 48,
    aggregation_strategy: str = "last",
    drop_samples: Iterable[str] | None = [
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

    If `interval_length_number` is 1, `interval_length_unit` is `"h"` (hour), and `num_intervals` is 48, this is the same as the `SAITS <https://arxiv.org/pdf/2202.08516>`_ preprocessing :cite:`du2023saits`.
    Truncated if a sample has more `num_intervals` steps; Padded if a sample has less than `num_intervals` steps.
    Further, by default the following 12 samples are dropped since they have no time series information at all: 147514, 142731, 145611, 140501, 155655, 143656, 156254, 150309,
    140936, 141264, 150649, 142998.
    Taken the defaults of `interval_length_number`, `interval_length_unit`, `num_intervals`, and `drop_samples`, the tensor stored in `.R` of `edata` is the same as when doing the `PyPOTS <https://github.com/WenjieDu/PyPOTS>`_ preprocessing :cite:`du2023pypots`.
    A simple deviation is that the tensor in `ehrdata` is of shape `n_obs x n_vars x n_intervals` (with defaults, 3000x37x48) while the tensor in PyPOTS is of shape `n_obs x n_intervals x n_vars` (3000x48x37).
    The tensor stored in `.R` is hence also fully compatible with the PyPOTS package, as the `.R` tensor of EHRData objects generally is.
    Note: In the original dataset, some missing values are encoded with a -1 for some entries of the variables `'DiasABP'`, `'NIDiasABP'`, and `'Weight'`. Here, these are replaced with `NaN` s.

    Args:
       data_path: Path to the raw data. If the path exists, the data is loaded from there.
           Else, the data is downloaded.
       interval_length_number: Numeric value of the length of one interval.
       interval_length_unit: Unit belonging to the interval length.
       num_intervals: Number of intervals.
       aggregation_strategy: Aggregation strategy for the time series data when multiple
           measurements for a person's parameter within a time interval is available.
           Available are `'first'` and `'last'`, as used in :meth:`~pandas.DataFrame.drop_duplicates`.
       drop_samples: Samples to drop from the dataset (indicate their RecordID).

    Returns:
        The processed physionet2012 dataset.
        The raw data is also downloaded, stored and available under the ``data_path``.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.physionet_2012()
    """
    from ehrdata import EHRData

    if data_path is None:
        data_path = DEFAULT_DATA_PATH / "physionet2012"

    # alternative in future?
    # config_parser = tsdb.utils.config.read_configs()
    # tsdb.utils.config.write_configs(
    #     config_parser=config_parser, key_value_set={"path": {"data_home": "ehrapy_data/physionet2012"}}
    # )

    # tsdb.load(dataset_name="physionet_2012")

    # # returns a dictionary
    # tsdb.data_processing.load_physionet2012(data_path)

    outcome_filenames = ["Outcomes-a.txt", "Outcomes-b.txt", "Outcomes-c.txt"]
    temp_data_set_names = ["set-a", "set-b", "set-c"]

    for filename in temp_data_set_names:
        _download(
            url=f"https://physionet.org/files/challenge-2012/1.0.0/{filename}.tar.gz?download",
            output_path=data_path,
            output_filename=f"{filename}.tar.gz",
            archive_format="tar.gz",
        )

    for filename in outcome_filenames:
        _download(
            url=f"https://physionet.org/files/challenge-2012/1.0.0/{filename}?download",
            output_path=data_path,
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
    for outcome_filename in outcome_filenames:
        outcome_df = pd.read_csv(data_path / outcome_filename)
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
    tem = xa["interval_step"].to_dataframe()
    r = xa["Value"].values

    # Three time series variables in the original dataset ['DiasABP', 'NIDiasABP', 'Weight'] have -1 instead of NaN for some missing values
    # No -1 value in the original dataset represents a valid measurement, so we can safely replace -1 with NaN
    r[r == -1] = np.nan

    obs.index = obs.index.astype(str)
    var.index = var.index.astype(str)

    edata = EHRData(R=r, obs=obs, var=var, tem=tem)

    return edata[~edata.obs.index.isin(drop_samples or [])]


def mimic_2(
    columns_obs_only: Iterable[str] | None = None,
) -> EHRData:
    """Loads the MIMIC-II dataset.

    This dataset was created for the purpose of a case study in the book: `Secondary Analysis of Electronic Health Records <https://link.springer.com/book/10.1007/978-3-319-43742-2>`_ :cite:`critical2016secondary`.
    In particular, the dataset was used to investigate the effectiveness of indwelling arterial catheters in hemodynamically stable patients with respiratory failure for mortality outcomes.
    The dataset is derived from MIMIC-II, the publicly-accessible critical care database.
    It contains summary clinical data and outcomes for 1,776 patients.

    More details on the data can be found on `physionet <https://physionet.org/content/mimic2-iaccd/1.0/>`_.

    Args:
        columns_obs_only: Columns to include only in obs and not X.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
    """
    _download(
        "https://www.physionet.org/files/mimic2-iaccd/1.0/full_cohort_data.csv?download",
        output_path=DEFAULT_DATA_PATH,
        output_filename="ehrapy_mimic2.csv",
    )
    edata = read_csv(
        filename=f"{DEFAULT_DATA_PATH}/ehrapy_mimic2.csv",
        columns_obs_only=columns_obs_only,
    )

    return edata


def mimic_2_preprocessed() -> EHRData:
    """Loads the preprocessed MIMIC-II dataset.

    This dataset is a preprocessed version of :func:`~ehrdata.dt.mimic_2`.
    The dataset was preprocessed according to: https://github.com/theislab/ehrapy-datasets/tree/main/mimic_2.

    This dataset was created for the purpose of a case study in the book: `Secondary Analysis of Electronic Health Records <https://link.springer.com/book/10.1007/978-3-319-43742-2>`_ :cite:`critical2016secondary`.
    In particular, the dataset was used to investigate the effectiveness of indwelling arterial catheters in hemodynamically stable patients with respiratory failure for mortality outcomes.
    The dataset is derived from MIMIC-II, the publicly-accessible critical care database.
    It contains summary clinical data and outcomes for 1,776 patients.

    More details on the data can be found on `physionet <https://physionet.org/content/mimic2-iaccd/1.0/>`_.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2_preprocessed()
    """
    _download(
        url="https://figshare.com/ndownloader/files/39727936",
        output_path=DEFAULT_DATA_PATH,
        output_filename="mimic_2_preprocessed.h5ad",
        raw_format="h5ad",
    )
    edata = read_h5ad(
        filename=f"{DEFAULT_DATA_PATH}/mimic_2_preprocessed.h5ad",
    )

    return edata


def diabetes_130_raw(
    columns_obs_only: Iterable[str] | None = None,
) -> EHRData:
    """Loads the raw diabetes-130 dataset.

    More details and the original dataset can be found `here <http://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008>`_ :cite:`strack2014impact`.

    Args:
        columns_obs_only: Columns to include in `obs` only and not `X`.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.diabetes_130_raw()

    """
    _download(
        url="https://figshare.com/ndownloader/files/45110029",
        output_path=DEFAULT_DATA_PATH,
        output_filename="diabetes_130_raw.csv",
        raw_format="csv",
    )
    adata = read_csv(
        filename=f"{DEFAULT_DATA_PATH}/diabetes_130_raw.csv",
        columns_obs_only=columns_obs_only,
    )

    return adata


def diabetes_130_fairlearn(
    columns_obs_only: Iterable[str] | None = None,
) -> EHRData:
    """Loads the preprocessed diabetes-130 dataset by fairlearn.

    This loads the dataset from the `fairlearn.datasets.fetch_diabetes_hospital <https://fairlearn.org/v0.10/api_reference/generated/fairlearn.datasets.fetch_diabetes_hospital.html#fairlearn.datasets.fetch_diabetes_hospital>`_ function. :cite:`bird2020fairlearn`

    More details and the original dataset can be found `here <http://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008>`_.

    Args:
        columns_obs_only: Columns to include in `obs` only and not `X`.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.diabetes_130_fairlearn()

    """
    _download(
        url="https://figshare.com/ndownloader/files/45110371",
        output_path=DEFAULT_DATA_PATH,
        output_filename="diabetes_130_fairlearn.csv",
        raw_format="csv",
    )
    edata = read_csv(
        filename=f"{DEFAULT_DATA_PATH}/diabetes_130_fairlearn.csv",
        columns_obs_only=columns_obs_only,
    )

    return edata
