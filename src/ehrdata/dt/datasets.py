from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ehrdata.core.constants import DEFAULT_DATA_PATH
from ehrdata.dt._dataloader import _download
from ehrdata.io import from_events, from_pandas, read_csv, read_h5ed, read_meds
from ehrdata.io.omop import setup_connection
from ehrdata.io.omop._queries import _generate_timedeltas

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from duckdb import DuckDBPyConnection

    from ehrdata import EHRData


def ehrdata_blobs(
    *,
    n_variables: int = 11,
    n_cat_vars: int = 0,
    n_categories: list[int] | None = None,
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
    layer: str | None = None,
) -> EHRData:
    """Generates time series example dataset suited for alignment tasks.

    Args:
        n_variables: Dimension of feature space.
        n_cat_vars: Number of categorical variables.
        n_categories: List of cardinalities for each categorical variable.
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
        layer: Name of the layer in the EHRData object that will store the time series data. If not specified, it uses `X`.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.ehrdata_blobs(
        ...     variable_length=True, time_shifts=True, seasonality=True, irregular_sampling=True
        ... )

    results in a dataset like:

    .. image:: /_static/tutorial_images/ehrdata_blobs.png
       :alt: EHR data blobs visualization

    Categorical variables can be generated with different cardinalities per variable
    (e.g. 2, 3, 4 categories). Different clusters (groups) can also exhibit different
    category distributions:

    .. image:: /_static/tutorial_images/ehrdata_blobs_categorical_5_groups.png
       :alt: Histograms of categorical variables by group (5 clusters)

    .. image:: /_static/tutorial_images/ehrdata_blobs_categorical_20_groups.png
       :alt: Histograms of categorical variables by group (20 clusters)
    """
    rng = np.random.default_rng(random_state if isinstance(random_state, int) else None)

    if n_cat_vars > 0:
        if n_cat_vars > n_variables:
            msg = "Number of categorical variables cannot be greater than number of variables."
            raise ValueError(msg)
        if n_categories is None:
            n_categories = rng.integers(2, 4, size=n_cat_vars).tolist()
        if n_categories is not None and len(n_categories) != n_cat_vars:
            msg = f"Length of n_categories ({len(n_categories)}) must match n_cat_vars ({n_cat_vars})"
            raise ValueError(msg)

    n_numeric = n_variables - n_cat_vars

    # Generate cluster centers and assignments
    centers = rng.normal(0, 5, size=(n_centers, n_numeric))
    y = rng.integers(0, n_centers, size=n_observations)

    # Generate base feature values the numeric time series are built from. These are an
    # intermediate of the generator only; the returned object exposes the time series tensor.
    base_values = np.zeros((n_observations, n_variables))
    base_values[:, :n_numeric] = centers[y] + rng.normal(0, cluster_std, size=(n_observations, n_numeric))

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
    tem_layer = np.zeros((n_observations, n_variables, n_total_timepoints))
    tem_layer.fill(np.nan)

    # Generate numeric time series for each observation
    for i in range(n_observations):
        # Map this observation's time points to the global time index
        obs_timepoints = all_timepoints[i]

        # Find indices of these timepoints in the global time array
        time_indices = np.searchsorted(all_unique_times, obs_timepoints)

        # Generate patterns for this observation
        for v in range(n_numeric):
            base_value = base_values[i, v]

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

            tem_layer[i, v, time_indices] = time_series

    # Generate categorical time series if requested
    if n_cat_vars > 0:
        for i in range(n_observations):
            obs_timepoints = all_timepoints[i]
            time_indices = np.searchsorted(all_unique_times, obs_timepoints)
            cluster = y[i]

            for cat_idx in range(n_cat_vars):
                # Variable index in the layer
                v = n_numeric + cat_idx
                cardinality = n_categories[cat_idx]

                # Determine cluster-preferred state for this categorical variable
                preferred_state = cluster % cardinality

                # Smaller cluster standard deviation = higher concentration around that cluster
                concentration = 1.0 / (1.0 + cluster_std)

                # Generate probabilities biased (concentration) toward the preferred state, rest split uniformly
                probs = (
                    np.ones(cardinality) * (1 - concentration) / (cardinality - 1) if cardinality > 1 else np.ones(1)
                )
                probs[preferred_state] = concentration

                # Randomly assign categorical values with cluster bias
                random_states = rng.choice(cardinality, size=len(time_indices), p=probs)

                tem_layer[i, v, time_indices] = random_states.astype(float)

    # Add random missing values if requested
    if missing_values > 0:
        # Create a mask for random missing values (ignoring already missing values)
        missing_mask = rng.random(tem_layer.shape) < missing_values
        not_nan_mask = ~np.isnan(tem_layer)
        tem_layer[missing_mask & not_nan_mask] = np.nan

    if sparse:
        # Handle both NaN and sparsity
        # First replace NaN with 0 where we're keeping values
        mask_r = rng.random(tem_layer.shape) > sparsity
        tem_layer_copy = tem_layer.copy()
        tem_layer_copy[np.isnan(tem_layer)] = 0
        tem_layer_copy[~mask_r] = 0

        # Get coordinates and values for non-zero entries
        coords = np.where(tem_layer_copy != 0)
        values = tem_layer_copy[coords]

        from sparse import COO

        tem_layer = COO(np.asarray(coords), values, shape=tem_layer.shape)

    from ehrdata import EHRData

    obs = pd.DataFrame({"cluster": pd.Categorical(y)}, index=pd.Index([str(i) for i in range(n_observations)]))
    var = pd.DataFrame(index=pd.Index([f"feature_{i}" for i in range(n_variables)]))

    return (
        EHRData(layers={layer: tem_layer}, obs=obs, var=var, tem=t_df)
        if layer is not None
        else EHRData(X=tem_layer, obs=obs, var=var, tem=t_df)
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
        for file_path in (data_path / nested_omop_tables_folder).glob("*.csv"):
            shutil.move(file_path, data_path)

    setup_connection(
        data_path,
        backend_handle,
        prefix=dataset_prefix,
    )


def mimic_iv_omop(backend_handle: DuckDBPyConnection, data_path: Path | None = None) -> None:
    """Loads the MIMIC-IV demo data in the OMOP Common Data model.

    Loads the MIMIC-IV demo dataset from its `physionet repository <https://physionet.org/content/mimic-iv-demo-omop/0.9/#files-panel>`_ :cite:`kallfelz2021mimic` :cite:`goldberger2000physiobank`.

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
        data_path = DEFAULT_DATA_PATH / "mimic-iv-demo-data-in-the-omop-common-data-model-0.9"

    _setup_eunomia_datasets(
        data_url=data_url,
        backend_handle=backend_handle,
        data_path=data_path,
        nested_omop_tables_folder="mimic-iv-demo-data-in-the-omop-common-data-model-0.9/1_omop_data_csv",
        dataset_prefix="2b_",
    )


def mimic_iv_meds(
    data_path: Path | str | None = None,
    *,
    interval_length_number: int = 1,
    interval_length_unit: str = "D",
    num_intervals: int = 14,
    aggregation_strategy: Literal["last", "first", "mean", "median", "min", "max", "sum", "count"] = "last",
    sparse: bool = False,
    layer: str | None = None,
) -> EHRData:
    """Loads the MIMIC-IV demo data in the Medical Event Data Standard (MEDS).

    Loads the 100 patients of the MIMIC-IV Clinical Database Demo in the `MEDS format from physionet <https://physionet.org/content/mimic-iv-demo-meds/0.0.1/>`_ :cite:`vandewater2025mimic` :cite:`johnson2023mimic` :cite:`goldberger2000physiobank`.
    The events, such as laboratory measurements, vital signs, medications, diagnoses, and procedures, are read with :func:`~ehrdata.io.read_meds`.
    The intervals start at the first hospital admission of each patient.
    `obs` holds the `split` of each patient, the time of the first admission in `anchor_time`, `gender` with the categories `"female"` and `"male"`, the `age` at the first admission in years, and `death`, which is 1 if the death of the patient is recorded and 0 otherwise.
    `var` holds the number of binned events of each code in `n_events` and its `description`.
    `tem['time_value']` is the start of every interval in `interval_length_unit`, for instance days since the first admission with the defaults.

    Args:
        data_path: Path to the raw data. If the path exists, the data is loaded from there.
            Else, the data is downloaded.
        interval_length_number: Numeric value of the length of one interval.
        interval_length_unit: Unit belonging to the interval length.
        num_intervals: Number of intervals.
        aggregation_strategy: Aggregation strategy for the numeric values of a code within one interval, as in :func:`~ehrdata.io.from_events`.
        sparse: Whether to store the data as a `sparse.COO` array instead of a :class:`numpy.ndarray`.
        layer: Name of the layer in the EHRData object that will store the time series data. If not specified, it uses `X`.

    Returns:
        The MIMIC-IV demo dataset of shape patients × codes × intervals.
        The raw data is also downloaded, stored and available under the ``data_path``.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_iv_meds()
        >>> edata
        EHRData object with n_obs × n_vars × n_t = 100 × 7033 × 14
            obs: 'split', 'anchor_time', 'gender', 'age', 'death'
            var: 'n_events', 'description'
            tem: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'
            shape of .X: (100, 7033, 14)
    """
    import duckdb

    if data_path is None:
        data_path = DEFAULT_DATA_PATH / "mimic-iv-demo-meds"
    data_path = Path(data_path)

    dataset_name = "mimic-iv-demo-data-in-the-medical-event-data-standard-meds-0.0.1"
    _download(
        f"https://physionet.org/static/published-projects/mimic-iv-demo-meds/{dataset_name}.zip",
        output_path=data_path,
    )
    root = data_path / dataset_name

    edata = read_meds(
        root,
        anchor_code="HOSPITAL_ADMISSION",
        interval_length_number=interval_length_number,
        interval_length_unit=interval_length_unit,
        num_intervals=num_intervals,
        aggregation_strategy=aggregation_strategy,
        sparse=sparse,
        layer=layer,
    )

    # every patient has exactly one of the two static gender codes, which are the only static codes
    del edata.uns["meds_static_codes"]
    is_male = edata.obs.pop("GENDER__M").to_numpy()
    edata.obs = edata.obs.drop(columns="GENDER__F")
    edata.obs["gender"] = pd.Categorical(np.where(is_male, "male", "female"), categories=["female", "male"])

    life_events = (
        duckdb.sql(
            f"SELECT subject_id::VARCHAR AS subject_id, code, MIN(time) AS time FROM read_parquet('{root / 'data' / '**' / '*.parquet'}') "
            "WHERE code IN ('MEDS_BIRTH', 'MEDS_DEATH') GROUP BY ALL"
        )
        .df()
        .pivot(index="subject_id", columns="code", values="time")
        .reindex(edata.obs_names)
    )
    edata.obs["age"] = (
        pd.to_datetime(edata.obs["anchor_time"]) - life_events["MEDS_BIRTH"]
    ).dt.days.to_numpy() / 365.25
    edata.obs["death"] = life_events["MEDS_DEATH"].notna().astype(np.int64).to_numpy()
    _add_time_value(edata, interval_length_number)

    return edata


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
    data_path: Path | str | None = None,
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
    layer: str | None = None,
) -> EHRData:
    """Loads the dataset of the `PhysioNet challenge 2012 (v1.0.0) <https://physionet.org/content/challenge-2012/1.0.0/>`_.

    This dataset was designed to encourage the development of algorithms for mortality rate prediction using physiological data :cite:`silva2012predicting` :cite:`goldberger2000physiobank`.

    If `interval_length_number` is 1, `interval_length_unit` is `"h"` (hour), and `num_intervals` is 48, this is the same as the `SAITS <https://arxiv.org/pdf/2202.08516>`_ preprocessing :cite:`du2023saits`.
    Truncated if a sample has more `num_intervals` steps; Padded if a sample has less than `num_intervals` steps.
    Further, by default the following 12 samples are dropped since they have no time series information at all: 147514, 142731, 145611, 140501, 155655, 143656, 156254, 150309,
    140936, 141264, 150649, 142998.
    Taken the defaults of `interval_length_number`, `interval_length_unit`, `num_intervals`, and `drop_samples`, the tensor stored in `.layers[layer_name]` of `edata` is the same as when doing the `PyPOTS <https://github.com/WenjieDu/PyPOTS>`_ preprocessing :cite:`du2023pypots`.
    A simple deviation is that the tensor in `ehrdata` is of shape `n_obs x n_vars x n_intervals` (with defaults, 3000x37x48) while the tensor in PyPOTS is of shape `n_obs x n_intervals x n_vars` (3000x48x37).
    The tensor stored in `.layers[layer_name]` is hence also fully compatible with the PyPOTS package, as the `.layers` field of EHRData objects generally is.
    In the original dataset, missing values are encoded as -1, for instance for `'Height'`, `'Survival'` (no death recorded), `'DiasABP'`, `'NIDiasABP'`, and `'Weight'`.
    Here, these are missing values (`NaN`) instead.
    `'Gender'` and `'ICUType'` hold the labels of their codes as categories, while the outcome `'In-hospital_death'` stays 0 (survivor) or 1 (died in hospital).
    `tem['time_value']` is the start of every interval in `interval_length_unit`, for instance hours since ICU admission with the defaults.

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
       layer: Name of the layer in the EHRData object that will store the time series data. If not specified, it uses `X`.

    Returns:
        The processed physionet2012 dataset.
        The raw data is also downloaded, stored and available under the ``data_path``.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.physionet_2012()
        >>> edata
        EHRData object with n_obs × n_vars × n_t = 11988 × 37 × 48
            obs: 'set', 'Age', 'Gender', 'Height', 'ICUType', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival', 'In-hospital_death'
            var: 'Parameter'
            tem: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47'
            shape of .X: (11988, 37, 48)

        Inspect static information

        >>> edata.obs.head()
                    set   Age  Gender  Height                   ICUType  SAPS-I  SOFA  Length_of_stay  Survival  In-hospital_death
        RecordID
        132539    set-a  54.0  female     NaN                  surgical     6.0   1.0             5.0       NaN                  0
        132540    set-a  76.0    male   175.3  cardiac surgery recovery    16.0   8.0             8.0       NaN                  0
        132541    set-a  44.0  female     NaN                   medical    21.0  11.0            19.0       NaN                  0
        132543    set-a  68.0    male   180.3                   medical     7.0   1.0             9.0     575.0                  0
        132545    set-a  88.0  female     NaN                   medical    17.0   2.0             4.0     918.0                  0

        Inspect the time axis

        >>> edata.tem.head(3)
                      interval_start_offset interval_end_offset  time_value
        interval_step
        0                   0 days 00:00:00     0 days 01:00:00         0.0
        1                   0 days 01:00:00     0 days 02:00:00         1.0
        2                   0 days 02:00:00     0 days 03:00:00         2.0

        Inspect the 48-hour trajectory of the variable ``RespRate``:

        >>> edata[edata.obs.index == "132539", edata.var_names == "RespRate"].X
        [[[19., 18., 19., 20., 20., 17., nan, 15., 14., 17., 15., 15.,
             12., 15., 15., 12., 14., 13., 18., 13., 12., 20., 15., 24.,
             nan, 16., 19., 18., nan, 16., nan, 18., nan, 18., nan, 20.,
             nan, 24., 21., 16., 18., 14., 23., 17., 20., 20., 20., 23.]]]
    """
    EXPECTED_PARAMETERS = [
        "ALP",
        "ALT",
        "AST",
        "Albumin",
        "BUN",
        "Bilirubin",
        "Cholesterol",
        "Creatinine",
        "DiasABP",
        "FiO2",
        "GCS",
        "Glucose",
        "HCO3",
        "HCT",
        "HR",
        "K",
        "Lactate",
        "MAP",
        "MechVent",
        "Mg",
        "NIDiasABP",
        "NIMAP",
        "NISysABP",
        "Na",
        "PaCO2",
        "PaO2",
        "Platelets",
        "RespRate",
        "SaO2",
        "SysABP",
        "Temp",
        "TroponinI",
        "TroponinT",
        "Urine",
        "WBC",
        "Weight",
        "pH",
    ]
    if data_path is None:
        data_path = DEFAULT_DATA_PATH / "physionet2012"

    elif isinstance(data_path, str):
        data_path = Path(data_path)

    outcome_filenames = ["Outcomes-a.txt", "Outcomes-b.txt", "Outcomes-c.txt"]
    data_set_names = ["set-a", "set-b", "set-c"]

    for filename in data_set_names:
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

    person_outcome_df = pd.concat([pd.read_csv(data_path / filename) for filename in outcome_filenames])

    static_features = ["Age", "Gender", "ICUType", "Height"]

    person_long_across_set_collector = []
    for data_subset_dir in data_set_names:
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

    # gather the static_features together with RecordID and set for each person into the obs table
    obs = (
        person_long_across_set_df[person_long_across_set_df["Parameter"].isin(static_features)]
        .pivot(index=["RecordID", "set"], columns=["Parameter"], values=["Value"])
        .reset_index(level="set", col_level=1)
    )
    obs.columns = obs.columns.droplevel(0)

    obs = obs.merge(person_outcome_df, how="left", left_on="RecordID", right_on="RecordID")
    obs.set_index("RecordID", inplace=True)

    # in order to conveniently save the produced EHRData object: infer to avoid h5ad error b.c. object columns
    obs = obs.infer_objects()
    placeholder_columns = ["Age", "Height", "SAPS-I", "SOFA", "Length_of_stay", "Survival"]
    obs[placeholder_columns] = obs[placeholder_columns].replace(-1, np.nan)
    _label_codes(
        obs,
        {
            "Gender": {0: "female", 1: "male"},
            "ICUType": {1: "coronary care", 2: "cardiac surgery recovery", 3: "medical", 4: "surgical"},
        },
    )

    # consider only time series features from now
    df_dynamic_long = person_long_across_set_df[
        ~person_long_across_set_df["Parameter"].isin(static_features) & (person_long_across_set_df["Value"] != -1)
    ]

    return _create_edata_from_physionet_long_format(
        df_dynamic_long=df_dynamic_long,
        obs=obs,
        expected_parameters=EXPECTED_PARAMETERS,
        interval_length_number=interval_length_number,
        interval_length_unit=interval_length_unit,
        num_intervals=num_intervals,
        aggregation_strategy=aggregation_strategy,
        drop_samples=drop_samples,
        layer=layer,
        dataset="physionet2012",
    )


def physionet2019(
    data_path: Path | str | None = None,
    *,
    interval_length_number: int = 1,
    interval_length_unit: str = "h",
    num_intervals: int = 48,
    aggregation_strategy: str = "last",
    drop_samples: Iterable[str] | None = None,
    n_samples: int | None = None,
    subsample_seed: int | None = 0,
    layer: str | None = None,
) -> EHRData:
    """Loads the dataset of the `PhysioNet challenge 2019 (v1.0.0) <https://physionet.org/content/challenge-2019/1.0.0/>`_.

    This dataset was designed to encourage the development of algorithms for sepsis prediction using physiological data :cite:`reyna2020early` :cite:`goldberger2000physiobank`.

    The data consists of 35 time dependent features and 5 static features (`Age`, `Gender`, `Unit1`, `Unit2`, `HospAdmTime`).
    More information on the features can be found on the link above.
    `'Gender'` holds the labels `'female'` and `'male'` as categories, while the ICU indicators `'Unit1'` (MICU) and `'Unit2'` (SICU) and the `'SepsisLabel'` stay 0 or 1.
    `tem['time_value']` is the start of every interval in `interval_length_unit`, for instance hours since ICU admission with the defaults.

    The full dataset consists of 40'336 patients, with values for the 35 dynamic features recorded hourly, and indicated missing if the value is not available.
    This amounts to a final dataset shape of 40'336 x 35 x number of considered time steps.

    The generated `EHRData` object truincates samples if a sample has more `num_intervals` steps; and pads with missing values if a sample has less than `num_intervals` steps.

    The tensor stored in `.layers[layer_name]` is fully compatible with e.g. the `PyPOTS <https://github.com/WenjieDu/PyPOTS>`_ :cite:`du2023pypots` package, as the `.layers` field of EHRData objects generally is.

    Args:
       data_path: Path to the raw data. If the path exists, the data is loaded from there.
           Else, the data is downloaded. Hint: if you have downloaded the data already from the link above, set this path to the `training` folder.
       interval_length_number: Numeric value of the length of one interval.
       interval_length_unit: Unit belonging to the interval length.
       num_intervals: Number of intervals.
       aggregation_strategy: Aggregation strategy for the time series data when multiple
           measurements for a person's parameter within a time interval is available.
           Available are `'first'` and `'last'`, as used in :meth:`~pandas.DataFrame.drop_duplicates`.
       drop_samples: Samples to drop from the dataset (indicate their RecordID).
       n_samples: Number of samples to subsample from the dataset. If not specified, all samples are used.
       subsample_seed: Seed for the subsampling. If not specified, a random seed is used.
       layer: Name of the layer in the EHRData object that will store the time series data. If not specified, it uses `X`.

    Returns:
        The processed physionet2019 dataset.
        The raw data is also downloaded, stored and available under the ``data_path``.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.physionet_2019()
        >>> edata
        EHRData object with n_obs × n_vars × n_t = 40336 × 35 × 48
            obs: 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'training_Set'
            var: 'Parameter'
            tem: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47'
            shape of .X: (40336, 35, 48)

        Inspect static information

        >>> edata.obs.head()
                    Age  Gender  Unit1  Unit2  HospAdmTime   training_Set
        RecordID
        p000001   83.14  female    NaN    NaN        -0.03  training_setA
        p000002   75.91  female    0.0    1.0       -98.60  training_setA
        p000003   45.82  female    1.0    0.0     -1195.71  training_setA
        p000004   65.71  female    0.0    1.0        -8.77  training_setA
        p000005   28.09    male    1.0    0.0        -0.05  training_setA

        Inspect the 48-hour trajectory of the variable ``SepsisLabel``:

        >>> edata[edata.obs.index == "p020378", edata.var_names == "SepsisLabel"].X
        [[[nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., nan,
             nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]]]
    """
    EXPECTED_PARAMETERS = [
        "AST",
        "Alkalinephos",
        "BUN",
        "BaseExcess",
        "Bilirubin_direct",
        "Bilirubin_total",
        "Calcium",
        "Chloride",
        "Creatinine",
        "DBP",
        "EtCO2",
        "FiO2",
        "Fibrinogen",
        "Glucose",
        "HCO3",
        "HR",
        "Hct",
        "Hgb",
        "Lactate",
        "MAP",
        "Magnesium",
        "O2Sat",
        "PTT",
        "PaCO2",
        "Phosphate",
        "Platelets",
        "Potassium",
        "Resp",
        "SBP",
        "SaO2",
        "SepsisLabel",
        "Temp",
        "TroponinI",
        "WBC",
        "pH",
    ]

    if data_path is None:
        data_path = DEFAULT_DATA_PATH / "physionet2019"

    elif isinstance(data_path, str):
        data_path = Path(data_path)

    _download(
        url="https://exampledata.scverse.org/ehrapy/training.zip",
        output_path=data_path,
        output_filename="training.zip",
        archive_format="zip",
    )

    temp_data_set_names = ["training_setA", "training_setB"]

    static_features = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]

    person_collector_static = {}
    person_collector_dynamic = {}

    for data_subset_dir in temp_data_set_names:
        if not (data_path / "training" / data_subset_dir).exists():
            err = f"Data path {data_path / 'training' / data_subset_dir} does not exist. Please make sure you point `data_path` to the 'training' folder of the downloaded data."
            raise FileNotFoundError(err)

    all_files = []
    for data_subset_dir in temp_data_set_names:
        all_files.extend((data_path / "training" / data_subset_dir).glob("*.psv"))

    all_files = sorted(all_files)

    if drop_samples is not None:
        drop_samples_set = set(drop_samples)
        all_files = [f for f in all_files if f.stem not in drop_samples_set]

    if n_samples is not None:
        if n_samples > len(all_files):
            msg = f"n_samples ({n_samples}) cannot be greater than the available number of samples ({len(all_files)})"
            raise ValueError(msg)
        rng = np.random.default_rng(subsample_seed)
        selected_files = np.sort(rng.choice(len(all_files), size=n_samples, replace=False, shuffle=False)).tolist()
        all_files = np.array(all_files)[selected_files].tolist()

    # Now read only the selected files
    for txt_file in all_files:
        data_subset_dir = txt_file.parent.name

        # each txt file is the data of a person, in wide format
        # the columns in the txt files are both the dynamic columns and the static columns (the latter repating the information for every recorded timepoint)
        person_wide = pd.read_csv(txt_file, sep="|")

        person_static = person_wide[static_features].loc[0]
        person_static["RecordID"] = txt_file.stem
        person_static["training_Set"] = data_subset_dir
        person_collector_static[txt_file.stem] = person_static

        person_dynamic_long = person_wide.iloc[:, ~person_wide.columns.isin(static_features)].melt(
            id_vars=["ICULOS"], var_name="Parameter", value_name="Value"
        )
        person_dynamic_long.dropna(inplace=True)
        person_dynamic_long["training_Set"] = data_subset_dir
        person_dynamic_long["RecordID"] = txt_file.stem

        person_collector_dynamic[txt_file.stem] = person_dynamic_long

    obs = pd.concat(person_collector_static.values(), axis=1).T.set_index("RecordID").infer_objects()
    _label_codes(obs, {"Gender": {0: "female", 1: "male"}})
    df_dynamic_long = pd.concat(person_collector_dynamic.values())

    return _create_edata_from_physionet_long_format(
        df_dynamic_long=df_dynamic_long,
        obs=obs,
        expected_parameters=EXPECTED_PARAMETERS,
        interval_length_number=interval_length_number,
        interval_length_unit=interval_length_unit,
        num_intervals=num_intervals,
        aggregation_strategy=aggregation_strategy,
        drop_samples=drop_samples,
        layer=layer,
        dataset="physionet2019",
    )


def _label_codes(obs: pd.DataFrame, labels: Mapping[str, Mapping[int, str]]) -> None:
    """Replace the documented codes of nominal `obs` columns by their labels, with codes outside of them as missing values."""
    for column, column_labels in labels.items():
        obs[column] = pd.Categorical(obs[column].map(column_labels), categories=list(column_labels.values()))


def _add_time_value(edata: EHRData, interval_length_number: int) -> None:
    """Store the start of every interval in units of the interval length unit in `tem["time_value"]`."""
    edata.tem["time_value"] = np.arange(edata.n_t, dtype=np.float64) * interval_length_number


def _create_edata_from_physionet_long_format(
    df_dynamic_long: pd.DataFrame,
    obs: pd.DataFrame,
    *,
    expected_parameters: Iterable[str],
    interval_length_number: int,
    interval_length_unit: str,
    num_intervals: int,
    aggregation_strategy: str,
    drop_samples: Iterable[str] | None,
    layer: str | None,
    dataset: Literal["physionet2012", "physionet2019"] = "physionet2012",
) -> EHRData:
    """Create an EHRData object from prepared physionet2012 or physionet2019 data parts.

    Creates an EHRData object from:
    1. a long dataframe
    2. a prepared obs dataframe
    3. instructions taken in the physionet2012 and physionet2019 dataset preparation functions
    """
    from ehrdata import EHRData

    interval_df = _generate_timedeltas(
        interval_length_number=interval_length_number,
        interval_length_unit=interval_length_unit,
        num_intervals=num_intervals,
    )

    if dataset == "physionet2012":
        df_long_time_seconds = np.array(pd.to_timedelta(df_dynamic_long["Time"] + ":00").dt.total_seconds())
    elif dataset == "physionet2019":
        df_long_time_seconds = np.array(pd.to_timedelta(df_dynamic_long["ICULOS"], unit="h").dt.total_seconds())
    else:
        msg = f"Dataset {dataset} not supported."
        raise ValueError(msg)

    interval_df_interval_end_offset_seconds = np.array(interval_df["interval_end_offset"].dt.total_seconds())

    # need to throw out entries that are later in time than the observation time
    outside_observation_time_mask = df_long_time_seconds <= interval_df_interval_end_offset_seconds.max()
    df_dynamic_long = df_dynamic_long[outside_observation_time_mask]
    df_long_time_seconds = df_long_time_seconds[outside_observation_time_mask]

    df_long_interval_step = np.argmax(df_long_time_seconds[:, None] <= interval_df_interval_end_offset_seconds, axis=1)
    df_dynamic_long.loc[:, ["interval_step"]] = df_long_interval_step

    # if one person for one feature (=Parameter) within one interval_step has multiple measurements, decide which one to keep
    df_long = df_dynamic_long.drop_duplicates(
        subset=["RecordID", "Parameter", "interval_step"], keep=aggregation_strategy
    )

    xa = df_long.set_index(["RecordID", "Parameter", "interval_step"]).to_xarray()

    # persons whose dynamic measurements all fall outside the observation window are dropped from the long->xarray  pivot;
    # reindex to every person in obs (in obs order) so the layer stays aligned with obs and missing persons are padded with missing values instead of producing a shape mismatch
    xa = xa.reindex(
        RecordID=obs.index.values,
        fill_value=np.nan,
    )
    # since NaNs are dropped, it can happen that a Parameter is completely dropped when it has no values for the subset of persons considered
    # to provide a full set of Parameters everytime, we reindex to add the missing Parameters back in, just with missing values
    xa = xa.reindex(
        Parameter=expected_parameters,
        fill_value=np.nan,
    )
    # to provide a full set of interval_steps everytime, evenif all the samples have less than num_interval steps, we add the missing steps and pad with missing values
    xa = xa.reindex(
        interval_step=np.arange(num_intervals),
        fill_value=np.nan,
    )
    var = xa["Parameter"].to_dataframe()
    tem = interval_df.set_index("interval_step")
    tem_layer = xa["Value"].values

    obs.index = obs.index.astype(str)
    var.index = var.index.astype(str)

    # in order to conveniently save the produced EHRData object: cast problematic types in .obs and .tem
    obs = obs.infer_objects()
    for col in tem.columns:
        tem[col] = tem[col].astype(str)

    edata = (
        EHRData(layers={layer: tem_layer}, obs=obs, var=var, tem=tem)
        if layer is not None
        else EHRData(X=tem_layer, obs=obs, var=var, tem=tem)
    )
    _add_time_value(edata, interval_length_number)

    return edata[~edata.obs.index.isin(drop_samples or [])].copy()


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
        "https://exampledata.scverse.org/ehrapy/full_cohort_data.csv",
        output_path=DEFAULT_DATA_PATH,
        output_filename="ehrapy_mimic2.csv",
    )
    edata = read_csv(
        filename=f"{DEFAULT_DATA_PATH}/ehrapy_mimic2.csv",
        columns_obs_only=columns_obs_only,
    )

    # In the raw dataset, the variable censor_flg is encoded inversely (0=death, 1=censored)
    # We flip it here so it follows the standard convention (0=censored, 1=event happened)

    censor_col = "censor_flg"
    if censor_col in edata.var.index:
        censor_idx = edata.var.index.get_loc(censor_col)
        edata.X[:, censor_idx] = np.where(edata.X[:, censor_idx] == 0, 1, 0)
    elif censor_col in edata.obs.columns:
        edata.obs[censor_col] = np.where(edata.obs[censor_col] == 0, 1, 0)

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
        url="https://exampledata.scverse.org/ehrapy/mimic_2_preprocessed.h5ad",
        output_path=DEFAULT_DATA_PATH,
        output_filename="mimic_2_preprocessed.h5ad",
        raw_format="h5ad",
    )
    edata = read_h5ed(
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
        url="https://exampledata.scverse.org/ehrapy/diabetes_130_raw.csv",
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
        url="https://exampledata.scverse.org/ehrapy/diabetes_130_fairlearn.csv",
        output_path=DEFAULT_DATA_PATH,
        output_filename="diabetes_130_fairlearn.csv",
        raw_format="csv",
    )
    edata = read_csv(
        filename=f"{DEFAULT_DATA_PATH}/diabetes_130_fairlearn.csv",
        columns_obs_only=columns_obs_only,
    )

    return edata


def eicu_crd(
    data_path: Path | str | None = None,
    *,
    interval_length_number: int = 1,
    interval_length_unit: str = "h",
    num_intervals: int = 48,
    aggregation_strategy: Literal["last", "first", "mean", "median", "min", "max"] = "last",
    layer: str | None = None,
) -> EHRData:
    """Loads the demo of the `eICU Collaborative Research Database (v2.0.1) <https://physionet.org/content/eicu-crd-demo/2.0.1/>`_.

    The eICU Collaborative Research Database holds the data of patients admitted to intensive care units (ICUs) across the United States in 2014 and 2015 :cite:`pollard2018eicu` :cite:`goldberger2000physiobank`.
    Its openly available demo holds 2520 ICU stays from 20 hospitals.
    The observations are the ICU stays, identified by `patientunitstayid`, with the stay and patient information of the `patient` table in `obs`.
    The variables are the vital signs of the `vitalPeriodic` and `vitalAperiodic` tables and the laboratory measurements of the `lab` table, with the time since ICU admission.
    Measurements before the ICU admission are not included.

    Ages above 89 years, which are `"> 89"` in the original dataset, are 90.
    `hospital_death` and `unit_death` are 1 if the patient died in the hospital or in the ICU, respectively, and 0 otherwise, instead of the `hospitaldischargestatus` and `unitdischargestatus` with `"Expired"` and `"Alive"`.
    `tem['time_value']` is the start of every interval in `interval_length_unit`, for instance hours since ICU admission with the defaults.

    Args:
        data_path: Path to the raw data. If the path exists, the data is loaded from there.
            Else, the data is downloaded.
        interval_length_number: Numeric value of the length of one interval.
        interval_length_unit: Unit belonging to the interval length.
        num_intervals: Number of intervals.
        aggregation_strategy: Aggregation strategy for the values of a variable within one interval, as in :func:`~ehrdata.io.from_events`.
        layer: Name of the layer in the EHRData object that will store the time series data. If not specified, it uses `X`.

    Returns:
        The eICU demo dataset of shape ICU stays × variables × intervals.
        The raw data is also downloaded, stored and available under the ``data_path``.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.eicu_crd()
        >>> edata.shape
        (2520, 170, 48)
        >>> edata.obs[["gender", "age", "unittype", "hospital_death"]].head(3)
                           gender   age      unittype  hospital_death
        patientunitstayid
        141764             Female  87.0  Med-Surg ICU             0.0
        141765             Female  87.0  Med-Surg ICU             0.0
        143870               Male  76.0          SICU             0.0
    """
    if data_path is None:
        data_path = DEFAULT_DATA_PATH / "eicu-crd-demo"
    data_path = Path(data_path)

    for table in ["patient", "vitalPeriodic", "vitalAperiodic", "lab"]:
        _download(
            f"https://physionet.org/files/eicu-crd-demo/2.0.1/{table}.csv.gz?download",
            output_path=data_path,
            output_filename=f"{table}.csv.gz",
            raw_format="csv",
        )

    obs = pd.read_csv(data_path / "patient.csv.gz", index_col="patientunitstayid")
    obs["age"] = pd.to_numeric(obs["age"].replace("> 89", "90"))
    deaths = {"hospitaldischargestatus": "hospital_death", "unitdischargestatus": "unit_death"}
    for status in deaths:
        obs[status] = obs[status].map({"Alive": 0, "Expired": 1})
    obs = obs.rename(columns=deaths)

    vitals = [
        pd.read_csv(data_path / f"{table}.csv.gz")
        .drop(columns=f"{table.lower()}id")
        .melt(id_vars=["patientunitstayid", "observationoffset"], var_name="code", value_name="numeric_value")
        .rename(columns={"observationoffset": "offset"})
        for table in ["vitalPeriodic", "vitalAperiodic"]
    ]
    lab = pd.read_csv(
        data_path / "lab.csv.gz", usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"]
    ).rename(columns={"labresultoffset": "offset", "labname": "code", "labresult": "numeric_value"})
    events = pd.concat([*vitals, lab], ignore_index=True).dropna(subset=["numeric_value"])
    events["offset"] = pd.to_timedelta(events["offset"], unit="min")

    edata = from_events(
        events,
        obs=obs,
        interval_length_number=interval_length_number,
        interval_length_unit=interval_length_unit,
        num_intervals=num_intervals,
        aggregation_strategy=aggregation_strategy,
        layer=layer,
        subject_id="patientunitstayid",
        time="offset",
    )
    _add_time_value(edata, interval_length_number)

    return edata


def pbcseq(
    data_path: Path | str | None = None,
    *,
    interval_length_number: int = 365,
    interval_length_unit: str = "D",
    num_intervals: int = 15,
    aggregation_strategy: Literal["last", "first", "mean", "median", "min", "max"] = "first",
    layer: str | None = None,
) -> EHRData:
    """Loads the repeated visits of the Mayo Clinic primary biliary cholangitis (PBC) trial.

    The 312 patients of the randomized placebo-controlled trial of D-penicillamine for primary biliary cholangitis (formerly primary biliary cirrhosis) at the Mayo Clinic between 1974 and 1984 were followed up at scheduled visits, at 6 months, 1 year, and then yearly :cite:`murtaugh1994primary`.
    The dataset holds the clinical and laboratory measurements of all 1945 visits, as provided by the `survival R package <https://github.com/therneau/survival>`_ as `pbcseq`.
    The variables are `ascites`, `hepato` (hepatomegaly), `spiders` (spider angiomata), `edema` (0 for no edema, 0.5 for untreated or successfully treated edema, 1 for edema despite diuretic therapy), `bili` (serum bilirubin in mg/dl), `chol` (serum cholesterol in mg/dl), `albumin` (in g/dl), `alk.phos` (alkaline phosphatase in U/liter), `ast` (aspartate aminotransferase in U/ml), `platelet` (platelet count), `protime` (prothrombin time in seconds) and the histologic `stage` of the disease.
    `obs` holds the `age` at enrollment in years, the `sex`, the treatment `trt` with the categories `"D-penicillamine"` and `"placebo"`, the follow-up time `futime` in days, and the `status` at the end of the follow-up with the categories `"censored"`, `"transplant"`, and `"death"`.
    The intervals start at enrollment, and `tem['time_value']` is the start of every interval in `interval_length_unit`, for instance days since enrollment with the defaults.

    Args:
        data_path: Path to the raw data. If the path exists, the data is loaded from there.
            Else, the data is downloaded.
        interval_length_number: Numeric value of the length of one interval.
        interval_length_unit: Unit belonging to the interval length.
        num_intervals: Number of intervals.
        aggregation_strategy: Aggregation strategy for the values of a variable when a patient has multiple visits within one interval, as in :func:`~ehrdata.io.from_events`.
            With the default `"first"` and yearly intervals, the first interval holds the measurements at enrollment.
        layer: Name of the layer in the EHRData object that will store the time series data. If not specified, it uses `X`.

    Returns:
        The PBC dataset of shape patients × variables × intervals.
        The raw data is also downloaded, stored and available under the ``data_path``.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.pbcseq()
        >>> edata
        EHRData object with n_obs × n_vars × n_t = 312 × 12 × 15
            obs: 'futime', 'status', 'trt', 'age', 'sex'
            var: 'n_events'
            tem: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'
            shape of .X: (312, 12, 15)
    """
    if data_path is None:
        data_path = DEFAULT_DATA_PATH / "pbcseq"
    data_path = Path(data_path)

    _download(
        "https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbcseq.csv",
        output_path=data_path,
    )
    visits = pd.read_csv(data_path / "pbcseq.csv")

    obs_columns = ["futime", "status", "trt", "age", "sex"]
    obs = visits.groupby("id")[obs_columns].first()
    _label_codes(
        obs,
        {
            "status": {0: "censored", 1: "transplant", 2: "death"},
            "trt": {1: "D-penicillamine", 2: "placebo"},
            "sex": {"f": "female", "m": "male"},
        },
    )

    variables = [
        "ascites",
        "hepato",
        "spiders",
        "edema",
        "bili",
        "chol",
        "albumin",
        "alk.phos",
        "ast",
        "platelet",
        "protime",
        "stage",
    ]
    events = visits.melt(id_vars=["id", "day"], value_vars=variables, var_name="code", value_name="numeric_value")
    events["day"] = pd.to_timedelta(events["day"], unit="D")
    edata = from_events(
        events.dropna(subset=["numeric_value"]),
        obs=obs,
        codes=variables,
        interval_length_number=interval_length_number,
        interval_length_unit=interval_length_unit,
        num_intervals=num_intervals,
        aggregation_strategy=aggregation_strategy,
        layer=layer,
        subject_id="id",
        time="day",
    )
    _add_time_value(edata, interval_length_number)

    return edata


def heart_failure(
    columns_obs_only: Iterable[str] | None = None,
) -> EHRData:
    """Loads the heart failure clinical records dataset.

    The dataset holds the medical records of 299 patients with heart failure collected at the Faisalabad Institute of Cardiology and at the Allied Hospital in Faisalabad, Pakistan, from April to December 2015 :cite:`ahmad2017survival` :cite:`chicco2020machine`.
    All patients had a left ventricular systolic dysfunction and were in the classes III or IV of the New York Heart Association classification.
    `time` is the follow-up period in days, and `DEATH_EVENT` is 1 if the patient died during the follow-up period and 0 otherwise.
    `sex` holds the categories `"female"` and `"male"`, while `anaemia`, `diabetes`, `high_blood_pressure`, and `smoking` stay 0 or 1.

    More details and the original dataset can be found in the `UCI Machine Learning Repository <https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records>`_.

    Args:
        columns_obs_only: Columns to include in `obs` only and not `X`.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.heart_failure(columns_obs_only=["time", "DEATH_EVENT"])
        >>> edata
        EHRData object with n_obs × n_vars × n_t = 299 × 11 × 1
            obs: 'time', 'DEATH_EVENT'
            shape of .X: (299, 11)
    """
    data_path = DEFAULT_DATA_PATH / "heart_failure"
    _download(
        "https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip",
        output_path=data_path,
        output_filename="heart_failure_clinical_records_dataset.csv.zip",
    )
    df = pd.read_csv(data_path / "heart_failure_clinical_records_dataset.csv")
    _label_codes(df, {"sex": {0: "female", 1: "male"}})

    return from_pandas(df, columns_obs_only=columns_obs_only)


def heart_disease(
    columns_obs_only: Iterable[str] | None = None,
) -> EHRData:
    """Loads the heart disease dataset.

    The dataset holds the 14 commonly used attributes of 920 patients referred for coronary angiography at the Cleveland Clinic Foundation, the Hungarian Institute of Cardiology in Budapest, the Veterans Administration Medical Center in Long Beach, California, and the University Hospitals of Zurich and Basel :cite:`detrano1989international`.
    `site` holds the clinic, with the categories `"cleveland"`, `"hungarian"`, `"long_beach_va"`, and `"switzerland"`.
    The diagnosis `num` is 0 if the diameter of all major vessels is narrowed by less than 50% and 1 to 4 otherwise.
    `sex`, the chest pain type `cp`, the resting electrocardiographic results `restecg`, the slope of the peak exercise ST segment `slope`, and the thallium scintigraphy result `thal` hold the labels of their documented codes as categories.
    The fasting blood sugar above 120 mg/dl `fbs` and the exercise induced angina `exang` stay 0 or 1.
    Missing values, which are `?` in the original dataset, and the physiologically impossible serum cholesterol `chol` and resting blood pressure `trestbps` of 0 are missing values (`NaN`).

    More details and the original dataset can be found in the `UCI Machine Learning Repository <https://archive.ics.uci.edu/dataset/45/heart+disease>`_.

    Args:
        columns_obs_only: Columns to include in `obs` only and not `X`.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.heart_disease(columns_obs_only=["site", "num"])
        >>> edata
        EHRData object with n_obs × n_vars × n_t = 920 × 13 × 1
            obs: 'site', 'num'
            shape of .X: (920, 13)
    """
    data_path = DEFAULT_DATA_PATH / "heart_disease"
    _download(
        "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
        output_path=data_path,
        output_filename="processed.cleveland.data.zip",
    )
    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",
    ]
    sites = {"cleveland": "cleveland", "hungarian": "hungarian", "va": "long_beach_va", "switzerland": "switzerland"}
    df = pd.concat(
        [
            pd.read_csv(data_path / f"processed.{filename}.data", names=columns, na_values="?").assign(site=site)
            for filename, site in sites.items()
        ],
        ignore_index=True,
    )
    df[["chol", "trestbps"]] = df[["chol", "trestbps"]].replace(0, np.nan)
    df["site"] = pd.Categorical(df["site"], categories=list(sites.values()))
    _label_codes(
        df,
        {
            "sex": {0: "female", 1: "male"},
            "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"},
            "restecg": {0: "normal", 1: "ST-T wave abnormality", 2: "left ventricular hypertrophy"},
            "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
            "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
        },
    )
    df.index = df.index.astype(str)

    return from_pandas(df[["site", *columns]], columns_obs_only=columns_obs_only)
