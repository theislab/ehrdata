import duckdb
import numpy as np
import pandas as pd
import pytest

import ehrdata as ed
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME


def test_mimic_2():
    edata = ed.dt.mimic_2()
    assert edata.shape == (1776, 46, 1)
    expected_first_two_vars = ["aline_flg", "icu_los_day"]
    assert list(edata.var.index.values[:2]) == expected_first_two_vars
    expected_first_four_X = np.array([[1, 7.63], [0, 1.14]])
    assert np.array_equal(edata.X[:2, :2], expected_first_four_X)


def test_mimic_2_preprocessed():
    edata = ed.dt.mimic_2_preprocessed()
    assert edata.shape == (1776, 46, 1)
    expected_first_two_vars = ["ehrapycat_service_unit", "ehrapycat_day_icu_intime"]
    assert list(edata.var.index.values[:2]) == expected_first_two_vars
    expected_first_four_X = np.array([[2.0, 0.0], [1.0, 2.0]])
    assert np.array_equal(edata.X[:2, :2], expected_first_four_X)


def test_diabetes_130_raw():
    edata = ed.dt.diabetes_130_raw()
    assert edata.shape == (101766, 50, 1)
    expected_first_two_vars = ["encounter_id", "patient_nbr"]
    assert list(edata.var.index.values[:2]) == expected_first_two_vars
    expected_first_four_X = np.array([[2278392, 8222157], [149190, 55629189]])
    assert np.array_equal(edata.X[:2, :2], expected_first_four_X)


def test_diabetes_130_fairlearn():
    edata = ed.dt.diabetes_130_fairlearn()
    assert edata.shape == (101766, 24, 1)
    expected_first_two_vars = ["race", "gender"]
    assert list(edata.var.index.values[:2]) == expected_first_two_vars
    expected_first_four_X = np.array([["Caucasian", "Female"], ["Caucasian", "Female"]])
    assert np.array_equal(edata.X[:2, :2], expected_first_four_X)


@pytest.fixture
def duckdb_connection():
    """Fixture to create and return a DuckDB connection for testing."""
    con = duckdb.connect()
    yield con
    con.close()


def test_mimic_iv_omop():
    duckdb_connection = duckdb.connect()
    ed.dt.mimic_iv_omop(backend_handle=duckdb_connection)
    assert len(duckdb_connection.execute("SHOW TABLES").df()) == 30
    # sanity check of one table
    assert duckdb_connection.execute("SELECT * FROM person").df().shape == (100, 18)
    duckdb_connection.close()


def test_gibleed_omop():
    duckdb_connection = duckdb.connect()
    ed.dt.gibleed_omop(backend_handle=duckdb_connection)
    assert len(duckdb_connection.execute("SHOW TABLES").df()) == 36
    # sanity check of one table
    assert duckdb_connection.execute("SELECT * FROM person").df().shape == (2694, 18)
    duckdb_connection.close()


def test_synthea27nj_omop():
    duckdb_connection = duckdb.connect()
    ed.dt.synthea27nj_omop(backend_handle=duckdb_connection)
    assert len(duckdb_connection.execute("SHOW TABLES").df()) == 38
    # sanity check of one table
    assert duckdb_connection.execute("SELECT * FROM person").df().shape == (28, 18)
    duckdb_connection.close()


def test_physionet2012():
    edata = ed.dt.physionet2012(layer=DEFAULT_TEM_LAYER_NAME)
    assert edata.shape == (11988, 37, 48)
    assert edata.tem.shape == (48, 2)
    assert edata.layers[DEFAULT_TEM_LAYER_NAME].shape == (11988, 37, 48)
    assert edata.obs.shape == (11988, 10)
    assert edata.var.shape == (37, 1)

    # check a few hand-picked values for a stricter test
    # first entry set a
    assert edata.obs.loc["132539"].values[0] == "set-a"
    assert np.allclose(
        edata.obs.loc["132539"].values[1:].astype(np.float32),
        np.array([54.0, 0.0, -1.0, 4.0, 6, 1, 5, -1, 0], dtype=np.float32),
    )

    # first entry set b
    assert edata.obs.loc["142675"].values[0] == "set-b"
    assert np.allclose(
        edata.obs.loc["142675"].values[1:].astype(np.float32),
        np.array([70.0, 1.0, 175.3, 2.0, 27, 14, 9, 7, 1], dtype=np.float32),
    )

    # first entry set c
    assert edata.obs.loc["152871"].values[0] == "set-c"
    assert np.allclose(
        edata.obs.loc["152871"].values[1:].astype(np.float32),
        np.array([71.0, 1.0, 167.6, 4.0, 19, 10, 23, -1, 0], dtype=np.float32),
    )

    # first entry c two different HR value
    assert np.isclose(edata[edata.obs.index.get_loc("152871"), "HR", 0].layers[DEFAULT_TEM_LAYER_NAME].item(), 65)
    assert np.isclose(edata[edata.obs.index.get_loc("152871"), "HR", 28].layers[DEFAULT_TEM_LAYER_NAME].item(), 68)


def test_physionet2012_arguments():
    edata = ed.dt.physionet2012(
        layer=DEFAULT_TEM_LAYER_NAME,
        interval_length_number=2,
        interval_length_unit="min",
        num_intervals=24,
        aggregation_strategy="first",
        drop_samples=None,
    )
    assert edata.shape == (12000, 37, 24)
    assert edata.tem.shape == (24, 2)
    assert edata.layers[DEFAULT_TEM_LAYER_NAME].shape == (12000, 37, 24)
    assert edata.obs.shape == (12000, 10)
    assert edata.var.shape == (37, 1)


def test_physionet2019():
    edata = ed.dt.physionet2019(layer=DEFAULT_TEM_LAYER_NAME, n_samples=10)
    assert edata.shape == (10, 35, 48)
    assert edata.tem.shape == (48, 2)
    assert edata.layers[DEFAULT_TEM_LAYER_NAME].shape == (10, 35, 48)
    assert edata.obs.shape == (10, 6)
    assert edata.var.shape == (35, 1)

    # check a few hand-picked values for a stricter test
    # a randomly picked entry, for which verified the values in the source data
    assert edata.obs.loc["p000667", "training_Set"] == "training_setA"
    np.allclose(
        edata.obs.loc["p000667", ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]].values,
        np.array([19.96, 0.0, 1.0, 0.0, -0.02]),
    )

    np.allclose(
        edata[edata.obs.index.get_loc("p000667"), "MAP", :5].layers["tem_data"],
        np.array([np.nan, np.nan, np.nan, 75.67, 87.33]),
        equal_nan=True,
    )


def test_physionet2019_arguments():
    edata = ed.dt.physionet2019(
        layer=DEFAULT_TEM_LAYER_NAME,
        interval_length_number=2,
        num_intervals=24,
        aggregation_strategy="first",
        drop_samples=None,
        n_samples=2,
    )
    assert edata.shape == (2, 35, 24)
    assert edata.tem.shape == (24, 2)
    assert edata.layers[DEFAULT_TEM_LAYER_NAME].shape == (2, 35, 24)
    assert edata.obs.shape == (2, 6)
    assert edata.var.shape == (35, 1)


@pytest.mark.parametrize("sparse_param", [False])  # [False, True]
def test_ehrdata_blobs(sparse_param):
    """Test the ehrdata_blobs function."""
    edata = ed.dt.ehrdata_blobs(
        layer=DEFAULT_TEM_LAYER_NAME,
        n_observations=100,
        n_variables=5,
        base_timepoints=10,
        n_cat_vars=2,
        sparse=sparse_param,
    )

    assert isinstance(edata, ed.EHRData)

    assert edata.shape == (100, 5, 10)
    assert edata.n_obs == 100
    assert edata.n_vars == 5
    assert edata.n_t == 10

    # Test X data
    if not sparse_param:
        assert isinstance(edata.X, np.ndarray)
        assert edata.X.shape == (100, 5)
    # else:
    #     assert sparse.issparse(ehr_data.X)
    #     assert ehr_data.X.shape == (100, 5)

    # Test 3D data
    if not sparse_param:
        assert isinstance(edata.layers[DEFAULT_TEM_LAYER_NAME], np.ndarray)
        assert edata.layers[DEFAULT_TEM_LAYER_NAME].shape == (100, 5, 10)
    # else:
    #     from sparse import COO
    #     assert isinstance(ehr_data.R, COO)
    #     assert ehr_data.R.shape == (100, 5, 10)

    # Test obs DataFrame
    assert isinstance(edata.obs, pd.DataFrame)
    assert "cluster" in edata.obs.columns
    assert edata.obs.shape == (100, 1)

    # Test var DataFrame
    assert isinstance(edata.var, pd.DataFrame)
    assert edata.var.shape == (5, 0)

    # Test t DataFrame
    assert isinstance(edata.tem, pd.DataFrame)
    assert "timepoint" in edata.tem.columns
    assert edata.tem.shape == (10, 2)


def test_ehrdata_blobs_categories():
    with pytest.raises(
        ValueError, match=r"Number of categorical variables cannot be greater than number of variables."
    ):
        ed.dt.ehrdata_blobs(n_variables=5, n_cat_vars=6)
    with pytest.raises(ValueError, match=r"Length of n_categories .* must match n_cat_vars"):
        ed.dt.ehrdata_blobs(n_variables=5, n_cat_vars=2, n_categories=[2, 3, 4])

    edata = ed.dt.ehrdata_blobs(
        n_observations=100,
        n_variables=10,
        n_cat_vars=2,
        n_categories=[3, 2],
        base_timepoints=10,
        sparse=False,
        random_state=42,
    )

    n_numeric = 10 - 2

    for cat_idx, k in enumerate([3, 2]):
        v = n_numeric + cat_idx
        vals = edata.layers[DEFAULT_TEM_LAYER_NAME][:, v, :].ravel()
        vals = vals[~np.isnan(vals)]
        assert vals.size > 0
        assert vals.min() >= 0
        assert vals.max() <= (k - 1)
        assert np.allclose(vals, np.round(vals))

    for i in range(100):
        valid_times = ~np.isnan(edata.layers[DEFAULT_TEM_LAYER_NAME][i, 0, :])
        valid_idx = np.where(valid_times)[0]
        assert valid_idx.size > 0
        mid_idx = valid_idx[len(valid_idx) // 2]
        for cat_idx in range(2):
            v = n_numeric + cat_idx
            assert edata.X[i, v] == edata.layers[DEFAULT_TEM_LAYER_NAME][i, v, mid_idx]

    assert isinstance(edata, ed.EHRData)
    assert edata.shape == (100, 10, 10)

    assert isinstance(edata.X, np.ndarray)
    assert edata.X.shape == (100, 10)

    assert isinstance(edata.layers[DEFAULT_TEM_LAYER_NAME], np.ndarray)
    assert edata.layers[DEFAULT_TEM_LAYER_NAME].shape == (100, 10, 10)


def test_ehrdata_blobs_distribution():
    edata = ed.dt.ehrdata_blobs(
        n_observations=500,
        n_variables=10,
        n_centers=3,
        base_timepoints=15,
        cluster_std=0.5,
        sparse=False,
        random_state=42,
    )

    assert isinstance(edata, ed.EHRData)
    assert edata.shape == (500, 10, 15)

    clusters = edata.obs["cluster"].astype(int).values

    # Test cluster separation in X
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        cluster_points = edata.X[cluster_mask]

        cluster_center = np.mean(cluster_points, axis=0)

        # Calculate average distance to center
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        avg_distance = np.mean(distances)

        # Check that points are reasonably clustered (average distance should be near cluster_std)
        expected_distance = 0.5 * np.sqrt(10)  # cluster_std * sqrt(dimensions)
        assert 0.3 * expected_distance < avg_distance < 3.0 * expected_distance

    # Test time evolution in R
    # Check that variation increases with time
    time_variations = []
    for t in range(edata.n_t):
        time_slice = edata.layers[DEFAULT_TEM_LAYER_NAME][:, :, t]
        variation = np.std(time_slice)
        time_variations.append(variation)

    # Verify increasing variation trend
    assert time_variations[-1] > time_variations[0]

    # Test that 3D data at t=0 is close to X
    first_timepoint = edata.layers[DEFAULT_TEM_LAYER_NAME][:, :, 0]
    correlation = np.corrcoef(first_timepoint.flatten(), edata.X.flatten())[0, 1]
    assert correlation > 0.5


def test_ehrdata_ts_blobs_irregular():
    edata = ed.dt.ehrdata_blobs(
        n_observations=300,
        n_variables=8,
        n_centers=4,
        base_timepoints=20,
        cluster_std=0.5,
        sparse=False,
        variable_length=True,
        time_shifts=True,
        seasonality=True,
        irregular_sampling=True,
        missing_values=0.1,
        random_state=42,
    )

    assert isinstance(edata, ed.EHRData)
    assert edata.n_obs == 300
    assert edata.n_vars == 8

    # Check that time dimension exists and has more points than base_timepoints
    assert edata.n_t > 20
    assert "time_value" in edata.tem.columns

    # Test for irregular time sampling
    time_values = edata.tem["time_value"].values
    time_diffs = np.diff(time_values)
    # Time differences should have meaningful variation with irregular sampling
    assert np.std(time_diffs) > 0.001

    # Test for missing values in 3D data
    nan_count = np.isnan(edata.layers[DEFAULT_TEM_LAYER_NAME]).sum()
    total_elements = np.prod(edata.layers[DEFAULT_TEM_LAYER_NAME].shape)
    missing_ratio = nan_count / total_elements
    assert missing_ratio > 0.05

    # Test for variable length time series
    valid_counts = []
    for i in range(edata.n_obs):
        valid_count = np.sum(~np.isnan(edata.layers[DEFAULT_TEM_LAYER_NAME][i, 0, :]))
        valid_counts.append(valid_count)

    valid_counts = np.array(valid_counts)
    # Check for variation in time series lengths
    assert np.std(valid_counts) > 0.5
    assert np.max(valid_counts) > np.min(valid_counts)

    # Test for time shifts
    clusters = edata.obs["cluster"].astype(int).values

    shift_detected = False
    for cluster in np.unique(clusters):
        cluster_mask = clusters == cluster
        obs_indices = np.where(cluster_mask)[0]

        if len(obs_indices) >= 2:
            obs1 = obs_indices[0]
            obs2 = obs_indices[1]

            valid_times1 = np.where(~np.isnan(edata.layers[DEFAULT_TEM_LAYER_NAME][obs1, 0, :]))[0]
            valid_times2 = np.where(~np.isnan(edata.layers[DEFAULT_TEM_LAYER_NAME][obs2, 0, :]))[0]

            if len(valid_times1) > 0 and len(valid_times2) > 0:
                first_time1 = valid_times1[0]
                first_time2 = valid_times2[0]

                if abs(first_time1 - first_time2) > 0:
                    shift_detected = True
                    break

    assert shift_detected

    # Test for seasonality
    seasonal_pattern_found = False
    for i in range(min(10, edata.n_obs)):  # Check first 10 observations max
        for v in range(edata.n_vars):
            values = edata.layers[DEFAULT_TEM_LAYER_NAME][i, v, :]
            valid_mask = ~np.isnan(values)

            if np.sum(valid_mask) >= 10:  # Need at least 10 valid points
                time_series = values[valid_mask]

                # Look for patterns by checking if the time series is non-monotonic
                # True seasonality will have ups and downs
                diffs = np.diff(time_series)
                sign_changes = np.sum((diffs[:-1] * diffs[1:]) < 0)

                # If we have multiple sign changes, there's likely a pattern
                if sign_changes >= 3:
                    seasonal_pattern_found = True
                    break
        if seasonal_pattern_found:
            break

    assert seasonal_pattern_found
