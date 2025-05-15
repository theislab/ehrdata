import duckdb
import numpy as np
import pandas as pd
import pytest

import ehrdata as ed


@pytest.fixture(scope="function")
def duckdb_connection():
    """Fixture to create and return a DuckDB connection for testing."""
    con = duckdb.connect()
    yield con
    con.close()


@pytest.mark.slow
def test_mimic_iv_omop(tmp_path):
    duckdb_connection = duckdb.connect()
    ed.dt.mimic_iv_omop(data_path=tmp_path, backend_handle=duckdb_connection)
    assert len(duckdb_connection.execute("SHOW TABLES").df()) == 30
    # sanity check of one table
    assert duckdb_connection.execute("SELECT * FROM person").df().shape == (100, 18)
    duckdb_connection.close()


@pytest.mark.slow
def test_gibleed_omop(tmp_path):
    duckdb_connection = duckdb.connect()
    ed.dt.gibleed_omop(data_path=tmp_path, backend_handle=duckdb_connection)
    assert len(duckdb_connection.execute("SHOW TABLES").df()) == 36
    # sanity check of one table
    assert duckdb_connection.execute("SELECT * FROM person").df().shape == (2694, 18)
    duckdb_connection.close()


@pytest.mark.slow
def test_synthea27nj_omop(tmp_path):
    duckdb_connection = duckdb.connect()
    ed.dt.synthea27nj_omop(data_path=tmp_path, backend_handle=duckdb_connection)
    assert len(duckdb_connection.execute("SHOW TABLES").df()) == 38
    # sanity check of one table
    assert duckdb_connection.execute("SELECT * FROM person").df().shape == (28, 18)
    duckdb_connection.close()


@pytest.mark.slow
def test_physionet2012():
    edata = ed.dt.physionet2012()
    assert edata.shape == (11988, 37)

    assert edata.R.shape == (11988, 37, 48)
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
    assert np.isclose(edata[edata.obs.index.get_loc("152871"), "HR", 0].R.item(), 65)
    assert np.isclose(edata[edata.obs.index.get_loc("152871"), "HR", 28].R.item(), 68)


def test_physionet2012_arguments():
    edata = ed.dt.physionet2012(
        interval_length_number=2,
        interval_length_unit="min",
        num_intervals=24,
        aggregation_strategy="first",
        drop_samples=None,
    )
    assert edata.shape == (12000, 37, 24)

    assert edata.R.shape == (12000, 37, 24)
    assert edata.obs.shape == (12000, 10)
    assert edata.var.shape == (37, 1)


@pytest.mark.parametrize("sparse_param", [False])  # [False, True]
def test_ehrdata_blobs(sparse_param):
    """Test the ehrdata_blobs function."""
    edata = ed.dt.ehrdata_blobs(n_observations=100, n_variables=5, n_timepoints=10, sparse=sparse_param)

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

    # Test R data
    if not sparse_param:
        assert isinstance(edata.R, np.ndarray)
        assert edata.R.shape == (100, 5, 10)
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
    assert isinstance(edata.t, pd.DataFrame)
    assert "timepoint" in edata.t.columns
    assert edata.t.shape == (10, 1)


def test_ehrdata_blobs_distribution():
    edata = ed.dt.ehrdata_blobs(
        n_observations=500, n_variables=10, n_centers=3, n_timepoints=15, cluster_std=0.5, sparse=False, random_state=42
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
        time_slice = edata.R[:, :, t]
        variation = np.std(time_slice)
        time_variations.append(variation)

    # Verify increasing variation trend
    assert time_variations[-1] > time_variations[0]

    # Test that R at t=0 is close to X
    first_timepoint = edata.R[:, :, 0]
    assert np.isclose(first_timepoint, edata.X, rtol=0.1, atol=0.1).mean() > 0.8
