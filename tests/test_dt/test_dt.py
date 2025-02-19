import duckdb
import numpy as np
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

    assert edata.r.shape == (11988, 37, 48)
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
    assert np.isclose(edata[edata.obs.index.get_loc("152871"), "HR", 0].r.item(), 65)
    assert np.isclose(edata[edata.obs.index.get_loc("152871"), "HR", 28].r.item(), 68)


def test_physionet2012_arguments():
    edata = ed.dt.physionet2012(
        interval_length_number=2,
        interval_length_unit="min",
        num_intervals=24,
        aggregation_strategy="first",
        drop_samples=None,
    )
    assert edata.shape == (12000, 37)

    assert edata.r.shape == (12000, 37, 24)
    assert edata.obs.shape == (12000, 10)
    assert edata.var.shape == (37, 1)
