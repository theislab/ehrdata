from pathlib import Path

import duckdb
import pytest

import ehrdata as ed

TEST_DATA_DIR = Path(__file__).parent / "ehrapy_data"


@pytest.fixture(scope="function")
def duckdb_connection():
    """Fixture to create and return a DuckDB connection for testing."""
    con = duckdb.connect()
    yield con
    con.close()


def test_mimic_iv_omop(duckdb_connection):
    ed.dt.mimic_iv_omop(backend_handle=duckdb_connection)
    assert len(duckdb_connection.execute("SHOW TABLES").df()) == 30
    # sanity check of one table
    assert duckdb_connection.execute("SELECT * FROM person").df().shape == (100, 18)


def test_gibleed_omop(duckdb_connection):
    ed.dt.gibleed_omop(backend_handle=duckdb_connection)
    assert len(duckdb_connection.execute("SHOW TABLES").df()) == 36
    # sanity check of one table
    assert duckdb_connection.execute("SELECT * FROM person").df().shape == (2694, 18)


def test_synthea27nj_omop(duckdb_connection):
    ed.dt.synthea27nj_omop(backend_handle=duckdb_connection)
    assert len(duckdb_connection.execute("SHOW TABLES").df()) == 37
    # sanity check of one table
    assert duckdb_connection.execute("SELECT * FROM person").df().shape == (28, 18)
