import shutil
from pathlib import Path

import duckdb
import pytest

# Assuming the dataset loading functions are in the ehrdata.dt module
from ehrdata.dt import gibleed_omop, synthea27nj_omop

TEST_DATA_DIR = Path("ehrapy_data")


@pytest.fixture(scope="function")
def duckdb_connection():
    """Fixture to create and return a DuckDB connection for testing."""
    con = duckdb.connect()
    yield con
    con.close()


def test_gibleed_omop(duckdb_connection):
    """Test loading the GIBleed dataset."""
    test_path = TEST_DATA_DIR / "gibleed_omop_test"
    gibleed_omop(backend_handle=duckdb_connection, data_path=test_path)

    # Verify that tables are created in DuckDB
    tables = duckdb_connection.execute("SHOW TABLES;").fetchall()
    assert len(tables) > 0, "No tables were loaded into DuckDB for GIBleed dataset."


def test_synthea27nj_omop(duckdb_connection):
    """Test loading the Synthe27Nj dataset."""
    test_path = TEST_DATA_DIR / "synthea27nj_test"
    synthea27nj_omop(backend_handle=duckdb_connection, data_path=test_path)

    # Verify that tables are created in DuckDB
    tables = duckdb_connection.execute("SHOW TABLES;").fetchall()
    assert len(tables) > 0, "No tables were loaded into DuckDB for Synthea27Nj dataset."


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_data():
    """Fixture to clean up test data directory after tests."""
    yield
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)


if __name__ == "__main__":
    pytest.main()
