import shutil
from pathlib import Path

import duckdb
import pytest

from ehrdata.dt import gibleed_omop, mimic_iv_omop, synthea27nj_omop

TEST_DATA_DIR = Path("ehrapy_data")


@pytest.fixture(scope="function")
def duckdb_connection():
    """Fixture to create and return a DuckDB connection for testing."""
    con = duckdb.connect()
    yield con
    con.close()


def test_mimic_iv_omop(duckdb_connection):
    """Test loading the GIBleed dataset."""
    test_path = TEST_DATA_DIR / "mimic-iv-demo-data-in-the-omop-common-data-model-0.9"
    mimic_iv_omop(backend_handle=duckdb_connection, data_path=test_path)

    # Verify that tables are created in DuckDB
    tables = duckdb_connection.execute("SHOW TABLES;").fetchall()
    assert len(tables) > 0, f"No tables were loaded into DuckDB for MIMIC-IV dataset. {list(test_path.iterdir())}"


def test_gibleed_omop(duckdb_connection):
    """Test loading the GIBleed dataset."""
    test_path = TEST_DATA_DIR / "GIBleed_dataset"
    gibleed_omop(backend_handle=duckdb_connection, data_path=test_path)

    # Verify that tables are created in DuckDB
    tables = duckdb_connection.execute("SHOW TABLES;").fetchall()
    assert len(tables) > 0, f"No tables were loaded into DuckDB for GIBleed dataset. {list(test_path.iterdir())}"


def test_synthea27nj_omop(duckdb_connection):
    """Test loading the Synthe27Nj dataset."""
    test_path = TEST_DATA_DIR / "Synthea27Nj"
    synthea27nj_omop(backend_handle=duckdb_connection, data_path=test_path)

    # Verify that tables are created in DuckDB
    tables = duckdb_connection.execute("SHOW TABLES;").fetchall()
    assert len(tables) > 0, f"No tables were loaded into DuckDB for Synthea27Nj dataset. {list(test_path.iterdir())}"


# @pytest.fixture(scope="session", autouse=True)
# def cleanup_test_data():
#     """Fixture to clean up test data directory after tests."""
#     yield
#     if TEST_DATA_DIR.exists():
#         shutil.rmtree(TEST_DATA_DIR)


if __name__ == "__main__":
    pytest.main()
