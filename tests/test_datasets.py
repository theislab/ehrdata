import shutil
from pathlib import Path
import duckdb
import pytest
from ehrdata.dt import gibleed_omop, synthea27nj_omop, mimic_iv_omop

TEST_DATA_DIR = Path(__file__).parent / "ehrapy_data"

@pytest.fixture(scope="function")
def duckdb_connection():
    """Fixture to create and return a DuckDB connection for testing."""
    con = duckdb.connect()
    yield con
    con.close()



def test_gibleed_omop(duckdb_connection, tmp_path):
    """Test loading the GIBleed dataset."""
    test_path = tmp_path / "gibleed_omop_test"
    gibleed_omop(backend_handle=duckdb_connection, data_path=test_path)

    # Verify that tables are created in DuckDB
    tables = duckdb_connection.execute("SHOW TABLES;").fetchall()
    assert len(tables) > 0, f"No tables were loaded into DuckDB for GIBleed dataset. {list(test_path.iterdir())}"


def test_synthea27nj_omop(duckdb_connection, tmp_path):
    """Test loading the Synthe27Nj dataset."""
    test_path = tmp_path / "synthea27nj_test"
    synthea27nj_omop(backend_handle=duckdb_connection, data_path=test_path)

    # Verify that tables are created in DuckDB
    tables = duckdb_connection.execute("SHOW TABLES;").fetchall()
    assert len(tables) > 0, f"No tables were loaded into DuckDB for Synthea27Nj dataset. {list(test_path.iterdir())}"


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_data():
    """Fixture to clean up test data directory after tests."""
    yield
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)


if __name__ == "__main__":
    pytest.main()
