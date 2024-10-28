import duckdb
import pytest

from ehrdata.io.omop import register_omop_to_db_connection


@pytest.fixture  # (scope="session")
def omop_connection_vanilla():
    con = duckdb.connect()
    register_omop_to_db_connection(path="tests/data/toy_omop/vanilla", backend_handle=con, source="csv")
    yield con
    con.close()
