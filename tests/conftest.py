import duckdb
import pytest

from ehrdata.io.omop import setup_connection


@pytest.fixture  # (scope="session")
def omop_connection_vanilla():
    con = duckdb.connect()
    setup_connection(path="tests/data/toy_omop/vanilla", backend_handle=con)
    yield con
    con.close()


@pytest.fixture  # (scope="session")
def omop_connection_capital_letters():
    con = duckdb.connect()
    setup_connection(path="tests/data/toy_omop/capital_letters", backend_handle=con)
    yield con
    con.close()


@pytest.fixture  # (scope="session")
def omop_connection_empty_observation():
    con = duckdb.connect()
    setup_connection(path="tests/data/toy_omop/empty_observation", backend_handle=con)
    yield con
    con.close()
