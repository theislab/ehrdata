import duckdb

import ehrdata as ed


def test_mimic_iv_omop():
    con = duckdb.connect()
    ed.dt.mimic_iv_omop(backend_handle=con)
    assert len(con.execute("SHOW TABLES").df()) == 30
    con.close()


# TODO
# def test_gibleed_omop():
#     con = duckdb.connect()
#     ed.dt.gibleed_omop(backend_handle=con)
#     assert len(con.execute("SHOW TABLES").df()) == 36
#     con.close()


# def test_synthea27nj_omop():
#     con = duckdb.connect()
#     ed.dt.synthea27nj_omop(backend_handle=con)
#     assert len(con.execute("SHOW TABLES").df()) == 37
#     con.close()
