import duckdb
import numpy as np

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


def test_physionet2012():
    edata = ed.dt.physionet2012()
    assert edata.shape == (11988, 38)

    assert edata.r.shape == (11988, 38, 48)
    assert edata.obs.shape == (11988, 10)
    assert edata.var.shape == (38, 1)

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
