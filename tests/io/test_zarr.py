from tests.conftest import TEST_DATA_PATH

from ehrdata.io import read_zarr

_TEST_PATH_H5AD = f"{TEST_DATA_PATH}/toy_zarr/"


def test_read_zarr():
    edata = read_zarr(file_name=f"{_TEST_PATH_H5AD}/zarr_numeric_basic.zarr")

    assert edata.shape == (4, 3, 0)
    assert edata.X.shape == (4, 3)
    assert set(edata.var_names) == {"col" + str(i) for i in range(1, 4)}
    assert set(edata.obs.columns) == set()


def test_write_zarr():
    import ehrdata as ed

    edata = ed.io.read_csv(f"{TEST_DATA_PATH}/toy_csv/csv_basic.csv", columns_obs_only=["survival"])
    edata = ed.io.read_csv(f"{TEST_DATA_PATH}/toy_csv/csv_basic.csv")

    ed.io.write_zarr(edata, f"{TEST_DATA_PATH}/toy_zarr/zarr_numeric_basic.zarr")
    assert True


# TODO: make this test run - how can dtype object be handled?
# @pytest.mark.parametrize("edata_name", ["edata_330", "edata_nonnumeric_missing_330", "edata_333"])
# def test_write_h5ad_basic(edata_name, request, tmp_path):
#     edata = request.getfixturevalue(edata_name)
#     write_h5ad(edata, tmp_path / f"test_write_{edata_name}.h5ad")
#     assert Path.exists(tmp_path / f"test_write_{edata_name}.h5ad")


# TODO: make this test run - how can dtype object be handled?
# @pytest.mark.parametrize("edata_name", ["edata_330", "edata_nonnumeric_missing_330", "edata_333"])
# def test_write_read_h5ad_basic(edata_name, request, tmp_path):
#     edata = request.getfixturevalue(edata_name)
#     write_h5ad(edata, tmp_path / f"test_write_{edata_name}.h5ad")

#     edata_read = read_h5ad(tmp_path / f"test_write_{edata_name}.h5ad")

#     assert edata_read.shape == edata.shape

#     assert np.array_equal(edata_read.X, edata.X)
#     pd.testing.assert_frame_equal(edata_read.obs, edata.obs)
#     pd.testing.assert_frame_equal(edata_read.var, edata.var)
#     if edata.R is None:
#         assert edata_read.R is None
#     else:
#         assert edata_read.R is not None
#         assert np.array_equal(edata.R, edata_read.R)
