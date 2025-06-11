from tests.conftest import TEST_DATA_PATH

from ehrdata.io import read_zarr

_TEST_PATH_H5AD = f"{TEST_DATA_PATH}/toy_zarr/"


def test_read_zarr_basic():
    edata = read_zarr(file_name=f"{_TEST_PATH_H5AD}/adata_basic.zarr")

    assert edata.shape == (5, 4, 0)
    assert edata.X.shape == (5, 4)
    assert "survival" in edata.obs.columns
    assert all(edata.obs["survival"].values == [1, 2, 3, 4, 5])
    assert "variables" in edata.var.columns
    assert all(edata.var["variables"].values == ["var_1", "var_2", "var_3", "var_4"])
    assert "obs_level_representation" in edata.obsm
    assert edata.obsm["obs_level_representation"].shape == (5, 2)
    assert "var_level_representation" in edata.varm
    assert edata.varm["var_level_representation"].shape == (4, 2)
    # shapes are enforced by AnnData/EHRData for the below, no need to test
    assert "other_layer" in edata.layers
    assert "obs_level_connectivities" in edata.obsp
    assert "var_level_connectivities" in edata.varp
    assert "information" in edata.uns


def test_read_zarr_basic_with_tem():
    edata = read_zarr(file_name=f"{_TEST_PATH_H5AD}/edata_basic_with_tem.zarr")

    assert edata.shape == (5, 4, 2)
    assert edata.X.shape == (5, 4)
    assert edata.R.shape == (5, 4, 2)
    assert "survival" in edata.obs.columns
    assert all(edata.obs["survival"].values == [1, 2, 3, 4, 5])
    assert "variables" in edata.var.columns
    assert all(edata.var["variables"].values == ["var_1", "var_2", "var_3", "var_4"])
    assert "timestep" in edata.tem.columns
    assert all(edata.tem["timestep"].values == ["t1", "t2"])
    assert "obs_level_representation" in edata.obsm
    assert edata.obsm["obs_level_representation"].shape == (5, 2)
    assert "var_level_representation" in edata.varm
    assert edata.varm["var_level_representation"].shape == (4, 2)
    # shapes are enforced by AnnData/EHRData for the below, no need to test
    assert "other_layer" in edata.layers
    assert "obs_level_connectivities" in edata.obsp
    assert "var_level_connectivities" in edata.varp
    assert "information" in edata.uns


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
