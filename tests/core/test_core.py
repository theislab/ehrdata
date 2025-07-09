import anndata as ad
import numpy as np
import pandas as pd
import pytest
from tests.conftest import _assert_shape_matches

from ehrdata import EHRData


def _assert_fields_are_view(edata: EHRData):
    assert edata.is_view
    assert isinstance(edata.X, ad._core.views.ArrayView)
    assert isinstance(edata.tem, ad._core.views.DataFrameView)


#################################################################
### Test combinations of X, layers, t during initialization
#################################################################
def test_ehrdata_init_vanilla_empty():
    edata = EHRData()

    _assert_shape_matches(edata, (0, 0, 1), check_X_None=True)
    assert edata.X is None
    assert edata.obs.shape == (0, 0)
    assert edata.var.shape == (0, 0)
    assert edata.tem.shape == (1, 0)


def test_ehrdata_init_vanilla_X(X_numpy_32):
    edata = EHRData(X=X_numpy_32)
    _assert_shape_matches(edata, (3, 2, 1))

    assert edata.obs.shape == (3, 0)

    assert edata.var.shape == (2, 0)

    assert edata.tem.shape == (1, 0)


def test_ehrdata_init_vanilla_3dlayer(X_numpy_322):
    edata = EHRData(layers={"tem_layer": X_numpy_322})
    _assert_shape_matches(edata, (3, 2, 2), check_X_None=True)

    assert edata.layers is not None
    assert edata.layers["tem_layer"] is not None
    assert edata.layers["tem_layer"].shape == (3, 2, 2)

    assert edata.obs.shape == (3, 0)

    assert edata.var.shape == (2, 0)

    assert edata.tem.shape == (2, 0)


def test_ehrdata_init_vanilla_X_and_3dlayer(X_numpy_32, X_numpy_322):
    edata = EHRData(X=X_numpy_32, layers={"tem_layer": X_numpy_322})
    _assert_shape_matches(edata, (3, 2, 2))

    assert edata.layers is not None
    assert edata.layers["tem_layer"] is not None

    assert edata.X.shape == (3, 2)
    assert edata.obs.shape == (3, 0)
    assert edata.var.shape == (2, 0)
    assert edata.tem.shape == (2, 0)
    assert edata.layers["tem_layer"].shape == (3, 2, 2)


def test_ehrdata_init_vanilla_X_and_t(X_numpy_32, tem_21):
    edata = EHRData(X=X_numpy_32, tem=tem_21)
    _assert_shape_matches(edata, (3, 2, 2))

    assert edata.layers is not None

    assert edata.X.shape == (3, 2)
    assert edata.obs.shape == (3, 0)
    assert edata.var.shape == (2, 0)
    assert edata.tem.shape == (2, 1)


def test_ehrdata_init_vanilla_X_and_3dlayer_and_t(X_numpy_32, X_numpy_322, tem_21):
    edata = EHRData(X=X_numpy_32, layers={"tem_layer": X_numpy_322}, tem=tem_21)
    _assert_shape_matches(edata, (3, 2, 2))

    assert edata.layers is not None
    assert edata.layers["tem_layer"] is not None

    assert edata.X.shape == (3, 2)
    assert edata.obs.shape == (3, 0)
    assert edata.var.shape == (2, 0)
    assert edata.tem.shape == (2, 1)
    assert edata.layers["tem_layer"].shape == (3, 2, 2)


def test_ehrdata_init_vanilla_X_and_layers(X_numpy_32):
    edata = EHRData(X=X_numpy_32, layers={"some_layer": X_numpy_32})

    _assert_shape_matches(edata, (3, 2, 1))

    assert edata.obs.shape == (3, 0)
    assert edata.var.shape == (2, 0)
    assert edata.tem.shape == (1, 0)


#################################################################
### Test aligned dataframes during intitialization
#################################################################
def test_ehrdata_init_vanilla_obs(obs_31):
    edata = EHRData(obs=obs_31)
    _assert_shape_matches(edata, (3, 0, 1), check_X_None=True)


def test_ehrdata_init_vanilla_var(var_31):
    edata = EHRData(var=var_31)
    _assert_shape_matches(edata, (0, 3, 1), check_X_None=True)


def test_ehrdata_init_vanilla_tem(tem_31):
    edata = EHRData(tem=tem_31)
    _assert_shape_matches(edata, (0, 0, 3), check_X_None=True)


#################################################################
### Test assignment of X, 3dlayer, t combinations
#################################################################
def test_ehrdata_init_3dlayer_assign_X_tem(X_numpy_32, X_numpy_322, tem_21):
    edata = EHRData(layers={"tem_layer": X_numpy_322})
    edata.X = X_numpy_32
    _assert_shape_matches(edata, (3, 2, 2))
    assert edata.tem.shape == (2, 0)

    edata = EHRData(layers={"tem_layer": X_numpy_322})
    edata.tem = tem_21
    _assert_shape_matches(edata, (3, 2, 2), check_X_None=True)
    assert edata.tem.shape == (2, 1)

    # for assignment of X and t, test both orders
    edata = EHRData(layers={"tem_layer": X_numpy_322})
    edata.X = X_numpy_32
    edata.tem = tem_21
    _assert_shape_matches(edata, (3, 2, 2))
    assert edata.tem.shape == (2, 1)

    edata = EHRData(layers={"tem_layer": X_numpy_322})
    edata.tem = tem_21
    edata.X = X_numpy_32
    _assert_shape_matches(edata, (3, 2, 2))
    assert edata.tem.shape == (2, 1)


# def test_ehrdata_init_X_assign_3dlayer_t(X_numpy_32, X_numpy_322, tem_21):
#     edata = EHRData(X=X_numpy_32)
#     with pytest.raises(ValueError):
#         edata.layers["tem_layer"] = X_numpy_322

#     edata = EHRData(X=X_numpy_32)
#     with pytest.raises(ValueError):
#         edata.tem = tem_21


# def test_ehrdata_init_t_assign_X_3dlayer(X_numpy_32, X_numpy_322, tem_21):
#     edata = EHRData(tem=tem_21)
#     with pytest.raises(ValueError):
#         edata.X = X_numpy_32

#     edata = EHRData(tem=tem_21)
#     with pytest.raises(ValueError):
#         edata.layers["tem_layer"] = X_numpy_322


def test_ehrdata_init_X_3dlayer_assign_t(X_numpy_32, X_numpy_322, tem_21):
    edata = EHRData(X=X_numpy_32, layers={"tem_layer": X_numpy_322})

    edata.tem = tem_21
    _assert_shape_matches(edata, (3, 2, 2))
    assert edata.tem.shape == (2, 1)


def test_ehrdata_init_X_t_assign_3dlayer(X_numpy_32, X_numpy_322, tem_21):
    edata = EHRData(X=X_numpy_32, tem=tem_21)

    edata.layers["tem_layer"] = X_numpy_322
    _assert_shape_matches(edata, (3, 2, 2))
    assert edata.tem.shape == (2, 1)


def test_ehrdata_init_3dlayer_t_assign_X(X_numpy_32, X_numpy_322, tem_21):
    edata = EHRData(layers={"tem_layer": X_numpy_322}, tem=tem_21)

    edata.X = X_numpy_32
    _assert_shape_matches(edata, (3, 2, 2))
    assert edata.tem.shape == (2, 1)


#################################################################
### Test illegal initializations
#################################################################
def test_ehrdata_init_fail_X_and_3dlayer_mismatch(X_numpy_32, obs_31, var_21):
    tem_layer = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    with pytest.raises(ValueError):
        EHRData(X=X_numpy_32, obs=obs_31, layers={"tem_layer": tem_layer}, var=var_21)


def test_ehrdata_init_fail_3dlayer_and_t_mismatch(X_numpy_32, X_numpy_322, obs_31, var_21):
    tem = pd.DataFrame({"tem1": [1]})

    with pytest.raises(ValueError):
        EHRData(X=X_numpy_32, obs=obs_31, layers={"tem_layer": X_numpy_322}, tem=tem, var=var_21)


#################################################################
### Test t is protected alike obs, var
#################################################################
def test_ehrdata_set_aligneddataframes(X_numpy_32, X_numpy_322):
    edata_Xonly = EHRData(X=X_numpy_32, layers={"tem_layer": X_numpy_322})

    # show that setting to None behavior for t alike obs, var
    with pytest.raises(ValueError):
        edata_Xonly.obs = None  # type: ignore
    with pytest.raises(ValueError):
        edata_Xonly.var = None  # type: ignore
    with pytest.raises(ValueError):
        edata_Xonly.tem = None  # type: ignore

    # show that setting df behavior for t alike obs, var
    with pytest.raises(ValueError):
        edata_Xonly.obs = pd.DataFrame([0, 1, 2, 3, 4])
    with pytest.raises(ValueError):
        edata_Xonly.var = pd.DataFrame([0, 1, 2, 3, 4])
    with pytest.raises(ValueError):
        edata_Xonly.tem = pd.DataFrame([0, 1, 2, 3, 4])


#################################################################
### Test different types for 3dlayer
#################################################################
# "X_sparse_322" currently disabled as sparse.COO support in AnnData is not yet implemented
@pytest.mark.parametrize("X_fixture_name", ["X_numpy_322", "X_dask_322"])
def test_ehrdata_X_data_types(X_fixture_name, request):
    X = request.getfixturevalue(X_fixture_name)
    edata = EHRData(layers={"tem_layer": X})
    _assert_shape_matches(edata, (3, 2, 2), check_X_None=True)


#################################################################
### Test assignment operations
#################################################################
def test_ehrdata_assignments(X_numpy_32, X_numpy_322, obs_31, var_21):
    edata = EHRData(X=X_numpy_32, obs=obs_31, var=var_21, layers={"tem_layer": X_numpy_322})

    edata.X[0, 0] = 100

    edata.obs["new_obs_col"] = [1, 2, 3]
    edata.var["new_var_col"] = ["a", "b"]
    edata.tem["new_tem_col"] = [1, 2]

    edata.obs["new_obs_col"].iloc[0] = "obs_entry"
    edata.var["new_var_col"].iloc[0] = "var_entry"
    edata.tem["new_tem_col"].iloc[0] = "tem_entry"

    edata.varm["varm_entry"] = np.array([[1, 2], [3, 4]])
    edata.obsm["obsm_entry"] = np.array([[1, 2], [3, 4], [5, 6]])
    edata.varp["varp_entry"] = np.array([[1, 2], [3, 4]])
    edata.obsp["obsp_entry"] = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])


def test_ehrdata_assignments_view(X_numpy_32, X_numpy_322, obs_31, var_21):
    # TODO: fix tests this checks ehrdata is robust for view behvaior
    edata = EHRData(X=X_numpy_32, obs=obs_31, var=var_21, layers={"tem_layer": X_numpy_322})
    edata_view = edata[:2, :1, :1]

    edata_view.X[0, 0] = 100

    edata_view = edata[:2, :1, :1]
    edata_view.obs["new_obs_col"] = [1, 2]

    edata_view = edata[:2, :1, :1]
    edata_view.var["new_var_col"] = ["a"]

    edata_view = edata[:2, :1, :1]
    edata_view.tem["new_tem_col"] = [1]

    edata_view = edata[:2, :1, :1]
    edata_view.obs["new_obs_col"].iloc[0] = "obs_entry"

    edata_view = edata[:2, :1, :1]
    edata_view.var["new_var_col"].iloc[0] = "var_entry"

    edata_view = edata[:2, :1, :1]
    edata_view.tem["new_tem_col"].iloc[0] = "tem_entry"

    edata_view = edata[:2, :1, :1]
    edata_view.varm["varm_entry"] = np.array([[1], [3]])

    edata_view = edata[:2, :1, :1]
    edata_view.obsm["obsm_entry"] = np.array([[1, 2], [5, 6]])

    edata_view = edata[:2, :1, :1]
    edata_view.varp["varp_entry"] = np.array([[1], [4]])

    edata_view = edata[:2, :1, :1]
    edata_view.obsp["obsp_entry"] = np.array([[1, 2], [3, 4]])


#################################################################
### Test slicing
#################################################################
def test_ehrdata_subset_slice_2D_vanilla(X_numpy_32, X_numpy_322, obs_31, var_21):
    edata = EHRData(X=X_numpy_32, layers={"tem_layer": X_numpy_322}, obs=obs_31, var=var_21)
    edata_sliced = edata[:2, :1]

    assert edata_sliced.is_view
    _assert_shape_matches(edata_sliced, (2, 1, 2))


def test_ehrdata_subset_slice_2D_repeated(X_numpy_32, X_numpy_322, obs_31, var_21):
    edata = EHRData(X=X_numpy_32, layers={"tem_layer": X_numpy_322}, obs=obs_31, var=var_21)
    edata_sliced = edata[1:]
    edata_sliced = edata_sliced[1:]

    _assert_fields_are_view(edata_sliced)
    _assert_shape_matches(edata_sliced, (1, 2, 2))

    assert np.array_equal(edata_sliced.X, X_numpy_32[2].reshape(-1, 2))
    assert np.array_equal(edata_sliced.layers["tem_layer"], X_numpy_322[2].reshape(-1, 2, 2))


def test_ehrdata_subset_slice_3D_vanilla(X_numpy_32, X_numpy_322, obs_31, var_21, tem_21):
    edata = EHRData(X=X_numpy_32, layers={"tem_layer": X_numpy_322}, obs=obs_31, var=var_21, tem=tem_21)
    edata_sliced = edata[:2, :1, :1]

    _assert_fields_are_view(edata_sliced)
    _assert_shape_matches(edata_sliced, (2, 1, 1))

    assert edata_sliced.tem.shape == (1, 1)


def test_ehrdata_subset_slice_3D_repeated(edata_333):
    edata = edata_333
    edata_sliced = edata[1:, 1:, 1:]
    edata_sliced = edata_sliced[1:, 1:, 1:]

    _assert_fields_are_view(edata_sliced)
    _assert_shape_matches(edata_sliced, (1, 1, 1))

    assert edata_sliced.tem.shape == (1, 1)

    # test that the true values are conserved
    assert np.array_equal(edata_sliced.X, edata.X[2, 2].reshape(-1, 1))
    assert np.array_equal(edata_sliced.layers["tem_layer"], edata.layers["tem_layer"][2, 2, 2].reshape(-1, 1, 1))
    assert np.array_equal(edata.tem.iloc[2].values.reshape(-1, 1), edata_sliced.tem.values)


def test_ehrdata_subset_obsvar_names_vanilla(edata_333):
    edata = edata_333
    edata_a = edata[["obs1"]]
    _assert_fields_are_view(edata_a)
    _assert_shape_matches(edata_a, (1, 3, 3))

    edata_ab = edata[["obs1", "obs2"]]
    _assert_fields_are_view(edata_ab)
    _assert_shape_matches(edata_ab, (2, 3, 3))

    edata_a = edata[:, ["var1"]]
    _assert_fields_are_view(edata_a)
    _assert_shape_matches(edata_a, (3, 1, 3))

    edata_ab = edata[:, ["var1", "var2"]]
    _assert_fields_are_view(edata_ab)
    _assert_shape_matches(edata_ab, (3, 2, 3))

    # TODO why does this fail
    edata_a = edata[["obs1", "obs2"], ["var1", "var2"]]

    _assert_fields_are_view(edata_a)
    _assert_shape_matches(edata_a, (2, 2, 3))
    assert edata_a.layers["tem_layer"].shape == (2, 2, 3)


def test_ehrdata_subset_obsvar_names_repeated(edata_333):
    edata = edata_333
    edata_ab = edata[["obs2", "obs3"]]
    edata_a = edata_ab[["obs3"]]

    _assert_fields_are_view(edata_a)
    _assert_shape_matches(edata_a, (1, 3, 3))

    assert np.array_equal(edata_a.layers["tem_layer"], edata.layers["tem_layer"][2, :, :].reshape(-1, 3, 3))

    edata = edata_333
    edata_ab = edata[:, ["var2", "var3"]]
    edata_a = edata_ab[:, ["var3"]]

    _assert_fields_are_view(edata_a)
    _assert_shape_matches(edata_a, (3, 1, 3))

    assert np.array_equal(edata_a.layers["tem_layer"], edata.layers["tem_layer"][:, 2, :].reshape(3, -1, 3))


def test_ehrdata_subset_boolindex_vanilla(edata_333):
    edata = edata_333
    edata_sliced = edata[[False, True, True], [True, False, False], [True, False, False]]

    _assert_fields_are_view(edata_sliced)
    _assert_shape_matches(edata_sliced, (2, 1, 1))

    assert edata_sliced.tem.shape == (1, 1)


def test_ehrdata_subset_boolindex_repeated(edata_333):
    edata = edata_333
    edata_sliced = edata[[False, True, True], [False, True, True], [False, True, True]]
    edata_sliced = edata_sliced[[False, True], [False, True], [False, True]]

    _assert_fields_are_view(edata_sliced)
    _assert_shape_matches(edata_sliced, (1, 1, 1))

    assert edata_sliced.tem.shape == (1, 1)

    # test that the true values are conserved
    assert np.array_equal(edata_sliced.X, edata.X[2, 2].reshape(-1, 1))
    assert np.array_equal(edata_sliced.layers["tem_layer"], edata.layers["tem_layer"][2, 2, 2].reshape(-1, 1, 1))
    assert np.array_equal(edata.tem.iloc[2].values.reshape(-1, 1), edata_sliced.tem.values)


def test_ehrdata_subset_numberindex_vanilla(edata_333):
    edata = edata_333
    edata_sliced = edata[[1, 2], [1], [1]]

    _assert_fields_are_view(edata_sliced)
    _assert_shape_matches(edata_sliced, (2, 1, 1))

    assert edata_sliced.tem.shape == (1, 1)


def test_ehrdata_subset_numberindex_repeated(edata_333):
    edata = edata_333
    edata_sliced = edata[[1, 2], [1, 2], [1, 2]]
    edata_sliced = edata_sliced[[1], [1], [1]]

    _assert_fields_are_view(edata_sliced)
    _assert_shape_matches(edata_sliced, (1, 1, 1))

    assert edata_sliced.tem.shape == (1, 1)

    # test that the true values are conserved
    assert np.array_equal(edata_sliced.X, edata.X[2, 2].reshape(-1, 1))
    assert np.array_equal(edata_sliced.layers["tem_layer"], edata.layers["tem_layer"][2, 2, 2].reshape(-1, 1, 1))
    assert np.array_equal(edata.tem.iloc[2].values.reshape(-1, 1), edata_sliced.tem.values)


def test_ehrdata_subset_mixedindices(edata_333):
    edata = edata_333

    edata_sliced = edata[["obs1", "obs2"], 1, [False, True, True]]

    _assert_fields_are_view(edata_sliced)
    _assert_shape_matches(edata_sliced, (2, 1, 2))

    edata_sliced = edata_sliced[np.array([False, True]), :, [1]]

    _assert_fields_are_view(edata_sliced)
    _assert_shape_matches(edata_sliced, (1, 1, 1))


def test_copy(edata_333):
    edata = EHRData()
    edata.copy()

    edata = edata_333
    edata_copy = edata.copy()

    assert isinstance(edata_copy.X, np.ndarray)
    assert isinstance(edata_copy.layers["tem_layer"], np.ndarray)
    assert isinstance(edata_copy.obs, pd.DataFrame)
    assert isinstance(edata_copy.var, pd.DataFrame)
    assert isinstance(edata_copy.tem, pd.DataFrame)

    assert edata.tem.shape == (3, 1)


def test_copy_of_slice(edata_333):
    edata = edata_333
    edata_sliced = edata[1:, 1:, 1:]
    edata_sliced_copy = edata_sliced.copy()

    _assert_shape_matches(edata_sliced_copy, (2, 2, 2))
    assert isinstance(edata_sliced_copy.X, np.ndarray)
    assert isinstance(edata_sliced_copy.layers["tem_layer"], np.ndarray)
    assert isinstance(edata_sliced_copy.obs, pd.DataFrame)
    assert isinstance(edata_sliced_copy.var, pd.DataFrame)
    assert isinstance(edata_sliced_copy.tem, pd.DataFrame)

    assert edata_sliced_copy.tem.shape == (2, 1)


# TODO why does this fail
def test_copy_of_obsvar_names(edata_333):
    edata = edata_333

    edata_obs_subset = edata[["obs1", "obs2"]]
    edata_obs_subset = edata_obs_subset.copy()
    assert not edata_obs_subset.is_view
    _assert_shape_matches(edata_obs_subset, (2, 3, 3))

    edata_var_subset = edata[:, ["var1", "var2"]]
    edata_var_subset = edata_var_subset.copy()
    assert not edata_var_subset.is_view
    _assert_shape_matches(edata_var_subset, (3, 2, 3))

    edata_obsvar_subset = edata[["obs1"], ["var1", "var2"]]
    edata_obsvar_subset = edata_obsvar_subset.copy()
    assert not edata_obsvar_subset.is_view
    _assert_shape_matches(edata_obsvar_subset, (1, 2, 3))
