import anndata as ad
import numpy as np
import pandas as pd
import pytest
from tests.conftest import _assert_shape_matches

from ehrdata import EHRData
from ehrdata.core.constants import R_LAYER_KEY


def _assert_fields_are_view(edata: EHRData):
    assert edata.is_view
    assert isinstance(edata.X, ad._core.views.ArrayView)
    assert isinstance(edata.R, ad._core.views.ArrayView)
    assert isinstance(edata.tem, ad._core.views.DataFrameView)


#################################################################
### Test combinations of X, R, t during initialization
#################################################################
def test_ehrdata_init_vanilla_empty():
    edata = EHRData()

    _assert_shape_matches(edata, (0, 0, 0), check_X_None=True, check_R_None=True)
    assert edata.X is None
    assert edata.R is None
    assert edata.obs.shape == (0, 0)
    assert edata.var.shape == (0, 0)
    assert edata.tem.shape == (0, 0)


def test_ehrdata_init_vanilla_X(X_numpy_32):
    edata = EHRData(X=X_numpy_32)
    _assert_shape_matches(edata, (3, 2, 0), check_R_None=True)

    assert edata.obs.shape == (3, 0)

    assert edata.var.shape == (2, 0)

    assert edata.tem.shape == (0, 0)


def test_ehrdata_init_vanilla_r(R_numpy_322):
    edata = EHRData(R=R_numpy_322)
    _assert_shape_matches(edata, (3, 2, 2))

    assert edata.layers is not None
    assert edata.layers[R_LAYER_KEY] is not None
    assert edata.layers[R_LAYER_KEY].shape == (3, 2, 2)

    assert edata.obs.shape == (3, 0)

    assert edata.var.shape == (2, 0)

    assert edata.tem.shape == (2, 0)


def test_ehrdata_init_vanilla_X_and_r(X_numpy_32, R_numpy_322):
    edata = EHRData(X=X_numpy_32, R=R_numpy_322)
    _assert_shape_matches(edata, (3, 2, 2))

    assert edata.layers is not None
    assert edata.R is not None

    assert edata.X.shape == (3, 2)
    assert edata.obs.shape == (3, 0)
    assert edata.var.shape == (2, 0)
    assert edata.tem.shape == (2, 0)
    assert edata.R.shape == (3, 2, 2)


def test_ehrdata_init_vanilla_X_and_t(X_numpy_32, tem_21):
    edata = EHRData(X=X_numpy_32, tem=tem_21)
    _assert_shape_matches(edata, (3, 2, 2), check_R_None=True)

    assert edata.layers is not None

    assert edata.X.shape == (3, 2)
    assert edata.obs.shape == (3, 0)
    assert edata.var.shape == (2, 0)
    assert edata.tem.shape == (2, 1)


def test_ehrdata_init_vanilla_X_and_r_and_t(X_numpy_32, R_numpy_322, tem_21):
    edata = EHRData(X=X_numpy_32, R=R_numpy_322, tem=tem_21)
    _assert_shape_matches(edata, (3, 2, 2))

    assert edata.layers is not None
    assert edata.R is not None

    assert edata.X.shape == (3, 2)
    assert edata.obs.shape == (3, 0)
    assert edata.var.shape == (2, 0)
    assert edata.tem.shape == (2, 1)
    assert edata.R.shape == (3, 2, 2)


def test_ehrdata_init_vanilla_X_and_layers(X_numpy_32):
    edata = EHRData(X=X_numpy_32, layers={"some_layer": X_numpy_32})

    _assert_shape_matches(edata, (3, 2, 0), check_R_None=True)

    assert edata.obs.shape == (3, 0)
    assert edata.var.shape == (2, 0)
    assert edata.tem.shape == (0, 0)


#################################################################
### Test aligned dataframes during intitialization
#################################################################
def test_ehrdata_init_vanilla_obs(obs_31):
    edata = EHRData(obs=obs_31)
    _assert_shape_matches(edata, (3, 0, 0), check_X_None=True, check_R_None=True)


def test_ehrdata_init_vanilla_var(var_31):
    edata = EHRData(var=var_31)
    _assert_shape_matches(edata, (0, 3, 0), check_X_None=True, check_R_None=True)


def test_ehrdata_init_vanilla_tem(tem_31):
    edata = EHRData(tem=tem_31)
    _assert_shape_matches(edata, (0, 0, 3), check_X_None=True, check_R_None=True)


#################################################################
### Test assignment of X, R, t combinations
#################################################################
def test_ehrdata_init_r_assign_X_tem(X_numpy_32, R_numpy_322, tem_21):
    edata = EHRData(R=R_numpy_322)
    edata.X = X_numpy_32
    _assert_shape_matches(edata, (3, 2, 2))
    assert edata.tem.shape == (2, 0)

    edata = EHRData(R=R_numpy_322)
    edata.tem = tem_21
    _assert_shape_matches(edata, (3, 2, 2))
    assert edata.tem.shape == (2, 1)

    # for assignment of X and t, test both orders
    edata = EHRData(R=R_numpy_322)
    edata.X = X_numpy_32
    edata.tem = tem_21
    _assert_shape_matches(edata, (3, 2, 2))
    assert edata.tem.shape == (2, 1)

    edata = EHRData(R=R_numpy_322)
    edata.tem = tem_21
    edata.X = X_numpy_32
    _assert_shape_matches(edata, (3, 2, 2))
    assert edata.tem.shape == (2, 1)


def test_ehrdata_init_X_assign_r_t(X_numpy_32, R_numpy_322, tem_21):
    edata = EHRData(X=X_numpy_32)
    with pytest.raises(ValueError):
        edata.R = R_numpy_322

    edata = EHRData(X=X_numpy_32)
    with pytest.raises(ValueError):
        edata.tem = tem_21


def test_ehrdata_init_t_assign_X_r(X_numpy_32, R_numpy_322, tem_21):
    edata = EHRData(tem=tem_21)
    with pytest.raises(ValueError):
        edata.X = X_numpy_32

    edata = EHRData(tem=tem_21)
    with pytest.raises(ValueError):
        edata.R = R_numpy_322


def test_ehrdata_init_X_r_assign_t(X_numpy_32, R_numpy_322, tem_21):
    edata = EHRData(X=X_numpy_32, R=R_numpy_322)

    edata.tem = tem_21
    _assert_shape_matches(edata, (3, 2, 2))
    assert edata.tem.shape == (2, 1)


def test_ehrdata_init_X_t_assign_r(X_numpy_32, R_numpy_322, tem_21):
    edata = EHRData(X=X_numpy_32, tem=tem_21)

    edata.R = R_numpy_322
    _assert_shape_matches(edata, (3, 2, 2))
    assert edata.tem.shape == (2, 1)


def test_ehrdata_init_r_t_assign_X(X_numpy_32, R_numpy_322, tem_21):
    edata = EHRData(R=R_numpy_322, tem=tem_21)

    edata.X = X_numpy_32
    _assert_shape_matches(edata, (3, 2, 2))
    assert edata.tem.shape == (2, 1)


#################################################################
### Test illegal initializations
#################################################################
def test_ehrdata_init_fail_r_2D(X_numpy_32):
    with pytest.raises(ValueError):
        EHRData(R=X_numpy_32)


def test_ehrdata_X_assign_fail_r_2D(X_numpy_32):
    edata = EHRData(X=X_numpy_32)
    with pytest.raises(ValueError):
        edata.R = X_numpy_32


def test_ehrdata_init_fail_X_and_R_mismatch(X_numpy_32, obs_31, var_21):
    r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    with pytest.raises(ValueError):
        EHRData(X=X_numpy_32, obs=obs_31, R=r, var=var_21)


def test_ehrdata_init_fail_R_and_t_mismatch(X_numpy_32, R_numpy_322, obs_31, var_21):
    tem = pd.DataFrame({"tem1": [1]})

    with pytest.raises(ValueError):
        EHRData(X=X_numpy_32, obs=obs_31, R=R_numpy_322, tem=tem, var=var_21)


def test_ehrdata_init_fail_R_and_layer_R_LAYER_KEY_exists(X_numpy_32, R_numpy_322):
    layers = {R_LAYER_KEY: R_numpy_322}

    with pytest.raises(ValueError):
        EHRData(X=X_numpy_32, R=R_numpy_322, layers=layers)


#################################################################
### Test t is protected alike obs, var
#################################################################
def test_ehrdata_set_aligneddataframes(X_numpy_32):
    edata_Xonly = EHRData(X_numpy_32)

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


def test_ehrdata_del_r(X_numpy_32, R_numpy_322):
    edata = EHRData(X=X_numpy_32, R=R_numpy_322)
    del edata.R

    assert edata.tem.shape == (0, 0)

    _assert_shape_matches(edata, (3, 2, 0), check_R_None=True)


#################################################################
### Test different types for R
#################################################################
# "R_sparse_322" currently disabled as sparse.COO support in AnnData is not yet implemented
@pytest.mark.parametrize("R_fixture_name", ["R_numpy_322", "R_dask_322"])
def test_ehrdata_R_data_types(R_fixture_name, request):
    R = request.getfixturevalue(R_fixture_name)
    edata = EHRData(R=R)
    _assert_shape_matches(edata, (3, 2, 2))


#################################################################
### Test sliceing
#################################################################
def test_ehrdata_subset_slice_2D_vanilla(X_numpy_32, R_numpy_322, obs_31, var_21):
    edata = EHRData(X=X_numpy_32, R=R_numpy_322, obs=obs_31, var=var_21)
    edata_sliced = edata[:2, :1]

    assert edata_sliced.is_view
    _assert_shape_matches(edata_sliced, (2, 1, 2))


def test_ehrdata_subset_slice_2D_repeated(X_numpy_32, R_numpy_322, obs_31, var_21):
    edata = EHRData(X=X_numpy_32, R=R_numpy_322, obs=obs_31, var=var_21)
    edata_sliced = edata[1:]
    edata_sliced = edata_sliced[1:]

    _assert_fields_are_view(edata_sliced)
    _assert_shape_matches(edata_sliced, (1, 2, 2))

    assert np.array_equal(edata_sliced.X, X_numpy_32[2].reshape(-1, 2))
    assert np.array_equal(edata_sliced.R, R_numpy_322[2].reshape(-1, 2, 2))


def test_ehrdata_subset_slice_3D_vanilla(X_numpy_32, R_numpy_322, obs_31, var_21, tem_21):
    edata = EHRData(X=X_numpy_32, R=R_numpy_322, obs=obs_31, var=var_21, tem=tem_21)
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
    assert np.array_equal(edata_sliced.R, edata.R[2, 2, 2].reshape(-1, 1, 1))
    assert np.array_equal(edata.tem.iloc[2].values.reshape(-1, 1), edata_sliced.tem.values)


def test_ehrdata_subset_obsvar_names_vanilla(edata_333, adata_33):
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

    edata_a = edata[["obs1", "obs2"], ["var1", "var2"]]
    _assert_fields_are_view(edata_a)
    _assert_shape_matches(edata_a, (2, 2, 3))
    assert edata_a.R.shape == (2, 2, 3)


def test_ehrdata_subset_obsvar_names_repeated(edata_333):
    edata = edata_333
    edata_ab = edata[["obs2", "obs3"]]
    edata_a = edata_ab[["obs3"]]

    _assert_fields_are_view(edata_a)
    _assert_shape_matches(edata_a, (1, 3, 3))

    assert np.array_equal(edata_a.R, edata.R[2, :, :].reshape(-1, 3, 3))

    edata = edata_333
    edata_ab = edata[:, ["var2", "var3"]]
    edata_a = edata_ab[:, ["var3"]]

    _assert_fields_are_view(edata_a)
    _assert_shape_matches(edata_a, (3, 1, 3))

    assert np.array_equal(edata_a.R, edata.R[:, 2, :].reshape(3, -1, 3))


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
    assert np.array_equal(edata_sliced.R, edata.R[2, 2, 2].reshape(-1, 1, 1))
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
    assert np.array_equal(edata_sliced.R, edata.R[2, 2, 2].reshape(-1, 1, 1))
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
    assert isinstance(edata_copy.R, np.ndarray)
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
    assert isinstance(edata_sliced_copy.R, np.ndarray)
    assert isinstance(edata_sliced_copy.obs, pd.DataFrame)
    assert isinstance(edata_sliced_copy.var, pd.DataFrame)
    assert isinstance(edata_sliced_copy.tem, pd.DataFrame)

    assert edata_sliced_copy.tem.shape == (2, 1)


def test_copy_of_obsvar_names(edata_333, adata_33):
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
