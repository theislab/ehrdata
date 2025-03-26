import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ehrdata import EHRData
from ehrdata.core.constants import R_LAYER_KEY


def _shape_test(edata: EHRData, shape: tuple[int, int, int]):
    assert edata.shape == shape

    assert isinstance(edata.obs, pd.DataFrame)
    assert len(edata.obs) == shape[0]
    assert edata.n_obs == shape[0]

    assert isinstance(edata.var, pd.DataFrame)
    assert len(edata.var) == shape[1]
    assert edata.n_vars == shape[1]

    assert isinstance(edata.t, pd.DataFrame)
    assert len(edata.t) == shape[2]
    assert edata.n_t == shape[2]


def _assert_fields_are_view(edata: EHRData):
    assert edata.is_view
    assert isinstance(edata.X, ad._core.views.ArrayView)
    assert isinstance(edata.r, ad._core.views.ArrayView)
    assert isinstance(edata.t, ad._core.views.DataFrameView)


#################################################################
### Test combinations of X, r, t during initialization
#################################################################
def test_ehrdata_init_vanilla_empty():
    edata = EHRData()

    _shape_test(edata, (0, 0, 0))

    assert edata.X is None
    assert edata.r is None

    assert hasattr(edata, "obs")
    assert edata.obs.shape == (0, 0)

    assert hasattr(edata, "var")
    assert edata.var.shape == (0, 0)

    assert hasattr(edata, "t")
    assert edata.t.shape == (0, 0)


def test_ehrdata_init_vanilla_X(X_32):
    edata = EHRData(X=X_32)
    _shape_test(edata, (3, 2, 0))

    assert edata.r is None

    assert hasattr(edata, "obs")
    assert edata.obs.shape == (3, 0)

    assert hasattr(edata, "var")
    assert edata.var.shape == (2, 0)

    assert hasattr(edata, "t")
    assert edata.t.shape == (0, 0)


def test_ehrdata_init_vanilla_r(r_322):
    edata = EHRData(r=r_322)
    _shape_test(edata, (3, 2, 2))

    assert edata.layers is not None
    assert edata.layers[R_LAYER_KEY] is not None
    assert edata.layers[R_LAYER_KEY].shape == (3, 2, 2)

    assert hasattr(edata, "obs")
    assert edata.obs.shape == (3, 0)

    assert hasattr(edata, "var")
    assert edata.var.shape == (2, 0)

    assert hasattr(edata, "t")
    assert edata.t.shape == (2, 0)


def test_ehrdata_init_vanilla_X_and_r(X_32, r_322):
    edata = EHRData(X=X_32, r=r_322)
    _shape_test(edata, (3, 2, 2))

    assert edata.layers is not None
    assert edata.r is not None

    assert edata.X.shape == (3, 2)
    assert edata.obs.shape == (3, 0)
    assert edata.var.shape == (2, 0)
    assert edata.t.shape == (2, 0)
    assert edata.r.shape == (3, 2, 2)


def test_ehrdata_init_vanilla_X_and_t(X_32, t_21):
    edata = EHRData(X=X_32, t=t_21)
    _shape_test(edata, (3, 2, 2))

    assert edata.layers is not None
    assert edata.r is None

    assert edata.X.shape == (3, 2)
    assert edata.obs.shape == (3, 0)
    assert edata.var.shape == (2, 0)
    assert edata.t.shape == (2, 1)


def test_ehrdata_init_vanilla_X_and_r_and_t(X_32, r_322, t_21):
    edata = EHRData(X=X_32, r=r_322, t=t_21)
    _shape_test(edata, (3, 2, 2))

    assert edata.layers is not None
    assert edata.r is not None

    assert edata.X.shape == (3, 2)
    assert edata.obs.shape == (3, 0)
    assert edata.var.shape == (2, 0)
    assert edata.t.shape == (2, 1)
    assert edata.r.shape == (3, 2, 2)


#################################################################
### Test aligned dataframes during intitialization
#################################################################
def test_ehrdata_init_vanilla_obs(obs_31):
    edata = EHRData(obs=obs_31)
    _shape_test(edata, (3, 0, 0))


def test_ehrdata_init_vanilla_var(var_31):
    edata = EHRData(var=var_31)
    _shape_test(edata, (0, 3, 0))


def test_ehrdata_init_vanilla_t(t_31):
    edata = EHRData(t=t_31)
    _shape_test(edata, (0, 0, 3))


#################################################################
### Test assignment of X, r, t combinations
#################################################################
def test_ehrdata_init_r_assign_X_t(X_32, r_322, t_21):
    edata = EHRData(r=r_322)
    edata.X = X_32
    _shape_test(edata, (3, 2, 2))
    assert edata.t.shape == (2, 0)

    edata = EHRData(r=r_322)
    edata.t = t_21
    _shape_test(edata, (3, 2, 2))
    assert edata.t.shape == (2, 1)

    # for assignment of X and t, test both orders
    edata = EHRData(r=r_322)
    edata.X = X_32
    edata.t = t_21
    _shape_test(edata, (3, 2, 2))
    assert edata.t.shape == (2, 1)

    edata = EHRData(r=r_322)
    edata.t = t_21
    edata.X = X_32
    _shape_test(edata, (3, 2, 2))
    assert edata.t.shape == (2, 1)


def test_ehrdata_init_X_assign_r_t(X_32, r_322, t_21):
    edata = EHRData(X=X_32)
    with pytest.raises(ValueError):
        edata.r = r_322

    edata = EHRData(X=X_32)
    with pytest.raises(ValueError):
        edata.t = t_21


def test_ehrdata_init_t_assign_X_r(X_32, r_322, t_21):
    edata = EHRData(t=t_21)
    with pytest.raises(ValueError):
        edata.X = X_32

    edata = EHRData(t=t_21)
    with pytest.raises(ValueError):
        edata.r = r_322


def test_ehrdata_init_X_r_assign_t(X_32, r_322, t_21):
    edata = EHRData(X=X_32, r=r_322)

    edata.t = t_21
    _shape_test(edata, (3, 2, 2))
    assert edata.t.shape == (2, 1)


def test_ehrdata_init_X_t_assign_r(X_32, r_322, t_21):
    edata = EHRData(X=X_32, t=t_21)

    edata.r = r_322
    _shape_test(edata, (3, 2, 2))
    assert edata.t.shape == (2, 1)


def test_ehrdata_init_r_t_assign_X(X_32, r_322, t_21):
    edata = EHRData(r=r_322, t=t_21)

    edata.X = X_32
    _shape_test(edata, (3, 2, 2))
    assert edata.t.shape == (2, 1)


#################################################################
### Test illegal initializations
#################################################################
def test_ehrdata_init_fail_r_2D(X_32):
    with pytest.raises(ValueError):
        EHRData(r=X_32)


def test_ehrdata_X_assign_fail_r_2D(X_32):
    edata = EHRData(X=X_32)
    with pytest.raises(ValueError):
        edata.r = X_32


def test_ehrdata_init_fail_X_and_r_mismatch(X_32, obs_31, var_21):
    r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    with pytest.raises(ValueError):
        EHRData(X=X_32, obs=obs_31, r=r, var=var_21)


def test_ehrdata_init_fail_r_and_t_mismatch(X_32, r_322, obs_31, var_21):
    t = pd.DataFrame({"t1": [1]})

    with pytest.raises(ValueError):
        EHRData(X=X_32, obs=obs_31, r=r_322, t=t, var=var_21)


def test_ehrdata_init_fail_r_and_layer_R_LAYER_KEY_exists(X_32, r_322):
    layers = {R_LAYER_KEY: r_322}

    with pytest.raises(ValueError):
        EHRData(X=X_32, r=r_322, layers=layers)


#################################################################
### Test t is protected alike obs, var
#################################################################
def test_ehrdata_set_aligneddataframes(X_32):
    edata_Xonly = EHRData(X_32)

    # show that setting to None behavior for t alike obs, var
    with pytest.raises(ValueError):
        edata_Xonly.obs = None  # type: ignore
    with pytest.raises(ValueError):
        edata_Xonly.var = None  # type: ignore
    with pytest.raises(ValueError):
        edata_Xonly.t = None  # type: ignore

    # show that setting df behavior for t alike obs, var
    with pytest.raises(ValueError):
        edata_Xonly.obs = pd.DataFrame([0, 1, 2, 3, 4])
    with pytest.raises(ValueError):
        edata_Xonly.var = pd.DataFrame([0, 1, 2, 3, 4])
    with pytest.raises(ValueError):
        edata_Xonly.t = pd.DataFrame([0, 1, 2, 3, 4])


def test_ehrdata_del_r(X_32, r_322):
    edata = EHRData(X=X_32, r=r_322)
    del edata.r

    assert edata.r is None
    assert edata.t.shape == (0, 0)

    _shape_test(edata, (3, 2, 0))


#################################################################
### Test sliceing
#################################################################
def test_ehrdata_subset_slice_2D_vanilla(X_32, r_322, obs_31, var_21):
    edata = EHRData(X=X_32, r=r_322, obs=obs_31, var=var_21)
    edata_sliced = edata[:2, :1]

    assert edata_sliced.is_view
    _shape_test(edata_sliced, (2, 1, 2))


def test_ehrdata_subset_slice_2D_repeated(X_32, r_322, obs_31, var_21):
    edata = EHRData(X=X_32, r=r_322, obs=obs_31, var=var_21)
    edata_sliced = edata[1:]
    edata_sliced = edata_sliced[1:]

    _assert_fields_are_view(edata_sliced)
    _shape_test(edata_sliced, (1, 2, 2))

    assert np.array_equal(edata_sliced.X, X_32[2].reshape(-1, 2))
    assert np.array_equal(edata_sliced.r, r_322[2].reshape(-1, 2, 2))


def test_ehrdata_subset_slice_3D_vanilla(X_32, r_322, obs_31, var_21, t_21):
    edata = EHRData(X=X_32, r=r_322, obs=obs_31, var=var_21, t=t_21)
    edata_sliced = edata[:2, :1, :1]

    _assert_fields_are_view(edata_sliced)
    _shape_test(edata_sliced, (2, 1, 1))

    assert edata_sliced.r.shape == (2, 1, 1)
    assert edata_sliced.t.shape == (1, 1)


def test_ehrdata_subset_slice_3D_repeated(edata_333):
    edata = edata_333
    edata_sliced = edata[1:, 1:, 1:]
    edata_sliced = edata_sliced[1:, 1:, 1:]

    _assert_fields_are_view(edata_sliced)
    _shape_test(edata_sliced, (1, 1, 1))
    assert hasattr(edata_sliced, "r")
    assert edata_sliced.r.shape == (1, 1, 1)
    assert edata_sliced.t.shape == (1, 1)

    # test that the true values are conserved
    assert np.array_equal(edata_sliced.X, edata.X[2, 2].reshape(-1, 1))
    assert np.array_equal(edata_sliced.r, edata.r[2, 2, 2].reshape(-1, 1, 1))
    assert np.array_equal(edata.t.iloc[2].values.reshape(-1, 1), edata_sliced.t.values)


def test_ehrdata_subset_obsvar_names_vanilla(edata_333, adata_33):
    edata = edata_333
    edata_a = edata[["obs1"]]
    _assert_fields_are_view(edata_a)
    _shape_test(edata_a, (1, 3, 3))

    edata_ab = edata[["obs1", "obs2"]]
    _assert_fields_are_view(edata_ab)
    _shape_test(edata_ab, (2, 3, 3))

    edata_a = edata[:, ["var1"]]
    _assert_fields_are_view(edata_a)
    _shape_test(edata_a, (3, 1, 3))

    edata_ab = edata[:, ["var1", "var2"]]
    _assert_fields_are_view(edata_ab)
    _shape_test(edata_ab, (3, 2, 3))

    edata_a = edata[["obs1", "obs2"], ["var1", "var2"]]
    _assert_fields_are_view(edata_a)
    _shape_test(edata_a, (2, 2, 3))
    assert edata_a.r.shape == (2, 2, 3)


def test_ehrdata_subset_obsvar_names_repeated(edata_333):
    edata = edata_333
    edata_ab = edata[["obs2", "obs3"]]
    edata_a = edata_ab[["obs3"]]

    _assert_fields_are_view(edata_a)
    _shape_test(edata_a, (1, 3, 3))

    assert np.array_equal(edata_a.r, edata.r[2, :, :].reshape(-1, 3, 3))

    edata = edata_333
    edata_ab = edata[:, ["var2", "var3"]]
    edata_a = edata_ab[:, ["var3"]]

    _assert_fields_are_view(edata_a)
    _shape_test(edata_a, (3, 1, 3))

    assert np.array_equal(edata_a.r, edata.r[:, 2, :].reshape(3, -1, 3))


def test_ehrdata_subset_boolindex_vanilla(edata_333):
    edata = edata_333
    edata_sliced = edata[[False, True, True], [True, False, False], [True, False, False]]

    _assert_fields_are_view(edata_sliced)
    _shape_test(edata_sliced, (2, 1, 1))

    assert edata_sliced.r.shape == (2, 1, 1)
    assert edata_sliced.t.shape == (1, 1)


def test_ehrdata_subset_boolindex_repeated(edata_333):
    edata = edata_333
    edata_sliced = edata[[False, True, True], [False, True, True], [False, True, True]]
    edata_sliced = edata_sliced[[False, True], [False, True], [False, True]]

    _assert_fields_are_view(edata_sliced)
    _shape_test(edata_sliced, (1, 1, 1))
    assert hasattr(edata_sliced, "r")
    assert edata_sliced.r.shape == (1, 1, 1)
    assert edata_sliced.t.shape == (1, 1)

    # test that the true values are conserved
    assert np.array_equal(edata_sliced.X, edata.X[2, 2].reshape(-1, 1))
    assert np.array_equal(edata_sliced.r, edata.r[2, 2, 2].reshape(-1, 1, 1))
    assert np.array_equal(edata.t.iloc[2].values.reshape(-1, 1), edata_sliced.t.values)


def test_ehrdata_subset_numberindex_vanilla(edata_333):
    edata = edata_333
    edata_sliced = edata[[1, 2], [1], [1]]

    _assert_fields_are_view(edata_sliced)
    _shape_test(edata_sliced, (2, 1, 1))

    assert edata_sliced.r.shape == (2, 1, 1)
    assert edata_sliced.t.shape == (1, 1)


def test_ehrdata_subset_numberindex_repeated(edata_333):
    edata = edata_333
    edata_sliced = edata[[1, 2], [1, 2], [1, 2]]
    edata_sliced = edata_sliced[[1], [1], [1]]

    _assert_fields_are_view(edata_sliced)
    _shape_test(edata_sliced, (1, 1, 1))
    assert hasattr(edata_sliced, "r")
    assert edata_sliced.r.shape == (1, 1, 1)
    assert edata_sliced.t.shape == (1, 1)

    # test that the true values are conserved
    assert np.array_equal(edata_sliced.X, edata.X[2, 2].reshape(-1, 1))
    assert np.array_equal(edata_sliced.r, edata.r[2, 2, 2].reshape(-1, 1, 1))
    assert np.array_equal(edata.t.iloc[2].values.reshape(-1, 1), edata_sliced.t.values)


def test_ehrdata_subset_mixedindices(edata_333):
    edata = edata_333

    edata_sliced = edata[["obs1", "obs2"], 1, [False, True, True]]

    _assert_fields_are_view(edata_sliced)
    _shape_test(edata_sliced, (2, 1, 2))

    edata_sliced = edata_sliced[np.array([False, True]), :, [1]]

    _assert_fields_are_view(edata_sliced)
    _shape_test(edata_sliced, (1, 1, 1))


def test_copy(edata_333):
    edata = EHRData()
    edata.copy()

    edata = edata_333
    edata_copy = edata.copy()

    assert isinstance(edata_copy.X, np.ndarray)
    assert isinstance(edata_copy.r, np.ndarray)
    assert isinstance(edata_copy.obs, pd.DataFrame)
    assert isinstance(edata_copy.var, pd.DataFrame)
    assert isinstance(edata_copy.t, pd.DataFrame)

    assert edata.t.shape == (3, 1)


def test_copy_of_slice(edata_333):
    edata = edata_333
    edata_sliced = edata[1:, 1:, 1:]
    edata_sliced_copy = edata_sliced.copy()

    _shape_test(edata_sliced_copy, (2, 2, 2))
    assert isinstance(edata_sliced_copy.X, np.ndarray)
    assert isinstance(edata_sliced_copy.r, np.ndarray)
    assert isinstance(edata_sliced_copy.obs, pd.DataFrame)
    assert isinstance(edata_sliced_copy.var, pd.DataFrame)
    assert isinstance(edata_sliced_copy.t, pd.DataFrame)

    assert edata_sliced_copy.t.shape == (2, 1)


def test_copy_of_obsvar_names(edata_333, adata_33):
    edata = edata_333

    edata_obs_subset = edata[["obs1", "obs2"]]
    edata_obs_subset = edata_obs_subset.copy()
    assert not edata_obs_subset.is_view
    _shape_test(edata_obs_subset, (2, 3, 3))

    edata_var_subset = edata[:, ["var1", "var2"]]
    edata_var_subset = edata_var_subset.copy()
    assert not edata_var_subset.is_view
    _shape_test(edata_var_subset, (3, 2, 3))

    edata_obsvar_subset = edata[["obs1"], ["var1", "var2"]]
    edata_obsvar_subset = edata_obsvar_subset.copy()
    assert not edata_obsvar_subset.is_view
    _shape_test(edata_obsvar_subset, (1, 2, 3))
