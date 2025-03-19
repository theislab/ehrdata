import numpy as np
import pandas as pd
import pytest

from ehrdata import EHRData
from ehrdata.core.constants import R_LAYER_KEY


def test_ehrdata_init_vanilla_empty():
    edata = EHRData()
    assert edata.X is None
    assert edata.r is None

    assert edata.obs is not None
    assert edata.var is not None
    assert edata.t is not None


def test_ehrdata_init_vanilla_X():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame({"obs1": [1, 2, 3]})
    var = pd.DataFrame({"var1": [1, 2]})
    adata = EHRData(X=X, obs=obs, var=var)

    assert adata.X is not None
    assert adata.obs is not None
    assert adata.var is not None
    assert adata.r is None

    assert hasattr(adata, "t")
    assert adata.t is not None

    assert adata.shape == (3, 2)
    assert adata.n_obs == 3
    assert adata.n_vars == 2
    assert adata.n_t is None


def test_ehrdata_init_vanilla_r():
    obs = pd.DataFrame({"obs1": [1, 2, 3]})
    var = pd.DataFrame({"var1": [1, 2]})
    r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

    adata = EHRData(obs=obs, r=r, var=var)
    assert adata.X is not None
    assert adata.obs is not None
    assert adata.var is not None
    assert adata.layers is not None
    assert adata.layers[R_LAYER_KEY] is not None

    assert hasattr(adata, "t")
    assert adata.t is not None

    assert adata.X.shape == (3, 2)
    assert adata.obs.shape == (3, 1)
    assert adata.var.shape == (2, 1)
    assert adata.layers[R_LAYER_KEY].shape == (3, 2, 2)

    assert adata.shape == (3, 2, 2)
    assert adata.n_obs == 3
    assert adata.n_vars == 2
    assert adata.n_t == 2


def test_ehrdata_init_vanilla_X_and_r():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame({"obs1": [1, 2, 3]})
    var = pd.DataFrame({"var1": [1, 2]})
    r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

    adata = EHRData(X=X, obs=obs, r=r, var=var)
    assert adata.X is not None
    assert adata.obs is not None
    assert adata.var is not None
    assert adata.layers is not None
    assert adata.layers[R_LAYER_KEY] is not None

    assert hasattr(adata, "t")
    assert adata.t is not None

    assert adata.X.shape == (3, 2)
    assert adata.obs.shape == (3, 1)
    assert adata.var.shape == (2, 1)
    assert adata.layers[R_LAYER_KEY].shape == (3, 2, 2)

    assert adata.shape == (3, 2, 2)
    assert adata.n_obs == 3
    assert adata.n_vars == 2
    assert adata.n_t == 2


def test_ehrdata_init_vanilla_r_add_X():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame({"obs1": [1, 2, 3]})
    var = pd.DataFrame({"var1": [1, 2]})
    r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

    adata = EHRData(obs=obs, r=r, var=var)
    adata.X = X  # TODO: now wrong shape of AnnData..?
    assert adata.X is not None
    assert adata.obs is not None
    assert adata.var is not None
    assert adata.layers is not None
    assert adata.layers[R_LAYER_KEY] is not None

    assert hasattr(adata, "t")
    assert adata.t is not None

    assert adata.X.shape == (3, 2)
    assert adata.obs.shape == (3, 1)
    assert adata.var.shape == (2, 1)
    assert adata.layers[R_LAYER_KEY].shape == (3, 2, 2)

    assert adata.shape == (3, 2, 2)
    assert adata.n_obs == 3
    assert adata.n_vars == 2
    assert adata.n_t == 2


def test_ehrdata_init_vanilla_X_add_r():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame({"obs1": [1, 2, 3]})
    var = pd.DataFrame({"var1": [1, 2]})
    r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

    adata = EHRData(X=X, obs=obs, var=var)
    adata.r = r
    assert adata.X is not None
    assert adata.obs is not None
    assert adata.var is not None
    assert adata.layers is not None
    assert adata.layers[R_LAYER_KEY] is not None

    assert hasattr(adata, "t")
    assert adata.t is not None

    assert adata.X.shape == (3, 2)
    assert adata.obs.shape == (3, 1)
    assert adata.var.shape == (2, 1)
    assert adata.layers[R_LAYER_KEY].shape == (3, 2, 2)

    assert adata.shape == (3, 2, 2)
    assert adata.n_obs == 3
    assert adata.n_vars == 2
    assert adata.n_t == 2


def test_ehrdata_init_fail_r_2D():
    r = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        EHRData(r=r)


def test_ehrdata_X_add_r_2D():
    X = np.array([[1, 2], [3, 4]])
    r = np.array([[1, 2], [3, 4]])
    adata = EHRData(X=X)
    with pytest.raises(ValueError):
        adata.r = r


def test_ehrdata_init_fail_X_and_r_mismatch():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame({"obs1": [1, 2, 3]})
    var = pd.DataFrame({"var1": [1, 2]})
    r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    with pytest.raises(ValueError):
        EHRData(X=X, obs=obs, r=r, var=var)


def test_ehrdata_init_fail_r_and_layer_R_LAYER_KEY_exists():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame({"obs1": [1, 2, 3]})
    var = pd.DataFrame({"var1": [1, 2]})
    r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    layers = {R_LAYER_KEY: r}

    with pytest.raises(ValueError):
        EHRData(X=X, obs=obs, r=r, layers=layers, var=var)


def test_ehrdata_init_fail_X_and_t_no_r():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame({"obs1": [1, 2, 3]})
    var = pd.DataFrame({"var1": [1, 2]})
    t = pd.DataFrame({"t1": [1, 2]})

    with pytest.raises(ValueError):
        EHRData(X=X, obs=obs, t=t, var=var)


def test_ehrdata_init_fail_r_and_t_mismatch():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame({"obs1": [1, 2, 3]})
    var = pd.DataFrame({"var1": [1, 2]})
    r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    t = pd.DataFrame({"t1": [1]})

    with pytest.raises(ValueError):
        EHRData(X=X, obs=obs, r=r, t=t, var=var)


def test_ehrdata_del_r():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    adata = EHRData(X=X, r=r)
    del adata.r

    assert adata.r is None
    assert adata.t is None

    assert adata.shape == (3, 2)
    assert adata.n_obs == 3
    assert adata.n_vars == 2
    assert adata.n_t is None


# def test_fail_set_t_invalid_shape():
#     r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
#     t = pd.DataFrame({"t1": [1, 2, 3]})
#     adata = EHRData(r=r)
#     with pytest.raises(ValueError):
#         adata.t = t


def test_fail_set_t_to_None():
    r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    adata = EHRData(r=r)
    with pytest.raises(ValueError):
        adata.t = None


def test_ehrdata_slice_2D_vanilla():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame({"obs1": [1, 2, 3]})
    var = pd.DataFrame({"var1": [1, 2]})
    r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

    adata = EHRData(X=X, obs=obs, r=r, var=var)
    adata_sliced = adata[:2, :1]

    assert adata_sliced.is_view
    assert adata_sliced.shape[0] == 2
    assert adata_sliced.shape[1] == 1
    assert adata_sliced.r.shape == (2, 1, 2)


def test_ehrdata_slice_2D_repeated():
    pass


def test_ehrdata_slice_3D_vanilla():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame({"obs1": [1, 2, 3]})
    var = pd.DataFrame({"var1": [1, 2]})
    t = pd.DataFrame({"t1": [1, 2]})
    r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

    adata = EHRData(X=X, obs=obs, r=r, var=var, t=t)
    adata_sliced = adata[:2, :1, :1]

    assert adata_sliced.is_view
    assert adata_sliced.shape[0] == 2
    assert adata_sliced.shape[1] == 1
    assert adata_sliced.r.shape == (2, 1, 1)
    assert adata_sliced.t.shape == (1, 1)


def test_ehrdata_slice_3D_repeated():
    pass


def test_copy():
    edata = EHRData()
    edata.copy()


def test_repr():
    pass


# TODO: test that r has shape 3

# TODO:
# - test that r is an arrayview?
# - test that t is a dataframeview?
# - test multiple sliceing?
# - allow explictly for r to be only a numpy array or dask array?
# - test that r and t are aligned
# - test repr
