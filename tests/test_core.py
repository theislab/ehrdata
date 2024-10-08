import numpy as np
import pandas as pd

from ehrdata import EHRData
from ehrdata.core.constants import R_LAYER_KEY


def test_ehrdata_init_empty():
    edata = EHRData()
    assert edata.r is None


def test_ehrdata_init_standard():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame({"obs1": [1, 2, 3]})
    var = pd.DataFrame({"var1": [1, 2]})
    adata = EHRData(X=X, obs=obs, var=var)

    assert adata.X is not None
    assert adata.obs is not None
    assert adata.var is not None

    assert adata.shape[0] == 3
    assert adata.shape[1] == 2


def test_ehrdata_init_standard_and_r():
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

    assert adata.shape[0] == 3
    assert adata.shape[1] == 2


def test_ehrdata_slice_2D():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame({"obs1": [1, 2, 3]})
    var = pd.DataFrame({"var1": [1, 2]})
    r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

    adata = EHRData(X=X, obs=obs, r=r, var=var)
    adata_sliced = adata[:2, :1]

    assert adata_sliced.shape[0] == 2
    assert adata_sliced.shape[1] == 1
    assert adata_sliced.layers[R_LAYER_KEY].shape == (2, 1, 2)


def test_ehrdata_slice_3D():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    obs = pd.DataFrame({"obs1": [1, 2, 3]})
    var = pd.DataFrame({"var1": [1, 2]})
    r = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

    adata = EHRData(X=X, obs=obs, r=r, var=var)
    adata_sliced = adata[:2, :1, :1]

    assert adata_sliced.shape[0] == 2
    assert adata_sliced.shape[1] == 1
    assert adata_sliced.layers[R_LAYER_KEY].shape == (2, 1, 1)
    assert adata_sliced.t.shape == (1, 0)


def test_copy():
    edata = EHRData()
    edata.copy()
