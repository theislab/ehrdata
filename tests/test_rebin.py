from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import sparse

import ehrdata as ed
from ehrdata import EHRData
from ehrdata._types import ARRAY_TYPES_NUMERIC_3D_ABLE
from ehrdata.io.omop._queries import _generate_timedeltas

if TYPE_CHECKING:
    from collections.abc import Callable

nan = np.nan

X = np.array(
    [
        [[1, 2, 6, nan, 5], [nan, 4, nan, 7, nan], [0, 0, 0, 0, 0]],
        [[0, nan, 3, 2, 1], [nan, nan, nan, nan, nan], [0, 0, 0, 0, 0]],
    ]
)

EXPECTED = {
    "last": [[[6, 5], [4, 7], [0, 0]], [[3, 1], [nan, nan], [0, 0]]],
    "first": [[[1, 5], [4, 7], [0, 0]], [[0, 2], [nan, nan], [0, 0]]],
    "mean": [[[3, 5], [4, 7], [0, 0]], [[1.5, 1.5], [nan, nan], [0, 0]]],
    "median": [[[2, 5], [4, 7], [0, 0]], [[1.5, 1.5], [nan, nan], [0, 0]]],
    "min": [[[1, 5], [4, 7], [0, 0]], [[0, 1], [nan, nan], [0, 0]]],
    "max": [[[6, 5], [4, 7], [0, 0]], [[3, 2], [nan, nan], [0, 0]]],
    "sum": [[[9, 5], [4, 7], [0, 0]], [[3, 3], [nan, nan], [0, 0]]],
    "count": [[[3, 1], [1, 1], [3, 2]], [[2, 2], [nan, nan], [3, 2]]],
}


def _coo_nan_fill(x: np.ndarray) -> sparse.COO:
    return sparse.COO.from_numpy(x, fill_value=nan)


def _dask_coo(x: np.ndarray) -> da.Array:
    return da.from_array(sparse.COO.from_numpy(x, fill_value=nan), chunks=(1, 2, 2), asarray=False)


def _dask_chunked(x: np.ndarray) -> da.Array:
    return da.from_array(x, chunks=(1, 2, 2))


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, da.Array):
        x = x.compute()
    if isinstance(x, sparse.COO):
        x = x.todense()
    return x


@pytest.mark.parametrize("array_type", [*ARRAY_TYPES_NUMERIC_3D_ABLE, _coo_nan_fill, _dask_chunked, _dask_coo])
@pytest.mark.parametrize("aggregation_strategy", EXPECTED)
def test_rebin_strategies(array_type: Callable, aggregation_strategy: str):
    edata = EHRData(array_type(X), layers={"other": array_type(X)})

    rebinned = ed.rebin(edata, 3, aggregation_strategy=aggregation_strategy)

    assert rebinned.shape == (2, 3, 2)
    for key in (None, "other"):
        assert type(rebinned.layers[key]) is type(edata.layers[key])
        if isinstance(edata.layers[key], da.Array):
            assert type(rebinned.layers[key]._meta) is type(edata.layers[key]._meta)
        np.testing.assert_array_equal(_to_numpy(rebinned.layers[key]), np.array(EXPECTED[aggregation_strategy]))


@pytest.mark.parametrize("array_type", [sparse.COO.from_numpy, _coo_nan_fill])
def test_rebin_sparse_fill_value(array_type: Callable):
    edata = EHRData(array_type(X))

    rebinned = ed.rebin(edata, 3, aggregation_strategy="sum")

    assert rebinned.X.fill_value == edata.X.fill_value or np.isnan(rebinned.X.fill_value)
    assert rebinned.X.nnz < np.prod(rebinned.shape)


def test_rebin_int():
    edata = EHRData(np.arange(10).reshape(1, 2, 5))

    rebinned = ed.rebin(edata, 2, aggregation_strategy="mean")

    np.testing.assert_array_equal(rebinned.X, [[[0.5, 2.5, 4], [5.5, 7.5, 9]]])


def test_rebin_bin_size_one_and_larger_than_n_t():
    edata = EHRData(X)

    np.testing.assert_array_equal(ed.rebin(edata, 1).X, X)
    np.testing.assert_array_equal(
        ed.rebin(edata, 10, aggregation_strategy="max").X, [[[6], [7], [0]], [[3], [nan], [0]]]
    )


def test_rebin_tem_offsets():
    tem = _generate_timedeltas(1, "h", 5).set_index("interval_step")
    tem.index = tem.index.astype(str)
    tem = tem.astype(str)
    tem["other"] = "x"
    edata = EHRData(X, tem=tem)

    rebinned = ed.rebin(edata, 3)

    expected = pd.DataFrame(
        {
            "interval_start_offset": ["0 days 00:00:00", "0 days 03:00:00"],
            "interval_end_offset": ["0 days 03:00:00", "0 days 05:00:00"],
        },
        index=pd.Index(["0", "1"], name="interval_step"),
    )
    pd.testing.assert_frame_equal(rebinned.tem, expected)


def test_rebin_tem_interval_step():
    edata = EHRData(X, tem=pd.DataFrame({"interval_step": np.arange(5)}))

    rebinned = ed.rebin(edata, 2)

    pd.testing.assert_frame_equal(
        rebinned.tem, pd.DataFrame({"interval_step": np.arange(3)}, index=pd.Index(["0", "1", "2"]))
    )


def test_rebin_keeps_annotations():
    edata = EHRData(
        X,
        obs=pd.DataFrame({"age": [30, 40]}, index=["p0", "p1"]),
        var=pd.DataFrame(index=["a", "b", "c"]),
        layers={"static": np.ones((2, 3))},
        obsm={"embedding": np.ones((2, 2))},
        uns={"key": "value"},
    )

    rebinned = ed.rebin(edata, 2)

    pd.testing.assert_frame_equal(rebinned.obs, edata.obs)
    pd.testing.assert_frame_equal(rebinned.var, edata.var)
    np.testing.assert_array_equal(rebinned.layers["static"], edata.layers["static"])
    np.testing.assert_array_equal(rebinned.obsm["embedding"], edata.obsm["embedding"])
    assert rebinned.uns["key"] == "value"
    assert edata.shape == (2, 3, 5)


def test_rebin_errors():
    with pytest.raises(ValueError, match="aggregation_strategy"):
        ed.rebin(EHRData(X), 2, aggregation_strategy="mode")
    with pytest.raises(ValueError, match="bin_size"):
        ed.rebin(EHRData(X), 0)
    with pytest.raises(ValueError, match="time axis"):
        ed.rebin(EHRData(np.ones((2, 3))), 2)
