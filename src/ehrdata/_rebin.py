from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import sparse

from ehrdata._types import DaskArray

if TYPE_CHECKING:
    from ehrdata import EHRData

REBIN_AGGREGATION_STRATEGIES = ("last", "first", "mean", "median", "min", "max", "sum", "count")


def rebin(
    edata: EHRData,
    bin_size: int,
    *,
    aggregation_strategy: Literal["last", "first", "mean", "median", "min", "max", "sum", "count"] = "last",
) -> EHRData:
    """Aggregate the time axis of an :class:`~ehrdata.EHRData` object into coarser intervals.

    Every `bin_size` consecutive timepoints are combined into one, for instance hourly intervals into 6-hourly ones with `bin_size=6`.
    If the number of timepoints is not a multiple of `bin_size`, the last interval combines the remaining timepoints.
    The aggregation applies to `.X` and every layer with a time axis, while 2D layers and all annotations are kept.

    Missing values (`NaN`) are ignored, and an interval without any observed value is missing.
    `"last"` and `"first"` keep the latest and the earliest observed value, and `"count"` is the number of observed values.

    In `.tem`, the `interval_start_offset` and `time_value` of a new interval are the ones of its first timepoint, and the `interval_end_offset` the one of its last timepoint.
    An `interval_step` column is renumbered, and other columns of `.tem` are dropped.

    Args:
        edata: Central data object.
        bin_size: Number of consecutive timepoints combined into one interval.
        aggregation_strategy: Strategy to use when aggregating the values of a variable within one interval.

    Returns:
        A new :class:`~ehrdata.EHRData` object with the aggregated time axis.

    Examples:
        >>> import ehrdata as ed
        >>> import numpy as np
        >>> X = np.array([[[1.0, 2.0, np.nan, np.nan, 5.0]]])
        >>> edata = ed.EHRData(X)
        >>> edata_rebinned = ed.rebin(edata, 2, aggregation_strategy="mean")
        >>> edata_rebinned.shape
        (1, 1, 3)
        >>> edata_rebinned.X
        array([[[1.5, nan, 5. ]]])
    """
    from ehrdata import EHRData

    if aggregation_strategy not in REBIN_AGGREGATION_STRATEGIES:
        msg = f"aggregation_strategy must be one of {REBIN_AGGREGATION_STRATEGIES}."
        raise ValueError(msg)
    if not isinstance(bin_size, int | np.integer) or bin_size < 1:
        msg = "bin_size must be a positive integer."
        raise ValueError(msg)

    layers = dict(edata.layers.items())
    time_keys = [key for key, value in layers.items() if value.ndim == 3]
    if not time_keys:
        msg = "Neither .X nor any layer has a time axis."
        raise ValueError(msg)
    for key in time_keys:
        layers[key] = _rebin_array(layers[key], bin_size, aggregation_strategy)

    return EHRData(
        X=layers.pop(None, None),
        layers=layers,
        obs=edata.obs.copy(),
        var=edata.var.copy(),
        tem=_rebin_tem(edata.tem, bin_size),
        uns=edata.uns.copy(),
        obsm=dict(edata.obsm.items()),
        varm=dict(edata.varm.items()),
        obsp=dict(edata.obsp.items()),
        varp=dict(edata.varp.items()),
    )


def _rebin_tem(tem: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    n_t = len(tem)
    starts = np.arange(0, n_t, bin_size)
    ends = np.minimum(starts + bin_size, n_t) - 1
    rebinned = pd.DataFrame(index=pd.Index(np.arange(len(starts)).astype(str), name=tem.index.name))
    if "interval_step" in tem.columns:
        rebinned["interval_step"] = np.arange(len(starts))
    for column, positions in (("interval_start_offset", starts), ("interval_end_offset", ends), ("time_value", starts)):
        if column in tem.columns:
            rebinned[column] = tem[column].to_numpy()[positions]
    return rebinned


@singledispatch
def _rebin_array(a, bin_size: int, aggregation_strategy: str):
    msg = f"Rebinning arrays of type {type(a)} is not implemented."
    raise NotImplementedError(msg)


@_rebin_array.register(np.ndarray)
def _(a: np.ndarray, bin_size: int, aggregation_strategy: str) -> np.ndarray:
    dtype = np.result_type(a.dtype, np.float32)
    n_t = a.shape[-1]
    n_bins = -(-n_t // bin_size)
    padded = np.full((*a.shape[:-1], n_bins * bin_size), np.nan, dtype=dtype)
    padded[..., :n_t] = a
    binned = padded.reshape(*a.shape[:-1], n_bins, bin_size)
    observed = ~np.isnan(binned)
    count = observed.sum(axis=-1)

    if aggregation_strategy == "count":
        result = count.astype(dtype)
    elif aggregation_strategy in ("sum", "mean"):
        result = np.where(observed, binned, 0).sum(axis=-1)
        if aggregation_strategy == "mean":
            result = result / np.maximum(count, 1)
    elif aggregation_strategy == "min":
        result = np.where(observed, binned, np.inf).min(axis=-1)
    elif aggregation_strategy == "max":
        result = np.where(observed, binned, -np.inf).max(axis=-1)
    elif aggregation_strategy == "median":
        ordered = np.sort(binned, axis=-1)
        lower = np.take_along_axis(ordered, np.maximum(count - 1, 0)[..., None] // 2, axis=-1)
        upper = np.take_along_axis(ordered, count[..., None] // 2, axis=-1)
        result = ((lower + upper) / 2)[..., 0]
    else:
        position = (
            observed.argmax(axis=-1)
            if aggregation_strategy == "first"
            else bin_size - 1 - observed[..., ::-1].argmax(axis=-1)
        )
        result = np.take_along_axis(binned, position[..., None], axis=-1)[..., 0]

    result = result.astype(dtype)
    result[count == 0] = np.nan
    return result


@_rebin_array.register(sparse.COO)
def _(a: sparse.COO, bin_size: int, aggregation_strategy: str) -> sparse.COO:
    lead_shape, n_t = a.shape[:-1], a.shape[-1]
    fibers, fiber_idx = np.unique(np.ravel_multi_index(a.coords[:-1], lead_shape), return_inverse=True)
    dense = np.full((len(fibers), n_t), a.fill_value, dtype=a.dtype)
    dense[fiber_idx, a.coords[-1]] = a.data
    values = _rebin_array(dense, bin_size, aggregation_strategy)
    fill = _rebin_array(np.full((1, n_t), a.fill_value, dtype=a.dtype), bin_size, aggregation_strategy)[0]
    fill_value = fill[0]

    keep = ~_equal_nan(values, fill_value)
    fiber_rows, bins = np.nonzero(keep)
    flat, data = [fibers[fiber_rows]], [values[keep]]
    flat_bins = [bins]
    deviating_bins = np.flatnonzero(~_equal_nan(fill, fill_value))
    unstored = np.setdiff1d(np.arange(np.prod(lead_shape, dtype=int)), fibers) if len(deviating_bins) else None
    for b in deviating_bins:
        flat.append(unstored)
        data.append(np.full(len(unstored), fill[b], dtype=values.dtype))
        flat_bins.append(np.full(len(unstored), b))

    coords = np.vstack([*np.unravel_index(np.concatenate(flat), lead_shape), np.concatenate(flat_bins)])
    return sparse.COO(
        coords, np.concatenate(data), shape=(*lead_shape, len(fill)), fill_value=fill_value, has_duplicates=False
    )


@_rebin_array.register(DaskArray)
def _(a: DaskArray, bin_size: int, aggregation_strategy: str) -> DaskArray:
    a = a.rechunk({a.ndim - 1: -1})
    n_bins = -(-a.shape[-1] // bin_size)
    dtype = np.result_type(a.dtype, np.float32)
    return a.map_blocks(
        _rebin_array,
        bin_size,
        aggregation_strategy,
        chunks=(*a.chunks[:-1], (n_bins,)),
        dtype=dtype,
        meta=a._meta.astype(dtype),
    )


def _equal_nan(a: np.ndarray, b: float | np.ndarray) -> np.ndarray:
    return (a == b) | (np.isnan(a) & np.isnan(b))
