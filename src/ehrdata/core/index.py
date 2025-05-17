from collections.abc import Iterable
from functools import singledispatch

import numpy as np
import pandas as pd
from anndata.compat import Index


@singledispatch
def _subset(a, subset_idx):
    msg = f"Subsetting of type {type(a)} in field .r is not implemented."
    raise NotImplementedError(msg)


@_subset.register(np.ndarray)
@_subset.register(pd.DataFrame)
def _(a: np.ndarray | pd.DataFrame, subset_idx: Index):
    # Select as combination of indexes, not coordinates
    # Correcting for indexing behaviour of np.ndarray
    if all(isinstance(x, Iterable) for x in subset_idx):
        subset_idx = np.ix_(*subset_idx)
    return a[subset_idx]
