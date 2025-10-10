from __future__ import annotations

from functools import cache
from importlib.util import find_spec
from types import EllipsisType
from typing import TYPE_CHECKING
from warnings import warn

import h5py
import numpy as np
import pandas as pd
import scipy

#############################
# scipy sparse array compat #
#############################


CSMatrix = scipy.sparse.csr_matrix | scipy.sparse.csc_matrix
CSArray = scipy.sparse.csr_array | scipy.sparse.csc_array


class Empty:
    pass


Index1D = slice | int | str | np.int64 | np.ndarray | pd.Series
IndexRest = Index1D | EllipsisType
Index = (
    IndexRest
    | tuple[Index1D, IndexRest]
    | tuple[IndexRest, Index1D]
    | tuple[Index1D, Index1D, EllipsisType]
    | tuple[EllipsisType, Index1D, Index1D]
    | tuple[Index1D, EllipsisType, Index1D]
    | CSMatrix
    | CSArray
)
H5Group = h5py.Group
H5Array = h5py.Dataset
H5File = h5py.File


#############################
# Optional deps
#############################
@cache
def is_zarr_v2() -> bool:
    import zarr
    from packaging.version import Version

    return Version(zarr.__version__) < Version("3.0.0")


if is_zarr_v2():
    msg = "anndata will no longer support zarr v2 in the near future. Please prepare to upgrade to zarr>=3."
    warn(msg, DeprecationWarning, stacklevel=2)


if find_spec("awkward") or TYPE_CHECKING:
    import awkward  # noqa: F401
    from awkward import Array as AwkArray
else:

    class AwkArray:
        @staticmethod
        def __repr__():
            return "mock awkward.highlevel.Array"


if find_spec("zappy") or TYPE_CHECKING:
    from zappy.base import ZappyArray
else:

    class ZappyArray:
        @staticmethod
        def __repr__():
            return "mock zappy.base.ZappyArray"


if TYPE_CHECKING:
    # type checkers are confused and can only see …core.Array
    from dask.array.core import Array as DaskArray
elif find_spec("dask"):
    from dask.array import Array as DaskArray
else:

    class DaskArray:
        @staticmethod
        def __repr__():
            return "mock dask.array.core.Array"


# https://github.com/scverse/anndata/issues/1749
def is_cupy_importable() -> bool:
    try:
        import cupy  # noqa: F401
    except ImportError:
        return False
    return True


if is_cupy_importable() or TYPE_CHECKING:
    from cupy import ndarray as CupyArray
    from cupyx.scipy.sparse import csc_matrix as CupyCSCMatrix
    from cupyx.scipy.sparse import csr_matrix as CupyCSRMatrix
    from cupyx.scipy.sparse import spmatrix as CupySparseMatrix

    try:
        import dask.array as da
    except ImportError:
        pass
    else:
        da.register_chunk_type(CupyCSRMatrix)
        da.register_chunk_type(CupyCSCMatrix)
else:

    class CupySparseMatrix:
        @staticmethod
        def __repr__():
            return "mock cupyx.scipy.sparse.spmatrix"

    class CupyCSRMatrix:
        @staticmethod
        def __repr__():
            return "mock cupyx.scipy.sparse.csr_matrix"

    class CupyCSCMatrix:
        @staticmethod
        def __repr__():
            return "mock cupyx.scipy.sparse.csc_matrix"

    class CupyArray:
        @staticmethod
        def __repr__():
            return "mock cupy.ndarray"


def lazy_import_torch() -> None:
    try:
        import torch

        return torch
    except ImportError:
        msg = "The optional module 'torch' is not installed. Please install it using 'pip install ehrdata[torch]'."
        raise ImportError(msg) from None
