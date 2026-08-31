from importlib.util import find_spec
from typing import TYPE_CHECKING, Literal

import h5py
import numpy as np
import scipy
import scipy.sparse as sp
import sparse
import zarr

if TYPE_CHECKING:
    # can only see core.Array
    from dask.array.core import Array as DaskArray
elif find_spec("dask"):
    from dask.array import Array as DaskArray
else:
    DaskArray = type("Array", (), {"__module__": "dask.array"})

from fast_array_utils.conv import to_dense
from numpy import ma

# from anndata.abc import CSCDataset, CSRDataset
# `h5py` and `zarr` are used directly: both are hard dependencies (h5py via anndata), so
# aliases would only rename them. These names used to come from the private `anndata.compat`,
# which dropped the storage aliases in anndata 0.13.3
# (see https://github.com/scverse/cellrank/issues/1377).
from ehrdata._compat import CupyArray, CupySparseMatrix, ZappyArray

type ArrayStorageType = zarr.Array | h5py.Dataset
type GroupStorageType = zarr.Group | h5py.Group
type StorageType = ArrayStorageType | GroupStorageType

CSMatrix = scipy.sparse.csr_matrix | scipy.sparse.csc_matrix
CSArray = scipy.sparse.csr_array | scipy.sparse.csc_array

type XDataType = (
    np.ndarray
    | ma.MaskedArray
    | CSMatrix
    | CSArray
    | sparse.COO
    | h5py.Dataset
    | zarr.Array
    | ZappyArray
    # | CSRDataset
    # | CSCDataset
    | DaskArray
    | CupyArray
    | CupySparseMatrix
)

EHRDataElem = Literal[
    "obs",
    "var",
    "tem",
    "obsm",
    "varm",
    "obsp",
    "varp",
    "layers",
    "X",
    "raw",
    "uns",
]

Join_T = Literal["inner", "outer"]


# as in ehrapy
def asarray(a):
    """Convert input to a dense NumPy array in CPU memory using fast-array-utils."""
    return to_dense(a, to_cpu_memory=True)


# as in ehrapy
def as_dense_dask_array(a, chunk_size=1000):
    """Convert input to a dense Dask array."""
    import dask.array as da

    return da.from_array(a, chunks=chunk_size)


# as in ehrapy
ARRAY_TYPES_NUMERIC = (
    asarray,
    as_dense_dask_array,
    sp.csr_array,
    sp.csc_array,
    sparse.COO.from_numpy,
)
ARRAY_TYPES_NUMERIC_3D_ABLE = (asarray, as_dense_dask_array, sparse.COO.from_numpy)
ARRAY_TYPES_NONNUMERIC = (asarray, as_dense_dask_array)
