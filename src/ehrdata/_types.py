from typing import Literal

import numpy as np
import scipy
import scipy.sparse as sp

# from anndata.abc import CSCDataset, CSRDataset
from anndata.compat import CupyArray, CupySparseMatrix, DaskArray, H5Array, H5Group, ZarrArray, ZarrGroup
from fast_array_utils.conv import to_dense
from numpy import ma

from ehrdata._compat import ZappyArray

type ArrayStorageType = ZarrArray | H5Array
type GroupStorageType = ZarrGroup | H5Group
type StorageType = ArrayStorageType | GroupStorageType

CSMatrix = scipy.sparse.csr_matrix | scipy.sparse.csc_matrix
CSArray = scipy.sparse.csr_array | scipy.sparse.csc_array

type XDataType = (
    np.ndarray
    | ma.MaskedArray
    | CSMatrix
    | CSArray
    | H5Array
    | ZarrArray
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
)  # add coo_array once supported in AnnData
ARRAY_TYPES_NUMERIC_3D_ABLE = (asarray, as_dense_dask_array)  # add coo_array once supported in AnnData
ARRAY_TYPES_NONNUMERIC = (asarray, as_dense_dask_array)
