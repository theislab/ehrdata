from typing import Literal

import numpy as np
import scipy

# from anndata.abc import CSCDataset, CSRDataset
from anndata.compat import CupyArray, CupySparseMatrix, DaskArray, H5Array, H5Group, ZarrArray, ZarrGroup
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

type RDataType = np.ndarray | scipy.sparse.coo_array | DaskArray

EHRDataElem = Literal[
    "obs",
    "var",
    "t",
    "obsm",
    "varm",
    "obsp",
    "varp",
    "layers",
    "X",
    "R",
    "raw",
    "uns",
]

Join_T = Literal["inner", "outer"]
