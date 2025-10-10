from typing import Literal, TypeAlias

import numpy as np
import scipy

# from anndata.abc import CSCDataset, CSRDataset
from anndata.compat import CupyArray, CupySparseMatrix, DaskArray, H5Array, H5Group, ZarrArray, ZarrGroup
from numpy import ma
from sparse import COO

from ehrdata._compat import ZappyArray

ArrayStorageType: TypeAlias = ZarrArray | H5Array
GroupStorageType: TypeAlias = ZarrGroup | H5Group
StorageType: TypeAlias = ArrayStorageType | GroupStorageType

CSMatrix = scipy.sparse.csr_matrix | scipy.sparse.csc_matrix
CSArray = scipy.sparse.csr_array | scipy.sparse.csc_array

XDataType: TypeAlias = (
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

RDataType: TypeAlias = np.ndarray | COO | DaskArray

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
