from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar

import numpy as np
import pandas as pd
from anndata import AnnData
from anndata._core.aligned_mapping import (
    AlignedMapping,
    AlignedMappingProperty,
    AlignedView,
    Layers,
    LayersBase,
    Value,
)
from anndata._core.index import _subset
from anndata._core.views import (
    DataFrameView,
    ElementRef,
    _resolve_idx,
    as_view,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from os import PathLike

    from anndata._core.index import Index as ADIndex
    from anndata._core.index import Index1D

    from ehrdata._types import XDataType

    Index: TypeAlias = ADIndex | tuple[Index1D, Index1D, Index1D]

T = TypeVar("T", bound=AlignedMapping)


class AlignedView3D(AlignedView):
    """AlignedView for 3D data."""

    def __getitem__(self, key: str) -> Value:
        # this is a hack to allow __getitem to work for 2D and 3D data
        subset_idx = self.subset_idx[:2] if self.parent_mapping[key].ndim == 2 else self.subset_idx
        return as_view(
            _subset(self.parent_mapping[key], subset_idx),
            ElementRef(self.parent, self.attrname, (key,)),
        )


class LayersView3D(AlignedView3D, LayersBase):
    """LayersView for 3D data."""


# overwrite the view class of LayersBase allows to use the required __getitem__ method of AlignedView3D
LayersBase._view_class = LayersView3D


@dataclass
class AlignedMappingProperty3D(AlignedMappingProperty):
    """A :class:`property` that subclasses the AlignedMappingproperty of AnnData.

     The subclass is used to add the _tidx for subsetting,
    and overridethe axes of the AlignedMapping in __get__.
    """

    def __get__(self, obj: None | AnnData, objtype: type | None = None) -> T:
        if obj is None:
            # When accessed from the class, e.g. via `AnnData.obs`,
            # this needs to return a `property` instance, e.g. for Sphinx
            return self  # type: ignore
        if not obj.is_view:
            return self.construct(obj, store=getattr(obj, f"_{self.name}"))
        parent_anndata = obj._adata_ref
        idxs = (obj._oidx, obj._vidx, obj._tidx)
        parent: AlignedMapping = getattr(parent_anndata, self.name)
        return parent._view(obj, (tuple(idxs[ax] for ax in (0, 1, 2))))


def _get_array_3d_dim(X: XDataType | None) -> int:
    """Get the 3rd dimension of an array."""
    if X is not None and len(X.shape) == 3 and X.shape[2] > 1:
        return X.shape[2]

    else:
        return 1


def _get_layers_3d_dim(layers: Mapping[str, Any] | None) -> int:
    """Get the 3rd dimension consensus of all arrays in the layers.

    All arrays in layers need to match on the first two axes, which is checked in AnnData.
    Further, all arrays in layers need to have the same 3rd axis dimension, unless they are 2D.
    """
    if layers is None or len(layers) == 0:
        return 1

    else:
        shape_3d = {}

        for key, value in layers.items():
            shape_3d[key] = _get_array_3d_dim(value)

        shape_3d_values = set(shape_3d.values())

        if len(shape_3d_values) == 1:
            return shape_3d_values.pop()

        elif 1 in shape_3d_values and len(shape_3d_values) == 2:
            shape_3d_values.discard(1)
            return shape_3d_values.pop()

        else:
            msg = f"The 3rd dimension of layers is not consistent: {', '.join(f'{key} : {value}' for key, value in shape_3d.items())}."

            raise ValueError(msg)


class EHRData(AnnData):
    """Model two and three dimensional electronic health record data.

    .. figure:: ../../_static/tutorial_images/logo.png
       :width: 260px
       :align: right
       :class: dark-light

    Extends :class:`~anndata.AnnData` to further support regular and irregular time-series data.

    Args:
        X: A #observations × #variables data array. A view of the data is used if the
            data type matches, otherwise, a copy is made.
        obs: Key-indexed one-dimensional observations annotation of length #observations.
        var: Key-indexed one-dimensional variables annotation of length #variables.
        tem: Key-indexed one-dimensional time annotation of length #timesteps.
        uns: Key-indexed unstructured annotation.
        obsm: Key-indexed multi-dimensional observations annotation of length #observations.
            If passing a :class:`numpy.ndarray`, it needs to have a structured datatype.
        varm: Key-indexed multi-dimensional variables annotation of length #variables.
            If passing a :class:`numpy.ndarray`, it needs to have a structured datatype.
        layers: Key-indexed multi-dimensional #observations × #variables × #timesteps data arrays, aligned to dimensions of `X`.
        shape: Shape tuple (#observations, #variables). Can only be provided if `X` is None.
        filename: Name of backing file. See :class:`h5py.File`.
        filemode: Open mode of backing file. See :class:`h5py.File`.
    """

    _t: pd.DataFrame | None
    _n_t: int

    layers: AlignedMappingProperty3D = AlignedMappingProperty3D("layers", Layers)

    def __init__(
        self,
        X: XDataType | pd.DataFrame | None = None,
        *,
        obs: pd.DataFrame | Mapping[str, Iterable[Any]] | None = None,
        var: pd.DataFrame | Mapping[str, Iterable[Any]] | None = None,
        tem: pd.DataFrame | None = None,
        uns: Mapping[str, Any] | None = None,
        obsm: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        varm: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        layers: Mapping[str, np.ndarray] | None = None,
        raw: Mapping[str, Any] | None = None,
        dtype: np.dtype | type | str | None = None,
        shape: tuple[int, int] | None = None,
        filename: PathLike[str] | str | None = None,
        filemode: Literal["r", "r+"] | None = None,
        asview: bool = False,
        obsp: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        varp: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        oidx: Index1D | None = None,
        vidx: Index1D | None = None,
        tidx: Index1D | None = None,
    ):
        self._tidx = None
        self._n_t = _get_layers_3d_dim(layers)

        super().__init__(
            X=X,
            obs=obs,
            var=var,
            uns=uns,
            obsm=obsm,
            varm=varm,
            layers=layers,
            raw=raw,
            dtype=dtype,
            shape=shape,
            filename=filename,
            filemode=filemode,
            asview=asview,
            obsp=obsp,
            varp=varp,
            oidx=oidx,
            vidx=vidx,
        )

        self._tidx = tidx

        if tem is None:
            self.tem = pd.DataFrame(index=pd.RangeIndex(self.n_t).astype(str))
        else:
            self.tem = tem

    @classmethod
    def from_adata(
        cls,
        adata: AnnData,
        *,
        tem: pd.DataFrame | None = None,
        tidx: slice | None = None,
    ) -> EHRData:
        """Create an EHRData object from an AnnData object.

        Args:
            adata: Annotated data object.
            tem: Time dataframe for describing third axis, see tem attribute.
            tidx: A slice for the 3rd dimension R. Usually, this will be None here.

        Returns:
            An EHRData object extending the AnnData object.
        """
        instance = cls(shape=adata.shape)

        if adata.is_view:
            # use _init_as_view of adata, but don't subset since already sliced in __getitem__
            instance._init_as_view(adata, slice(None), slice(None))
            # The tidx is not part of AnnData, so we need to set it separately. Setting it is required for the getter of r
            instance._tidx = tidx
            # _n_t is not part of AnnData, so need to set it separately
            instance._n_t = 1 if tem is None else len(tem)

            # tem is not part of AnnData, so we need to set it separately
            if tem is not None:
                instance.tem = DataFrameView(tem, tem.index)

        else:
            # For actual objects, initialize normally
            instance._init_as_actual(
                X=adata.X,
                obs=adata.obs,
                var=adata.var,
                uns=adata.uns,
                obsm=adata.obsm,
                varm=adata.varm,
                obsp=adata.obsp,
                varp=adata.varp,
                raw=adata.raw,
                layers=adata.layers,
                shape=adata.shape if adata.X is None else None,
                filename=adata.filename,
                filemode=adata.file._filemode,
            )

            if tem is not None:
                instance.tem = tem

        return instance

    @property
    def tem(self) -> pd.DataFrame:
        """Time dataframe for describing third axis."""
        return self._tem

    @tem.setter
    def tem(self, input: pd.DataFrame) -> None:
        if not isinstance(input, pd.DataFrame):
            msg = "Can only assign pd.DataFrame to tem."
            raise ValueError(msg)
        if self.n_t != len(input):
            msg = f"Length of passed value for tem is {len(input)}, but this EHRData has shape: {self.shape}"
            raise ValueError(msg)

        self._tem = input
        self._n_t = len(input)

    @property
    def n_t(self) -> int:
        """Number of time points."""
        return self._n_t if hasattr(self, "_n_t") else 1

    @property
    def _tidx(self) -> slice | None:
        return self.__tidx

    @_tidx.setter
    def _tidx(self, input) -> None:
        self.__tidx = input

    @property
    def X(self):
        """Data matrix."""
        return super().X

    @X.setter
    def X(self, value):
        # this is a bit hacky, but anndata checks its own shape to match the shape of X
        n_t = self.n_t
        self._n_t = None  # type: ignore

        super(EHRData, self.__class__).X.fset(self, value)

        self._n_t = n_t

    @property
    def shape(self) -> tuple[int, int] | tuple[int, int, int]:
        """Shape of data (`n_obs`, `n_vars`, `n_t`)."""
        return self.n_obs, self.n_vars, self.n_t

    def __repr__(self) -> str:
        parent_repr = super().__repr__()

        if "View of" in parent_repr:
            parent_repr = parent_repr.replace("View of AnnData", "View of EHRData")
        else:
            parent_repr = parent_repr.replace("AnnData", "EHRData")

        lines_anndata = parent_repr.splitlines()

        # Filter out R_LAYER_KEY from layers line because it's a special layer called r
        lines_ehrdata = []
        position_of_t = 1
        for line in lines_anndata:
            if "obs:" in line or "var:" in line:
                position_of_t += 1

            lines_ehrdata.append(line)

        if not self.tem.empty:
            lines_ehrdata.insert(
                position_of_t, f"    tem: {list(self.tem.index.astype(str))}".replace("[", "").replace("]", "")
            )

        shape_info = []
        if self.X is not None:
            shape_info.append(f"shape of .X: {self.X.shape}")
        shape_info.extend(f"shape of .{layer}: {self.layers[layer].shape}" for layer in self.layers)

        if shape_info:
            lines_ehrdata.extend(["    " + info for info in shape_info])

        return "\n".join(lines_ehrdata)

    def __getitem__(self, index: Index | None) -> EHRData:
        """Slice the EHRData object along 1-3 axes.

        Args:
            index: 1D, 2D, or 3D index.

        Returns:
            An EHRData view object.
        """
        oidx, vidx, tidx = self._unpack_index(index)
        adata_sliced = super().__getitem__((oidx, vidx))

        tem_sliced = None if self.tem is None else self.tem.iloc[tidx]

        if self._tidx is None:
            # the input tidx might be of various kinds, and we want to store
            # a resolved version in AnnData style
            tidx = _resolve_idx(slice(None), tidx, self.n_t)
        if self._tidx is not None:
            tidx = _resolve_idx(self._tidx, tidx, self._adata_ref.n_t)

        # if tidx is an integer, numpy's automatic dimension reduction by drops an axis
        if isinstance(tidx, (int | np.integer)):
            tidx = slice(tidx, tidx + 1)

        return EHRData.from_adata(adata=adata_sliced, tem=tem_sliced, tidx=tidx)

    def _unpack_index(self, index: Index) -> tuple[Index1D, Index1D, Index1D]:
        if not isinstance(index, tuple):
            return index, slice(None), slice(None)
        elif len(index) == 3:
            return index
        elif len(index) == 2:
            return index[0], index[1], slice(None)
        elif len(index) == 1:
            return index[0], slice(None), slice(None)
        else:
            msg = "invalid number of indices"
            raise IndexError(msg)

    def copy(self) -> EHRData:
        """Returns a copy of the EHRData object."""
        return EHRData.from_adata(
            super().copy(),
            tem=None if self.tem is None else self.tem.copy(),
            tidx=self._tidx,
        )
