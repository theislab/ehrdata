from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from anndata import AnnData
from anndata._core.views import DataFrameView, _resolve_idx
from sparse import COO

from ehrdata._types import DaskArray, RDataType, XDataType
from ehrdata.core.constants import R_LAYER_KEY

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from os import PathLike

    from anndata._core.index import Index as ADIndex
    from anndata._core.index import Index1D

    Index: TypeAlias = ADIndex | tuple[Index1D, Index1D, Index1D]


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
        R: A #observations × #variables × #timesteps data array. A view of the data is used if
            the data type matches, otherwise, a copy is made.
        obs: Key-indexed one-dimensional observations annotation of length #observations.
        var: Key-indexed one-dimensional variables annotation of length #variables.
        tem: Key-indexed one-dimensional time annotation of length #timesteps.
        uns: Key-indexed unstructured annotation.
        obsm: Key-indexed multi-dimensional observations annotation of length #observations.
            If passing a :class:`numpy.ndarray`, it needs to have a structured datatype.
        varm: Key-indexed multi-dimensional variables annotation of length #variables.
            If passing a :class:`numpy.ndarray`, it needs to have a structured datatype.
        layers: Key-indexed multi-dimensional arrays aligned to dimensions of `X`.
        shape: Shape tuple (#observations, #variables). Can only be provided if `X` is None.
        filename: Name of backing file. See :class:`h5py.File`.
        filemode: Open mode of backing file. See :class:`h5py.File`.
    """

    _t: pd.DataFrame | None

    def __init__(
        self,
        X: XDataType | pd.DataFrame | None = None,
        *,
        R: RDataType | None = None,
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

        # Check if r is already present in layers
        R_existing = layers.get(R_LAYER_KEY) if layers is not None else None
        if R is not None and R_existing is not None:
            msg = f"`R` is both specified and already present in `layers[{R_LAYER_KEY}]`."
            raise ValueError(msg)

        R = R if R is not None else R_existing

        # Type checking for r
        if R is not None and not isinstance(R, (np.ndarray, COO, DaskArray)):  # type: ignore  # noqa: UP038
            msg = f"`R` must be numpy.ndarray, sparse.COO, or dask.array.Array, got {type(R)}"
            raise TypeError(msg)

        # Shape handling
        if R is not None:
            if len(R.shape) != 3:
                msg = f"`R` must be 3-dimensional, got shape {R.shape}"
                raise ValueError(msg)

            self._n_t = R.shape[2]

            if X is not None:
                if X.shape[:2] != R.shape[:2]:
                    msg = f"Shape mismatch between X {X.shape} and R {R.shape}"
                    raise ValueError(msg)
            else:
                # Create empty X matching R's shape
                X = np.nan * np.empty((R.shape[0], R.shape[1]))

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

        # Can set r, tidx only after AnnData initialization
        self.R = R
        self._tidx = tidx

        # Handle tem
        if R is None and tem is None:
            self.tem = pd.DataFrame([])

        elif R is None and tem is not None:
            self._n_t = len(tem)
            self.tem = tem

        elif R is not None and tem is not None:
            if not isinstance(tem, pd.DataFrame):
                msg = f"`tem` must be pandas.DataFrame, got {type(tem)}"
                raise TypeError(msg)

            if len(tem) != R.shape[2]:
                msg = f"Shape mismatch between R's third dimension {R.shape[2]} and tem {len(tem)}"
                raise ValueError(msg)

            self.tem = tem

        elif R is not None:
            # Default tem with RangeIndex
            l = 1 if R is None or len(R.shape) <= 2 else R.shape[2]
            self.tem = pd.DataFrame(index=pd.RangeIndex(l))

    @classmethod
    def from_adata(
        cls,
        adata: AnnData,
        *,
        R: np.ndarray | None = None,
        tem: pd.DataFrame | None = None,
        tidx: slice | None = None,
    ) -> EHRData:
        """Create an EHRData object from an AnnData object.

        Args:
            adata: Annotated data object.
            R: 3-Dimensional tensor, see R attribute.
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
            instance._n_t = None if R is None else R.shape[2]

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
            # Set R and tem directly for actual objects
            if R is None:
                instance._n_t = 0
            else:
                instance._n_t = R.shape[2]
                instance.layers[R_LAYER_KEY] = R
            if tem is not None:
                instance.tem = tem

        return instance

    @property
    def R(self) -> np.ndarray | None:
        """3-Dimensional tensor, aligned with obs along first axis, var along second axis, with a third time axis."""
        if self.is_view:
            if self._adata_ref.layers.get(R_LAYER_KEY) is None:
                return None
            else:
                return self.layers.get(R_LAYER_KEY)[:, :, self._tidx]
        else:
            return self.layers.get(R_LAYER_KEY)

    @R.setter
    def R(self, input: RDataType | None) -> None:
        if input is not None and len(input.shape) != 3:
            msg = f"`R` must be 3-dimensional, got shape {input.shape}"
            raise ValueError(msg)
        if input is not None and input.shape != self.shape:
            msg = f"`R` must be of shape of EHRData {self.shape}, but is {input.shape}"
            raise ValueError(msg)

        self._n_t = 0 if input is None else input.shape[2]

        if input is None:
            del self.R
        else:
            self.layers[R_LAYER_KEY] = input

    @R.deleter
    def R(self) -> None:
        self.layers.pop(R_LAYER_KEY, None)
        self._n_t = 0
        self.tem = pd.DataFrame([])

    @property
    def tem(self) -> pd.DataFrame:
        """Time dataframe for describing third axis."""
        return self._tem

    @tem.setter
    def tem(self, input: pd.DataFrame) -> None:
        if not isinstance(input, pd.DataFrame):
            msg = "Can only assign pd.DataFrame to tem."
            raise ValueError(msg)
        if self.n_t is not None and len(input) != self.n_t:
            msg = f"Length of passed value for tem is {len(input)}, but this EHRData has shape: {self.shape}"
            raise ValueError(msg)

        self._tem = input
        self._n_t = len(input)

    @property
    def n_t(self) -> int | None:
        """Number of time points."""
        return self._n_t

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
        self._n_t = None

        super(EHRData, self.__class__).X.fset(self, value)

        self._n_t = n_t

    @property
    def shape(self) -> tuple[int, int] | tuple[int, int, int]:
        """Shape of data (`n_obs`, `n_vars`, `n_t`)."""
        if not hasattr(self, "n_t") or self.n_t is None:
            return self.n_obs, self.n_vars
        else:
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
            if "layers:" in line and R_LAYER_KEY in line:
                # Handle case where r_layer is the only layer
                if line == f"    layers: '{R_LAYER_KEY}'":
                    # Skip this line completely
                    continue
                # Handle case where r_layer is among other layers
                line = line.replace(f"'{R_LAYER_KEY}', ", "")
                line = line.replace(f", '{R_LAYER_KEY}'", "")

            if self.R is not None and "n_obs × n_vars" in line:
                line_splits = line.split("object with")
                line = line_splits[0] + f"object with n_obs × n_vars × n_t = {self.n_obs} × {self.n_vars} × {self.n_t}"

            if "obs:" in line or "var:" in line:
                position_of_t += 1

            lines_ehrdata.append(line)

        if not self.tem.empty:
            lines_ehrdata.insert(
                position_of_t, f"    tem: {list(self.tem.index.astype(str))}".replace("[", "").replace("]", "")
            )

        # Add shape info only if X or R are present
        shape_info = []
        if self.X is not None:
            shape_info.append(f"shape of .X: {self.X.shape}")
        if self.R is not None:
            shape_info.append(f"shape of .R: {self.R.shape}")

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

        r_sliced = None if self.R is None else adata_sliced.layers[R_LAYER_KEY][:, :, tidx]
        return EHRData.from_adata(adata=adata_sliced, R=r_sliced, tem=tem_sliced, tidx=tidx)

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
            R=None if self.R is None else self.R.copy(),
            tem=None if self.tem is None else self.tem.copy(),
            tidx=self._tidx,
        )
