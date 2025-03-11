from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import pandas as pd
from anndata import AnnData
from anndata._core.access import ElementRef
from anndata._core.views import DataFrameView, as_view

from ehrdata.core.constants import R_LAYER_KEY
from ehrdata.core.index import _subset

if TYPE_CHECKING:
    from anndata._core.index import Index as ADIndex
    from anndata._core.index import Index1D

    Index: TypeAlias = ADIndex | tuple[Index1D, Index1D, Index1D]


class EHRData(AnnData):
    """EHRData object.

    Inherits `__init__` parameters, methods, and properties from :class:`~anndata.AnnData`.

    Parameters
    ----------
    X
        See :attr:`~anndata.AnnData.X`.
    r
        3-Dimensional tensor. See :attr:`r`.
    t
        Time dataframe for describing third axis. See :attr:`t`.
    """

    _t: pd.DataFrame | None

    def __init__(
        self,
        X=None,
        r: np.ndarray | None = None,
        *,
        t: pd.DataFrame | None = None,
        **kwargs,
    ):
        # Check if r is already present in layers
        r_existing = kwargs.get("layers", {}).get(R_LAYER_KEY)
        if r is not None and r_existing is not None:
            msg = f"`r` is both specified and already present in `layers[{R_LAYER_KEY}]`."
            raise ValueError(msg)

        # Use existing r if present
        r = r if r is not None else r_existing

        # Type checking for r
        if r is not None and not isinstance(r, (np.ndarray | "dask.array.Array")):  # type: ignore
            msg = f"`r` must be numpy.ndarray or dask.array.Array, got {type(r)}"
            raise TypeError(msg)

        # Shape handling
        if r is not None:
            if len(r.shape) != 3:
                msg = f"`r` must be 3-dimensional, got shape {r.shape}"
                raise ValueError(msg)

            if X is not None:
                if X.shape[:2] != r.shape[:2]:
                    msg = f"Shape mismatch between X {X.shape} and r {r.shape}"
                    raise ValueError(msg)
            else:
                # Create empty X matching r's shape
                X = np.nan * np.empty((r.shape[0], r.shape[1]))

        # once here, verified a) r has shape[2] if exists. below will verify b) t has same shape as r[2]
        self.n_t = 0 if r is None else r.shape[2]

        # Initialize AnnData with X
        super().__init__(X=X, **kwargs)

        # Set r after AnnData initialization
        if r is not None:
            self.layers[R_LAYER_KEY] = r

        # Handle t
        if r is None and t is not None:
            msg = "`t` can only be specified if `r` is specified"
            raise ValueError(msg)

        if t is not None:
            if not isinstance(t, pd.DataFrame):
                msg = f"`t` must be pandas.DataFrame, got {type(t)}"
                raise TypeError(msg)

            if r is not None and len(t) != r.shape[2]:
                msg = f"Shape mismatch between r's third dimension {r.shape[2]} and t {len(t)}"
                raise ValueError(msg)

            self.t = t
        else:
            # Default t with RangeIndex
            l = 1 if r is None or len(r.shape) <= 2 else r.shape[2]
            self.t = pd.DataFrame(index=pd.RangeIndex(l))

    @classmethod
    def from_adata(
        cls,
        adata: AnnData,
        *,
        r: np.ndarray | None = None,
        t: pd.DataFrame | None = None,
    ) -> EHRData:
        """Create an EHRData object from an AnnData object.

        Parameters
        ----------
        adata
            Annotated data object.
        r
            3-Dimensional tensor, see :attr:`r`.
        t
            Time dataframe for describing third axis, see :attr:`t`.

        Returns
        -------
        An EHRData object.
        """
        instance = cls(shape=adata.shape)

        if adata.is_view:
            # use _init_as_view of adata, but don't subset since already sliced in __getitem__
            instance._init_as_view(adata, slice(None), slice(None))
            # The tidx is not part of AnnData, so we need to set it separately. Setting it is required for the getter of r
            instance._tidx = adata._tidx

            # t is not part of AnnData, so we need to set it separately
            if t is not None:
                instance.t = DataFrameView(t, t.index)

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
            # Set r and t directly for actual objects
            if r is not None:
                instance.layers[R_LAYER_KEY] = r
            if t is not None:
                instance.t = t

        return instance

    @property
    def r(self) -> np.ndarray | None:
        """3-Dimensional tensor, aligned with obs along first axis, var along second axis, and allowing a 3rd axis."""
        if self.is_view:
            if self._adata_ref.layers.get(R_LAYER_KEY) is None:
                return None
            else:
                r = as_view(
                    _subset(
                        self._adata_ref.layers.get(R_LAYER_KEY),
                        (self._oidx, self._vidx, self._tidx if self._tidx is not None else slice(None)),
                    ),
                    ElementRef(self, "layers", (R_LAYER_KEY,)),
                )
                return r
        else:
            return self.layers.get(R_LAYER_KEY)

    # TODO: allow for r to be numpy and dask only?
    # TODO: check that r and t are aligned
    @r.setter
    def r(self, input: np.ndarray | None) -> None:
        # assert self.shape == input.shape

        if input is None:
            del self.r
        else:
            self.layers[R_LAYER_KEY] = input

    @r.deleter
    def r(self) -> None:
        self.layers.pop(R_LAYER_KEY, None)

    @property
    def t(self) -> pd.DataFrame | None:
        """Time dataframe for describing third axis."""
        return self._t

    @t.setter
    def t(self, input: pd.DataFrame | None) -> None:
        self._t = input

    @t.deleter
    def t(self) -> None:
        del self._t

    @property
    def n_t(self) -> int:
        """Number of time points."""
        return self._n_t

    @n_t.setter
    def n_t(self, input: int) -> None:
        self._n_t = input

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of data matrix (:attr:`n_obs`, :attr:`n_vars`)."""
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
        for line in lines_anndata:
            if line == f"    layers: '{R_LAYER_KEY}'":
                line.replace(f"'{R_LAYER_KEY}', ", "")
            if self.r is not None and "n_obs × n_vars" in line:
                line_splits = line.split("object with")
                line = (
                    line_splits[0]
                    + f"object with n_obs × n_vars × n_t = {self.r.shape[0]} × {self.r.shape[1]} × {self.r.shape[2]}"
                )
            lines_ehrdata.append(line)

        if self.r is not None:
            lines_ehrdata.insert(2, f"    t: {list(self.t.index.astype(str))}".replace("[", "").replace("]", ""))

        # Add shape info only if X or r are present
        shape_info = []
        if self.X is not None:
            shape_info.append(f"shape of .X: {self.X.shape}")
        if self.r is not None:
            shape_info.append(f"shape of .r: {self.r.shape}")

        if shape_info:
            lines_ehrdata.extend(["    " + info for info in shape_info])

        return "\n".join(lines_ehrdata)

    def __getitem__(self, index: Index | None) -> EHRData:
        """Slice the EHRData object along 1–3 axes.

        Parameters
        ----------
        index
            1D, 2D, or 3D index.

        Returns
        -------
        An EHRData view object.
        """
        oidx, vidx, tidx = self._unpack_index(index)
        adata_sliced = super().__getitem__((oidx, vidx))
        adata_sliced._tidx = tidx
        r_sliced = None if self.r is None else adata_sliced.layers[R_LAYER_KEY][:, :, tidx]
        t_sliced = None if self.t is None else self.t.iloc[tidx]
        return EHRData.from_adata(adata=adata_sliced, r=r_sliced, t=t_sliced)

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
            raise IndexError("invalid number of indices")

    def copy(self) -> EHRData:
        """Returns a copy of the EHRData object."""
        return EHRData.from_adata(
            super().copy(),
            r=None if self.r is None else self.r.copy(),
            t=None if self.t is None else self.t.copy(),
        )
