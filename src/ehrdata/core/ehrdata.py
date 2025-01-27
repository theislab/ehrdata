from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import pandas as pd
from anndata import AnnData

from ehrdata.core.constants import R_LAYER_KEY

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
        super().__init__(X=X, **kwargs)

        if r is not None:
            if (r2 := self.layers.get(R_LAYER_KEY)) is not None and r2 is not r:
                msg = "`r` is both specified and already present in `adata.layers`."
                raise ValueError(msg)
            self.layers[R_LAYER_KEY] = r
        # else:
        #     self.layers[R_LAYER_KEY] = np.zeros((self._adata.shape[0], self._adata.shape[1], 0))

        if t is None:
            l = 1 if r is None or len(r.shape) <= 2 else r.shape[2]
            self.t = pd.DataFrame(index=pd.RangeIndex(l))
        elif isinstance(t, pd.DataFrame):
            self.t = t
        else:
            raise ValueError("t must be a pandas.DataFrame")

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
            instance._init_as_view(adata, slice(None), slice(None))
        else:
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
        instance.r = r
        instance.t = t
        return instance

    @property
    def r(self) -> np.ndarray | None:
        """3-Dimensional tensor, aligned with obs along first axis, var along second axis, and allowing a 3rd axis."""
        return self.layers.get(R_LAYER_KEY)

    @r.setter
    def r(self, input: np.ndarray | None) -> None:
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

    def __repr__(self) -> str:
        # TODO: derive from `r` if self.t is None
        timeseries = "no timeseries" if self.t is None else f"a timeseries of {len(self.t)} steps"
        return f"EHRData object with n_obs x n_var = {self.n_obs} x {self.n_vars}, and {timeseries}.\n \
            shape of .X: {self.X.shape if self.X is not None else (0, 0)} \n \
            shape of .r: {self.r.shape if self.r is not None else (0, 0, 0)} \n"

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
