from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from ehrdata.core.constants import R_LAYER_KEY


class EHRData:
    """EHRData object."""

    def __init__(
        self,
        adata=None,
        X=None,
        r=None,
        obs=None,
        var=None,
        t=None,
        uns=None,
        obsm=None,
        varm=None,
        layers=None,
        raw=None,
        dtype=None,
        shape=None,
        filename=None,
        filemode=None,
        asview=None,
        *,
        obsp=None,
        varp=None,
        oidx=None,
        vidx=None,
        tidx=None,
    ):
        """EHRData object."""
        if adata is None:
            self._adata = AnnData(
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

        elif isinstance(adata, AnnData):
            self._adata = adata

        else:
            raise ValueError("adata must be an AnnData object")

        if r is not None:
            self.layers[R_LAYER_KEY] = r
        else:
            self.layers[R_LAYER_KEY] = np.empty((self._adata.shape[0], 0, 0))

        if t is not None:
            if isinstance(t, pd.DataFrame):
                self.t = t
            else:
                raise ValueError("t must be a pandas.DataFrame")

        else:
            if R_LAYER_KEY not in self.layers.keys():
                self.t = pd.DataFrame(pd.RangeIndex(1))
            elif len(self.layers[R_LAYER_KEY].shape) <= 2:
                self.t = pd.DataFrame(pd.RangeIndex(1))
            else:
                self.t = pd.DataFrame(index=pd.RangeIndex(self.layers[R_LAYER_KEY].shape[2]))

        if tidx is not None:
            self.t = t.iloc[tidx]

    @property
    def X(self):
        """Field from AnnData."""
        return self._adata.X

    @X.setter
    def X(self, input):
        self._adata.X = input

    @property
    def obs(self):
        """Field from AnnData."""
        return self._adata.obs

    @obs.setter
    def obs(self, input):
        self._adata.obs = input

    @property
    def var(self):
        """Field from AnnData."""
        return self._adata.var

    @var.setter
    def var(self, input):
        self._adata.obs = input

    @property
    def obsm(self):
        """Field from AnnData."""
        return self._adata.obsm

    @obsm.setter
    def obsm(self, input):
        self._adata.obsm = input

    @property
    def varm(self):
        """Field from AnnData."""
        return self._adata.varm

    @varm.setter
    def varm(self, input):
        self._adata.varm = input

    @property
    def obsp(self):
        """Field from AnnData."""
        return self._adata.obsp

    @obsp.setter
    def obsp(self, input):
        self._adata.obsp = input

    @property
    def varp(self):
        """Field from AnnData."""
        return self._adata.varp

    @varp.setter
    def varp(self, input):
        self._adata.varp = input

    @property
    def var_names(self):
        """Field from AnnData."""
        return self._adata.var_names

    @var_names.setter
    def var_names(self, input):
        self._adata.var_names = input

    @property
    def obs_names(self):
        """Field from AnnData."""
        return self._adata.obs_names

    @obs_names.setter
    def obs_names(self, input):
        self._adata.obs_names = input

    @property
    def uns(self):
        """Field from AnnData."""
        return self._adata.uns

    @uns.setter
    def uns(self, input):
        self._adata.uns = input

    @property
    def layers(self):
        """Field from AnnData."""
        # TODO: dont allow writing a specific key to layers..?
        return self._adata.layers

    @layers.setter
    def layers(self, key, input):
        # TODO: if 3D: need to match time dimension?
        self._adata.layers[key] = input

    @property
    def r(self):
        """3-Dimensional tensor, aligned with obs along first axis, var along second axis, and allowing a 3rd axis."""
        return self._adata.layers[R_LAYER_KEY]

    @r.setter
    def r(self, input):
        self._adata.layers[R_LAYER_KEY] = input

    @property
    def t(self):
        """Time dataframe for describing third axis."""
        return self._t

    @t.setter
    def t(self, input):
        self._t = input

    def __repr__(self):
        return f"EHRData object with n_obs x n_var = {self._adata.n_obs} x {self._adata.n_vars}, and a timeseries of {len(self.t)} steps.\n \
            shape of .X: {self._adata.shape} \n \
            shape of .r: ({self.r.shape if self.r is not None else (0,0,0)}) \n"

    @property
    def shape(self):
        """Shape of the data matrix without r's third dimension."""
        return self._adata.shape

    def __getitem__(self, index):
        oidx, vidx, tidx = self._unpack_index(index)
        adata_sliced = self._adata[oidx, vidx]
        t_sliced = self._t.iloc[tidx]

        adata_2D_sliced = EHRData(adata=adata_sliced, t=t_sliced)
        adata_2D_sliced.r = adata_2D_sliced.r[:, :, tidx]

        return adata_2D_sliced

    def _unpack_index(self, index):
        if not isinstance(index, tuple):
            return index, slice(None)
        elif len(index) == 3:
            return index
        elif len(index) == 2:
            return index[0], index[1], slice(None)
        elif len(index) == 1:
            return index[0], slice(None)
        else:
            raise IndexError("invalid number of indices")

    def copy(self):
        """Returns a copy of the EHRData object."""
        return EHRData(adata=self._adata.copy(), t=self._t.copy())
