from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import pandas as pd

from ehrdata._logger import logger

if TYPE_CHECKING:
    from ehrdata import EHRData


def move_to_obs(
    edata: EHRData,
    var_names: list[str] | str,
    *,
    layer: str | None = None,
    copy_columns: bool = False,
    copy: bool = False,
) -> EHRData | None:
    """Move variables from `.X`/`.layers` to `.obs`.

    This function moves the `var_names` specified from the indicated `layer` to `.obs`.

    Important to note:

    - The `layer` must be 2D.
    - If `copy_columns` is set to `False`, `var_names` will be removed across `.X`/`.layers`.

    Args:
        edata: Central data object.
        var_names: The columns to move to `.obs`.
        layer: The 2D layer to use from the :class:`~ehrdata.EHRData` object. If `None`, the `X` layer is used. Must be 2D.
        copy_columns: If False, the columns are moved to `obs` and deleted from `.X`/`.layers`. If `True`, the values are copied to obs (and therefore kept in `.X`/`.layers`).
        copy: If `True`, a new :class:`~ehrdata.EHRData` object is returned. If `False`, the original object is modified inplace and `None` is returned.

    Returns:
        A new :class:`~ehrdata.EHRData` object with moved or copied columns from `.X`/`.layers` to `obs`.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.move_to_obs(edata, ["age"], copy=True)
        EHRData object with n_obs × n_vars × n_t = 1776 × 45 × 1
            obs: 'age'
            shape of .X: (1776, 45)

        where

        >>> edata
        EHRData object with n_obs × n_vars × n_t = 1776 × 46 × 1
            shape of .X: (1776, 46)
    """
    if layer is not None and (len(edata.layers[layer].shape) > 2) and (edata.layers[layer].shape[2] > 1):
        msg = "Layer is 3D, but move_to_obs only supports 2D layers."
        raise ValueError(msg)

    if copy:
        edata = edata.copy()

    if isinstance(var_names, str):
        var_names = [var_names]

    if not all(elem in edata.var_names.values for elem in var_names):
        err = f"Columns `{[col for col in var_names if col not in edata.var_names.values]}` are not in var_names."
        raise ValueError(err)

    cols_to_obs_indices = edata.var_names.isin(var_names)

    if copy_columns:
        cols_to_obs = edata[:, cols_to_obs_indices].to_df()
        edata.obs = edata.obs.join(cols_to_obs)

    else:
        df = edata[:, cols_to_obs_indices].to_df()
        edata._inplace_subset_var(~cols_to_obs_indices)
        edata.obs = edata.obs.join(df)

    if copy:
        return edata
    else:
        return None


def move_to_x(
    edata: EHRData,
    features: list[str] | str,
    *,
    layer: str | None = None,
    copy_columns: bool = False,
) -> EHRData:
    """Move variables from `.obs` to `.X`/`.layers`.

    This function moves the `features` specified from `.obs` to `.X` or the indicated `layer`.

    The column names in `.obs` are preserved in `.var_names` of the new object.

    The new object only consists of the specified `layer` and `.obs` from the original object.
    That means other layers and fields such as `.obsm`, `.varm` etc. are not in the new object, providing a clean and minimal object for further analysis.

    Important: The `layer` must be 2D.

    Args:
        edata: Central data object.
        features: The columns to move to `.X`/`.layers`.
        layer: The 2D layer to use from the :class:`~ehrdata.EHRData` object. If `None`, `.X` is used.
        copy_columns: The values are copied to `.X`/`.layers` (and therefore kept in `.obs`) instead of being moved completely.

    Returns:
        A new data object with moved columns from `.obs` to `.X`/`.layers`, with

        - `.X`/`.layers` containing the original plus moved columns
        - `.obs`

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2(columns_obs_only=["age"])
        >>> ed.move_to_x(edata, ["age"])
        EHRData object with n_obs × n_vars × n_t = 1776 × 46 × 1
            shape of .X: (1776, 46)

        where

        >>> edata
        EHRData object with n_obs × n_vars × n_t = 1776 × 45 × 1
            obs: 'age'
            shape of .X: (1776, 45)
    """
    from ehrdata import EHRData

    if isinstance(features, str):  # pragma: no cover
        features = [features]

    if not all(elem in edata.obs.columns.values for elem in features):
        err = f"Columns `{[col for col in features if col not in edata.obs.columns.values]}` are not in obs."
        raise ValueError(err)

    if layer is not None and (len(edata.layers[layer].shape) > 2) and (edata.layers[layer].shape[2] > 1):
        msg = "Layer is 3D, but move_to_x only supports 2D layers."
        raise ValueError(msg)

    cols_present_in_x = []
    cols_not_in_x = []

    for col in features:
        if col in set(edata.var_names):
            cols_present_in_x.append(col)
        else:
            cols_not_in_x.append(col)

    if cols_present_in_x:
        logger(f"Columns `{cols_present_in_x}` are already in var_names. Skipped moving `{cols_present_in_x}`. ")

    if cols_not_in_x:
        edata_from_obs = EHRData(X=edata.obs[cols_not_in_x])
        if layer is None:
            new_edata = ad.concat([edata, edata_from_obs], axis=1)
            new_edata = EHRData.from_adata(new_edata)
        else:
            # anndata can only concatenate if both objects have X.
            # if one has, it raises an Error: if neither has, it returns an object with 0 observations
            edata.X = edata.layers[layer]
            new_edata = ad.concat([edata, edata_from_obs], axis=1)
            new_edata.layers[layer] = new_edata.X
            new_edata.X = None
            new_edata = EHRData.from_adata(new_edata)

        if copy_columns:
            new_edata.obs = edata.obs
        else:
            new_edata.obs = edata.obs[edata.obs.columns[~edata.obs.columns.isin(cols_not_in_x)]]

        # AnnData's concat discards var if they don't match in their keys, so we need to create a new var
        created_var = pd.DataFrame(index=cols_not_in_x)
        new_edata.var = pd.concat([edata.var, created_var], axis=0)
    else:
        logger("No columns moved, move_to_x returning original object.")
        new_edata = edata

    return new_edata
