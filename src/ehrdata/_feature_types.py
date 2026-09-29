from __future__ import annotations

import warnings
from functools import singledispatch, wraps
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sparse
from dateutil.parser import isoparse  # type: ignore
from fast_array_utils.conv import to_dense
from rich import print
from rich.tree import Tree
from scipy.sparse import issparse

from ehrdata._logger import logger
from ehrdata._types import CSArray, CSMatrix
from ehrdata.core.constants import CATEGORICAL_TAG, DATE_TAG, FEATURE_TYPE_KEY, MISSING_VALUES, NUMERIC_TAG

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ehrdata import EHRData

# sparse array types
_FAST_SPARSE_TYPES = (sparse.COO, CSMatrix, CSArray)


def _detect_feature_type(
    col: pd.Series,
    *,
    binary_as: Literal["categorical", "numeric"] = "categorical",
) -> tuple[Literal["date", "categorical", "numeric"], bool]:
    """Detect the feature type of a :class:`~pandas.Series`.

    Args:
        col: The series to detect the feature type of.
        binary_as: How to classify binary (0/1) features.

    Returns:
        The detected feature type (one of 'date', 'categorical', or 'numeric') and a boolean, which is True if the feature type is uncertain.
    """
    col[col.isin(MISSING_VALUES)] = np.nan
    col = col.infer_objects()
    col = col.dropna()
    if len(col) == 0:
        err_msg = f"Feature '{col.name}' contains only NaN values. Please drop this feature to infer the feature type."
        raise ValueError(err_msg)
    majority_type = col.apply(type).value_counts().idxmax()

    if majority_type == pd.Timestamp:
        return DATE_TAG, False  # type: ignore

    if majority_type is str:
        try:
            col.apply(isoparse)
            return DATE_TAG, False  # type: ignore
        except ValueError:
            try:
                col = pd.to_numeric(col, errors="raise")  # Could be an encoded categorical or a numeric feature
                majority_type = float
            except ValueError:
                # Features stored as Strings that cannot be converted to float are assumed to be categorical
                return CATEGORICAL_TAG, False  # type: ignore

    if majority_type not in [int, float]:
        return CATEGORICAL_TAG, False  # type: ignore

    # Guess categorical if the feature is binary (values are exactly {0, 1})
    if set(col.unique()) == {0, 1}:
        if binary_as == "numeric":
            return NUMERIC_TAG, False  # type: ignore
        return CATEGORICAL_TAG, True  # type: ignore

    return NUMERIC_TAG, False  # type: ignore


def _classify_sparse_binary_columns(
    var_coord: np.ndarray,
    data: np.ndarray,
    n_vars: int,
    entries_per_var: int,
    *,
    fill_value,
    fill_is_nan: bool,
) -> np.ndarray:
    """Per variable, decides if its values (explicit + implicit) are exactly `{0, 1}`."""
    is_float = np.issubdtype(data.dtype, np.floating)
    nan_mask = np.isnan(data) if is_float else np.zeros(data.shape, dtype=bool)

    has_zero = np.zeros(n_vars, dtype=bool)
    has_one = np.zeros(n_vars, dtype=bool)
    has_other = np.zeros(n_vars, dtype=bool)
    stored_count = np.zeros(n_vars, dtype=np.int64)
    non_nan_count = np.zeros(n_vars, dtype=np.int64)

    np.logical_or.at(has_zero, var_coord, (data == 0) & ~nan_mask)
    np.logical_or.at(has_one, var_coord, (data == 1) & ~nan_mask)
    np.logical_or.at(has_other, var_coord, ~nan_mask & (data != 0) & (data != 1))
    np.add.at(stored_count, var_coord, 1)
    np.add.at(non_nan_count, var_coord, ~nan_mask)

    has_implicit = stored_count < entries_per_var
    if not fill_is_nan:
        non_nan_count += has_implicit * (entries_per_var - stored_count)
        if fill_value == 0:
            has_zero |= has_implicit
        elif fill_value == 1:
            has_one |= has_implicit
        else:
            has_other |= has_implicit

    all_nan = np.nonzero(non_nan_count == 0)[0]
    if all_nan.size:
        err_msg = f"Feature at position {all_nan[0]} contains only NaN values. Please drop this feature to infer the feature type."
        raise ValueError(err_msg)

    return has_zero & has_one & ~has_other


def _binary_mask_to_results(
    is_binary: np.ndarray, binary_as: Literal["categorical", "numeric"]
) -> list[tuple[Literal["categorical", "numeric"], bool]]:
    binary_result = (NUMERIC_TAG, False) if binary_as == "numeric" else (CATEGORICAL_TAG, True)
    numeric_result = (NUMERIC_TAG, False)
    return [binary_result if b else numeric_result for b in is_binary]


def _detect_feature_types_sparse_coo_non_numeric(
    X: sparse.COO,
    *,
    binary_as: Literal["categorical", "numeric"] = "categorical",
) -> list[tuple[Literal["date", "categorical", "numeric"], bool]]:
    """Detect the feature type of every variable of a non-numeric `sparse.COO` array."""
    n_vars = X.shape[1]
    entries_per_var = X.size // n_vars

    order = np.argsort(X.coords[1], kind="stable")
    sorted_vars = X.coords[1][order]
    sorted_data = X.data[order]
    starts = np.searchsorted(sorted_vars, np.arange(n_vars))
    ends = np.searchsorted(sorted_vars, np.arange(n_vars), side="right")

    results = []
    for j in range(n_vars):
        if ends[j] - starts[j] < entries_per_var:
            # rare: a genuine implicit fill value among non-numeric data, densify just this column
            col = pd.Series(X[:, j].todense().reshape(-1))
        else:
            col = pd.Series(sorted_data[starts[j] : ends[j]])
        results.append(_detect_feature_type(col, binary_as=binary_as))

    return results


@singledispatch
def _detect_feature_types_sparse(X, *, binary_as: Literal["categorical", "numeric"] = "categorical"):
    """Detect the feature type of every variable (axis 1) of a sparse array, without densifying."""
    msg = f"_detect_feature_types_sparse does not support array type {type(X)}."
    raise NotImplementedError(msg)


@_detect_feature_types_sparse.register(sparse.COO)
def _(X: sparse.COO, *, binary_as: Literal["categorical", "numeric"] = "categorical"):
    # sparse.COO isn't restricted to numeric dtypes in memory (only ehrdata's on-disk binsparse
    # writer enforces that); route non-numeric arrays through the dedicated per-column path.
    if not (np.issubdtype(X.dtype, np.number) or np.issubdtype(X.dtype, np.bool_)):
        return _detect_feature_types_sparse_coo_non_numeric(X, binary_as=binary_as)

    n_vars = X.shape[1]
    entries_per_var = X.size // n_vars  # rows * time steps (for 3D) per variable
    fill_is_nan = np.issubdtype(X.dtype, np.floating) and np.isnan(X.fill_value)
    is_binary = _classify_sparse_binary_columns(
        X.coords[1], X.data, n_vars, entries_per_var, fill_value=X.fill_value, fill_is_nan=fill_is_nan
    )
    return _binary_mask_to_results(is_binary, binary_as)


@_detect_feature_types_sparse.register(sp.csr_array)
@_detect_feature_types_sparse.register(sp.csr_matrix)
def _(X, *, binary_as: Literal["categorical", "numeric"] = "categorical"):
    # CSR stores each entry's column index in .indices
    n_vars = X.shape[1]
    is_binary = _classify_sparse_binary_columns(X.indices, X.data, n_vars, X.shape[0], fill_value=0, fill_is_nan=False)
    return _binary_mask_to_results(is_binary, binary_as)


@_detect_feature_types_sparse.register(sp.csc_array)
@_detect_feature_types_sparse.register(sp.csc_matrix)
def _(X, *, binary_as: Literal["categorical", "numeric"] = "categorical"):
    # CSC stores entries grouped by column via .indptr
    n_vars = X.shape[1]
    var_coord = np.repeat(np.arange(n_vars), np.diff(X.indptr))
    is_binary = _classify_sparse_binary_columns(var_coord, X.data, n_vars, X.shape[0], fill_value=0, fill_is_nan=False)
    return _binary_mask_to_results(is_binary, binary_as)


def infer_feature_types(
    edata: EHRData,
    *,
    layer: str | None = None,
    output: Literal["tree", "dataframe"] | None = "tree",
    binary_as: Literal["categorical", "numeric"] = "categorical",
    verbose: bool = True,
) -> pd.DataFrame | None:
    """Infer feature types of an :class:`~ehrdata.EHRData` object.

    For each feature in `edata.var_names`, the method infers one of the following types: `'date'`, `'categorical'`, or `'numeric'`.
    The inferred types are stored in `edata.var['feature_type']`.
    Please check the inferred types and adjust if necessary using `edata.var['feature_type']['feature1']='corrected_type'` or with :func:`~ehrdata.replace_feature_types`.
    Be aware that not all features stored numerically are of `'numeric'` type, as categorical features might be stored in a numerically encoded format.
    For example, a feature with values [0, 1, 2] might be a categorical feature with three categories.
    This is accounted for in the method, but it is recommended to check the inferred types.

    Args:
        edata: Data object.
        layer: The layer to use from the EHRData object. If `None`, the `X` field is used.
        output: The output format. Choose between `'tree'`, `'dataframe'`, or `None`.
            If `'tree'`, the feature types will be printed to the console in a tree format.
            If `'dataframe'`, a :class:`~pandas.DataFrame` with the feature types will be returned.
            If `None`, nothing will be returned.
        binary_as: How to classify binary features with values 0 and 1.
            If `'categorical'` (default), binary features are classified as categorical.
            If `'numeric'`, binary features are classified as numeric.
        verbose: Whether to print warnings for uncertain feature types.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.infer_feature_types(edata)
    """
    feature_types = {}
    uncertain_features = []

    X = edata.X if layer is None else edata.layers[layer]

    if X is None:
        # X not set, fall back to first available layer
        first_layer = sorted(edata.layers)[0]
        X = edata.layers[first_layer]
        warnings.warn(
            f"No layer specified and `edata.X` is None. Falling back to layer '{first_layer}' for feature type inference. "
            f"To be explicit, pass `layer='{first_layer}'` or the desired layer name.",
            UserWarning,
            stacklevel=2,
        )

    # The non-densifying fast path applies to the array types registered on _detect_feature_types_sparse
    # (sparse.COO handles numeric and non-numeric dtypes internally; scipy.sparse is always numeric).
    if isinstance(X, _FAST_SPARSE_TYPES):
        sparse_results = _detect_feature_types_sparse(X, binary_as=binary_as)
    else:
        sparse_results = None

        if X.ndim == 3:
            n_obs, n_vars, n_t = X.shape
            X = X.transpose(0, 2, 1).reshape(n_obs * n_t, n_vars)

        df = pd.DataFrame(X.reshape(-1, edata.shape[1]), columns=edata.var_names)

    for i, feature in enumerate(edata.var_names):
        if (
            FEATURE_TYPE_KEY in edata.var
            and edata.var[FEATURE_TYPE_KEY][feature] is not None
            and not pd.isna(edata.var[FEATURE_TYPE_KEY][feature])
        ):
            feature_types[feature] = edata.var[FEATURE_TYPE_KEY][feature]
        else:
            if sparse_results is not None:
                feature_types[feature], raise_warning = sparse_results[i]
            else:
                feature_types[feature], raise_warning = _detect_feature_type(df[feature], binary_as=binary_as)
            if raise_warning:
                uncertain_features.append(feature)

    edata.var[FEATURE_TYPE_KEY] = pd.Series(feature_types)[edata.var_names]

    if verbose:
        if uncertain_features:
            names = ", ".join(f"'{feature}'" for feature in uncertain_features)
            logger.warning(
                f"{'Features' if len(uncertain_features) > 1 else 'Feature'} {names} {'were' if len(uncertain_features) > 1 else 'was'} detected as categorical features stored numerically. "
                f"Adjust using `ed.replace_feature_types` if needed."
            )

        logger.info(
            f"Stored feature types in edata.var['{FEATURE_TYPE_KEY}']. "
            f"Adjust using `ed.replace_feature_types` if needed."
        )

    if output == "tree":
        feature_type_overview(edata)
    elif output == "dataframe":
        return edata.var[FEATURE_TYPE_KEY].to_frame()
    elif output is not None:
        err_msg = f"Output format {output} not recognized. Choose between 'tree', 'dataframe', or None."
        raise ValueError(err_msg)


def _check_feature_types(func):
    @wraps(func)
    def wrapper(edata, *args, **kwargs):
        from ehrdata import EHRData

        # Account for class methods that pass self as first argument
        _self = None
        if not isinstance(edata, EHRData) and len(args) > 0 and isinstance(args[0], EHRData):
            _self = edata
            edata = args[0]
            args = args[1:]

        if FEATURE_TYPE_KEY not in edata.var:
            infer_feature_types(edata, output=None)
            logger.warning(
                f"Feature types were inferred and stored in edata.var[{FEATURE_TYPE_KEY}]. Verify using `ed.feature_type_overview` and adjust using `ed.replace_feature_types` if needed."
            )

        for feature in edata.var_names:
            feature_type = edata.var[FEATURE_TYPE_KEY][feature]
            if (
                feature_type is not None
                and (not pd.isna(feature_type))
                and feature_type not in [CATEGORICAL_TAG, NUMERIC_TAG, DATE_TAG]
            ):
                logger.warning(
                    f"Feature '{feature}' has an invalid feature type '{feature_type}'. Correct using `ed.replace_feature_types`."
                )

        if _self is not None:
            return func(_self, edata, *args, **kwargs)
        return func(edata, *args, **kwargs)

    return wrapper


@_check_feature_types
def feature_type_overview(edata: EHRData) -> None:
    """Print an overview of the feature types and encoding modes in the :class:`~ehrdata.EHRData` object.

    Args:
        edata: Data object.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.feature_type_overview(edata)
    """
    tree = Tree(
        f"[b] Detected feature types for EHRData object with {len(edata.obs_names)} obs and {len(edata.var_names)} vars",
        guide_style="underline2",
    )

    branch = tree.add("📅[b] Date features")
    for date in sorted(edata.var_names[edata.var[FEATURE_TYPE_KEY] == DATE_TAG]):
        branch.add(date)

    branch = tree.add("📐[b] Numerical features")
    for numeric in sorted(edata.var_names[edata.var[FEATURE_TYPE_KEY] == NUMERIC_TAG]):
        branch.add(numeric)

    branch = tree.add("🗂️[b] Categorical features")
    cat_features = edata.var_names[edata.var[FEATURE_TYPE_KEY] == CATEGORICAL_TAG]

    df = pd.DataFrame(
        to_dense(edata[:, cat_features].X) if issparse(edata[:, cat_features].X) else edata[:, cat_features].X,
        columns=cat_features,
    )

    if "encoding_mode" in edata.var:
        unencoded_vars = edata.var.loc[cat_features, "unencoded_var_names"].unique().tolist()

        for unencoded in sorted(unencoded_vars):
            if unencoded in edata.var_names:
                branch.add(f"{unencoded} ({df.loc[:, unencoded].nunique()} categories)")
            else:
                enc_mode = edata.var.loc[edata.var["unencoded_var_names"] == unencoded, "encoding_mode"].values[0]
                branch.add(f"{unencoded} ({edata.obs[unencoded].nunique()} categories); {enc_mode} encoded")

    else:
        for categorical in sorted(cat_features):
            categorical_feature = df.loc[:, categorical]
            categorical_feature[categorical_feature.isin(MISSING_VALUES)] = np.nan
            branch.add(f"{categorical} ({categorical_feature.nunique()} categories)")

    print(tree)


def replace_feature_types(
    edata: EHRData,
    features: Iterable[str],
    corrected_type: Literal["categorical", "numeric", "date"],
) -> None:
    """Correct the feature types for a list of features inplace.

    Args:
        edata: Data object.
        features: The features to correct.
        corrected_type: The corrected feature type. One of `'date'`, `'categorical'`, or `'numeric'`.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.diabetes_130_fairlearn()
        >>> ed.infer_feature_types(edata)
        >>> ed.replace_feature_types(edata, ["time_in_hospital", "number_diagnoses", "num_procedures"], "numeric")
    """
    if FEATURE_TYPE_KEY not in edata.var:
        err_msg = "Feature types were not inferred. Please infer feature types using 'ed.infer_feature_types' before correcting."
        raise ValueError(err_msg)

    if corrected_type not in [CATEGORICAL_TAG, NUMERIC_TAG, DATE_TAG]:
        err_msg = f"Corrected type {corrected_type} not recognized. Choose between '{DATE_TAG}', '{CATEGORICAL_TAG}', or '{NUMERIC_TAG}'."
        raise ValueError(err_msg)

    if FEATURE_TYPE_KEY not in edata.var:
        err_msg = (
            "Feature types were not inferred. Please infer feature types using 'infer_feature_types' before correcting."
        )
        raise ValueError(err_msg)

    if isinstance(features, str):
        features = [features]

    edata.var.loc[features, FEATURE_TYPE_KEY] = corrected_type


@singledispatch
def _harmonize_missing_values_numeric(X, *, var_names: pd.Index, vars: Iterable[str] | None) -> tuple[object, bool]:
    """Treat the implicit zero fill value of a numeric array as missing, if that array type is ambiguous about it."""
    return X, False


@_harmonize_missing_values_numeric.register(sparse.COO)
def _(X: sparse.COO, *, var_names: pd.Index, vars: Iterable[str] | None) -> tuple[sparse.COO, bool]:
    """Swap a numeric :class:`sparse.COO` array's zero fill value to `np.nan`, keeping `vars` columns real."""
    if np.issubdtype(X.dtype, np.bool_) or X.fill_value != 0:
        return X, False

    excluded = (
        np.asarray([var_names.get_loc(v) for v in vars], dtype=np.intp)
        if vars is not None
        else np.array([], dtype=np.intp)
    )

    var_coord = X.coords[1]
    is_excluded = np.isin(var_coord, excluded) if excluded.size else np.zeros(var_coord.shape, dtype=bool)

    data = X.data.astype(np.float64, copy=True)
    data[(data == 0) & ~is_excluded] = np.nan
    coords = X.coords

    if excluded.size:
        # Materialize every position of the excluded columns that isn't already stored explicitly,
        # since those positions would otherwise silently read back as the new np.nan fill value.
        excluded_shape = list(X.shape)
        excluded_shape[1] = excluded.size
        stored = np.zeros(excluded_shape, dtype=bool)

        pos_in_excluded = np.full(X.shape[1], -1, dtype=np.intp)
        pos_in_excluded[excluded] = np.arange(excluded.size)

        if is_excluded.any():
            index = (
                coords[0][is_excluded],
                pos_in_excluded[var_coord[is_excluded]],
                *(coords[ax][is_excluded] for ax in range(2, X.ndim)),
            )
            stored[index] = True

        missing = np.argwhere(~stored)
        if missing.size:
            missing = missing.T.copy()
            missing[1] = excluded[missing[1]]
            coords = np.concatenate([coords, missing], axis=1)
            data = np.concatenate([data, np.zeros(missing.shape[1], dtype=np.float64)])

    return sparse.COO(coords, data, shape=X.shape, fill_value=np.nan), True


@_harmonize_missing_values_numeric.register(sp.csr_array)
@_harmonize_missing_values_numeric.register(sp.csr_matrix)
@_harmonize_missing_values_numeric.register(sp.csc_array)
@_harmonize_missing_values_numeric.register(sp.csc_matrix)
def _(X, *, var_names: pd.Index, vars: Iterable[str] | None) -> tuple[object, bool]:
    err_msg = (
        f"ed.harmonize_missing_values cannot treat the implicit zero fill value as missing for "
        f"array type {type(X)}. Please convert to a dense array or sparse.COO first."
    )
    raise NotImplementedError(err_msg)


def harmonize_missing_values(
    edata: EHRData,
    *,
    layer: str | None = None,
    missing_values: Iterable[str] | None = ["nan", "np.nan", "<NA>", "pd.NA"],
    vars: Iterable[str] | None = None,
    copy: bool = False,
) -> EHRData | None:
    """Harmonize missing values in the :class:`~ehrdata.EHRData` object.

    This function will replace strings that are considered to represent missing values with `np.nan`.

    For a numeric :class:`sparse.COO` layer, the implicit fill value `0` is ambiguous between "not
    measured" and "measured as zero". This function treats it as missing for every variable except
    those listed in `vars`.
    Columns not in `vars` stay sparse; columns in `vars` keep `0` as a real value,
    which requires materializing their previously-implicit zeros as explicit stored entries so they
    don't pick up the new `np.nan` fill value. Boolean layers (sparse or dense) are also
    left untouched, as `0`/`False` is not ambiguous for them.

    Args:
        edata: Data object.
        layer: The layer to use from the :class:`~ehrdata.EHRData` object. If `None`, the `X` layer is used.
        missing_values: The strings that are considered to represent missing values and should be replaced with np.nan
        vars: For a sparse :class:`sparse.COO` layer only: variable names whose `0` values represent a
            real, measured zero rather than a missing value, and so are excluded from the
            zero-as-missing treatment described above.
        copy: Whether to return a copy of the :class:`~ehrdata.EHRData` object with the missing values replaced.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.harmonize_missing_values(edata)
    """
    if copy:
        edata = edata.copy()
    X = edata.X if layer is None else edata.layers[layer]

    # reject non numeric sparse.COO explicitly rather than silently densifying
    if isinstance(X, sparse.COO) and not (np.issubdtype(X.dtype, np.number) or np.issubdtype(X.dtype, np.bool_)):
        err_msg = (
            f"ed.harmonize_missing_values can only be used on a numeric sparse.COO layer "
            f"({'X' if layer is None else layer!r} has dtype {X.dtype})."
        )
        raise NotImplementedError(err_msg)

    # every scipy sparse array is of a numeric dtype and will enter this if block
    if np.issubdtype(X.dtype, np.number) or np.issubdtype(X.dtype, np.bool_):
        harmonized, changed = _harmonize_missing_values_numeric(X, var_names=edata.var_names, vars=vars)

        if changed:
            logger.debug(
                f"ed.harmonize_missing_values treats the implicit zero fill value as missing in sparse.COO layer {'X' if layer is None else layer}."
            )
            if layer is None:
                edata.X = harmonized
            else:
                edata.layers[layer] = harmonized
        else:
            logger.debug(
                f"ed.harmonize_missing_values does not affect numeric layer {'X' if layer is None else layer}."
            )

        return edata if copy else None

    df = pd.DataFrame(X.reshape(-1, edata.shape[1]), columns=edata.var_names)
    df[df.isin(missing_values)] = np.nan

    if layer is None:
        edata.X = df.values.reshape(X.shape)
    else:
        edata.layers[layer] = df.values.reshape(X.shape)

    return edata if copy else None


def _harmonize_on_read(edata: EHRData) -> None:
    layers = [None] if edata.X is not None else []
    # anndata 0.13's unified `.X` slot shows up as key None in edata.layers too; skip the duplicate
    layers += [key for key in edata.layers if key is not None]

    for layer in layers:
        label = "X" if layer is None else f"layer {layer}"
        try:
            harmonize_missing_values(edata, layer=layer)
            logger.info(f"Harmonizing missing values of {label}")
        except NotImplementedError:
            # scipy.sparse (CSR/CSC) can't be harmonized without densifying
            logger.debug(
                f"Skipping missing-value harmonization of {label}: not supported for scipy.sparse without densifying."
            )
