from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from dateutil.parser import isoparse  # type: ignore
from fast_array_utils.conv import to_dense
from lamin_utils import logger
from rich import print
from rich.tree import Tree
from scipy.sparse import issparse

from ehrdata.core.constants import CATEGORICAL_TAG, DATE_TAG, FEATURE_TYPE_KEY, MISSING_VALUES, NUMERIC_TAG

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ehrdata import EHRData


def _detect_feature_type(col: pd.Series) -> tuple[Literal["date", "categorical", "numeric"], bool]:
    """Detect the feature type of a :class:`~pandas.Series`.

    Args:
        col: The series to detect the feature type of.

    Returns:
        The detected feature type (one of 'date', 'categorical', or 'numeric') and a boolean, which is True if the feature type is uncertain.
    """
    n_elements = len(col)
    col = col.replace(MISSING_VALUES, np.nan)
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

    # Guess categorical if the feature is an integer and the values are 0/1 to n-1/n with no gaps
    if (
        (majority_type is int or (np.all(i.is_integer() for i in col)))
        and (n_elements != col.nunique())
        and (
            (col.min() == 0 and np.all(np.sort(col.unique()) == np.arange(col.nunique())))
            or (col.min() == 1 and np.all(np.sort(col.unique()) == np.arange(1, col.nunique() + 1)))
        )
    ):
        return CATEGORICAL_TAG, True  # type: ignore

    return NUMERIC_TAG, False  # type: ignore


def infer_feature_types(
    edata: EHRData,
    *,
    layer: str | None = None,
    output: Literal["tree", "dataframe"] | None = "tree",
    verbose: bool = True,
) -> pd.DataFrame | None:
    """Infer feature types from an :class:`~ehrdata.EHRData` object.

    For each feature in `edata.var_names`, the method infers one of the following types: `'date'`, `'categorical'`, or `'numeric'`.
    The inferred types are stored in `edata.var['feature_type']`. Please check the inferred types and adjust if necessary using
    `edata.var['feature_type']['feature1']='corrected_type'` or with :func:`~ehrdata.replace_feature_types`.
    Be aware that not all features stored numerically are of `'numeric'` type, as categorical features might be stored in a numerically encoded format.
    For example, a feature with values [0, 1, 2] might be a categorical feature with three categories. This is accounted for in the method, but it is
    recommended to check the inferred types.

    Args:
        edata: Data object.
        layer: The layer to use from the EHRData object. If `None`, the `X` field is used.
        output: The output format. Choose between `'tree'`, `'dataframe'`, or `None`. If `'tree'`, the feature types will be printed to the console in a tree format.
            If `'dataframe'`, a :class:`~pandas.DataFrame` with the feature types will be returned. If `None`, nothing will be returned.
        verbose: Whether to print warnings for uncertain feature types.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.infer_feature_types(edata)
    """
    feature_types = {}
    uncertain_features = []

    X = edata.X if layer is None else edata.layers[layer]

    if issparse(X):
        X = to_dense(X)

    df = pd.DataFrame(X.reshape(-1, edata.shape[1]), columns=edata.var_names)

    for feature in edata.var_names:
        if (
            FEATURE_TYPE_KEY in edata.var
            and edata.var[FEATURE_TYPE_KEY][feature] is not None
            and not pd.isna(edata.var[FEATURE_TYPE_KEY][feature])
        ):
            feature_types[feature] = edata.var[FEATURE_TYPE_KEY][feature]
        else:
            feature_types[feature], raise_warning = _detect_feature_type(df[feature])
            if raise_warning:
                uncertain_features.append(feature)

    edata.var[FEATURE_TYPE_KEY] = pd.Series(feature_types)[edata.var_names]

    if verbose:
        logger.warning(
            f"{'Features' if len(uncertain_features) > 1 else 'Feature'} {str(uncertain_features)[1:-1]} {'were' if len(uncertain_features) > 1 else 'was'} detected as categorical features stored numerically."
            f"Please verify and adjust if necessary using `ed.replace_feature_types`."
        )

        logger.info(
            f"Stored feature types in edata.var['{FEATURE_TYPE_KEY}']."
            f" Please verify and adjust if necessary using `ed.replace_feature_types`."
        )

    if output == "tree":
        feature_type_overview(edata)
    elif output == "dataframe":
        return edata.var[FEATURE_TYPE_KEY].to_frame()
    elif output is not None:
        err_msg = f"Output format {output} not recognized. Choose between 'tree', 'dataframe', or None."
        raise ValueError(err_msg)


# TODO: this function is a different flavor of inferring feature types. We should decide on a single implementation in the future.
def _infer_numerical_column_indices(
    edata: EHRData, layer: str | None = None, column_indices: Iterable[int] | None = None
) -> list[int]:
    mtx = edata.X if layer is None else edata[layer]
    indices = (
        list(range(mtx.shape[1])) if column_indices is None else [i for i in column_indices if i < mtx.shape[1] - 1]
    )
    non_numerical_indices = []
    for i in indices:
        # The astype("float64") call will throw only if the feature's data type cannot be cast to float64, meaning in
        # practice it contains non-numeric values. Consequently, it won't throw if the values are numeric but stored
        # as an "object" dtype, as astype("float64") can successfully convert them to floats.
        try:
            mtx[::, i].astype("float64")
        except ValueError:
            non_numerical_indices.append(i)

    return [idx for idx in indices if idx not in non_numerical_indices]


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
                f"Feature types were inferred and stored in edata.var[{FEATURE_TYPE_KEY}]. Please verify using `ed.feature_type_overview` and adjust if necessary using `ed.replace_feature_types`."
            )

        for feature in edata.var_names:
            feature_type = edata.var[FEATURE_TYPE_KEY][feature]
            if (
                feature_type is not None
                and (not pd.isna(feature_type))
                and feature_type not in [CATEGORICAL_TAG, NUMERIC_TAG, DATE_TAG]
            ):
                logger.warning(
                    f"Feature '{feature}' has an invalid feature type '{feature_type}'. Please correct using `ed.replace_feature_types`."
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
            categorical_feature_nans_cleaned = categorical_feature.replace(MISSING_VALUES, np.nan)
            branch.add(f"{categorical} ({categorical_feature_nans_cleaned.nunique()} categories)")

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


def harmonize_missing_values(
    edata: EHRData,
    *,
    layer: str | None = None,
    missing_values: Iterable[str] | None = ["nan", "np.nan", "<NA>", "pd.NA"],
    copy: bool = False,
) -> EHRData | None:
    """Harmonize missing values in the :class:`~ehrdata.EHRData` object.

    This function will replace strings that are considered to represent missing values with `np.nan`.

    Args:
        edata: Data object.
        layer: The layer to use from the :class:`~ehrdata.EHRData` object. If `None`, the `X` layer is used.
        missing_values: The strings that are considered to represent missing values and should be replaced with np.nan
        copy: Whether to return a copy of the :class:`~ehrdata.EHRData` object with the missing values replaced.

    Examples:
        >>> import ehrdata as ed
        >>> edata = ed.dt.mimic_2()
        >>> ed.harmonize_missing_values(edata)
    """
    if copy:
        edata = edata.copy()
    X = edata.X if layer is None else edata.layers[layer]

    # note that every sparse array is of a numeric dtype and will enter this if block
    if np.issubdtype(X.dtype, np.number):
        logger.warning(f"This operation does not affect numeric layer {'X' if layer is None else layer}.")
        return edata if copy else None

    df = pd.DataFrame(X.reshape(-1, edata.shape[1]), columns=edata.var_names)
    df.replace(missing_values, np.nan, inplace=True)

    if layer is None:
        edata.X = df.values.reshape(X.shape)
    else:
        edata.layers[layer] = df.values.reshape(X.shape)

    return edata if copy else None
