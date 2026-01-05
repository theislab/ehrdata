from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import ehrdata as ed
from ehrdata import EHRData
from ehrdata._types import ARRAY_TYPES_NUMERIC
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME

if TYPE_CHECKING:
    from collections.abc import Callable
import scipy.sparse as sp


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NUMERIC)
@pytest.mark.parametrize("copy_columns", [True, False])
@pytest.mark.parametrize("copy", [True, False])
def test_move_to_obs_vanilla(edata_330: EHRData, array_type: Callable, *, copy_columns: bool, copy: bool):
    edata_330.X = array_type(edata_330.X)
    edata_reference = edata_330.copy()

    original_shape = edata_330.X.shape
    original_obs_cols = len(edata_330.obs.columns)
    var_names = ["var1", "var2"]

    result = ed.move_to_obs(edata_330, var_names, copy_columns=copy_columns, copy=copy)

    edata_to_check = result if copy else edata_330

    expected_n_vars = original_shape[1] if copy_columns else (original_shape[1] - len(var_names))
    assert edata_to_check.n_vars == expected_n_vars
    assert edata_to_check.n_obs == original_shape[0]

    if not copy_columns:
        assert list(edata_to_check.var_names) == ["var3"]

    assert edata_to_check.obs.shape[1] == original_obs_cols + 2
    assert "var1" in edata_to_check.obs.columns
    assert "var2" in edata_to_check.obs.columns

    assert edata_to_check.obs.loc["obs1", "var1"] == 1
    assert edata_to_check.obs.loc["obs2", "var1"] == 4
    assert edata_to_check.obs.loc["obs3", "var1"] == 7

    assert isinstance(edata_to_check.X, type(edata_reference.X))


def test_move_to_obs_layer(edata_330: EHRData):
    edata_330.layers["layer1"] = edata_330.X.copy()

    original_shape = edata_330.X.shape
    var_names = ["var1", "var2"]

    ed.move_to_obs(edata_330, var_names, layer="layer1")

    assert edata_330.shape[1] == original_shape[1] - len(var_names)
    assert edata_330.X.shape[1] == original_shape[1] - len(var_names)

    assert "var1" in edata_330.obs.columns
    assert "var2" in edata_330.obs.columns


@pytest.mark.parametrize("array_type", ARRAY_TYPES_NUMERIC)
@pytest.mark.parametrize("copy_columns", [True, False])
def test_move_to_x_vanilla(edata_330: EHRData, array_type: Callable, *, copy_columns: bool):
    edata_330.X = array_type(edata_330.X)
    edata_reference = edata_330.copy()

    original_obs_cols = len(edata_330.obs.columns)

    new_edata = ed.move_to_x(edata_330, ["obs_col_1"], copy_columns=copy_columns)

    assert "obs_col_1" in new_edata.var_names
    assert new_edata.n_vars == edata_reference.n_vars + 1

    if copy_columns:
        assert "obs_col_1" in new_edata.obs.columns
        assert new_edata.obs.shape[1] == original_obs_cols
    else:
        assert "obs_col_1" not in new_edata.obs.columns
        assert new_edata.obs.shape[1] == original_obs_cols - 1

    # for sparse csc arrays, anndata's concat transforms to csr.
    if sp.issparse(edata_reference.X):
        assert sp.issparse(new_edata.X)
    else:
        assert isinstance(new_edata.X, type(edata_reference.X))


def test_move_to_x_layer(edata_330: EHRData):
    edata_330.layers["layer1"] = edata_330.X.copy()

    original_shape = edata_330.X.shape
    features = ["obs_col_1"]

    edata_result = ed.move_to_x(edata_330, features, layer="layer1")

    assert edata_result.shape[1] == original_shape[1] + len(features)
    assert edata_result.X is None


def test_move_to_obs_3Dlayer(edata_333: EHRData):
    with pytest.raises(ValueError, match=r"Layer is 3D, but move_to_obs only supports 2D layers."):
        ed.move_to_obs(edata_333, ["var1"], layer=DEFAULT_TEM_LAYER_NAME)


def test_move_to_x_3Dlayer(edata_333: EHRData):
    with pytest.raises(ValueError, match=r"Layer is 3D, but move_to_x only supports 2D layers."):
        ed.move_to_x(edata_333, ["obs_col_1"], layer=DEFAULT_TEM_LAYER_NAME)


def test_move_to_obs_invalid_column_name(edata_330: EHRData):
    with pytest.raises(ValueError, match="are not in var_names"):
        ed.move_to_obs(edata_330, "invalid_var")

    with pytest.raises(ValueError, match="are not in var_names"):
        ed.move_to_obs(edata_330, ["invalid_var1", "invalid_var2"])

    with pytest.raises(ValueError, match="are not in var_names"):
        ed.move_to_obs(edata_330, ["var1", "invalid_var"])


def test_move_to_x_invalid_column_name(edata_330: EHRData):
    with pytest.raises(ValueError, match="are not in obs"):
        ed.move_to_x(edata_330, "invalid_col")

    with pytest.raises(ValueError, match="are not in obs"):
        ed.move_to_x(edata_330, ["invalid_col1", "invalid_col2"])

    with pytest.raises(ValueError, match="are not in obs"):
        ed.move_to_x(edata_330, ["obs1", "invalid_col"])
