import numpy as np
import pandas as pd
import pytest
import sparse
from scipy.sparse import csr_matrix

from ehrdata import EHRData, feature_type_overview, harmonize_missing_values, infer_feature_types, replace_feature_types
from ehrdata._logger import logger
from ehrdata.core.constants import DEFAULT_TEM_LAYER_NAME, MISSING_VALUES


@pytest.mark.parametrize(
    "sample_dataset",
    [
        "variable_type_samples",
        "variable_type_samples_string_format",
    ],
)
def test_harmonize_missing_values(sample_dataset, request):
    data, _ = request.getfixturevalue(sample_dataset)
    data = pd.DataFrame(data)
    edata = EHRData(X=data.values, var=pd.DataFrame(data.columns))
    harmonize_missing_values(edata)

    for missing_value_string in MISSING_VALUES:
        assert missing_value_string not in edata.X.flatten()


@pytest.mark.parametrize(
    "sample_dataset",
    [
        "variable_type_samples",
        "variable_type_samples_string_format",
    ],
)
def test_harmonize_missing_values_layer(sample_dataset, request):
    data, _ = request.getfixturevalue(sample_dataset)
    data = pd.DataFrame(data)
    edata = EHRData(X=np.full(data.values.shape, "nan"), layers={"layer1": data.values})
    harmonize_missing_values(edata, layer="layer1")
    for missing_value_string in MISSING_VALUES:
        assert missing_value_string not in edata.layers["layer1"].flatten()


@pytest.mark.parametrize(
    "sample_dataset",
    [
        "variable_type_samples",
        "variable_type_samples_string_format",
    ],
)
def test_harmonize_missing_values_3D(sample_dataset, request):
    data, _ = request.getfixturevalue(sample_dataset)
    data = pd.DataFrame(data)
    tem_layer = data.values.reshape(2, -1, 2)
    edata = EHRData(layers={DEFAULT_TEM_LAYER_NAME: tem_layer})
    harmonize_missing_values(edata, layer=DEFAULT_TEM_LAYER_NAME)
    for missing_value_string in MISSING_VALUES:
        assert missing_value_string not in edata.layers[DEFAULT_TEM_LAYER_NAME].flatten()


def test_harmonize_missing_values_sparse_coo():
    dense = np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    X = sparse.COO.from_numpy(dense)
    edata = EHRData(X=X)

    harmonize_missing_values(edata)

    assert isinstance(edata.X, sparse.COO)
    assert edata.X.nnz == X.nnz  # stays sparse
    assert np.isnan(edata.X.fill_value)
    np.testing.assert_array_equal(edata.X.todense(), [[1.0, np.nan, 2.0], [np.nan, np.nan, 3.0]])


def test_harmonize_missing_values_sparse_coo_3d():
    dense = np.array([[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], [[0.0, 0.0], [4.0, 0.0], [0.0, 5.0]]])
    X = sparse.COO.from_numpy(dense)
    edata = EHRData(layers={DEFAULT_TEM_LAYER_NAME: X})

    harmonize_missing_values(edata, layer=DEFAULT_TEM_LAYER_NAME)

    result = edata.layers[DEFAULT_TEM_LAYER_NAME]
    assert isinstance(result, sparse.COO)
    assert result.nnz == X.nnz
    assert np.isnan(result.fill_value)


def test_harmonize_missing_values_sparse_coo_vars_excluded():
    dense = np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 0.0, 0.0]])
    X = sparse.COO.from_numpy(dense)
    edata = EHRData(X=X, var=pd.DataFrame(index=["v0", "v1", "v2"]))

    harmonize_missing_values(edata, vars=["v1"])

    result = edata.X
    assert isinstance(result, sparse.COO)
    assert np.isnan(result.fill_value)
    np.testing.assert_array_equal(
        result.todense(),
        [[1.0, 0.0, 2.0], [np.nan, 0.0, 3.0], [4.0, 0.0, np.nan]],
    )


def test_harmonize_missing_values_sparse_coo_3d_vars_excluded():
    dense = np.array([[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], [[0.0, 0.0], [4.0, 0.0], [0.0, 5.0]]])
    X = sparse.COO.from_numpy(dense)
    edata = EHRData(layers={DEFAULT_TEM_LAYER_NAME: X}, var=pd.DataFrame(index=["v0", "v1", "v2"]))

    harmonize_missing_values(edata, layer=DEFAULT_TEM_LAYER_NAME, vars=["v1"])

    result = edata.layers[DEFAULT_TEM_LAYER_NAME]
    assert isinstance(result, sparse.COO)
    assert np.isnan(result.fill_value)
    np.testing.assert_array_equal(
        result.todense(),
        [[[1.0, np.nan], [0.0, 2.0], [3.0, np.nan]], [[np.nan, np.nan], [4.0, 0.0], [np.nan, 5.0]]],
    )


def test_harmonize_missing_values_sparse_coo_unknown_var_raises():
    dense = np.array([[1.0, 0.0], [0.0, 2.0]])
    X = sparse.COO.from_numpy(dense)
    edata = EHRData(X=X, var=pd.DataFrame(index=["v0", "v1"]))

    with pytest.raises(KeyError):
        harmonize_missing_values(edata, vars=["bogus"])


def test_harmonize_missing_values_scipy_sparse_unaffected():
    dense = np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    X = csr_matrix(dense)
    edata = EHRData(X=X)

    harmonize_missing_values(edata)

    assert isinstance(edata.X, csr_matrix)
    np.testing.assert_array_equal(edata.X.toarray(), dense)


@pytest.mark.parametrize(
    "sample_dataset",
    [
        "variable_type_samples",
        "variable_type_samples_string_format",
    ],
)
def test_feature_type_inference_vanilla(sample_dataset, request):
    data, target_types = request.getfixturevalue(sample_dataset)
    data = pd.DataFrame(data)
    edata = EHRData(X=data.values, var=pd.DataFrame(data.columns))
    infer_feature_types(edata)

    assert "feature_type" in edata.var.columns
    assert all(edata.var["feature_type"] == list(target_types.values()))


@pytest.mark.parametrize(
    "sample_dataset",
    [
        "variable_type_samples",
        "variable_type_samples_string_format",
    ],
)
def test_feature_type_inference_layer(sample_dataset, request):
    data, target_types = request.getfixturevalue(sample_dataset)
    data = pd.DataFrame(data)
    edata = EHRData(X=np.ones(data.values.shape), layers={"layer1": data.values})
    infer_feature_types(edata, layer="layer1")

    assert "feature_type" in edata.var.columns
    assert all(edata.var["feature_type"] == list(target_types.values()))


@pytest.mark.parametrize(
    "sample_dataset",
    [
        "variable_type_samples",
        "variable_type_samples_string_format",
    ],
)
def test_feature_type_inference_3D(sample_dataset, request):
    data, target_types = request.getfixturevalue(sample_dataset)

    data = pd.DataFrame(data)  # (4, 11)
    arr = data.values  # (4, 11)
    tem_layer = np.stack([arr[:2], arr[2:]], axis=2)  # (2,11,2)
    edata = EHRData(layers={DEFAULT_TEM_LAYER_NAME: tem_layer})
    infer_feature_types(edata, layer=DEFAULT_TEM_LAYER_NAME)

    assert "feature_type" in edata.var.columns
    assert all(edata.var["feature_type"] == list(target_types.values()))


@pytest.mark.parametrize(
    ("binary_as", "expected"),
    [("categorical", "categorical"), ("numeric", "numeric")],
)
def test_feature_type_inference_float_encoded_binary(binary_as, expected):
    edata = EHRData(
        X=np.array([[0.0], [1.0], [0.0], [1.0]]),
        var=pd.DataFrame(index=["binary_feature"]),
    )
    infer_feature_types(edata, binary_as=binary_as, output=None)

    assert edata.var["feature_type"]["binary_feature"] == expected


def test_feature_type_inference_sparse_coo():
    dense = np.array([[1.0, 2.0, 0.0], [4.0, 0.0, 6.0], [0.0, 5.0, 7.0], [8.0, 0.0, 0.0]])
    X = sparse.COO.from_numpy(dense)
    edata = EHRData(X=X, var=pd.DataFrame(index=["v0", "v1", "v2"]))

    infer_feature_types(edata, output=None)

    assert list(edata.var["feature_type"]) == ["numeric", "numeric", "numeric"]
    assert isinstance(edata.X, sparse.COO)
    assert edata.X.nnz == X.nnz  # inference must not densify or otherwise mutate the stored array


def test_feature_type_inference_sparse_coo_binary():
    dense = np.array([[0.0, 5.0], [1.0, 0.0], [0.0, 3.0], [1.0, 0.0]])
    X = sparse.COO.from_numpy(dense)
    edata = EHRData(X=X, var=pd.DataFrame(index=["bin", "num"]))

    infer_feature_types(edata, output=None)

    assert list(edata.var["feature_type"]) == ["categorical", "numeric"]


def test_feature_type_inference_sparse_coo_binary_as_numeric():
    dense = np.array([[0.0, 5.0], [1.0, 0.0], [0.0, 3.0], [1.0, 0.0]])
    X = sparse.COO.from_numpy(dense)
    edata = EHRData(X=X, var=pd.DataFrame(index=["bin", "num"]))

    infer_feature_types(edata, binary_as="numeric", output=None)

    assert list(edata.var["feature_type"]) == ["numeric", "numeric"]


def test_feature_type_inference_sparse_coo_all_nan_raises():
    dense = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    X = sparse.COO.from_numpy(dense)
    edata = EHRData(X=X, var=pd.DataFrame(index=["v0", "v1"]))

    with pytest.raises(ValueError, match="only NaN"):
        infer_feature_types(edata, output=None)


@pytest.mark.parametrize(
    "sample_dataset",
    [
        "variable_type_samples",
        "variable_type_samples_string_format",
    ],
)
def test_feature_type_overview_vanilla(sample_dataset, request, capsys):
    data, _ = request.getfixturevalue(sample_dataset)
    data = pd.DataFrame(data)
    edata = EHRData(X=data.values, var=pd.DataFrame(data.columns))
    feature_type_overview(edata)
    assert (
        " Detected feature types for EHRData object with 4 obs and 11 vars\n╠══ 📅 Date features\n╠══ 📐 Numerical features\n║   ╠══ 0\n║   ╠══ 1\n║   ╠══ 2\n║   ╠══ 3\n║   ╚══ 4\n╚══ 🗂️ Categorical features\n    ╠══ 10 (2 categories)\n    ╠══ 5 (4 categories)\n    ╠══ 6 (3 categories)\n    ╠══ 7 (2 categories)\n    ╠══ 8 (2 categories)\n    ╚══ 9 (2 categories)"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "sample_dataset",
    [
        "variable_type_samples",
        "variable_type_samples_string_format",
    ],
)
def test_replace_feature_types(sample_dataset, request):
    data, target_types = request.getfixturevalue(sample_dataset)
    data = pd.DataFrame(data)
    edata = EHRData(X=data.values, var=pd.DataFrame(data.columns).set_index(data.columns))
    infer_feature_types(edata)
    replace_feature_types(edata, ["int_column", "int_column_with_missing"], "categorical")

    target_types["int_column"] = "categorical"
    target_types["int_column_with_missing"] = "categorical"
    assert all(edata.var["feature_type"] == list(target_types.values()))


def test_infer_feature_types_warns_with_feature_name(monkeypatch):
    messages = []
    monkeypatch.setattr(logger, "warning", lambda msg, **kwargs: messages.append(msg))

    edata = EHRData(
        X=np.array([[0.0, 1.1], [1.0, 2.2], [0.0, 3.3], [1.0, 4.4]]),
        var=pd.DataFrame(index=["binary_feature", "numeric_feature"]),
    )
    infer_feature_types(edata, output=None)

    uncertain = [msg for msg in messages if "stored numerically" in msg]
    assert len(uncertain) == 1
    assert "'binary_feature'" in uncertain[0]
    assert "Feature  " not in uncertain[0]


def test_infer_feature_types_no_warning_without_uncertain_features(monkeypatch):
    messages = []
    monkeypatch.setattr(logger, "warning", lambda msg, **kwargs: messages.append(msg))

    edata = EHRData(
        X=np.array([[1.1], [2.2], [3.3], [4.4]]),
        var=pd.DataFrame(index=["numeric_feature"]),
    )
    infer_feature_types(edata, output=None)

    assert not [msg for msg in messages if "stored numerically" in msg]


def test_replace_feature_types_not_inferred_raises_error(variable_type_samples):
    data, _ = variable_type_samples
    data = pd.DataFrame(data)
    edata = EHRData(X=data.values, var=pd.DataFrame(data.columns).set_index(data.columns))

    with pytest.raises(ValueError):
        replace_feature_types(edata, ["int_column", "int_column_with_missing"], "categorical")


def test_replace_feature_types_invalid_type_raises_error(variable_type_samples):
    data, _ = variable_type_samples
    data = pd.DataFrame(data)
    edata = EHRData(X=data.values, var=pd.DataFrame(data.columns).set_index(data.columns))
    infer_feature_types(edata)
    with pytest.raises(KeyError):
        replace_feature_types(edata, ["misspelt_column"], "categorical")


def test_replace_feature_types_unknown_feature_raises_error(variable_type_samples):
    data, _ = variable_type_samples
    data = pd.DataFrame(data)
    edata = EHRData(X=data.values, var=pd.DataFrame(data.columns).set_index(data.columns))
    infer_feature_types(edata)
    with pytest.raises(ValueError):
        replace_feature_types(edata, ["int_column"], "invalid_target")
