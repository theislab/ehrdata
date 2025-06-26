import numpy as np
import pandas as pd
import pytest

from ehrdata import EHRData, feature_type_overview, harmonize_missing_values, infer_feature_types, replace_feature_types
from ehrdata.core.constants import MISSING_VALUES


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
    R = data.values.reshape(2, -1, 2)
    edata = EHRData(R=R)
    harmonize_missing_values(edata, layer="R_layer")
    for missing_value_string in MISSING_VALUES:
        assert missing_value_string not in edata.R.flatten()


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
    data = pd.DataFrame(data)
    R = data.values.reshape(2, -1, 2)
    edata = EHRData(R=R)
    infer_feature_types(edata, layer="R_layer")

    assert "feature_type" in edata.var.columns
    assert all(edata.var["feature_type"] == list(target_types.values()))


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
