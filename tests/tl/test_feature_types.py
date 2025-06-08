import numpy as np
import pandas as pd
import pytest

from ehrdata import EHRData
from ehrdata.core.constants import MISSING_VALUES
from ehrdata.tl import feature_type_overview, harmonize_missing_values, infer_feature_types, replace_feature_types


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


def test_feature_type_overview_vanilla(variable_type_samples):
    edata = EHRData(X=pd.DataFrame(variable_type_samples))
    feature_type_overview(edata)
    # assert something


def test_replace_feature_types(variable_type_samples):
    edata = EHRData(X=pd.DataFrame(variable_type_samples))
    replace_feature_types(edata, ["feature1", "feature2", "feature3"], "categorical")
    # assert something


# def test_feature_type_inference(adata):
#     ep.ad.infer_feature_types(adata, output=None)
#     assert adata.var[FEATURE_TYPE_KEY]["feature1"] == CATEGORICAL_TAG
#     assert adata.var[FEATURE_TYPE_KEY]["feature2"] == CATEGORICAL_TAG
#     assert adata.var[FEATURE_TYPE_KEY]["feature3"] == CATEGORICAL_TAG
#     assert adata.var[FEATURE_TYPE_KEY]["feature4"] == NUMERIC_TAG
#     assert adata.var[FEATURE_TYPE_KEY]["feature5"] == CATEGORICAL_TAG
#     assert adata.var[FEATURE_TYPE_KEY]["feature6"] == NUMERIC_TAG
#     assert adata.var[FEATURE_TYPE_KEY]["feature7"] == DATE_TAG


# def test_check_feature_types(adata):
#     @check_feature_types
#     def test_func(adata):
#         pass

#     assert FEATURE_TYPE_KEY not in adata.var.keys()
#     test_func(adata)
#     assert FEATURE_TYPE_KEY in adata.var.keys()

#     ep.ad.infer_feature_types(adata, output=None)
#     test_func(adata)
#     assert FEATURE_TYPE_KEY in adata.var.keys()

#     @check_feature_types
#     def test_func_with_return(adata):
#         return adata

#     adata = test_func_with_return(adata)
#     assert FEATURE_TYPE_KEY in adata.var.keys()


# def test_feature_types_impute_num_adata(impute_num_adata):
#     ep.ad.infer_feature_types(impute_num_adata, output=None)
#     assert np.all(impute_num_adata.var[FEATURE_TYPE_KEY] == [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG])


# def test_feature_types_impute_adata(impute_adata):
#     ep.ad.infer_feature_types(impute_adata, output=None)
#     assert np.all(impute_adata.var[FEATURE_TYPE_KEY] == [NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG, CATEGORICAL_TAG])


# def test_feature_types_impute_iris(impute_iris_adata):
#     ep.ad.infer_feature_types(impute_iris_adata, output=None)
#     assert np.all(
#         impute_iris_adata.var[FEATURE_TYPE_KEY] == [NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, NUMERIC_TAG, CATEGORICAL_TAG]
#     )


# def test_feature_types_impute_feature_types_titanic(impute_titanic_adata):
#     ep.ad.infer_feature_types(impute_titanic_adata, output=None)
#     impute_titanic_adata.var[FEATURE_TYPE_KEY] = [
#         CATEGORICAL_TAG,
#         CATEGORICAL_TAG,
#         CATEGORICAL_TAG,
#         CATEGORICAL_TAG,
#         CATEGORICAL_TAG,
#         NUMERIC_TAG,
#         NUMERIC_TAG,
#         NUMERIC_TAG,
#         NUMERIC_TAG,
#         NUMERIC_TAG,
#         CATEGORICAL_TAG,
#         CATEGORICAL_TAG,
#     ]


# def test_date_detection():
#     df = pd.DataFrame(
#         {
#             "date1": pd.to_datetime(["2021-01-01", "2024-04-16", "2021-01-03"]),
#             "date2": ["2021-01-01", "2024-04-16", "2021-01-03"],
#             "date3": ["2024-04-16 07:45:13", "2024-04-16", "2024-04"],
#             "not_date": ["not_a_date", "2024-04-16", "2021-01-03"],
#         }
#     )
#     adata = df_to_anndata(df)
#     ep.ad.infer_feature_types(adata, output=None)
#     assert np.all(adata.var[FEATURE_TYPE_KEY] == [DATE_TAG, DATE_TAG, DATE_TAG, CATEGORICAL_TAG])


# def test_all_possible_types():
#     df = pd.DataFrame(
#         {
#             "f1": [42, 17, 93, 235],
#             "f2": ["apple", "banana", "cherry", "date"],
#             "f3": [1, 2, 3, 1],
#             "f4": [1.0, 2.0, 1.0, 2.0],
#             "f5": ["20200101", "20200102", "20200103", "20200104"],
#             "f6": [True, False, True, False],
#             "f7": [np.nan, 1, np.nan, 2],
#             "f8": ["apple", 1, "banana", 2],
#             "f9": ["001", "002", "003", "002"],
#             "f10": ["5", "5", "5", "5"],
#             "f11": ["A1", "A2", "B1", "B2"],
#             "f12": [90210, 10001, 60614, 80588],
#             "f13": [0.25, 0.5, 0.75, 1.0],
#             "f14": ["2125551234", "2125555678", "2125559012", "2125553456"],
#             "f15": ["$100", "€150", "£200", "¥250"],
#             "f16": [101, 102, 103, 104],
#             "f17": [1e3, 5e-2, 3.1e2, 2.7e-1],
#             "f18": ["23.5", "324", "4.5", "0.5"],
#             "f19": [1, 2, 3, 4],
#             "f20": ["001", "002", "003", "004"],
#         }
#     )

#     adata = df_to_anndata(df)
#     ep.ad.infer_feature_types(adata, output=None)

#     assert np.all(
#         adata.var[FEATURE_TYPE_KEY]
#         == [
#             NUMERIC_TAG,
#             CATEGORICAL_TAG,
#             CATEGORICAL_TAG,
#             CATEGORICAL_TAG,
#             DATE_TAG,
#             CATEGORICAL_TAG,
#             CATEGORICAL_TAG,
#             CATEGORICAL_TAG,
#             CATEGORICAL_TAG,
#             NUMERIC_TAG,
#             CATEGORICAL_TAG,
#             NUMERIC_TAG,
#             NUMERIC_TAG,
#             NUMERIC_TAG,
#             CATEGORICAL_TAG,
#             NUMERIC_TAG,
#             NUMERIC_TAG,
#             NUMERIC_TAG,
#             NUMERIC_TAG,
#             NUMERIC_TAG,
#         ]
#     )


# def test_partial_annotation(adata):
#     adata.var[FEATURE_TYPE_KEY] = ["dummy", np.nan, np.nan, NUMERIC_TAG, None, np.nan, None]
#     ep.ad.infer_feature_types(adata, output=None)
#     assert np.all(
#         adata.var[FEATURE_TYPE_KEY]
#         == ["dummy", CATEGORICAL_TAG, CATEGORICAL_TAG, NUMERIC_TAG, CATEGORICAL_TAG, NUMERIC_TAG, DATE_TAG]
#     )
