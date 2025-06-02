import numpy as np
import pandas as pd
import pytest

from ehrdata import EHRData
from ehrdata.tl import from_dataframe, to_dataframe


@pytest.mark.parametrize("df", ["dataset_basic", "dataset_non_num_with_missing", "dataset_num_with_missing"])
def test_from_dataframe_basic(df, request):
    df = request.getfixturevalue(df)
    edata = from_dataframe(df)

    assert edata.shape == (df.shape[0], df.shape[1], 0)
    assert edata.X.shape == (df.shape[0], df.shape[1])
    assert edata.obs.shape == (df.shape[0], 0)
    assert edata.var.shape == (df.shape[1], 0)
    assert edata.tem.shape == (0, 0)

    assert np.array_equal(edata.var_names.values, df.columns.values)

    pd.testing.assert_frame_equal(pd.DataFrame(edata.X), pd.DataFrame(df.to_numpy()))
    desired_default_index = pd.RangeIndex(start=0, stop=len(df), step=1).astype(str)
    assert edata.obs.index.equals(desired_default_index)


@pytest.mark.parametrize("df", ["dataset_basic", "dataset_non_num_with_missing", "dataset_num_with_missing"])
@pytest.mark.parametrize("index_column", ["patient_id", 0])
def test_from_dataframe_basic_index(df, index_column, request):
    df = request.getfixturevalue(df)
    edata = from_dataframe(df, index_column=index_column)

    assert edata.shape == (df.shape[0], df.shape[1] - 1, 0)
    assert edata.X.shape == (df.shape[0], df.shape[1] - 1)
    assert edata.obs.shape == (df.shape[0], 0)
    assert edata.var.shape == (df.shape[1] - 1, 0)
    assert edata.tem.shape == (0, 0)

    assert np.array_equal(edata.var_names.values, df.columns.values[1:])

    pd.testing.assert_frame_equal(pd.DataFrame(edata.X), pd.DataFrame(df.iloc[:, 1:].to_numpy()))

    assert np.array_equal(
        edata.obs.index.values,
        df[index_column].values.astype(str)
        if isinstance(index_column, str)
        else df.iloc[:, index_column].values.astype(str),
    )


def test_from_dataframe_basic_index_invalid_index_throws_error(dataset_basic):
    with pytest.raises(ValueError):
        from_dataframe(dataset_basic, index_column="mistyped_column_name")

    with pytest.raises(IndexError):
        from_dataframe(dataset_basic, index_column=6)


@pytest.mark.parametrize("df", ["dataset_basic", "dataset_non_num_with_missing", "dataset_num_with_missing"])
def test_from_dataframe_basic_column_obs_only(df, request):
    df = request.getfixturevalue(df)
    edata = from_dataframe(df, columns_obs_only=["patient_id"])
    assert edata.shape == (df.shape[0], df.shape[1] - 1, 0)
    assert edata.X.shape == (df.shape[0], df.shape[1] - 1)
    assert edata.obs.shape == (df.shape[0], 1)
    assert edata.var.shape == (df.shape[1] - 1, 0)
    assert edata.tem.shape == (0, 0)

    assert np.array_equal(edata.var_names.values, df.columns.values[1:])

    pd.testing.assert_frame_equal(pd.DataFrame(edata.X), pd.DataFrame(df.iloc[:, 1:].to_numpy()))
    desired_default_index = pd.RangeIndex(start=0, stop=len(df), step=1).astype(str)
    assert edata.obs.index.equals(desired_default_index)


# avoid using from_dataframe here to avoid affecting tests
# of from_dataframe and to_dataframe
@pytest.mark.parametrize("dataset", ["dataset_basic", "dataset_non_num_with_missing", "dataset_num_with_missing"])
def test_to_dataframe_basic(dataset, request):
    original_df = request.getfixturevalue(dataset)
    edata = EHRData(X=original_df, var=original_df.columns.to_frame())
    df = to_dataframe(edata)

    assert df.shape == (original_df.shape[0], original_df.shape[1])
    expected_index = pd.RangeIndex(start=0, stop=len(original_df), step=1)
    assert df.index.equals(expected_index)


@pytest.mark.parametrize("dataset", ["dataset_basic", "dataset_non_num_with_missing", "dataset_num_with_missing"])
def test_to_dataframe_basic_layer(dataset, request):
    original_df = request.getfixturevalue(dataset)
    modified_df = original_df.copy()
    modified_df.iloc[1, 1] = "modified_value"

    edata = EHRData(X=original_df, var=original_df.columns.to_frame(), layers={"modified_df": modified_df})
    df = to_dataframe(edata, layer="modified_df")

    assert df.shape == (original_df.shape[0], original_df.shape[1])
    expected_index = pd.RangeIndex(start=0, stop=len(original_df), step=1)
    assert df.index.equals(expected_index)

    assert df.iloc[1, 1] == "modified_value"


def test_to_dataframe_basic_obs_col():
    pass


def test_to_dataframe_basic_var_col():
    pass
