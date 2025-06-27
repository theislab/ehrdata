import numpy as np
import pandas as pd
import pytest
from tests.conftest import _assert_shape_matches

from ehrdata import EHRData
from ehrdata.io import from_pandas, to_pandas


@pytest.mark.parametrize("df", ["csv_basic", "csv_non_num_with_missing", "csv_num_with_missing"])
def test_from_pandas_basic(df, request):
    df = request.getfixturevalue(df)
    edata = from_pandas(df)

    _assert_shape_matches(edata, (df.shape[0], df.shape[1], 0), check_R_None=True)
    assert edata.obs.shape == (df.shape[0], 0)
    assert edata.var.shape == (df.shape[1], 0)
    assert edata.tem.shape == (0, 0)

    assert np.array_equal(edata.var_names.values, df.columns.values)

    pd.testing.assert_frame_equal(pd.DataFrame(edata.X), pd.DataFrame(df.to_numpy()))
    desired_default_index = pd.RangeIndex(start=0, stop=len(df), step=1).astype(str)
    assert edata.obs.index.equals(desired_default_index)


@pytest.mark.parametrize("df", ["csv_basic", "csv_non_num_with_missing", "csv_num_with_missing"])
@pytest.mark.parametrize("index_column", ["patient_id", 0])
def test_from_pandas_basic_index(df, index_column, request):
    df = request.getfixturevalue(df)
    edata = from_pandas(df, index_column=index_column)
    _assert_shape_matches(edata, (df.shape[0], df.shape[1] - 1, 0), check_R_None=True)

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


def test_from_pandas_basic_index_invalid_index_throws_error(csv_basic):
    with pytest.raises(ValueError):
        from_pandas(csv_basic, index_column="mistyped_column_name")

    with pytest.raises(IndexError):
        from_pandas(csv_basic, index_column=6)


@pytest.mark.parametrize("df", ["csv_basic", "csv_non_num_with_missing", "csv_num_with_missing"])
def test_from_pandas_basic_column_obs_only(df, request):
    df = request.getfixturevalue(df)
    edata = from_pandas(df, columns_obs_only=["patient_id"])
    _assert_shape_matches(edata, (df.shape[0], df.shape[1] - 1, 0), check_R_None=True)
    assert edata.obs.shape == (df.shape[0], 1)
    assert edata.var.shape == (df.shape[1] - 1, 0)
    assert edata.tem.shape == (0, 0)

    assert np.array_equal(edata.var_names.values, df.columns.values[1:])

    pd.testing.assert_frame_equal(pd.DataFrame(edata.X), pd.DataFrame(df.iloc[:, 1:].to_numpy()))
    desired_default_index = pd.RangeIndex(start=0, stop=len(df), step=1).astype(str)
    assert edata.obs.index.equals(desired_default_index)


# avoid using from_pandas here to avoid affecting tests
# of from_pandas and to_pandas
@pytest.mark.parametrize("dataset", ["csv_basic", "csv_non_num_with_missing", "csv_num_with_missing"])
def test_to_pandas_basic(dataset, request):
    original_df = request.getfixturevalue(dataset)
    edata = EHRData(X=original_df, var=original_df.columns.to_frame())
    df = to_pandas(edata)

    assert df.shape == (original_df.shape[0], original_df.shape[1])
    assert df.index.equals(edata.obs.index)


@pytest.mark.parametrize("dataset", ["csv_basic", "csv_non_num_with_missing", "csv_num_with_missing"])
def test_to_pandas_basic_layer(dataset, request):
    original_df = request.getfixturevalue(dataset)
    modified_df = original_df.copy()
    modified_df.iloc[1, 1] = 123

    edata = EHRData(X=original_df, var=original_df.columns.to_frame(), layers={"modified_df": modified_df})
    df = to_pandas(edata, layer="modified_df")

    assert df.shape == (original_df.shape[0], original_df.shape[1])
    assert df.index.equals(edata.obs.index)

    assert df.iloc[1, 1] == 123


@pytest.mark.parametrize("dataset", ["csv_basic", "csv_non_num_with_missing", "csv_num_with_missing"])
@pytest.mark.parametrize("obs_cols", [["patient_id"], {"patient_id"}])
def test_to_pandas_basic_obs_cols(dataset, obs_cols, request):
    original_df = request.getfixturevalue(dataset)
    df_no_patient_id_col = original_df.drop(columns=["patient_id"])

    edata = EHRData(
        X=df_no_patient_id_col, var=df_no_patient_id_col.columns.to_frame(), obs=original_df[["patient_id"]]
    )
    # do this because want to ensure that edata has been set up properly
    assert edata.X.shape == (original_df.shape[0], original_df.shape[1] - 1)
    df = to_pandas(edata, obs_cols=obs_cols)

    assert df.shape == (original_df.shape[0], original_df.shape[1])
    assert df.index.equals(edata.obs.index)

    assert "patient_id" in df.columns


@pytest.mark.parametrize("dataset", ["csv_basic", "csv_non_num_with_missing", "csv_num_with_missing"])
def test_to_pandas_basic_var_col(dataset, request):
    original_df = request.getfixturevalue(dataset)
    edata = EHRData(X=original_df, var=original_df.columns.to_frame())

    new_var_col_name = "new_var_annotation"
    new_var_col_value = [f"some_annotation_{i}" for i in range(edata.shape[1])]
    edata.var[new_var_col_name] = new_var_col_value
    df = to_pandas(edata, var_col=new_var_col_name)

    assert df.shape == original_df.shape
    assert df.index.equals(edata.obs.index)

    assert np.array_equal(df.columns.values, new_var_col_value)
    assert df.shape == (original_df.shape[0], original_df.shape[1])


def test_from_pandas_longitudinal_wide():
    df = pd.DataFrame(
        {
            "var1_t_timestep1": [1, 2, 3],
            "var2_t_timestep1": [4, 5, 6],
            "var1_t_timestep2": [7, 8, 9],
            "var2_t_timestep2": [10, 11, 12],
        },
        index=["patient_1", "patient_2", "patient_3"],
    )
    edata = from_pandas(df, format="wide")
    _assert_shape_matches(edata, (3, 2, 2))

    assert edata.obs.shape == (3, 0)
    assert edata.var.shape == (2, 0)
    assert edata.tem.shape == (2, 0)

    assert np.array_equal(edata.obs_names.values, ["patient_1", "patient_2", "patient_3"])
    assert np.array_equal(edata.var_names.values, ["var1", "var2"])
    assert np.array_equal(edata.tem.index.values, ["timestep1", "timestep2"])

    assert np.array_equal(
        edata.R[1],
        np.array(
            [
                [2, 8],
                [5, 11],
            ]
        ),
    )


def test_from_pandas_longitudinal_wide_missing_timestep():
    df = pd.DataFrame(
        {
            "var1_t_timestep1": [1, 2, 3],
            "var2_t_timestep1": [4, 5, 6],
            "var1_t_timestep2": [7, 8, 9],
            "obs_column": ["obs_1", "obs_2", "obs_3"],
        },
        index=["patient_1", "patient_2", "patient_3"],
    )
    edata = from_pandas(df, format="wide", columns_obs_only=["obs_column"])
    _assert_shape_matches(edata, (3, 2, 2))

    assert edata.obs.shape == (3, 1)
    assert edata.var.shape == (2, 0)
    assert edata.tem.shape == (2, 0)

    assert np.array_equal(edata.obs_names.values, ["patient_1", "patient_2", "patient_3"])
    assert np.array_equal(edata.var_names.values, ["var1", "var2"])
    assert np.array_equal(edata.tem.index.values, ["timestep1", "timestep2"])

    assert np.array_equal(
        edata.R[1],
        np.array(
            [
                [2, 8],
                [5, np.nan],
            ]
        ),
        equal_nan=True,
    )

    assert np.array_equal(edata.obs["obs_column"].values, ["obs_1", "obs_2", "obs_3"])


@pytest.mark.parametrize("columns_obs_only", [["obs_column"], {"obs_column"}])
def test_from_pandas_longitudinal_wide_column_obs_only(columns_obs_only):
    df = pd.DataFrame(
        {
            "var1_t_timestep1": [1, 2, 3],
            "var2_t_timestep1": [4, 5, 6],
            "var1_t_timestep2": [7, 8, 9],
            "var2_t_timestep2": [10, 11, 12],
            "obs_column": ["obs_1", "obs_2", "obs_3"],
        },
        index=["patient_1", "patient_2", "patient_3"],
    )
    edata = from_pandas(df, format="wide", columns_obs_only=columns_obs_only)
    _assert_shape_matches(edata, (3, 2, 2))

    assert edata.obs.shape == (3, 1)
    assert edata.var.shape == (2, 0)
    assert edata.tem.shape == (2, 0)

    assert np.array_equal(edata.obs_names.values, ["patient_1", "patient_2", "patient_3"])
    assert np.array_equal(edata.var_names.values, ["var1", "var2"])
    assert np.array_equal(edata.tem.index.values, ["timestep1", "timestep2"])

    assert np.array_equal(
        edata.R[1],
        np.array(
            [
                [2, 8],
                [5, 11],
            ]
        ),
    )

    assert np.array_equal(edata.obs["obs_column"].values, ["obs_1", "obs_2", "obs_3"])


def test_from_pandas_longitudinal_long():
    df = pd.DataFrame(
        {
            "observation_id": [
                "patient_1",
                "patient_1",
                "patient_1",
                "patient_1",
                "patient_2",
                "patient_2",
                "patient_2",
                "patient_2",
                "patient_3",
                "patient_3",
                "patient_3",
                "patient_3",
            ],
            "variable": [
                "var1",
                "var1",
                "var2",
                "var2",
                "var1",
                "var1",
                "var2",
                "var2",
                "var1",
                "var1",
                "var2",
                "var2",
            ],
            "time": ["t1", "t2", "t1", "t2", "t1", "t2", "t1", "t2", "t1", "t2", "t1", "t2"],
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
    )
    edata = from_pandas(df, format="long")
    _assert_shape_matches(edata, (3, 2, 2))
    assert edata.obs.shape == (3, 0)
    assert edata.var.shape == (2, 0)
    assert edata.tem.shape == (2, 0)

    assert np.array_equal(edata.obs_names.values, ["patient_1", "patient_2", "patient_3"])
    assert np.array_equal(edata.var_names.values, ["var1", "var2"])
    assert np.array_equal(edata.tem.index.values, ["t1", "t2"])


def test_from_pandas_invalid_format():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        from_pandas(df, format="invalid_format")


def test_from_pandas_longitudinal_long_index_column_not_implemented():
    df = pd.DataFrame(
        {
            "observation_id": ["patient_1"],
            "variable": ["var1"],
            "time": ["t1"],
            "value": [1],
        }
    )
    with pytest.raises(ValueError):
        from_pandas(df, index_column="observation_id", format="long")


def test_to_pandas_longitudinal_wide(edata_333):
    df = to_pandas(edata_333, format="wide", layer="R_layer")
    assert df.shape == (3, 9)
    assert df.index.equals(edata_333.obs.index)
    assert np.array_equal(
        df.columns.values,
        [
            "var1_t_t1",
            "var1_t_t2",
            "var1_t_t3",
            "var2_t_t1",
            "var2_t_t2",
            "var2_t_t3",
            "var3_t_t1",
            "var3_t_t2",
            "var3_t_t3",
        ],
    )
    assert np.array_equal(df[["var1_t_t3"]].values.flatten(), edata_333.R[:, 0, 2])


def test_to_pandas_longitudinal_wide_obs_cols(edata_333):
    df = to_pandas(edata_333, format="wide", layer="R_layer", obs_cols=["obs_col_1"])
    assert df.shape == (3, 10)
    assert df.index.equals(edata_333.obs.index)
    assert np.array_equal(
        df.columns.values,
        [
            "var1_t_t1",
            "var1_t_t2",
            "var1_t_t3",
            "var2_t_t1",
            "var2_t_t2",
            "var2_t_t3",
            "var3_t_t1",
            "var3_t_t2",
            "var3_t_t3",
            "obs_col_1",
        ],
    )
    assert np.array_equal(df[["var1_t_t3"]].values.flatten(), edata_333.R[:, 0, 2])
    assert np.array_equal(df[["obs_col_1"]].values.flatten(), edata_333.obs["obs_col_1"].values)


def test_to_pandas_longitudinal_long(edata_333):
    df = to_pandas(edata_333, format="long", layer="R_layer")
    assert df.shape == (27, 4)
    assert np.array_equal(df.iloc[13, :3].values, np.array(["obs2", "var2", "t2"]).astype(object))
    assert df.iloc[13, 3] == 14


def test_to_pandas_invalid_format(edata_333):
    with pytest.raises(ValueError):
        to_pandas(edata_333, format="invalid_format")


def test_to_pandas_longitudinal_long_obs_cols_raises_error(edata_333):
    with pytest.raises(NotImplementedError):
        to_pandas(edata_333, format="long", obs_cols=["obs_col_1"])
