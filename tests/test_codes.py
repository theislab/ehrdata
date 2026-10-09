import dask.array as da
import duckdb
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import sparse

import ehrdata as ed

VAR_NAMES = ["ICD10CM/I21.0", "ICD10CM/I21.4", "ICD10/E119", "ATC/C07AB02", "ATC/C07AA05", "LOINC/8480-6", "age"]
X_2D = np.array(
    [
        [1.0, 2.0, 0.0, 1.0, np.nan, 120.0, 50.0],
        [np.nan, np.nan, 3.0, 0.0, 1.0, np.nan, 60.0],
    ]
)
X_3D = np.stack([X_2D, np.where(np.isnan(X_2D), 1.0, np.nan)], axis=2)


@pytest.fixture
def edata_codes() -> ed.EHRData:
    edata = ed.EHRData(X_3D, var=pd.DataFrame(index=VAR_NAMES), tem=pd.DataFrame(index=["t0", "t1"]))
    ed.annotate_codes(edata)
    return edata


def test_annotate_codes(edata_codes):
    var = edata_codes.var
    assert var["vocabulary"].tolist()[:-1] == ["ICD10CM", "ICD10CM", "ICD10", "ATC", "ATC", "LOINC"]
    assert var["code"].tolist()[:-1] == ["I21.0", "I21.4", "E119", "C07AB02", "C07AA05", "8480-6"]
    assert var["vocabulary"].isna()["age"]
    assert var["icd_category"].tolist()[:3] == ["ICD10CM/I21", "ICD10CM/I21", "ICD10/E11"]
    assert var["icd_chapter"].tolist()[:3] == ["ICD10CM/I00-I99", "ICD10CM/I00-I99", "ICD10/E00-E89"]
    assert var.loc["ATC/C07AB02", [f"atc_level_{level}" for level in range(1, 6)]].tolist() == [
        "ATC/C",
        "ATC/C07",
        "ATC/C07A",
        "ATC/C07AB",
        "ATC/C07AB02",
    ]
    assert var.loc[["LOINC/8480-6", "age"], ["icd_category", "atc_level_1", "description"]].isna().all().all()


@pytest.mark.parametrize(
    ("code", "chapter"),
    [("C50.9", "C00-D49"), ("D50.0", "D50-D89"), ("H65.0", "H60-H95"), ("O9A.1", "O00-O9A"), ("T50.9", "S00-T88")],
)
def test_annotate_codes_icd_chapter_boundaries(code, chapter):
    edata = ed.EHRData(np.ones((1, 1)), var=pd.DataFrame(index=[f"ICD10CM/{code}"]))
    ed.annotate_codes(edata)
    assert edata.var["icd_chapter"].iloc[0] == f"ICD10CM/{chapter}"


@pytest.mark.parametrize(
    ("name", "category", "chapter"),
    [
        ("ICD9CM/410.01", "ICD9CM/410", "ICD9CM/390-459"),
        ("ICD9/41001", "ICD9/410", "ICD9/390-459"),
        ("ICD9CM/038.9", "ICD9CM/038", "ICD9CM/001-139"),
        ("ICD9CM/E849.7", "ICD9CM/E849", "ICD9CM/E000-E999"),
        ("ICD9CM/V4511", "ICD9CM/V45", "ICD9CM/V01-V91"),
        ("ICD9CM/996.81", "ICD9CM/996", "ICD9CM/800-999"),
    ],
)
def test_annotate_codes_icd9(name, category, chapter):
    edata = ed.EHRData(np.ones((1, 1)), var=pd.DataFrame(index=[name]))
    ed.annotate_codes(edata)
    assert edata.var[["icd_category", "icd_chapter"]].iloc[0].tolist() == [category, chapter]
    assert edata.var["icd_codes"].iloc[0] == name.replace(".", "")


@pytest.mark.parametrize(
    ("name", "vocabulary", "code", "icd_codes"),
    [
        ("DIAGNOSIS//ICD//10//I210", "ICD10CM", "I210", "ICD10CM/I210"),
        ("DIAGNOSIS//ICD//9//41001", "ICD9CM", "41001", "ICD9CM/41001"),
        ("PROCEDURE//ICD//9//3995", "ICD9Proc", "3995", pd.NA),
        ("PROCEDURE//ICD//10//5A1D70Z", "ICD10PCS", "5A1D70Z", pd.NA),
        ("MEDICATION//Heparin//Administered//63323026201", "NDC", "63323026201", pd.NA),
        ("MEDICATION//START//Heparin//63323026201", "NDC", "63323026201", pd.NA),
        ("MEDICATION//STOP//Heparin//UNK", pd.NA, pd.NA, pd.NA),
        ("DRG//HCFA//3", "MS-DRG", "003", pd.NA),
        ("LAB//RESULT//50931//mg/dL", pd.NA, pd.NA, pd.NA),
        ("MEDS_BIRTH", pd.NA, pd.NA, pd.NA),
    ],
)
def test_annotate_codes_meds_mimic_iv(name, vocabulary, code, icd_codes):
    edata = ed.EHRData(np.ones((1, 1)), var=pd.DataFrame(index=[name]))
    ed.annotate_codes(edata)
    assert edata.var[["vocabulary", "code", "icd_codes"]].iloc[0].tolist() == [vocabulary, code, icd_codes]


def test_annotate_codes_copy(edata_codes):
    edata = ed.EHRData(X_2D, var=pd.DataFrame(index=VAR_NAMES))
    annotated = ed.annotate_codes(edata, copy=True)
    assert "vocabulary" not in edata.var.columns
    assert annotated.var.equals(edata_codes.var)


def test_annotate_codes_keeps_descriptions():
    edata = ed.EHRData(
        np.ones((1, 2)),
        var=pd.DataFrame({"description": ["Heart attack", None]}, index=["ICD10CM/I21.0", "ICD10CM/I21"]),
    )
    ed.annotate_codes(edata)
    assert edata.var["description"].tolist() == ["Heart attack", pd.NA]


@pytest.fixture
def omop_vocabulary():
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE concept AS SELECT * FROM (VALUES
            (1, 'Acute myocardial infarction', 'SNOMED', '22298006'),
            (2, 'Systolic blood pressure', 'LOINC', '8480-6'),
            (3, 'metoprolol 50 MG Oral Tablet', 'RxNorm', '866514'),
            (4, 'metoprolol', 'ATC', 'C07AB02'),
            (5, 'Beta blocking agents, selective', 'ATC', 'C07AB'),
            (6, 'hydrochlorothiazide / metoprolol', 'RxNorm', '1'),
            (7, 'metoprolol and thiazides', 'ATC', 'C07BB02'),
            (8, 'Acute myocardial infarction of anterolateral wall', 'ICD10CM', 'I21.09'),
            (9, 'Acute myocardial infarction of anterolateral wall, initial episode of care', 'ICD9CM', '410.01')
        ) AS t(concept_id, concept_name, vocabulary_id, concept_code)
        """
    )
    con.execute(
        """
        CREATE TABLE concept_ancestor AS SELECT * FROM (VALUES (5, 3), (4, 3), (3, 3), (4, 6), (7, 6))
        AS t(ancestor_concept_id, descendant_concept_id)
        """
    )
    con.execute(
        """
        CREATE TABLE concept_relationship AS SELECT * FROM (VALUES (8, 1, 'Maps to'), (1, 9, 'Mapped from'))
        AS t(concept_id_1, concept_id_2, relationship_id)
        """
    )
    yield con
    con.close()


def test_annotate_codes_omop_codes(omop_vocabulary):
    edata = ed.EHRData(np.ones((1, 3)), var=pd.DataFrame(index=["SNOMED/22298006", "LOINC/8480-6", "RxNorm/866514"]))
    ed.annotate_codes(edata, backend_handle=omop_vocabulary)
    var = edata.var
    assert var["concept_id"].tolist() == [1, 2, 3]
    assert var["description"].tolist() == [
        "Acute myocardial infarction",
        "Systolic blood pressure",
        "metoprolol 50 MG Oral Tablet",
    ]
    assert var["atc_level_5"].tolist() == [pd.NA, pd.NA, "ATC/C07AB02"]
    assert var["icd_codes"].tolist() == ["ICD10CM/I2109|ICD9CM/41001", pd.NA, pd.NA]


def test_annotate_codes_omop_concept_ids(omop_vocabulary):
    edata = ed.EHRData(np.ones((1, 3)), var=pd.DataFrame({"data_table_concept_id": [6, 2, 99]}, index=["0", "1", "2"]))
    ed.annotate_codes(edata, backend_handle=omop_vocabulary)
    var = edata.var
    assert var["vocabulary"].tolist() == ["RxNorm", "LOINC", pd.NA]
    assert var["code"].tolist() == ["1", "8480-6", pd.NA]
    assert var.loc["0", ["atc_level_1", "atc_level_2", "atc_level_3"]].tolist() == ["ATC/C", "ATC/C07", pd.NA]


def test_annotate_codes_omop_setup_variables(omop_connection_vanilla):
    edata = ed.io.omop.setup_obs(omop_connection_vanilla, "person_observation_period")
    edata = ed.io.omop.setup_variables(
        edata,
        backend_handle=omop_connection_vanilla,
        data_tables=["measurement"],
        data_field_to_keep=["value_as_number"],
        interval_length_number=1,
        interval_length_unit="day",
        num_intervals=2,
        enrich_var_with_feature_info=True,
    )
    ed.annotate_codes(edata, backend_handle=omop_connection_vanilla)
    assert edata.var["concept_id"].tolist() == [2000030004, 2000001003]
    assert edata.var["code"].tolist() == ["220048", "50804"]
    assert edata.var["description"].tolist() == ["Heart Rhythm", "Calculated Total CO2|Blood|Blood Gas"]


def test_annotate_codes_omop_without_concept_table():
    con = duckdb.connect()
    with pytest.raises(ValueError, match="concept"):
        ed.annotate_codes(ed.EHRData(np.ones((1, 1)), var=pd.DataFrame(index=["LOINC/8480-6"])), backend_handle=con)


EXPECTED_SUM = np.array([[3.0, 0.0, 1.0, 120.0, 50.0], [np.nan, 3.0, 0.0, np.nan, 60.0]])
EXPECTED_MAX = np.array([[2.0, 0.0, 1.0, 120.0, 50.0], [np.nan, 3.0, 0.0, np.nan, 60.0]])


@pytest.mark.parametrize(("aggregation_strategy", "expected"), [("sum", EXPECTED_SUM), ("max", EXPECTED_MAX)])
@pytest.mark.parametrize(
    "array_type",
    [
        np.asarray,
        lambda x: da.from_array(x, chunks=(1, 3, 2)),
        lambda x: sparse.COO.from_numpy(x, fill_value=np.nan),
        lambda x: da.from_array(sparse.COO.from_numpy(x, fill_value=np.nan), chunks=(1, 3, 2)),
    ],
)
def test_aggregate_codes(edata_codes, aggregation_strategy, expected, array_type):
    edata_codes.X = array_type(edata_codes.X)
    aggregated = ed.aggregate_codes(edata_codes, "icd_category", aggregation_strategy=aggregation_strategy)
    assert aggregated.var_names.tolist() == [
        "ICD10CM/I21",
        "ICD10/E11",
        "ATC/C07AB02",
        "ATC/C07AA05",
        "LOINC/8480-6",
        "age",
    ]
    assert type(aggregated.X) is type(edata_codes.X)
    X = aggregated.X
    X = X.compute() if isinstance(X, da.Array) else X
    X = X.todense() if isinstance(X, sparse.COO) else X
    np.testing.assert_array_equal(X[:, [0, 1, 2, 4, 5], 0], expected)
    assert aggregated.shape == (2, 6, 2)
    assert aggregated.tem.equals(edata_codes.tem)
    assert aggregated.var.loc["ICD10CM/I21", "code"] == "I21"
    assert aggregated.var.loc["ICD10CM/I21", "icd_chapter"] == "ICD10CM/I00-I99"
    assert aggregated.var.loc["ATC/C07AB02", "atc_level_4"] == "ATC/C07AB"
    assert aggregated.var.loc["ICD10CM/I21", "icd_codes"] == "ICD10CM/I210|ICD10CM/I214"


def test_aggregate_codes_atc_level(edata_codes):
    aggregated = ed.aggregate_codes(edata_codes, "atc_level_3", aggregation_strategy="max")
    assert aggregated.var_names.tolist()[-3:] == ["ATC/C07A", "LOINC/8480-6", "age"]
    np.testing.assert_array_equal(aggregated[:, "ATC/C07A"].X[:, 0, 0], [1.0, 1.0])
    assert aggregated.var.loc["ATC/C07A", ["atc_level_3", "atc_level_4"]].tolist() == ["ATC/C07A", pd.NA]


def test_aggregate_codes_chapter_description(edata_codes):
    aggregated = ed.aggregate_codes(edata_codes, "icd_chapter")
    assert aggregated.var.loc["ICD10CM/I00-I99", "description"] == "Diseases of the circulatory system"
    assert aggregated.var.loc["ICD10/E00-E89", "description"] == "Endocrine, nutritional and metabolic diseases"


@pytest.mark.parametrize("aggregation_strategy", ["sum", "max"])
@pytest.mark.parametrize("array_type", [sp.csr_array, sp.csc_matrix, lambda x: da.from_array(sp.csr_matrix(x))])
def test_aggregate_codes_sparse_2d(aggregation_strategy, array_type):
    X = np.array([[1, 0, 2, 0], [0, 0, 0, 3], [-1, 0, 0, 0]], dtype=np.float64)
    edata = ed.EHRData(array_type(X), var=pd.DataFrame(index=["ATC/A01", "ATC/A02", "ATC/B01", "age"]))
    ed.annotate_codes(edata)
    aggregated = ed.aggregate_codes(edata, "atc_level_1", aggregation_strategy=aggregation_strategy)
    result = aggregated.X.compute() if isinstance(aggregated.X, da.Array) else aggregated.X
    assert type(result) is type(array_type(X).compute() if isinstance(aggregated.X, da.Array) else array_type(X))
    expected = [[1, 2, 0], [0, 0, 3], [-1 if aggregation_strategy == "sum" else 0, 0, 0]]
    np.testing.assert_array_equal(result.toarray(), expected)


def test_aggregate_codes_layer(edata_codes):
    edata_codes.layers["counts"] = (edata_codes.X > 0).astype(bool)
    aggregated = ed.aggregate_codes(edata_codes, "icd_category", layer="counts")
    assert aggregated.layers["counts"].dtype == np.int64
    assert aggregated.layers["counts"][0, 0, 0] == 2


def test_aggregate_codes_unannotated():
    edata = ed.EHRData(np.ones((1, 1)), var=pd.DataFrame(index=["ICD10CM/I21.0"]))
    with pytest.raises(ValueError, match="annotate_codes"):
        ed.aggregate_codes(edata, "icd_category")


def test_annotate_codes_gibleed():
    con = duckdb.connect()
    ed.dt.gibleed_omop(backend_handle=con)
    edata = ed.EHRData(np.ones((1, 2)), var=pd.DataFrame(index=["SNOMED/74474003", "ICD10CM/K92.2"]))
    ed.annotate_codes(edata, backend_handle=con)
    assert edata.var["description"].tolist() == [
        "Gastrointestinal hemorrhage",
        "Gastrointestinal hemorrhage, unspecified",
    ]
    assert edata.var["icd_codes"].tolist() == ["ICD10CM/K922", "ICD10CM/K922"]
    assert edata.var["icd_category"].tolist() == [pd.NA, "ICD10CM/K92"]
    con.close()


@pytest.mark.parametrize(("write", "read"), [("write_h5ed", "read_h5ed"), ("write_zarr", "read_zarr")])
def test_annotate_codes_roundtrip(edata_codes, tmp_path, write, read):
    getattr(ed.io, write)(edata_codes, tmp_path / "edata")
    var = getattr(ed.io, read)(tmp_path / "edata").var
    assert var["icd_codes"].astype("string").tolist() == edata_codes.var["icd_codes"].tolist()
    assert var["atc_level_5"].astype("string").tolist() == edata_codes.var["atc_level_5"].tolist()
