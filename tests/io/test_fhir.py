import gzip
import json

import numpy as np
import pandas as pd
import pytest

from ehrdata.core.constants import ANCHOR_TIME_KEY
from ehrdata.io import read_fhir
from ehrdata.io.fhir import FHIR_RESOURCES

LOINC = "http://loinc.org"
SNOMED = "http://snomed.info/sct"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
ICD10CM = "http://hl7.org/fhir/sid/icd-10-cm"


def _concept(*codings, text=None):
    concept = {"coding": [{"system": system, "code": code, "display": display} for system, code, display in codings]}
    return {**concept, "text": text} if text else concept


def _observation(subject, code, time, **fields):
    return {
        "resourceType": "Observation",
        "status": fields.pop("status", "final"),
        "subject": {"reference": subject},
        "code": code,
        "effectiveDateTime": time,
        **fields,
    }


RESOURCES = {
    "Patient": [
        {
            "resourceType": "Patient",
            "id": "p1",
            "gender": "female",
            "birthDate": "1980-05-02",
            "deceasedDateTime": "2020-01-01T10:00:00+02:00",
        },
        {"resourceType": "Patient", "id": "p2", "gender": "male", "birthDate": "1975"},
        {"resourceType": "Patient", "id": "p4", "deceasedBoolean": True},
    ],
    "Observation": [
        _observation(
            "Patient/p1",
            _concept((LOINC, "2339-0", "Glucose")),
            "2015-09-26T20:02:28-03:00",
            valueQuantity={"value": 7.2, "unit": "mmol/L", "code": "mmol/L"},
        ),
        {
            "resourceType": "Observation",
            "status": "final",
            "subject": {"reference": "urn:uuid:p1"},
            "code": _concept((SNOMED, "229819007", "Tobacco use"), (LOINC, "72166-2", "Tobacco smoking status")),
            "effectivePeriod": {"start": "2016-01-01T00:00:00Z", "end": "2016-01-02T00:00:00Z"},
            "valueCodeableConcept": _concept((SNOMED, "266919005", "Never smoked tobacco")),
        },
        {
            "resourceType": "Observation",
            "status": "final",
            "subject": {"reference": "http://example.org/fhir/Patient/p2/_history/3"},
            "code": _concept((LOINC, "1234-5", "Count")),
            "effectiveInstant": "2017-03-04T05:06:07.123Z",
            "valueInteger": 5,
        },
        _observation("Patient/p2", _concept((LOINC, "2222-2", "Flag")), "2018-07", valueBoolean=True),
        _observation("Patient/p2", _concept((LOINC, "2222-2", "Flag")), "2018-08", valueBoolean=False),
        _observation("Patient/p2", {"text": "Free text"}, "2018-07-15", valueString="positive"),
        _observation(
            "Patient/p1",
            _concept((LOINC, "2339-0", "Glucose")),
            "2015-09-27",
            status="entered-in-error",
            valueQuantity={"value": 999.0},
        ),
        _observation(
            "Patient/p1",
            _concept((LOINC, "85354-9", "Blood pressure panel")),
            "2019-02-02T08:00:00Z",
            component=[
                {"code": _concept((LOINC, "8480-6", "Systolic")), "valueQuantity": {"value": 120, "code": "mm[Hg]"}},
                {"code": _concept((LOINC, "8462-4", "Diastolic")), "valueQuantity": {"value": 80, "code": "mm[Hg]"}},
            ],
        ),
        _observation("Group/g1", _concept((LOINC, "2339-0", "Glucose")), "2019-03-03", valueQuantity={"value": 1.0}),
        _observation("Patient/p3", _concept((LOINC, "2339-0", "Glucose")), "2019-03-03", valueQuantity={"value": 2.0}),
        _observation(
            "Patient/p1",
            _concept(("http://example.org/local", "HR", "Heart rate")),
            "2015",
            valueQuantity={"value": 61.5, "unit": "beats/min"},
        ),
    ],
    "Condition": [
        {
            "resourceType": "Condition",
            "subject": {"reference": "Patient/p1"},
            "code": _concept((ICD10CM, "E11.9", "T2DM"), (SNOMED, "44054006", "Diabetes mellitus type 2")),
            "onsetPeriod": {"start": "2010-01-01"},
        },
        {
            "resourceType": "Condition",
            "subject": {"reference": "Patient/p2"},
            "verificationStatus": _concept(
                ("http://terminology.hl7.org/CodeSystem/condition-ver-status", "refuted", "")
            ),
            "code": _concept((SNOMED, "38341003", "Hypertension")),
            "onsetDateTime": "2011-01-01",
        },
        {
            "resourceType": "Condition",
            "subject": {"reference": "Patient/p2"},
            "code": _concept((SNOMED, "38341003", "Hypertension")),
            "recordedDate": "2011-02-03",
        },
    ],
    "Medication": [{"resourceType": "Medication", "id": "m1", "code": _concept((RXNORM, "197361", "Amlodipine"))}],
    "MedicationRequest": [
        {
            "resourceType": "MedicationRequest",
            "status": "active",
            "subject": {"reference": "Patient/p1"},
            "medicationCodeableConcept": _concept((RXNORM, "860975", "Metformin")),
            "authoredOn": "2014-01-01T08:00:00+01:00",
        },
        {
            "resourceType": "MedicationRequest",
            "status": "active",
            "subject": {"reference": "Patient/p2"},
            "medicationReference": {"reference": "Medication/m1"},
            "authoredOn": "2014-02-02",
        },
    ],
    "MedicationAdministration": [
        {
            "resourceType": "MedicationAdministration",
            "status": "completed",
            "subject": {"reference": "Patient/p1"},
            "medicationCodeableConcept": _concept((RXNORM, "1719286", "Insulin")),
            "effectivePeriod": {"start": "2020-01-01T01:00:00Z", "end": "2020-01-01T02:00:00Z"},
            "dosage": {"dose": {"value": 4, "unit": "U", "code": "[iU]"}},
        },
        {
            "resourceType": "MedicationAdministration",
            "status": "not-done",
            "subject": {"reference": "Patient/p2"},
            "medicationReference": {"reference": "Medication/m1"},
            "effectiveDateTime": "2020-02-02T02:02:02Z",
        },
        {
            "resourceType": "MedicationAdministration",
            "status": "completed",
            "subject": {"reference": "Patient/p2"},
            "contained": [
                {"resourceType": "Medication", "id": "med", "code": _concept((RXNORM, "197361", "Amlodipine"))}
            ],
            "medicationReference": {"reference": "#med"},
            "effectiveDateTime": "2020-03-03T03:03:03Z",
        },
    ],
    "Procedure": [
        {
            "resourceType": "Procedure",
            "status": "completed",
            "subject": {"reference": "Patient/p1"},
            "code": _concept((SNOMED, "73761001", "Colonoscopy")),
            "performedDateTime": "2013-05-05",
        },
        {
            "resourceType": "Procedure",
            "status": "not-done",
            "subject": {"reference": "Patient/p2"},
            "code": _concept((SNOMED, "73761001", "Colonoscopy")),
            "performedPeriod": {"start": "2013-06-06", "end": "2013-06-07"},
        },
    ],
    "Encounter": [
        {
            "resourceType": "Encounter",
            "status": "finished",
            "subject": {"reference": "Patient/p1"},
            "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB"},
            "period": {"start": "2015-09-26T19:00:00-03:00", "end": "2015-09-26T21:00:00-03:00"},
        },
    ],
    "Immunization": [
        {
            "resourceType": "Immunization",
            "status": "completed",
            "patient": {"reference": "Patient/p1"},
            "vaccineCode": _concept(("http://hl7.org/fhir/sid/cvx", "140", "Influenza")),
            "occurrenceDateTime": "2019-10-01",
        },
    ],
}

# code: (description, unit, values of p1, p2 and p4 in the first interval)
EXPECTED_VAR = {
    "ActCode/AMB": (None, None, [1.0, np.nan, np.nan]),
    "CVX/140": ("Influenza", None, [1.0, np.nan, np.nan]),
    "Free text": ("Free text", None, [np.nan, 1.0, np.nan]),
    "LOINC/1234-5": ("Count", None, [np.nan, 5.0, np.nan]),
    "LOINC/2222-2": ("Flag", None, [np.nan, 0.0, np.nan]),
    "LOINC/2339-0": ("Glucose", "mmol/L", [7.2, np.nan, np.nan]),
    "LOINC/72166-2": ("Tobacco smoking status", None, [1.0, np.nan, np.nan]),
    "LOINC/8462-4": ("Diastolic", "mm[Hg]", [80.0, np.nan, np.nan]),
    "LOINC/8480-6": ("Systolic", "mm[Hg]", [120.0, np.nan, np.nan]),
    "RxNorm/1719286": ("Insulin", "[iU]", [np.nan, np.nan, np.nan]),
    "RxNorm/197361": ("Amlodipine", None, [np.nan, 2.0, np.nan]),
    "RxNorm/860975": ("Metformin", None, [1.0, np.nan, np.nan]),
    "SNOMED/38341003": ("Hypertension", None, [np.nan, 1.0, np.nan]),
    "SNOMED/44054006": ("Diabetes mellitus type 2", None, [1.0, np.nan, np.nan]),
    "SNOMED/73761001": ("Colonoscopy", None, [1.0, np.nan, np.nan]),
    "http://example.org/local/HR": ("Heart rate", "beats/min", [61.5, np.nan, np.nan]),
}


@pytest.fixture(params=["ndjson", "ndjson.gz", "bundle"])
def fhir_dir(request, tmp_path):
    if request.param == "bundle":
        entries = [{"fullUrl": f"urn:uuid:{r.get('id')}", "resource": r} for rs in RESOURCES.values() for r in rs]
        (tmp_path / "bundle.json").write_text(json.dumps({"resourceType": "Bundle", "type": "batch", "entry": entries}))
    for resource_type, resources in RESOURCES.items():
        lines = "".join(json.dumps(resource) + "\n" for resource in resources)
        if request.param == "ndjson":
            (tmp_path / f"{resource_type}.000.ndjson").write_text(lines)
        elif request.param == "ndjson.gz":
            (tmp_path / f"{resource_type}.000.ndjson.gz").write_bytes(gzip.compress(lines.encode()))
    return tmp_path


def test_read_fhir(fhir_dir):
    edata = read_fhir(fhir_dir, resources=FHIR_RESOURCES, interval_length_number=3650, interval_length_unit="D")

    assert list(edata.obs_names) == ["p1", "p2", "p4"]
    expected_obs = pd.DataFrame(
        {
            "gender": ["female", "male", None],
            "birth_date": ["1980-05-02 00:00:00", "1975-01-01 00:00:00", np.nan],
            "deceased": [True, False, True],
            "deceased_time": ["2020-01-01 08:00:00", np.nan, np.nan],
            ANCHOR_TIME_KEY: ["2010-01-01 00:00:00", "2011-02-03 00:00:00", np.nan],
        },
        index=["p1", "p2", "p4"],
    )
    pd.testing.assert_frame_equal(edata.obs, expected_obs, check_dtype=False)
    assert list(edata.var_names) == list(EXPECTED_VAR)
    assert list(edata.var["description"]) == [description for description, _, _ in EXPECTED_VAR.values()]
    assert list(edata.var["unit"]) == [unit for _, unit, _ in EXPECTED_VAR.values()]
    assert edata.var.loc["LOINC/2222-2", "n_events"] == 2
    assert edata.shape == (3, len(EXPECTED_VAR), 2)
    np.testing.assert_array_equal(edata.X[:, :, 0].T, [values for _, _, values in EXPECTED_VAR.values()])
    np.testing.assert_array_equal(
        edata.X[:, :, 1], np.where(edata.var_names == "RxNorm/1719286", [[4.0], [np.nan], [np.nan]], np.nan)
    )


def test_read_fhir_preferred_systems(fhir_dir):
    edata = read_fhir(fhir_dir, resources=["Condition", "Observation"], preferred_systems=[ICD10CM])

    assert "ICD10CM/E11.9" in edata.var_names
    assert "SNOMED/229819007" in edata.var_names
    assert edata.var.loc["ICD10CM/E11.9", "description"] == "T2DM"


def test_read_fhir_anchor_birth_date(fhir_dir):
    edata = read_fhir(
        fhir_dir,
        resources=["Patient", "Observation"],
        codes=["LOINC/2339-0"],
        anchor="birth_date",
        interval_length_unit="D",
    )

    birth_to_glucose = pd.Timestamp("2015-09-26 23:02:28") - pd.Timestamp("1980-05-02")
    np.testing.assert_array_equal(np.flatnonzero(~np.isnan(edata.X[0, 0])), [birth_to_glucose.days])
    assert edata.X[0, 0, birth_to_glucose.days] == 7.2


def test_read_fhir_without_patient(fhir_dir):
    edata = read_fhir(
        fhir_dir,
        resources=["Observation"],
        codes=["LOINC/2339-0"],
        interval_length_number=3650,
        interval_length_unit="D",
    )

    assert list(edata.obs_names) == ["p1", "p2", "p3"]
    assert list(edata.obs.columns) == [ANCHOR_TIME_KEY]
    np.testing.assert_array_equal(edata.X[:, 0, 0], [7.2, np.nan, 2.0])


def test_read_fhir_errors(tmp_path):
    with pytest.raises(ValueError, match="Unsupported resources"):
        read_fhir(tmp_path, resources=["Patient", "Specimen"])
    with pytest.raises(FileNotFoundError, match="No Patient resources"):
        read_fhir(tmp_path)
    (tmp_path / "Patient.ndjson").write_text(json.dumps(RESOURCES["Patient"][0]) + "\n")
    with pytest.raises(FileNotFoundError, match="No Observation resources"):
        read_fhir(tmp_path, resources=["Patient", "Observation"])
