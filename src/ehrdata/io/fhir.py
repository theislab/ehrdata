from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from ehrdata.io.events import from_events

if TYPE_CHECKING:
    from collections.abc import Collection
    from os import PathLike

    from ehrdata import EHRData

FHIR_RESOURCES = (
    "Patient",
    "Observation",
    "Condition",
    "MedicationRequest",
    "MedicationAdministration",
    "Procedure",
    "Encounter",
    "Immunization",
)

PREFERRED_SYSTEMS = (
    "http://loinc.org",
    "http://snomed.info/sct",
    "http://www.nlm.nih.gov/research/umls/rxnorm",
    "http://hl7.org/fhir/sid/cvx",
    "http://hl7.org/fhir/sid/icd-10-cm",
)

# The OMOP vocabulary_id of each system, so that codes from FHIR and OMOP coincide.
VOCABULARIES: Mapping[str, str] = {
    "http://loinc.org": "LOINC",
    "http://snomed.info/sct": "SNOMED",
    "http://www.nlm.nih.gov/research/umls/rxnorm": "RxNorm",
    "http://hl7.org/fhir/sid/cvx": "CVX",
    "http://hl7.org/fhir/sid/icd-10-cm": "ICD10CM",
    "http://hl7.org/fhir/sid/icd-10": "ICD10",
    "http://hl7.org/fhir/sid/icd-9-cm": "ICD9CM",
    "http://hl7.org/fhir/sid/ndc": "NDC",
    "http://www.whocc.no/atc": "ATC",
    "http://www.ama-assn.org/go/cpt": "CPT4",
    "http://terminology.hl7.org/CodeSystem/v3-ActCode": "ActCode",
}

_CODING = {"system": "VARCHAR", "code": "VARCHAR", "display": "VARCHAR"}
_CODEABLE_CONCEPT = {"coding": [_CODING], "text": "VARCHAR"}
_REFERENCE = {"reference": "VARCHAR"}
_PERIOD = {"start": "VARCHAR", "end": "VARCHAR"}
_QUANTITY = {"value": "DOUBLE", "unit": "VARCHAR", "code": "VARCHAR"}
_VALUE = {
    "valueQuantity": _QUANTITY,
    "valueCodeableConcept": _CODEABLE_CONCEPT,
    "valueString": "VARCHAR",
    "valueBoolean": "BOOLEAN",
    "valueInteger": "DOUBLE",
    "valueDateTime": "VARCHAR",
}
_MEDICATION = {
    "medicationCodeableConcept": _CODEABLE_CONCEPT,
    "medicationReference": _REFERENCE,
    "contained": [{"id": "VARCHAR", "code": _CODEABLE_CONCEPT}],
}

_SCHEMAS: Mapping[str, Mapping[str, Any]] = {
    "Patient": {
        "id": "VARCHAR",
        "gender": "VARCHAR",
        "birthDate": "VARCHAR",
        "deceasedDateTime": "VARCHAR",
        "deceasedBoolean": "BOOLEAN",
    },
    "Observation": {
        "status": "VARCHAR",
        "subject": _REFERENCE,
        "code": _CODEABLE_CONCEPT,
        "effectiveDateTime": "VARCHAR",
        "effectivePeriod": _PERIOD,
        "effectiveInstant": "VARCHAR",
        **_VALUE,
        "component": [{"code": _CODEABLE_CONCEPT, **_VALUE}],
    },
    "Condition": {
        "subject": _REFERENCE,
        "code": _CODEABLE_CONCEPT,
        "verificationStatus": _CODEABLE_CONCEPT,
        "onsetDateTime": "VARCHAR",
        "onsetPeriod": _PERIOD,
        "recordedDate": "VARCHAR",
    },
    "MedicationRequest": {"status": "VARCHAR", "subject": _REFERENCE, "authoredOn": "VARCHAR", **_MEDICATION},
    "MedicationAdministration": {
        "status": "VARCHAR",
        "subject": _REFERENCE,
        "effectiveDateTime": "VARCHAR",
        "effectivePeriod": _PERIOD,
        "dosage": {"dose": _QUANTITY},
        **_MEDICATION,
    },
    "Procedure": {
        "status": "VARCHAR",
        "subject": _REFERENCE,
        "code": _CODEABLE_CONCEPT,
        "performedDateTime": "VARCHAR",
        "performedPeriod": _PERIOD,
    },
    "Encounter": {"status": "VARCHAR", "subject": _REFERENCE, "class": _CODING, "period": _PERIOD},
    "Immunization": {
        "status": "VARCHAR",
        "patient": _REFERENCE,
        "vaccineCode": _CODEABLE_CONCEPT,
        "occurrenceDateTime": "VARCHAR",
    },
    "Medication": {"id": "VARCHAR", "code": _CODEABLE_CONCEPT},
}

_VALUE_COLUMNS = ", ".join(_VALUE)
_OBSERVATION_FILTER = "coalesce(status, '') NOT IN ('entered-in-error', 'cancelled')"

# Every query selects subject_id, time, code, numeric_value, unit and description, in this order.
_EVENT_QUERIES: Mapping[str, str] = {
    "Observation": f"""
        WITH o AS (
            SELECT
                fhir_ref(subject.reference) AS subject_id,
                fhir_time(coalesce(effectiveDateTime, effectivePeriod.start, effectiveInstant)) AS time,
                code, {_VALUE_COLUMNS}, component
            FROM {{source}}
            WHERE {_OBSERVATION_FILTER}
        ), v AS (
            SELECT subject_id, time, code, {_VALUE_COLUMNS} FROM o
            WHERE coalesce(len(component), 0) = 0 OR {" OR ".join(f"{column} IS NOT NULL" for column in _VALUE)}
            UNION ALL
            SELECT subject_id, time, c.code, {", ".join(f"c.{column}" for column in _VALUE)}
            FROM (SELECT subject_id, time, unnest(component) AS c FROM o)
        )
        SELECT
            subject_id, time, fhir_concept(code), coalesce(valueQuantity.value, valueInteger, valueBoolean::DOUBLE),
            coalesce(valueQuantity.code, valueQuantity.unit), fhir_description(code)
        FROM v
    """,
    "Condition": """
        SELECT
            fhir_ref(subject.reference), fhir_time(coalesce(onsetDateTime, onsetPeriod.start, recordedDate)),
            fhir_concept(code), NULL, NULL, fhir_description(code)
        FROM {source}
        WHERE coalesce(verificationStatus.coding[1].code, '') NOT IN ('entered-in-error', 'refuted')
    """,
    "MedicationRequest": """
        SELECT
            fhir_ref(subject.reference), fhir_time(authoredOn), fhir_concept({medication}), NULL, NULL,
            fhir_description({medication})
        FROM {source} {medication_join}
        WHERE coalesce(status, '') <> 'entered-in-error'
    """,
    "MedicationAdministration": """
        SELECT
            fhir_ref(subject.reference), fhir_time(coalesce(effectiveDateTime, effectivePeriod.start)),
            fhir_concept({medication}), dosage.dose.value, coalesce(dosage.dose.code, dosage.dose.unit),
            fhir_description({medication})
        FROM {source} {medication_join}
        WHERE coalesce(status, '') NOT IN ('entered-in-error', 'not-done')
    """,
    "Procedure": """
        SELECT
            fhir_ref(subject.reference), fhir_time(coalesce(performedDateTime, performedPeriod.start)),
            fhir_concept(code), NULL, NULL, fhir_description(code)
        FROM {source}
        WHERE coalesce(status, '') NOT IN ('entered-in-error', 'not-done')
    """,
    "Encounter": """
        SELECT
            fhir_ref(subject.reference), fhir_time(period.start), fhir_vocabulary(class.system) || '/' || class.code,
            NULL, NULL, class.display
        FROM {source}
        WHERE coalesce(status, '') NOT IN ('entered-in-error', 'cancelled')
    """,
    "Immunization": """
        SELECT
            fhir_ref(patient.reference), fhir_time(occurrenceDateTime), fhir_concept(vaccineCode), NULL, NULL,
            fhir_description(vaccineCode)
        FROM {source}
        WHERE coalesce(status, '') NOT IN ('entered-in-error', 'not-done')
    """,
}

_MEDICATION_CONCEPT = (
    "coalesce(medicationCodeableConcept, "
    "list_filter(contained, lambda m: '#' || m.id = medicationReference.reference)[1].code{medication_table})"
)
_MEDICATION_JOIN = (
    "LEFT JOIN (SELECT id AS medication_id, code AS medication_code FROM {source}) "
    "ON medication_id = fhir_ref(medicationReference.reference, 'Medication')"
)


def read_fhir(
    path: str | PathLike,
    *,
    resources: Collection[str] = ("Patient", "Observation", "Condition", "MedicationRequest", "Procedure", "Encounter"),
    preferred_systems: Sequence[str] | None = None,
    codes: Collection[str] | None = None,
    **binning: Any,
) -> EHRData:
    """Read `FHIR R4 <https://hl7.org/fhir/R4>`_ resources into an :class:`~ehrdata.EHRData` object with a time axis.

    The resources are read from `Bulk Data <https://hl7.org/fhir/uv/bulkdata>`_ export files, named after their resource type such as `Observation.000.ndjson`, or from Bundles in `.json` files.
    Both may be compressed, such as `Observation.000.ndjson.gz`.
    Every Observation, Condition, MedicationRequest, MedicationAdministration, Procedure, Encounter, and Immunization is an event of its subject, which is binned into intervals with :func:`~ehrdata.io.from_events`.
    Resources entered in error, refuted, cancelled, or not done are ignored, and so are resources whose subject is not a Patient.

    The code of an event is the system and the code of its coding, such as `LOINC/8480-6`, with common systems shortened to their OMOP vocabulary name, such as `LOINC`, `SNOMED`, `RxNorm`, or `ICD10CM`.
    If there are several codings, the first one whose system comes first in `preferred_systems` is used.
    A medication that refers to a Medication resource takes the code of this resource.
    An Observation with components, such as a blood pressure panel, gives one event per component.
    The numeric value of an event is the value of a quantity, an integer, or a boolean as 1 or 0, or the dose of a MedicationAdministration.
    Codes with other values, such as codings or strings, count the events.
    Times are in UTC, and partial dates, such as `2018-07`, start at their first day.

    The subjects are the Patient resources, sorted by their id, and events of other subjects are ignored.
    If `resources` does not contain `"Patient"`, the subjects are the ones with events instead.
    `obs` holds the `gender`, `birth_date`, whether the patient is `deceased`, and the `deceased_time`.
    `var` holds the `description` of each code, as given by its coding, and the most common `unit` of its numeric values.

    Args:
        path: A directory, which is searched recursively, or a single file.
        resources: The FHIR resource types to read.
        preferred_systems: The URIs of the coding systems to prefer if a concept has several codings, in this order.
            If not specified, LOINC, SNOMED CT, RxNorm, CVX, and ICD-10-CM are preferred in this order.
        codes: The codes to use as variables, in this order.
            If not specified, all codes of events with a time are used, sorted.
        **binning: Passed to :func:`~ehrdata.io.from_events`, such as `anchor`, `interval_length_number`, `interval_length_unit`, `num_intervals`, `aggregation_strategy`, `sparse`, and `layer`.
            With `anchor="birth_date"`, the intervals start at birth.

    Returns:
        An :class:`~ehrdata.EHRData` object of shape subjects × codes × intervals.

    Examples:
        >>> import json
        >>> import ehrdata as ed
        >>> heart_rate = {
        ...     "resourceType": "Observation",
        ...     "status": "final",
        ...     "subject": {"reference": "Patient/p1"},
        ...     "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4", "display": "Heart rate"}]},
        ...     "valueQuantity": {"value": 72, "unit": "/min"},
        ... }
        >>> resources = [
        ...     {"resourceType": "Patient", "id": "p1", "gender": "female", "birthDate": "1980-05-02"},
        ...     {**heart_rate, "effectiveDateTime": "2020-01-01T08:00:00Z"},
        ...     {**heart_rate, "effectiveDateTime": "2020-01-01T10:30:00+01:00"},
        ... ]
        >>> with open("fhir_bundle.json", "w") as f:
        ...     json.dump({"resourceType": "Bundle", "entry": [{"resource": r} for r in resources]}, f)
        >>> edata = ed.io.read_fhir("fhir_bundle.json", interval_length_number=1, interval_length_unit="h")
        >>> edata
        EHRData object with n_obs × n_vars × n_t = 1 × 1 × 2
            obs: 'gender', 'birth_date', 'deceased', 'deceased_time', 'anchor_time'
            var: 'n_events', 'description', 'unit'
            tem: '0', '1'
            shape of .X: (1, 1, 2)
        >>> edata.X
        array([[[72., 72.]]])
    """
    unknown = sorted(set(resources) - set(FHIR_RESOURCES))
    if unknown:
        msg = f"Unsupported resources {unknown}, choose from {list(FHIR_RESOURCES)}."
        raise ValueError(msg)

    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    _create_macros(con, PREFERRED_SYSTEMS if preferred_systems is None else preferred_systems)
    sources = _register_sources(con, Path(path))

    obs = None
    if "Patient" in resources:
        if "Patient" not in sources:
            msg = f"No Patient resources found in {path}."
            raise FileNotFoundError(msg)
        obs = (
            con.sql(
                f"""
                SELECT
                    id,
                    gender,
                    fhir_time(birthDate) AS birth_date,
                    coalesce(deceasedBoolean, deceasedDateTime IS NOT NULL) AS deceased,
                    fhir_time(deceasedDateTime) AS deceased_time
                FROM {sources["Patient"]}
                WHERE id IS NOT NULL
                ORDER BY id
                """
            )
            .df()
            .set_index("id")
            .rename_axis(None)
        )
        for column in ("birth_date", "deceased_time"):
            obs[column] = obs[column].map(str).mask(obs[column].isna())

    event_resources = [resource for resource in resources if resource != "Patient" and resource in sources]
    if not event_resources:
        msg = f"No {', '.join(resource for resource in resources if resource != 'Patient')} resources found in {path}."
        raise FileNotFoundError(msg)
    events = _event_relation(con, sources, event_resources).to_arrow_table()
    edata = from_events(events, obs=obs, codes=codes, **binning)
    con.register("fhir_events", events)
    var = (
        con.sql("SELECT code, mode(description) AS description, mode(unit) AS unit FROM fhir_events GROUP BY code")
        .df()
        .set_index("code")
        .reindex(edata.var_names)
    )
    edata.var["description"] = var["description"].to_numpy()
    edata.var["unit"] = var["unit"].to_numpy()
    return edata


def _duckdb_type(schema: str | Sequence[Any] | Mapping[str, Any]) -> str:
    if isinstance(schema, str):
        return schema
    if isinstance(schema, Mapping):
        return "STRUCT(" + ", ".join(f'"{key}" {_duckdb_type(value)}' for key, value in schema.items()) + ")"
    return f"{_duckdb_type(schema[0])}[]"


def _literal(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _create_macros(con: duckdb.DuckDBPyConnection, preferred_systems: Sequence[str]) -> None:
    vocabularies = " ".join(f"WHEN {_literal(url)} THEN {_literal(name)}" for url, name in VOCABULARIES.items())
    preferred = "".join(
        f"list_filter(codings, lambda c: c.system = {_literal(system)})[1], " for system in preferred_systems
    )
    con.execute(
        f"""
        CREATE MACRO fhir_time(s) AS CASE length(s)
            WHEN 4 THEN try_cast(s || '-01-01' AS TIMESTAMP)
            WHEN 7 THEN try_cast(s || '-01' AS TIMESTAMP)
            WHEN 10 THEN try_cast(s AS TIMESTAMP)
            ELSE timezone('UTC', try_cast(s AS TIMESTAMPTZ))
        END;
        CREATE MACRO fhir_ref(reference, resource_type := 'Patient') AS CASE
            WHEN starts_with(reference, resource_type || '/') THEN split_part(reference, '/', 2)
            WHEN starts_with(reference, 'urn:uuid:') THEN reference[10:]
            ELSE nullif(regexp_extract(reference, '/' || resource_type || '/([^/]+)', 1), '')
        END;
        CREATE MACRO fhir_vocabulary(system) AS CASE system {vocabularies} ELSE system END;
        CREATE MACRO fhir_pick(codings) AS CASE WHEN len(codings) > 1 THEN coalesce({preferred}codings[1]) ELSE codings[1] END;
        CREATE MACRO fhir_concept(concept) AS coalesce(
            fhir_vocabulary(fhir_pick(concept.coding).system) || '/' || fhir_pick(concept.coding).code, concept.text
        );
        CREATE MACRO fhir_description(concept) AS coalesce(fhir_pick(concept.coding).display, concept.text);
        """
    )


def _register_sources(con: duckdb.DuckDBPyConnection, path: Path) -> dict[str, str]:
    """Find the files under `path` and return a query of each resource type with the columns of its schema."""
    files = sorted(file for file in path.rglob("*") if file.is_file()) if path.is_dir() else [path]
    bundles = [str(file) for file in files if re.search(r"\.json(\.\w+)?$", file.name)]
    ndjson = [file for file in files if re.search(r"\.ndjson(\.\w+)?$", file.name)]

    bundle_resources = set()
    if bundles:
        con.execute(
            f"""
            CREATE TABLE bundle_resources AS
            SELECT resource->>'resourceType' AS resource_type, resource
            FROM (
                SELECT unnest(json_extract(json, '$.entry[*].resource')) AS resource
                FROM read_json_objects([{", ".join(map(_literal, bundles))}], maximum_object_size=2147483647)
            )
            """
        )
        bundle_resources = {row[0] for row in con.sql("SELECT DISTINCT resource_type FROM bundle_resources").fetchall()}

    sources = {}
    for resource_type, schema in _SCHEMAS.items():
        resource_files = [
            str(file) for file in ndjson if re.search(rf"(^|[^A-Za-z]){resource_type}([^A-Za-z]|$)", file.name)
        ]
        if resource_files:
            columns = ", ".join(f"{_literal(key)}: {_literal(_duckdb_type(value))}" for key, value in schema.items())
            sources[resource_type] = (
                f"read_json([{', '.join(map(_literal, resource_files))}], format='newline_delimited', columns={{{columns}}})"
            )
        elif resource_type in bundle_resources:
            sources[resource_type] = (
                f"(SELECT unnest(json_transform(resource, {_literal(json.dumps(schema))})) FROM bundle_resources "
                f"WHERE resource_type = {_literal(resource_type)})"
            )
    return sources


def _event_relation(
    con: duckdb.DuckDBPyConnection, sources: Mapping[str, str], resources: Sequence[str]
) -> duckdb.DuckDBPyRelation:
    """Return the events of `resources` with the columns subject_id, time, code, numeric_value, unit, and description."""
    medication_table = ", medication_code" if "Medication" in sources else ""
    medication_join = _MEDICATION_JOIN.format(source=sources["Medication"]) if "Medication" in sources else ""
    queries = [
        _EVENT_QUERIES[resource].format(
            source=sources[resource],
            medication=_MEDICATION_CONCEPT.format(medication_table=medication_table),
            medication_join=medication_join,
        )
        for resource in resources
    ]
    return con.sql(
        f"""
        SELECT
            #1::VARCHAR AS subject_id,
            #2::TIMESTAMP AS time,
            #3::VARCHAR AS code,
            #4::DOUBLE AS numeric_value,
            #5::VARCHAR AS unit,
            #6::VARCHAR AS description
        FROM ({" UNION ALL ".join(f"({query})" for query in queries)})
        WHERE #1 IS NOT NULL
        """
    )
