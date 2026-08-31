from __future__ import annotations

from typing import Literal


def get_table_catalog_dict(version: Literal["5.4"] = "5.4") -> dict[str, str]:
    """Get the table catalog dictionary of the OMOP CDM.

    Args:
        version:
            The version of the OMOP CDM. Currently, only 5.4 is supported.

    Returns:
        A dictionary of the table catalog. The key is the category of the table, and the value is a list of table names.
    """
    table_catalog_dict = {}
    table_catalog_dict["Clinical data"] = [
        "person",
        "observation_period",
        "specimen",
        "death",
        "visit_occurrence",
        "visit_detail",
        "procedure_occurrence",
        "drug_exposure",
        "device_exposure",
        "condition_occurrence",
        "measurement",
        "note",
        "note_nlp",
        "observation",
        "fact_relationship",
        "episode",
    ]

    table_catalog_dict["Health system data"] = ["location", "care_site", "provider"]
    table_catalog_dict["Health economics data"] = ["payer_plan_period", "cost"]
    table_catalog_dict["Standardized derived elements"] = [
        "cohort",
        "cohort_definition",
        "drug_era",
        "dose_era",
        "condition_era",
    ]
    table_catalog_dict["Metadata"] = ["cdm_source", "metadata"]
    table_catalog_dict["Vocabulary"] = [
        "concept",
        "vocabulary",
        "domain",
        "concept_class",
        "concept_relationship",
        "relationship",
        "concept_synonym",
        "concept_ancestor",
        "source_to_concept_map",
        "drug_strength",
    ]

    return table_catalog_dict
