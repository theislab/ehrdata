from __future__ import annotations

import os
import re
from bisect import bisect_right
from functools import singledispatch
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import scipy.sparse
import sparse

from ehrdata._compat import DaskArray
from ehrdata.core import EHRData

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from duckdb import DuckDBPyConnection

ICD9_VOCABULARIES = frozenset({"ICD9", "ICD9CM"})
ICD10_VOCABULARIES = frozenset({"ICD10", "ICD10CM", "ICD10GM"})

# (first category, chapter, title) of the ICD-9-CM chapters, sorted by first category.
ICD9_CHAPTERS: Sequence[tuple[str, str, str]] = (
    ("001", "001-139", "Infectious and parasitic diseases"),
    ("140", "140-239", "Neoplasms"),
    ("240", "240-279", "Endocrine, nutritional and metabolic diseases, and immunity disorders"),
    ("280", "280-289", "Diseases of the blood and blood-forming organs"),
    ("290", "290-319", "Mental disorders"),
    ("320", "320-389", "Diseases of the nervous system and sense organs"),
    ("390", "390-459", "Diseases of the circulatory system"),
    ("460", "460-519", "Diseases of the respiratory system"),
    ("520", "520-579", "Diseases of the digestive system"),
    ("580", "580-629", "Diseases of the genitourinary system"),
    ("630", "630-679", "Complications of pregnancy, childbirth, and the puerperium"),
    ("680", "680-709", "Diseases of the skin and subcutaneous tissue"),
    ("710", "710-739", "Diseases of the musculoskeletal system and connective tissue"),
    ("740", "740-759", "Congenital anomalies"),
    ("760", "760-779", "Certain conditions originating in the perinatal period"),
    ("780", "780-799", "Symptoms, signs, and ill-defined conditions"),
    ("800", "800-999", "Injury and poisoning"),
    ("E000", "E000-E999", "Supplementary classification of external causes of injury and poisoning"),
    (
        "V01",
        "V01-V91",
        "Supplementary classification of factors influencing health status and contact with health services",
    ),
)

# (first category, chapter, title) of the ICD-10-CM chapters, sorted by first category.
ICD10_CHAPTERS: Sequence[tuple[str, str, str]] = (
    ("A00", "A00-B99", "Certain infectious and parasitic diseases"),
    ("C00", "C00-D49", "Neoplasms"),
    (
        "D50",
        "D50-D89",
        "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism",
    ),
    ("E00", "E00-E89", "Endocrine, nutritional and metabolic diseases"),
    ("F00", "F01-F99", "Mental, behavioral and neurodevelopmental disorders"),
    ("G00", "G00-G99", "Diseases of the nervous system"),
    ("H00", "H00-H59", "Diseases of the eye and adnexa"),
    ("H60", "H60-H95", "Diseases of the ear and mastoid process"),
    ("I00", "I00-I99", "Diseases of the circulatory system"),
    ("J00", "J00-J99", "Diseases of the respiratory system"),
    ("K00", "K00-K95", "Diseases of the digestive system"),
    ("L00", "L00-L99", "Diseases of the skin and subcutaneous tissue"),
    ("M00", "M00-M99", "Diseases of the musculoskeletal system and connective tissue"),
    ("N00", "N00-N99", "Diseases of the genitourinary system"),
    ("O00", "O00-O9A", "Pregnancy, childbirth and the puerperium"),
    ("P00", "P00-P96", "Certain conditions originating in the perinatal period"),
    ("Q00", "Q00-Q99", "Congenital malformations, deformations and chromosomal abnormalities"),
    ("R00", "R00-R99", "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified"),
    ("S00", "S00-T88", "Injury, poisoning and certain other consequences of external causes"),
    ("U00", "U00-U85", "Codes for special purposes"),
    ("V00", "V00-Y99", "External causes of morbidity"),
    ("Z00", "Z00-Z99", "Factors influencing health status and contact with health services"),
)

_ICD9_CATEGORY = re.compile(r"[0-9]{3}|V[0-9]{2}|E[0-9]{3}")
_ICD10_CATEGORY = re.compile(r"[A-Z][0-9][0-9A-Z]")

ATC_LEVEL_LENGTHS = (1, 3, 4, 5, 7)
_ATC_CODE = re.compile(r"[A-Z]([0-9]{2}([A-Z]([A-Z]([0-9]{2})?)?)?)?")

# The MEDS MIMIC-IV ETL codes diagnoses and procedures as DIAGNOSIS//ICD//<version>//<code> and PROCEDURE//ICD//<version>//<code>.
_MEDS_ICD = re.compile(r"(DIAGNOSIS|PROCEDURE)//ICD//(9|10)//(.+)")
_MEDS_ICD_VOCABULARIES = {
    ("DIAGNOSIS", "9"): "ICD9CM",
    ("DIAGNOSIS", "10"): "ICD10CM",
    ("PROCEDURE", "9"): "ICD9Proc",
    ("PROCEDURE", "10"): "ICD10PCS",
}
_MEDS_MEDICATION = re.compile(r"MEDICATION//.*//([^/]+)")
_MEDS_DRG = re.compile(r"DRG//(HCFA|APR)//([0-9]+)")
_VOCABULARY_CODE = re.compile(r"([^/]+)/([^/].*)")

CODE_COLUMNS = (
    "vocabulary",
    "code",
    "description",
    "icd_chapter",
    "icd_category",
    "icd_codes",
    *(f"atc_level_{level}" for level in range(1, len(ATC_LEVEL_LENGTHS) + 1)),
)


def _icd9_category(code: str) -> str:
    return code[:4] if code.startswith("E") else code[:3]


def _icd_rollups(
    vocabulary: str, code: str, chapters: Sequence[tuple[str, str, str]], category: str, category_pattern: re.Pattern
) -> dict[str, str]:
    titles = {chapter: title for _, chapter, title in chapters}
    if code in titles:
        return {"icd_chapter": f"{vocabulary}/{code}", "description": titles[code]}
    starts = [start for start, _, _ in chapters]
    if not category_pattern.fullmatch(category) or category < starts[0]:
        return {}
    chapter = chapters[bisect_right(starts, category) - 1][1]
    return {
        "icd_chapter": f"{vocabulary}/{chapter}",
        "icd_category": f"{vocabulary}/{category}",
        "icd_codes": f"{vocabulary}/{code}",
    }


def _atc_rollups(vocabulary: str, code: str) -> dict[str, str]:
    code = code.strip().upper()
    if not _ATC_CODE.fullmatch(code):
        return {}
    return {
        f"atc_level_{level}": f"{vocabulary}/{code[:length]}"
        for level, length in enumerate(ATC_LEVEL_LENGTHS, start=1)
        if len(code) >= length
    }


def _rollups(vocabulary: object, code: object) -> dict[str, str]:
    if not isinstance(vocabulary, str) or not isinstance(code, str):
        return {}
    icd_code = code.strip().upper().replace(".", "")
    if vocabulary.upper() in ICD9_VOCABULARIES:
        return _icd_rollups(vocabulary, icd_code, ICD9_CHAPTERS, _icd9_category(icd_code), _ICD9_CATEGORY)
    if vocabulary.upper() in ICD10_VOCABULARIES:
        return _icd_rollups(vocabulary, icd_code, ICD10_CHAPTERS, icd_code[:3], _ICD10_CATEGORY)
    if vocabulary.upper() == "ATC":
        return _atc_rollups(vocabulary, code)
    return {}


def _parse_code(name: object) -> tuple[object, object]:
    if not isinstance(name, str):
        return pd.NA, pd.NA
    if match := _MEDS_ICD.fullmatch(name):
        return _MEDS_ICD_VOCABULARIES[match[1], match[2]], match[3]
    if match := _MEDS_MEDICATION.fullmatch(name):
        return ("NDC", match[1]) if match[1] != "UNK" else (pd.NA, pd.NA)
    if match := _MEDS_DRG.fullmatch(name):
        return "MS-DRG" if match[1] == "HCFA" else "APR-DRG", match[2].zfill(3)
    if match := _VOCABULARY_CODE.fullmatch(name):
        return match[1], match[2]
    return pd.NA, pd.NA


def _parse_codes(names: Sequence[str]) -> tuple[pd.Series, pd.Series]:
    parsed = [_parse_code(name) for name in names]
    return (
        pd.Series([vocabulary for vocabulary, _ in parsed], dtype="string"),
        pd.Series([code for _, code in parsed], dtype="string"),
    )


def _flatten(values: Iterable[object]) -> list[object]:
    return [code for value in values for code in (value if isinstance(value, list | np.ndarray) else [value])]


def _join_codes(codes: Iterable[object]) -> object:
    unique = sorted({code for value in codes if isinstance(value, str) for code in value.split("|")})
    return "|".join(unique) if unique else pd.NA


def _annotate_var(var: pd.DataFrame, vocabulary: pd.Series, code: pd.Series) -> pd.DataFrame:
    var = var.copy()
    var["vocabulary"] = pd.array(vocabulary, dtype="string")
    var["code"] = pd.array(code, dtype="string")
    rollups = pd.DataFrame(
        [_rollups(v, c) for v, c in zip(var["vocabulary"], var["code"], strict=True)],
        index=var.index,
        columns=CODE_COLUMNS[2:],
    ).astype("string")
    description = var.get("description", pd.Series(pd.NA, index=var.index)).astype("string")
    if "concept_name" in var.columns:
        description = description.fillna(var["concept_name"].astype("string"))
    var["description"] = description.fillna(rollups.pop("description"))
    for column in rollups.columns:
        var[column] = rollups[column]
    return var


def _table_exists(backend_handle: DuckDBPyConnection, table: str) -> bool:
    query = "SELECT count(*) FROM information_schema.tables WHERE lower(table_name) = ?"
    return backend_handle.execute(query, [table]).fetchone()[0] > 0


def _omop_concepts(
    backend_handle: DuckDBPyConnection, concept_ids: pd.Series, vocabulary: pd.Series, code: pd.Series
) -> pd.DataFrame:
    """Look up the concept of each variable by its concept id or else by its vocabulary and code.

    Also returns the codes of the ATC ancestors and of the ICD source concepts that map to each concept.
    """
    keys = pd.DataFrame(
        {
            "position": np.arange(len(code)),
            "concept_id": pd.to_numeric(concept_ids, errors="coerce").astype("Int64").array,
            "vocabulary_id": pd.array(vocabulary, dtype="string"),
            "concept_code": pd.array(code, dtype="string"),
        }
    )
    query = """
        SELECT k.position, c.concept_id, c.vocabulary_id, c.concept_code::VARCHAR AS concept_code, c.concept_name
        FROM _ehrdata_code_keys k JOIN concept c ON c.concept_id = coalesce(
            k.concept_id::BIGINT,
            (SELECT min(c2.concept_id) FROM concept c2
             WHERE c2.vocabulary_id = k.vocabulary_id::VARCHAR AND c2.concept_code::VARCHAR = k.concept_code::VARCHAR)
        )
    """
    icd_vocabularies = ", ".join(f"'{v}'" for v in sorted(ICD9_VOCABULARIES | ICD10_VOCABULARIES))
    related_queries = {
        "concept_ancestor": """
            SELECT ca.descendant_concept_id AS concept_id, list(a.concept_code::VARCHAR) AS atc_codes
            FROM concept_ancestor ca JOIN concept a ON ca.ancestor_concept_id = a.concept_id
            WHERE a.vocabulary_id = 'ATC' AND ca.descendant_concept_id IN (SELECT concept_id FROM _ehrdata_concepts)
            GROUP BY ca.descendant_concept_id
        """,
        "concept_relationship": f"""
            SELECT r.concept_id, list(s.vocabulary_id || '/' || replace(upper(s.concept_code::VARCHAR), '.', ''))
                AS icd_source_codes
            FROM (
                SELECT concept_id_2 AS concept_id, concept_id_1 AS source_concept_id FROM concept_relationship
                WHERE relationship_id = 'Maps to'
                UNION
                SELECT concept_id_1 AS concept_id, concept_id_2 AS source_concept_id FROM concept_relationship
                WHERE relationship_id = 'Mapped from'
            ) r JOIN concept s ON r.source_concept_id = s.concept_id
            WHERE s.vocabulary_id IN ({icd_vocabularies}) AND r.concept_id IN (SELECT concept_id FROM _ehrdata_concepts)
            GROUP BY r.concept_id
        """,
    }
    backend_handle.register("_ehrdata_code_keys", keys)
    try:
        concepts = backend_handle.execute(query).df()
    finally:
        backend_handle.unregister("_ehrdata_code_keys")
    backend_handle.register("_ehrdata_concepts", concepts)
    try:
        for table, related_query in related_queries.items():
            related = backend_handle.execute(related_query).df() if _table_exists(backend_handle, table) else None
            column = "atc_codes" if table == "concept_ancestor" else "icd_source_codes"
            concepts[column] = (
                concepts["concept_id"].map(related.set_index("concept_id")[column]) if related is not None else None
            )
    finally:
        backend_handle.unregister("_ehrdata_concepts")
    return concepts.set_index("position").reindex(keys["position"])


def _atc_lowest_common_level(codes: object) -> object:
    """The most specific ATC code shared by all ATC ancestors of a concept that have no more specific ATC descendant."""
    if not isinstance(codes, list | np.ndarray) or len(codes) == 0:
        return np.nan
    codes = [str(code) for code in codes]
    leaves = [code for code in codes if not any(other != code and other.startswith(code) for other in codes)]
    common = os.path.commonprefix(leaves)
    length = max((length for length in ATC_LEVEL_LENGTHS if length <= len(common)), default=0)
    return common[:length] if length else np.nan


def annotate_codes(
    edata: EHRData,
    *,
    backend_handle: DuckDBPyConnection | None = None,
    copy: bool = False,
) -> EHRData | None:
    """Annotate the variables with the vocabulary, code, description and hierarchy of their codes.

    Variables are named by codes like `ICD10CM/I21.0`, `ICD9CM/410.01`, `ATC/C07AB02`, `LOINC/8480-6`, or `SNOMED/22298006`, that is, an OMOP vocabulary name and a code of this vocabulary.
    Codes of the MEDS MIMIC-IV ETL are understood as well: `DIAGNOSIS//ICD//10//I210` and `DIAGNOSIS//ICD//9//41001` as `ICD10CM` and `ICD9CM`, `PROCEDURE//ICD//10//...` and `PROCEDURE//ICD//9//...` as `ICD10PCS` and `ICD9Proc`, the NDC at the end of `MEDICATION//...` codes as `NDC`, and `DRG//HCFA//...` and `DRG//APR//...` as `MS-DRG` and `APR-DRG`.
    Variables read from an OMOP CDM database are identified by their `data_table_concept_id` or, if enriched with feature information, by their `vocabulary_id` and `concept_code` instead.

    The following columns are written to `.var`:

    - `vocabulary` and `code`: the vocabulary and the code of the variable.
    - `description`: the description of the code, if already known from `.var` or from the OMOP CDM database.
    - `icd_chapter` and `icd_category`: the chapter, such as `ICD10CM/I00-I99`, and the category, such as `ICD10CM/I21` or `ICD9CM/410`, of ICD-9-CM and ICD-10 diagnosis codes.
    - `icd_codes`: the ICD-9-CM and ICD-10 diagnosis codes of the variable in upper case without dots, such as `ICD10CM/I210`, separated by `|`.
    - `atc_level_1` to `atc_level_5`: the ATC code at each level, such as `ATC/C`, `ATC/C07`, `ATC/C07A`, `ATC/C07AB`, and `ATC/C07AB02`.

    The ICD-10 chapters of ICD-10-CM are used for `ICD10`, `ICD10CM`, and `ICD10GM` codes alike, and those of ICD-9-CM for `ICD9` and `ICD9CM` codes.
    Descriptions of codes in licensed vocabularies like LOINC or SNOMED CT are only available from an OMOP CDM database.
    Likewise, SNOMED CT and other standard concepts can only be mapped to ICD codes with an OMOP CDM database.
    If the database has a `concept_relationship` table, the `icd_codes` of a concept also contain the ICD source codes that map to it.
    If the database has a `concept_ancestor` table, the ATC levels of drugs coded in other vocabularies, such as RxNorm, are taken from their ATC ancestors.
    Use :func:`~ehrdata.aggregate_codes` to aggregate the variables to one of these levels.

    Args:
        edata: Central data object.
        backend_handle: A DuckDB connection to an OMOP CDM database set up with :func:`~ehrdata.io.omop.setup_connection`, used to look up the concepts of the variables.
        copy: If `True`, a new :class:`~ehrdata.EHRData` object is returned. If `False`, the original object is modified inplace and `None` is returned.

    Returns:
        If `copy` is `True`, the :class:`~ehrdata.EHRData` object with the annotated `.var`.

    Examples:
        >>> import numpy as np
        >>> import ehrdata as ed
        >>> edata = ed.EHRData(np.ones((2, 3)), var=dict(var_names=["ICD10CM/I21.0", "ATC/C07AB02", "LOINC/8480-6"]))
        >>> ed.annotate_codes(edata)
        >>> edata.var[["vocabulary", "code", "icd_category", "atc_level_2"]]
                      vocabulary     code icd_category atc_level_2
        ICD10CM/I21.0    ICD10CM    I21.0  ICD10CM/I21        <NA>
        ATC/C07AB02          ATC  C07AB02         <NA>     ATC/C07
        LOINC/8480-6       LOINC   8480-6         <NA>        <NA>
    """
    if copy:
        edata = edata.copy()

    var = edata.var
    if {"vocabulary_id", "concept_code"} <= set(var.columns):
        vocabulary, code = var["vocabulary_id"], var["concept_code"]
    elif "data_table_concept_id" in var.columns:
        vocabulary = code = pd.Series(pd.NA, index=var.index, dtype="string")
    else:
        vocabulary, code = _parse_codes(var.index)
        vocabulary.index = code.index = var.index

    if backend_handle is not None:
        if not _table_exists(backend_handle, "concept"):
            msg = "The OMOP CDM database has no `concept` table."
            raise ValueError(msg)
        concept_ids = var.get("concept_id", var.get("data_table_concept_id", pd.Series(pd.NA, index=var.index)))
        concepts = _omop_concepts(backend_handle, concept_ids, vocabulary, code)
        concepts.index = var.index
        vocabulary = vocabulary.where(vocabulary.notna(), concepts["vocabulary_id"])
        code = code.where(code.notna(), concepts["concept_code"])

    var = _annotate_var(var, vocabulary, code)

    if backend_handle is not None:
        var["description"] = var["description"].fillna(concepts["concept_name"].astype("string"))
        atc = concepts["atc_codes"].map(_atc_lowest_common_level)
        is_atc = var["vocabulary"].astype(str).str.upper() == "ATC"
        for name in var.index[~is_atc & atc.notna()]:
            for column, value in _atc_rollups("ATC", atc[name]).items():
                var.loc[name, column] = value
        icd_codes = pd.concat([var["icd_codes"], concepts["icd_source_codes"]], axis=1)
        var["icd_codes"] = pd.array(
            [_join_codes(_flatten(row)) for row in icd_codes.itertuples(index=False)], dtype="string"
        )
        var["concept_id"] = concepts["concept_id"].astype("Int64").fillna(pd.to_numeric(concept_ids, errors="coerce"))

    edata.var = var
    return edata if copy else None


def _reduce_dense(x: np.ndarray, groups: np.ndarray, sizes: np.ndarray, strategy: Literal["sum", "max"]) -> np.ndarray:
    order = np.argsort(groups, kind="stable")
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    x = np.take(x, order, axis=1)
    if x.dtype == bool and strategy == "sum":
        x = x.astype(np.int64)
    if strategy == "max":
        return np.fmax.reduceat(x, starts, axis=1)
    if not np.issubdtype(x.dtype, np.floating):
        return np.add.reduceat(x, starts, axis=1)
    missing = np.isnan(x)
    out = np.add.reduceat(np.where(missing, 0, x), starts, axis=1)
    out[np.logical_and.reduceat(missing, starts, axis=1)] = np.nan
    return out


def _reduce_coords(
    coords: np.ndarray,
    data: np.ndarray,
    shape: Sequence[int],
    *,
    fill_value: float,
    groups: np.ndarray,
    sizes: np.ndarray,
    strategy: Literal["sum", "max"],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Reduce the explicit entries of a sparse array over the groups of axis 1, with `fill_value` at all other entries."""
    out_shape = (shape[0], len(sizes), *shape[2:])
    if data.dtype == bool and strategy == "sum":
        data = data.astype(np.int64)
    if len(data) == 0:
        return np.empty((len(shape), 0), dtype=np.intp), data, out_shape
    coords = np.array(coords, dtype=np.intp)
    coords[1] = groups[coords[1]]
    flat = np.ravel_multi_index(tuple(coords), out_shape)
    order = np.argsort(flat, kind="stable")
    flat, data = flat[order], data[order]
    cells, starts = np.unique(flat, return_index=True)
    cell_coords = np.stack(np.unravel_index(cells, out_shape))
    n_implicit = sizes[cell_coords[1]] - np.diff(np.append(starts, len(flat)))
    if strategy == "max":
        values = np.fmax.reduceat(data, starts)
        has_implicit = n_implicit > 0
        values[has_implicit] = np.fmax(values[has_implicit], fill_value)
        return cell_coords, values, out_shape
    if not np.issubdtype(data.dtype, np.floating):
        return cell_coords, np.add.reduceat(data, starts), out_shape
    missing = np.isnan(data)
    values = np.add.reduceat(np.where(missing, 0, data), starts)
    n_missing = np.add.reduceat(missing, starts) + (n_implicit if np.isnan(fill_value) else 0)
    values[n_missing == sizes[cell_coords[1]]] = np.nan
    return cell_coords, values, out_shape


@singledispatch
def _reduce_groups(x, groups: np.ndarray, sizes: np.ndarray, strategy: Literal["sum", "max"]):
    msg = f"Aggregating codes of an array of type {type(x)} is not implemented."
    raise NotImplementedError(msg)


@_reduce_groups.register(np.ndarray)
def _(x: np.ndarray, groups: np.ndarray, sizes: np.ndarray, strategy: Literal["sum", "max"]) -> np.ndarray:
    return _reduce_dense(x, groups, sizes, strategy)


@_reduce_groups.register(scipy.sparse.sparray)
@_reduce_groups.register(scipy.sparse.spmatrix)
def _(x, groups: np.ndarray, sizes: np.ndarray, strategy: Literal["sum", "max"]):
    coo = x.tocoo()
    coords, values, shape = _reduce_coords(
        np.stack([coo.row, coo.col]), coo.data, x.shape, fill_value=0, groups=groups, sizes=sizes, strategy=strategy
    )
    return type(x)((values, tuple(coords)), shape=shape)


@_reduce_groups.register(sparse.COO)
def _(x: sparse.COO, groups: np.ndarray, sizes: np.ndarray, strategy: Literal["sum", "max"]) -> sparse.COO:
    fill_value = x.fill_value
    if not (fill_value == 0 or np.isnan(fill_value)):
        return sparse.COO.from_numpy(_reduce_dense(x.todense(), groups, sizes, strategy), fill_value=fill_value)
    coords, values, shape = _reduce_coords(
        x.coords, x.data, x.shape, fill_value=fill_value, groups=groups, sizes=sizes, strategy=strategy
    )
    return sparse.COO(coords, values, shape=shape, fill_value=fill_value, has_duplicates=False, sorted=True)


@_reduce_groups.register(DaskArray)
def _(x: DaskArray, groups: np.ndarray, sizes: np.ndarray, strategy: Literal["sum", "max"]) -> DaskArray:
    x = x.rechunk({1: -1})
    meta = x._meta.astype(np.int64) if x.dtype == bool and strategy == "sum" else x._meta
    return x.map_blocks(
        _reduce_groups,
        groups,
        sizes,
        strategy,
        chunks=(x.chunks[0], (len(sizes),), *x.chunks[2:]),
        meta=meta,
    )


def aggregate_codes(
    edata: EHRData,
    by: str,
    *,
    aggregation_strategy: Literal["sum", "max"] = "sum",
    layer: str | None = None,
) -> EHRData:
    """Aggregate the variables to a level of their code hierarchy.

    Variables with the same value in `.var[by]` are aggregated into one variable named by this value.
    Variables without a value in `.var[by]` are kept as they are.
    Missing values are ignored, so an aggregated value is only missing if it is missing in all aggregated variables.
    Typically, `by` is one of the columns written by :func:`~ehrdata.annotate_codes`, such as `icd_category` or `atc_level_3`.

    Args:
        edata: Central data object.
        by: The column of `.var` with the level to aggregate the variables to.
        aggregation_strategy: How to aggregate the values of the variables.
            Use `"sum"` to add up counts and `"max"` to tell whether any of the variables is present.
        layer: The layer to aggregate. If `None`, `.X` is aggregated.

    Returns:
        An :class:`~ehrdata.EHRData` object with the aggregated `layer` and the same `.obs` and `.tem`.
        Its `.var` keeps the columns that are the same for all variables aggregated into one, and is annotated with :func:`~ehrdata.annotate_codes`.

    Examples:
        >>> import numpy as np
        >>> import ehrdata as ed
        >>> edata = ed.EHRData(
        ...     np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]]),
        ...     var=dict(var_names=["ICD10CM/I21.0", "ICD10CM/I21.4", "ICD10CM/E11.9"]),
        ... )
        >>> ed.annotate_codes(edata)
        >>> edata_categories = ed.aggregate_codes(edata, "icd_category")
        >>> edata_categories.to_df()
             ICD10CM/I21  ICD10CM/E11
        0          3.0          0.0
        1          1.0          3.0
    """
    if by not in edata.var.columns:
        msg = f"`{by}` is not a column of `.var`. Annotate the codes with `ed.annotate_codes` first."
        raise ValueError(msg)
    if aggregation_strategy not in ("sum", "max"):
        msg = "aggregation_strategy must be one of ('sum', 'max')."
        raise ValueError(msg)

    var = edata.var
    labels = var[by].astype(object).where(var[by].notna(), var.index.to_series())
    groups, names = pd.factorize(labels)
    sizes = np.bincount(groups, minlength=len(names))

    x = edata.X if layer is None else edata.layers[layer]
    aggregated = _reduce_groups(x, groups, sizes, aggregation_strategy)

    grouped = var.groupby(groups, sort=True)
    new_var = grouped.first(skipna=False).where(grouped.nunique(dropna=False).eq(1))
    new_var.index = pd.Index(names.astype(str), name=var.index.name)
    new_var = new_var.reindex(columns=new_var.columns.union(CODE_COLUMNS, sort=False))
    new_var[list(CODE_COLUMNS)] = new_var[list(CODE_COLUMNS)].astype("string")
    is_aggregated = var[by].notna().groupby(groups).any().to_numpy()
    aggregated_var = _annotate_var(
        new_var.loc[is_aggregated].drop(columns="description"), *_parse_codes(new_var.index[is_aggregated])
    )
    new_var.loc[is_aggregated, list(CODE_COLUMNS)] = aggregated_var[list(CODE_COLUMNS)]
    if "icd_codes" in var.columns:
        icd_codes = var["icd_codes"].groupby(groups).agg(_join_codes).to_numpy()
        new_var.loc[is_aggregated, "icd_codes"] = icd_codes[is_aggregated]

    return EHRData(
        X=aggregated if layer is None else None,
        layers=None if layer is None else {layer: aggregated},
        obs=edata.obs,
        var=new_var,
        tem=edata.tem,
        obsm=edata.obsm,
        obsp=edata.obsp,
        uns=edata.uns,
    )
