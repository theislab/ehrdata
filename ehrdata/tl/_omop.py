from __future__ import annotations

import numbers

from anndata import AnnData
from rich import print as rprint

from ehrdata.utils._omop_utils import df_to_dict, get_column_types, read_table


# TODO: overhaul
def get_concept_name(
    adata: AnnData | dict,
    concept_id: str | list,
    raise_error: bool = False,
) -> str | list[str]:
    """Get concept name from concept_id using concept table

    Args:
        adata: Anndata object or adata.uns
        concept_id: concept_id or list of concept_id
        raise_error: If True, raise error if concept_id not found. Defaults to False.

    Returns
    -------
        concept_name: concept name or list of concept names
    """
    if isinstance(concept_id, numbers.Integral):
        concept_id = [concept_id]

    if isinstance(adata, AnnData):
        adata_dict = adata.uns
    else:
        adata_dict = adata

    column_types = get_column_types(adata_dict, table_name="concept")
    df_concept = read_table(adata_dict, table_name="concept", dtype=column_types)
    # TODO dask Support
    # df_concept.compute().dropna(subset=["concept_id", "concept_name"], inplace=True, ignore_index=True)  # usecols=vocabularies_tables_columns["concept"]
    df_concept.dropna(
        subset=["concept_id", "concept_name"], inplace=True, ignore_index=True
    )  # usecols=vocabularies_tables_columns["concept"]
    concept_dict = df_to_dict(df=df_concept, key="concept_id", value="concept_name")
    concept_name = []
    concept_name_not_found = []
    for id in concept_id:
        try:
            concept_name.append(concept_dict[id])
        except KeyError:
            concept_name.append(id)
            concept_name_not_found.append(id)
    if len(concept_name_not_found) > 0:
        # warnings.warn(f"Couldn't find concept {id} in concept table!")
        rprint(f"Couldn't find concept {concept_name_not_found} in concept table!")
        if raise_error:
            raise KeyError
    if len(concept_name) == 1:
        return concept_name[0]
    else:
        return concept_name
