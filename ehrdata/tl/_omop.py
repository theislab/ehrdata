import numbers
from typing import Union

from anndata import AnnData
from rich import print as rprint

from ehrdata.utils.omop_utils import df_to_dict, get_column_types, read_table


def get_concept_name(adata: Union[AnnData, dict], concept_id: Union[str, list], raise_error=False, verbose=True):
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
        if verbose:
            rprint(f"Couldn't find concept {concept_name_not_found} in concept table!")
        if raise_error:
            raise KeyError
    if len(concept_name) == 1:
        return concept_name[0]
    else:
        return concept_name


# TODO
def get_concept_id():
    pass
