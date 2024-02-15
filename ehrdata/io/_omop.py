import os
from typing import Literal, Union

import awkward as ak
import ehrapy as ep
import pandas as pd
from rich import print as rprint

from ehrdata.utils.omop_utils import check_with_omop_cdm, get_column_types, get_table_catalog_dict, read_table


def init_omop(
    folder_path,
    delimiter=None,
    make_filename_lowercase=True,
    use_dask=False,
    level: Literal["stay_level", "patient_level"] = "stay_level",
    tables: Union[str, list[str]] = None,
    remove_empty_column=True,
):
    filepath_dict = check_with_omop_cdm(
        folder_path=folder_path, delimiter=delimiter, make_filename_lowercase=make_filename_lowercase
    )
    tables = list(filepath_dict.keys())
    adata_dict = {}
    adata_dict["filepath_dict"] = filepath_dict
    adata_dict["tables"] = tables
    adata_dict["delimiter"] = delimiter
    adata_dict["use_dask"] = use_dask

    table_catalog_dict = get_table_catalog_dict()

    color_map = {
        "Clinical data": "blue",
        "Health system data": "green",
        "Health economics data": "red",
        "Standardized derived elements": "magenta",
        "Metadata": "white",
        "Vocabulary": "dark_orange",
    }
    # Object description
    print_str = f"OMOP Database ([red]{os.path.basename(folder_path)}[/]) with {len(tables)} tables.\n"

    # Tables information
    for key, value in table_catalog_dict.items():
        table_list = [table_name for table_name in tables if table_name in value]
        if len(table_list) != 0:
            print_str = print_str + f"[{color_map[key]}]{key} tables[/]: [black]{', '.join(table_list)}[/]\n"
            # table_list_str = ', '.join(table_list)

            # text = Text(f"{key} tables: ", style=color_map[key])
            # text.append(table_list_str)
            # yield None, f"{key} tables", "red"
    rprint(print_str)

    tables = ["person", "death", "visit_occurrence"]
    # TODO patient level and hospital level
    if level == "stay_level":
        # index = {"visit_occurrence": "visit_occurrence_id", "person": "person_id", "death": "person_id"}
        # TODO Only support clinical_tables_columns
        table_dict = {}
        for table in tables:
            print(f"reading table [{table}]")
            column_types = get_column_types(adata_dict, table_name=table)
            df = read_table(adata_dict, table_name=table, dtype=column_types, index="person_id")
            if remove_empty_column:
                # TODO dask Support
                # columns = [column for column in df.columns if not df[column].compute().isna().all()]
                columns = [column for column in df.columns if not df[column].isna().all()]
            df = df.loc[:, columns]
            table_dict[table] = df

        # concept_id_list = list(self.concept.concept_id)
        # concept_name_list = list(self.concept.concept_id)
        # concept_domain_id_list = list(set(self.concept.domain_id))

        # self.loaded_tabel = ['visit_occurrence', 'person', 'death', 'measurement', 'observation', 'drug_exposure']
        # TODO dask Support
        joined_table = pd.merge(
            table_dict["visit_occurrence"], table_dict["person"], left_index=True, right_index=True, how="left"
        )

        joined_table = pd.merge(joined_table, table_dict["death"], left_index=True, right_index=True, how="left")

        # TODO dask Support
        # joined_table = joined_table.compute()

        # TODO check this earlier
        joined_table = joined_table.drop_duplicates(subset="visit_occurrence_id")
        joined_table = joined_table.set_index("visit_occurrence_id")
        # obs_only_list = list(self.joined_table.columns)
        # obs_only_list.remove('visit_occurrence_id')
        columns_obs_only = list(set(joined_table.columns) - {"year_of_birth", "gender_source_value"})
        adata = ep.ad.df_to_anndata(joined_table, index_column="visit_occurrence_id", columns_obs_only=columns_obs_only)
        # TODO this needs to be fixed because anndata set obs index as string by default
        # adata.obs.index = adata.obs.index.astype(int)

        """
        for column in self.measurement.columns:
            if column != 'visit_occurrence_id':
                obs_list = []
                for visit_occurrence_id in adata.obs.index:
                    obs_list.append(list(self.measurement[self.measurement['visit_occurrence_id'] == int(visit_occurrence_id)][column]))
                adata.obsm[column]= ak.Array(obs_list)

        for column in self.drug_exposure.columns:
            if column != 'visit_occurrence_id':
                obs_list = []
                for visit_occurrence_id in adata.obs.index:
                    obs_list.append(list(self.drug_exposure[self.drug_exposure['visit_occurrence_id'] == int(visit_occurrence_id)][column]))
                adata.obsm[column]= ak.Array(obs_list)

        for column in self.observation.columns:
            if column != 'visit_occurrence_id':
                obs_list = []
                for visit_occurrence_id in adata.obs.index:
                    obs_list.append(list(self.observation[self.observation['visit_occurrence_id'] == int(visit_occurrence_id)][column]))
                adata.obsm[column]= ak.Array(obs_list)
        """

        adata.uns.update(adata_dict)
    elif level == "patient_level":
        # TODO patient level
        # Each row in anndata would be a patient
        pass
    else:
        raise ValueError("level should be 'stay_level' or 'patient_level'")

    return adata


def from_dataframe(adata, feature: str, df):
    grouped = df.groupby("visit_occurrence_id")
    unique_visit_occurrence_ids = set(adata.obs.index)

    # Use set difference and intersection more efficiently
    feature_ids = unique_visit_occurrence_ids.intersection(grouped.groups.keys())
    empty_entry = {
        source_table_column: []
        for source_table_column in set(df.columns)
        if source_table_column not in ["visit_occurrence_id"]
    }
    columns_in_ak_array = list(set(df.columns) - {"visit_occurrence_id"})
    # Creating the array more efficiently
    ak_array = ak.Array(
        [
            grouped.get_group(visit_occurrence_id)[columns_in_ak_array].to_dict(orient="list")
            if visit_occurrence_id in feature_ids
            else empty_entry
            for visit_occurrence_id in unique_visit_occurrence_ids
        ]
    )
    adata.obsm[feature] = ak_array

    return adata


# TODO add function to check feature and add concept
# More IO functions


def to_dataframe(
    adata,
    features: Union[str, list[str]],  # TODO also support list of features
    # patient str or List,  # TODO also support subset of patients/visit
):
    # TODO
    # can be viewed as patient level - only select some patient
    # TODO change variable name here
    if isinstance(features, str):
        features = [features]
    df_concat = pd.DataFrame([])
    for feature in features:
        df = ak.to_dataframe(adata.obsm[feature])

        df.reset_index(drop=False, inplace=True)
        df["entry"] = adata.obs.index[df["entry"]]
        df = df.rename(columns={"entry": "visit_occurrence_id"})
        del df["subentry"]
        for col in df.columns:
            if col.endswith("time"):
                df[col] = pd.to_datetime(df[col])

        df["feature_name"] = feature
        df_concat = pd.concat([df_concat, df], axis=0)

    return df_concat
