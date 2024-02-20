import os
from typing import Literal, Optional, Union

import awkward as ak
import ehrapy as ep
import pandas as pd
from rich import print as rprint

from ehrdata.utils._omop_utils import check_with_omop_cdm, get_column_types, get_table_catalog_dict, read_table


def init_omop(
    folder_path,
    delimiter=None,
    make_filename_lowercase=True,
    use_dask=False,
    level: Literal["stay_level", "patient_level"] = "stay_level",
    load_tables: Optional[Union[str, list[str], tuple[str], Literal["auto"]]] = ("visit_occurrence", "person"),
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
    rprint(print_str)

    if load_tables == "auto" or load_tables is None:
        load_tables = tables
    elif isinstance(load_tables, str):
        load_tables = [load_tables]
    else:
        pass

    # TODO patient level and hospital level
    if level == "stay_level":
        table_dict = {}
        for table in ["visit_occurrence", "person"]:
            column_types = get_column_types(adata_dict, table_name=table)
            table_dict[table] = read_table(
                adata_dict, table_name=table, dtype=column_types, remove_empty_column=remove_empty_column
            )

        joined_table = pd.merge(
            table_dict["visit_occurrence"],
            table_dict["person"],
            left_on="person_id",
            right_on="person_id",
            how="left",
            suffixes=("_visit_occurrence", "_person"),
        )

        if "death" in load_tables:
            column_types = get_column_types(adata_dict, table_name="death")
            table_dict["death"] = read_table(
                adata_dict, table_name="death", dtype=column_types, remove_empty_column=remove_empty_column
            )
            joined_table = pd.merge(
                joined_table,
                table_dict["death"],
                left_on="person_id",
                right_on="person_id",
                how="left",
                suffixes=(None, "_death"),
            )

        if "visit_detail" in load_tables:
            column_types = get_column_types(adata_dict, table_name="visit_detail")
            table_dict["visit_detail"] = read_table(
                adata_dict, table_name="visit_detail", dtype=column_types, remove_empty_column=remove_empty_column
            )
            joined_table = pd.merge(
                joined_table,
                table_dict["visit_detail"],
                left_on="visit_occurrence_id",
                right_on="visit_occurrence_id",
                how="left",
                suffixes=(None, "_visit_detail"),
            )

        if "provider" in load_tables:
            column_types = get_column_types(adata_dict, table_name="provider")
            table_dict["provider"] = read_table(
                adata_dict, table_name="provider", dtype=column_types, remove_empty_column=remove_empty_column
            )
            joined_table = pd.merge(
                joined_table,
                table_dict["provider"],
                left_on="provider_id",
                right_on="provider_id",
                how="left",
                suffixes=(None, "_provider"),
            )

        if "location" in load_tables:
            column_types = get_column_types(adata_dict, table_name="location")
            table_dict["location"] = read_table(
                adata_dict, table_name="location", dtype=column_types, remove_empty_column=remove_empty_column
            )
            joined_table = pd.merge(
                joined_table,
                table_dict["location"],
                left_on="location_id",
                right_on="location_id",
                how="left",
                suffixes=(None, "_location"),
            )

        # TODO dask Support
        # joined_table = joined_table.compute()

        # TODO check this earlier
        # joined_table = joined_table.drop_duplicates(subset="visit_occurrence_id")
        joined_table = joined_table.set_index("visit_occurrence_id")
        # obs_only_list = list(self.joined_table.columns)
        # obs_only_list.remove('visit_occurrence_id')
        if remove_empty_column:
            columns = [column for column in joined_table.columns if not joined_table[column].isna().all()]
            joined_table = joined_table.loc[:, columns]
        columns_obs_only = list(set(joined_table.columns))
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
        table_dict = {}
        # for table in ['visit_occurrence', 'person']:
        column_types = get_column_types(adata_dict, table_name="person")
        table_dict[table] = read_table(
            adata_dict, table_name="person", dtype=column_types, remove_empty_column=remove_empty_column
        )

        joined_table = table_dict["person"]

        if "death" in load_tables:
            column_types = get_column_types(adata_dict, table_name="death")
            table_dict["death"] = read_table(
                adata_dict, table_name="death", dtype=column_types, remove_empty_column=remove_empty_column
            )
            joined_table = pd.merge(
                joined_table,
                table_dict["death"],
                left_on="person_id",
                right_on="person_id",
                how="left",
                suffixes=(None, "_death"),
            )

        if "visit_detail" in load_tables:
            column_types = get_column_types(adata_dict, table_name="visit_detail")
            table_dict["visit_detail"] = read_table(
                adata_dict, table_name="visit_detail", dtype=column_types, remove_empty_column=remove_empty_column
            )
            joined_table = pd.merge(
                joined_table,
                table_dict["visit_detail"],
                left_on="visit_occurrence_id",
                right_on="visit_occurrence_id",
                how="left",
                suffixes=(None, "_visit_detail"),
            )

        if "provider" in load_tables:
            column_types = get_column_types(adata_dict, table_name="provider")
            table_dict["provider"] = read_table(
                adata_dict, table_name="provider", dtype=column_types, remove_empty_column=remove_empty_column
            )
            joined_table = pd.merge(
                joined_table,
                table_dict["provider"],
                left_on="provider_id",
                right_on="provider_id",
                how="left",
                suffixes=(None, "_provider"),
            )

        if "location" in load_tables:
            column_types = get_column_types(adata_dict, table_name="location")
            table_dict["location"] = read_table(
                adata_dict, table_name="location", dtype=column_types, remove_empty_column=remove_empty_column
            )
            joined_table = pd.merge(
                joined_table,
                table_dict["location"],
                left_on="location_id",
                right_on="location_id",
                how="left",
                suffixes=(None, "_location"),
            )

        if remove_empty_column:
            columns = [column for column in joined_table.columns if not joined_table[column].isna().all()]
            joined_table = joined_table.loc[:, columns]
        columns_obs_only = list(set(joined_table.columns))
        adata = ep.ad.df_to_anndata(joined_table, index_column="person_id", columns_obs_only=columns_obs_only)

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
            (
                grouped.get_group(visit_occurrence_id)[columns_in_ak_array].to_dict(orient="list")
                if visit_occurrence_id in feature_ids
                else empty_entry
            )
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
