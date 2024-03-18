import os
from typing import Literal, Optional, Union

import awkward as ak
import pandas as pd
import pyarrow as pa
from anndata import AnnData
from ehrapy.anndata import df_to_anndata
from rich import print as rprint

from ehrdata.utils._omop_utils import (
    check_with_omop_cdm,
    get_column_types,
    get_feature_info,
    get_table_catalog_dict,
    read_table,
)


def init_omop(
    folder_path: str,
    delimiter: str = ",",
    make_filename_lowercase: bool = True,
    use_dask: bool = False,
    level: Literal["stay_level", "patient_level"] = "stay_level",
    load_tables: Optional[Union[str, list[str], tuple[str], Literal["auto"]]] = None,
    remove_empty_column: bool = True,
) -> AnnData:
    """Initialize an OMOP database, load tables and create anndata object

    Args:
        folder_path (str): Path to the folder containing the OMOP CDM tables.
        delimiter (str, optional): If data is in csv format, delimiter can be specified. Defaults to ','.
        make_filename_lowercase (bool, optional): If True, the filename will be converted to lowercase. Defaults to True.
        use_dask (bool, optional): If True, dask will be used to read the tables. For large tables, it is recommended to use dask. Defaults to False.
        level (Literal[&quot;stay_level&quot;, &quot;patient_level&quot;], optional): For stay level, each row in anndata would be a visit_occurrence. For patient level, each row in anndata would be a patient. Defaults to "stay_level".
        load_tables (Optional[Union[str, list[str], tuple[str], Literal[&quot;auto&quot;]]], optional): Basic tables to load. Support loading one of those: ["visit_occurrence", "person", "death", "visit_detail", "provider"]. If is None, it will try to load all non-empty supported tables.
        remove_empty_column (bool, optional): If True, columns with all missing values will be removed when loading tables. Defaults to True.

    Returns
    -------
        AnnData: Anndata object
    """
    if delimiter is None:
        delimiter = ","
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
        load_tables = {"visit_occurrence", "person", "death", "visit_detail", "provider"}.intersection(set(tables))
    elif isinstance(load_tables, str):
        load_tables = [load_tables]
    else:
        pass
    not_supported_load_tables = set(load_tables) - {"visit_occurrence", "person", "death", "visit_detail", "provider"}
    if len(not_supported_load_tables) > 0:
        raise ValueError(
            f"Loading {not_supported_load_tables} not supported. Only loading visit_occurrence, person, death, visit_detail, provider tables are supported"
        )

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
        adata = df_to_anndata(joined_table, index_column="visit_occurrence_id", columns_obs_only=columns_obs_only)
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

        if remove_empty_column:
            columns = [column for column in joined_table.columns if not joined_table[column].isna().all()]
            joined_table = joined_table.loc[:, columns]
        columns_obs_only = list(set(joined_table.columns))
        adata = df_to_anndata(joined_table, index_column="person_id", columns_obs_only=columns_obs_only)

    else:
        raise ValueError("level should be 'stay_level' or 'patient_level'")
    print(f"Loading tables: {load_tables}")
    return adata


def extract_features(
    adata: AnnData,
    source: Literal[
        "observation",
        "measurement",
        "procedure_occurrence",
        "specimen",
        "device_exposure",
        "drug_exposure",
        "condition_occurrence",
    ],
    features: Union[str, int, list[Union[str, int]]] = None,
    source_table_columns: Union[str, list[str]] = None,
    dropna: Optional[bool] = True,
    verbose: Optional[bool] = True,
    use_dask: bool = None,
) -> AnnData:
    """Extract features from OMOP CDM tables and add them to .obsm of anndata object

    Args:
        adata (AnnData): Anndata object
        source (Literal[ &quot;observation&quot;, &quot;measurement&quot;, &quot;procedure_occurrence&quot;, &quot;specimen&quot;, &quot;device_exposure&quot;, &quot;drug_exposure&quot;, &quot;condition_occurrence&quot;, ]): source table name. Defaults to None.
        features (Union[str, int, list[Union[str, int]]], optional): feature_id or feature_name, or list of feature_id or feature_name. Defaults to None.
        source_table_columns (Union[str, list[str]], optional): columns to be extracted from source table. If None, all columns will be extracted. Defaults to None.
        dropna (Optional[bool], optional): drop rows with missing values. Defaults to True.
        verbose (Optional[bool], optional): print progress. Defaults to True.
        use_dask (bool, optional): If True, dask will be used to read the tables. For large tables, it is highly recommended to use dask. If None, it will be set to adata.uns[&quot;use_dask&quot;]. Defaults to None.

    Returns
    -------
        AnnData: Anndata object
    """
    if source in ["measurement", "observation", "specimen"]:
        key = f"{source}_concept_id"
    elif source in ["device_exposure", "procedure_occurrence", "drug_exposure", "condition_occurrence"]:
        key = f"{source.split('_')[0]}_concept_id"
    else:
        raise KeyError(f"Extracting data from {source} is not supported yet")

    if source_table_columns is None:
        if source == "measurement":
            source_table_columns = ["visit_occurrence_id", "measurement_datetime", "value_as_number", key]
        elif source == "observation":
            source_table_columns = [
                "visit_occurrence_id",
                "value_as_number",
                "value_as_string",
                "observation_datetime",
                key,
            ]
        elif source == "condition_occurrence":
            source_table_columns = None
        else:
            raise KeyError(f"Extracting data from {source} is not supported yet")
    if use_dask is None:
        use_dask = use_dask = adata.uns["use_dask"]
    # TODO load using Dask or Dask-Awkward
    # Load source table using dask
    source_column_types = get_column_types(adata.uns, table_name=source)
    df_source = read_table(
        adata.uns, table_name=source, dtype=source_column_types, usecols=source_table_columns, use_dask=use_dask
    )
    info_df = get_feature_info(adata.uns, features=features, verbose=False)
    info_dict = info_df[["feature_id_2", "feature_name"]].set_index("feature_id_2").to_dict()["feature_name"]

    # Select featrues
    df_source = df_source[df_source[key].isin(list(info_df.feature_id_2))]
    # TODO select time period
    # df_source = df_source[(df_source.time >= 0) & (df_source.time <= 48*60*60)]
    # da_measurement['measurement_name'] = da_measurement.measurement_concept_id.replace(info_dict)

    # TODO dask caching
    """
    from dask.cache import Cache
    cache = Cache(2e9)
    cache.register()
    """
    print(f"Reading {source} table")
    if use_dask:
        if dropna:
            df_source = df_source.compute().dropna()
        else:
            df_source = df_source.compute()
    else:
        if dropna:
            df_source = df_source.dropna()

    # Filter data once, if possible
    filtered_data = {feature_id: df_source[df_source[key] == feature_id] for feature_id in info_dict.keys()}

    for feature_id, feature_name in info_dict.items():
        if verbose:
            print(f"Adding feature [{feature_name}] from source table")
        adata = from_dataframe(adata, feature_name, filtered_data[feature_id])

    return adata


def extract_note(
    adata: AnnData,
    use_dask: bool = None,
    columns: Optional[list[str]] = None,
) -> AnnData:
    """Extract note from OMOP CDM Note table and add them to .obsm of anndata object

    Args:
        adata (AnnData): Anndata object
        use_dask (bool, optional): If True, dask will be used to read the tables. For large tables, it is recommended to use dask. If None, it will be set to adata.uns[&quot;use_dask&quot;]. Defaults to None.
        columns (Optional[list[str]], optional): columns to be extracted from note table. If None, all columns will be extracted. Defaults to None.

    Returns
    -------
        AnnData: Anndata object
    """
    if use_dask is None:
        use_dask = use_dask = adata.uns["use_dask"]
    source_column_types = get_column_types(adata.uns, table_name="note")
    df_source = read_table(adata.uns, table_name="note", dtype=source_column_types, use_dask=use_dask)
    if columns is None:
        columns = df_source.columns
    # TODO improve this using from_dataframe
    obs_dict = [
        {
            column: list(df_source[df_source["visit_occurrence_id"] == int(visit_occurrence_id)][column])
            for column in columns
        }
        for visit_occurrence_id in adata.obs.index
    ]
    adata.obsm["note"] = ak.Array(obs_dict)
    return adata


def from_dataframe(adata: AnnData, feature: str, df: pd.DataFrame) -> AnnData:
    """Add data from a dataframe to .obsm of anndata object

    Args:
        adata (AnnData): Anndata object
        feature (str): feature name. It will be used as the key in .obsm
        df (pd.DataFrame): dataframe containing the data. It should have a column named 'visit_occurrence_id'

    Returns
    -------
        AnnData: Anndata object with an awkward array in obsm[feature]
    """
    # Add new rows for those visit_occurrence_id that don't have any data
    new_row_dict = {col: [] for col in df.columns}
    new_row_dict["visit_occurrence_id"] = list(set(adata.obs.index) - set(df.visit_occurrence_id.unique()))
    for key in new_row_dict.keys():
        if key != "visit_occurrence_id":
            new_row_dict[key] = [None] * len(new_row_dict["visit_occurrence_id"])
    new_rows = pd.DataFrame(new_row_dict)
    df = pd.concat([df, new_rows], ignore_index=True)
    df["visit_occurrence_id"] = df["visit_occurrence_id"].astype(int)
    ak_array = ak.from_arrow(pa.Table.from_pandas(df), highlevel=True)
    ak_array = ak.unflatten(ak_array, df["visit_occurrence_id"].value_counts(sort=False).values)

    # Need to sort the visit_occurrence_id in awkward array accoring to the sequence in the indices in the adata
    id_in_df = list(df["visit_occurrence_id"].unique())
    id_in_adata = list(adata.obs.index.astype(int))
    index_dict = {value: index for index, value in enumerate(id_in_df)}
    index = [index_dict[x] for x in id_in_adata]

    # Sort the ak_array to align with the adata
    ak_array = ak_array[index]
    columns_in_ak_array = list(set(df.columns) - {"visit_occurrence_id"})
    adata.obsm[feature] = ak_array[columns_in_ak_array]

    return adata


def to_dataframe(
    adata: AnnData,
    features: Union[str, list[str]],
    visit_occurrence_id: Optional[Union[str, list[str]]] = None,
) -> pd.DataFrame:
    """Convert data in .obsm of anndata object to dataframe

    Args:
        adata (AnnData): Anndata object with data in .obsm
        features (Union[str, list[str]]): feature name or list of feature names
        visit_occurrence_id (Optional[Union[str, list[str]]], optional): visit_occurrence_id or list of visit_occurrence_id. If None, all visit_occurrence_id will be selected. Defaults to None.

    Returns
    -------
        pd.DataFrame: dataframe containing the data
    """
    # TODO
    # can be viewed as patient level - only select some patient
    # TODO change variable name here
    if isinstance(features, str):
        features = [features]
    df_concat = pd.DataFrame([])
    for feature in features:
        if visit_occurrence_id is not None:
            if isinstance(visit_occurrence_id, str):
                visit_occurrence_id = [visit_occurrence_id]
            index_list = adata.obs.index.to_list()
            ids = [index_list.index(id) for id in visit_occurrence_id]
            df = ak.to_dataframe(adata.obsm[feature][ids])
            #
            df.reset_index(drop=False, inplace=True)
            if len(visit_occurrence_id) == 1:
                df["visit_occurrence_id"] = visit_occurrence_id[0]
                del df["entry"]
                del df["subentry"]
            else:
                mapping = dict(zip(range(len(visit_occurrence_id)), visit_occurrence_id))
                df["visit_occurrence_id"] = df["entry"].map(mapping)
                del df["subentry"]
                del df["entry"]
        else:
            df = ak.to_dataframe(adata.obsm[feature])

            df.reset_index(drop=False, inplace=True)
            df["entry"] = adata.obs.index[df["entry"]]
            df = df.rename(columns={"entry": "visit_occurrence_id"})
            del df["subentry"]

        """
        for col in df.columns:
            if col.endswith("time"):
                df[col] = pd.to_datetime(df[col])
        """
        df["feature_name"] = feature
        df_concat = pd.concat([df_concat, df], axis=0)

    return df_concat
