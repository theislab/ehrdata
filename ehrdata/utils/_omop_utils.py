import csv
import glob
import numbers
import os
import warnings
from pathlib import Path
from typing import Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from rich import print as rprint


def get_table_catalog_dict():
    """Get the table catalog dictionary of the OMOP CDM v5.4

    Returns
    -------
        Dictionary: a dictionary of the table catalog. The key is the category of the table, and the value is a list of table names
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


def get_dtype_mapping():
    """Get the data type mapping of the OMOP CDM v5.4

    Returns
    -------
        Dictionary: a dictionary of the data type mapping from OMOP CDM v5.4 to Python
    """
    dtype_mapping = {
        "integer": "Int64",
        "Integer": "Int64",
        "float": float,
        "bigint": "Int64",
        "varchar(MAX)": str,
        "varchar(2000)": str,
        "varchar(1000)": str,
        "varchar(255)": str,
        "varchar(250)": str,
        "varchar(80)": str,
        "varchar(60)": str,
        "varchar(50)": str,
        "varchar(25)": str,
        "varchar(20)": str,
        "varchar(10)": str,
        "varchar(9)": str,
        "varchar(3)": str,
        "varchar(2)": str,
        "varchar(1)": str,
        "datetime": object,
        "date": object,
    }

    return dtype_mapping


def get_omop_cdm_field_level():
    """Get the field level table sof the OMOP CDM v5.4

    Returns
    -------
        Pandas DataFrame
    """
    pth = f"{Path(__file__).resolve().parent}/OMOP_CDMv5.4_Field_Level.csv"
    df = pd.read_csv(pth)
    return df


# TODO also check column data type
def check_with_omop_cdm(folder_path: str, delimiter: str = None, make_filename_lowercase: bool = True) -> dict:
    """Check if the data adheres to the OMOP Common Data Model (CDM) version 5.4 standards

    Check if the table name and column names adhere to the OMOP CDM v5.4

    Args:
        folder_path (str): The path of the folder containing the OMOP data
        delimiter (str, optional): The delimiter of the CSV file. Defaults to None.
        make_filename_lowercase (bool, optional): Whether to make the filename into lowercase. Defaults to True.

    Returns
    -------
        dict: a dictionary of the table path. The key is the table name, and the value is the path of the table
    """
    print("Checking if your data adheres to the OMOP Common Data Model (CDM) version 5.4 standards.")
    filepath_list = glob.glob(os.path.join(folder_path, "*.csv")) + glob.glob(os.path.join(folder_path, "*.parquet"))
    filepath_list = glob.glob(os.path.join(folder_path, "*"))

    filepath_dict = {}
    for path in filepath_list:
        if os.path.isfile(path):
            is_single_file = True
        else:
            is_single_file = False

        # TODO support table stored in a folder
        """
        # If not a single file, only check the first one's column names
        if not os.path.isfile(path):
            folder_walk = os.walk(path)
            first_file_in_folder = next(folder_walk)[2][0]
            file = os.path.join(path, first_file_in_folder)
            is_single_file = False
        """
        if is_single_file:
            # Make filename into lowercase
            if make_filename_lowercase:
                new_path = os.path.join(folder_path, path.split("/")[-1].lower())
                if path != new_path:
                    warnings.warn(f"Rename file [{path}] to [{new_path}]", stacklevel=2)
                    os.rename(path, new_path)
                    path = new_path
            file_name = os.path.basename(path).split(".")[0]
            if not check_csv_has_only_header(path):
                filepath_dict[file_name] = path
        else:
            # path(s) to individual parquet/csv files
            folder_walk = os.walk(path)
            files_in_folder = next(folder_walk)[2]
            if not all(filename.endswith((".csv", ".parquet")) for filename in files_in_folder):
                raise TypeError("Only support CSV and Parquet file!")
            file_name = os.path.basename(path).split("/")[-1]
            filepath_dict[file_name] = np.char.add(f"{path}/", files_in_folder).tolist()
            # If not a single file, take the first one as sample
            path = os.path.join(path, files_in_folder[0])

        if not check_csv_has_only_header(path):
            # check if table name adheres to the OMOP CDM
            field_level = get_omop_cdm_field_level()
            if file_name not in set(field_level.cdmTableName):
                raise KeyError(
                    f"Table [{file_name}] is not defined in OMOP CDM v5.4! Please change the table name manually!"
                )

            # check if column names adhere to the OMOP CDM
            if path.endswith("csv"):
                with open(path) as f:
                    dict_reader = csv.DictReader(f, delimiter=delimiter)
                    columns = dict_reader.fieldnames
                    columns = list(filter(None, columns))
            elif path.endswith("parquet"):
                df = dd.read_parquet(path)
                columns = list(df.columns)
            else:
                raise TypeError("Only support CSV and Parquet file!")

            invalid_column_name = []
            for _, column in enumerate(columns):
                cdm_columns = set(field_level[field_level.cdmTableName == file_name]["cdmFieldName"])
                if column not in cdm_columns:
                    invalid_column_name.append(column)
            if len(invalid_column_name) > 0:
                print(
                    f"Column {invalid_column_name} is not defined in Table [{file_name}] in OMOP CDM v5.4! Please change the column name manually!\nFor more information, please refer to: https://ohdsi.github.io/CommonDataModel/cdm54.html#{file_name.upper()}"
                )
                raise KeyError

    return filepath_dict


def check_csv_has_only_header(file_path: str) -> bool:
    """Check if the CSV file has only header

    Args:
        file_path (str): The path of the CSV file

    Returns
    -------
        bool: True if the CSV file has only header, False otherwise
    """
    if file_path.endswith("csv"):
        with open(file_path) as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if header is not None:
                second_row = next(reader, None)
                return second_row is None
            else:
                return False
    else:
        return False


def get_column_types(adata_dict: dict, table_name: str) -> dict:
    """Get the column types of the table

    Args:
        adata_dict (dict): a dictionary containing filepath_dict and delimiter information
        table_name (str): Table name in OMOP CDM v5.4.

    Returns
    -------
        dict: a dictionary of the column types. The key is the column name, and the value is the column type
    """
    path = adata_dict["filepath_dict"][table_name]
    column_types = {}
    # If not a single file, read the first one
    if isinstance(path, list):
        path = path[0]
    if path.endswith("csv"):
        with open(path) as f:
            dict_reader = csv.DictReader(f, delimiter=adata_dict["delimiter"])
            columns = dict_reader.fieldnames
            columns = list(filter(None, columns))
    elif path.endswith("parquet"):
        df = dd.read_parquet(path)
        columns = list(df.columns)
    else:
        raise TypeError("Only support CSV and Parquet file!")
    columns_lowercase = [column.lower() for column in columns]
    for _, column in enumerate(columns_lowercase):
        dtype_mapping = get_dtype_mapping()
        field_level = get_omop_cdm_field_level()
        column_types[column] = dtype_mapping[
            field_level[(field_level.cdmTableName == table_name) & (field_level.cdmFieldName == column)][
                "cdmDatatype"
            ].values[0]
        ]
    return column_types


def get_primary_key(table_name: str) -> str:
    """Get the primary key of the table

    Args:
        table_name (str, optional): Table name in OMOP CDM v5.4.

    Returns
    -------
        str: the primary key of the table
    """
    field_level = get_omop_cdm_field_level()
    primary_key = field_level[(field_level.cdmTableName == table_name) & (field_level.isPrimaryKey == "Yes")][
        "cdmFieldName"
    ].values[0]
    return primary_key


def read_table(
    adata_dict: dict,
    table_name: str,
    dtype: dict = None,
    parse_dates: Union[list[str], str] = None,
    index: str = None,
    usecols: Union[list[str], str] = None,
    remove_empty_column: bool = True,
    use_dask: bool = None,
) -> Union[pd.DataFrame, dd.DataFrame]:
    """Read the table either in CSV or Parquet format using pandas or dask

    Args:
        adata_dict (dict): a dictionary containing filepath_dict, delimiter, use_dask, tables information
        table_name (str, optional): Table name in OMOP CDM v5.4.
        dtype (dict, optional): Data type of the columns. Defaults to None.
        parse_dates (Union[List[str], str], optional): Columns to parse as dates. Defaults to None.
        index (str, optional): set the index of the DataFrame. Defaults to None.
        usecols (Union[List[str], str], optional): Columns to read. Defaults to None.
        use_dask (bool, optional): Whether to use dask. It is recommended to use dask when the table is large. Defaults to None.

    Returns
    -------
        Union[pd.DataFrame, dd.DataFrame]: a pandas or dask DataFrame
    """
    path = adata_dict["filepath_dict"][table_name]
    if isinstance(path, list):
        if not use_dask or use_dask is None:
            use_dask = True
            warnings.warn(
                f"Multiple files detected for table [{table_name}]. Using dask to read the table.", stacklevel=2
            )
    if not use_dask:
        use_dask = adata_dict["use_dask"]
    if use_dask:
        if isinstance(path, list):
            filetype = path[0].split(".")[-1]
        else:
            filetype = path.split(".")[-1]
        if filetype == "csv":
            if usecols:
                dtype = {key: dtype[key] for key in usecols if key in dtype}
                if parse_dates:
                    parse_dates = {key: parse_dates[key] for key in usecols if key in parse_dates}
            df = dd.read_csv(
                path, delimiter=adata_dict["delimiter"], dtype=dtype, parse_dates=parse_dates, usecols=usecols
            )
        elif filetype == "parquet":
            if usecols:
                dtype = {key: dtype[key] for key in usecols if key in dtype}
                if parse_dates:
                    parse_dates = {key: parse_dates[key] for key in usecols if key in parse_dates}
            df = dd.read_parquet(path, dtype=dtype, parse_dates=parse_dates, columns=usecols)
        else:
            raise TypeError("Only support CSV and Parquet file!")
    else:
        if not os.path.isfile(path):
            raise TypeError("Only support reading a single file!")
        filetype = path.split(".")[-1]
        if filetype == "csv":
            if usecols:
                dtype = {key: dtype[key] for key in usecols if key in dtype}
                if parse_dates:
                    parse_dates = {key: parse_dates[key] for key in usecols if key in parse_dates}
            # TODO dtype and parse_dates has been disabled
            df = pd.read_csv(
                path, delimiter=adata_dict["delimiter"], usecols=usecols
            )  # dtype=dtype, parse_dates=parse_dates,
        elif filetype == "parquet":
            df = pd.read_parquet(path, columns=usecols)

        else:
            raise TypeError("Only support CSV and Parquet file!")
    if remove_empty_column and not use_dask:
        # TODO dask Support
        # columns = [column for column in df.columns if not df[column].compute().isna().all()]
        columns = [column for column in df.columns if not df[column].isna().all()]
        df = df.loc[:, columns]

    if index:
        df = df.set_index(index)
    return df


def map_concept_id(
    adata_dict: dict, concept_id: Union[str, list[int]], verbose: bool = True
) -> tuple[list[int], list[int]]:
    """Map between concept_id_1 and concept_id_2 using concept_relationship table

    Args:
        adata_dict (dict): a dictionary containing filepath_dict, delimiter, tables information.
        concept_id (Union[str, list[int]]): It could be a single concept_id or a list of concept_id.
        verbose (bool, optional): Defaults to True.

    Returns
    -------
        Tuple[list[int], list[int]]: a tuple of list of concept_id_1 and list of concept_id_2. If no map is found, the concept_id_1 and concept_id_2 will be the same.
    """
    filepath_dict = adata_dict["filepath_dict"]
    tables = adata_dict["tables"]
    delimiter = adata_dict["delimiter"]

    if isinstance(concept_id, numbers.Integral):
        concept_id = [concept_id]
    concept_id_1 = []
    concept_id_2 = []
    concept_id_mapped_not_found = []

    if "concept_relationship" in tables:
        column_types = get_column_types(adata_dict, table_name="concept_relationship")
        df_concept_relationship = pd.read_csv(
            filepath_dict["concept_relationship"], dtype=column_types, delimiter=delimiter
        )
        # TODO dask Support
        # df_concept_relationship.compute().dropna(subset=["concept_id_1", "concept_id_2", "relationship_id"], inplace=True)  # , usecols=vocabularies_tables_columns["concept_relationship"],
        df_concept_relationship.dropna(
            subset=["concept_id_1", "concept_id_2", "relationship_id"], inplace=True
        )  # , usecols=vocabularies_tables_columns["concept_relationship"],
        concept_relationship_dict = df_to_dict(
            df=df_concept_relationship[df_concept_relationship["relationship_id"] == "Maps to"],
            key="concept_id_1",
            value="concept_id_2",
        )
        concept_relationship_dict_reverse = df_to_dict(
            df=df_concept_relationship[df_concept_relationship["relationship_id"] == "Mapped from"],
            key="concept_id_1",
            value="concept_id_2",
        )
        for id in concept_id:
            try:
                concept_id_2.append(concept_relationship_dict[id])
                concept_id_1.append(id)
            except KeyError:
                try:
                    concept_id_1.append(concept_relationship_dict_reverse[id])
                    concept_id_2.append(id)
                except KeyError:
                    concept_id_1.append(id)
                    concept_id_2.append(id)
                    concept_id_mapped_not_found.append(id)
        if len(concept_id_mapped_not_found) > 0:
            # warnings.warn(f"Couldn't find a map for concept {id} in concept_relationship table!")
            if verbose:
                rprint(f"Couldn't find a map for concept {concept_id_mapped_not_found} in concept_relationship table!")
    else:
        concept_id_1 = concept_id
        concept_id_2 = concept_id

    if len(concept_id_1) == 1:
        return concept_id_1[0], concept_id_2[0]
    else:
        return concept_id_1, concept_id_2


def df_to_dict(df: pd.DataFrame, key: str, value: str) -> dict:
    """Convert a DataFrame to a dictionary

    Args:
        df (pd.DataFrame): a DataFrame
        key (str): the column name to be used as the key of the dictionary
        value (str): the column name to be used as the value of the dictionary

    Returns
    -------
        dict: a dictionary
    """
    if isinstance(df, dd.DataFrame):
        return pd.Series(df[value].compute().values, index=df[key].compute()).to_dict()
    else:
        return pd.Series(df[value].values, index=df[key]).to_dict()


# def get_close_matches_using_dict(word, possibilities, n=2, cutoff=0.6):
#     """Use SequenceMatcher to return a list of the indexes of the best
#     "good enough" matches. word is a sequence for which close matches
#     are desired (typically a string).
#     possibilities is a dictionary of sequences.
#     Optional arg n (default 2) is the maximum number of close matches to
#     return.  n must be > 0.
#     Optional arg cutoff (default 0.6) is a float in [0, 1].  Possibilities
#     that don't score at least that similar to word are ignored.
#     """
#     if not n > 0:
#         raise ValueError("n must be > 0: %r" % (n,))
#     if not 0.0 <= cutoff <= 1.0:
#         raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))
#     result = []
#     s = SequenceMatcher()
#     s.set_seq2(word)
#     for _, (key, value) in enumerate(possibilities.items()):
#         s.set_seq1(value)
#         if s.real_quick_ratio() >= cutoff and s.quick_ratio() >= cutoff and s.ratio() >= cutoff:
#             result.append((s.ratio(), value, key))

#     # Move the best scorers to head of list
#     result = _nlargest(n, result)

#     # Strip scores for the best n matches
#     return [(value, key, score) for score, value, key in result]


def get_feature_info(
    adata_dict: dict,
    features: Union[str, int, list[Union[str, int]]] = None,
    ignore_not_shown_in_concept_table: bool = True,
    exact_match: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Get the feature information from the concept table

    Args:
        adata_dict (dict): a dictionary containing filepath_dict, delimiter, tables information.
        features (Union[str, int, list[Union[str, int]]], optional): a feature name or a feature id. Defaults to None.
        ignore_not_shown_in_concept_table (bool, optional): If True, it will ignore the features that are not shown in the concept table. Defaults to True.
        exact_match (bool, optional): If True, it will only return the exact match if the feature name is input. Defaults to True.
        verbose (bool, optional): Defaults to True.

    Returns
    -------
        pd.DataFrame: a DataFrame containing the feature information
    """
    if "concept" in adata_dict["tables"]:
        column_types = get_column_types(adata_dict, table_name="concept")

        df_concept = read_table(adata_dict, table_name="concept", dtype=column_types).dropna(
            subset=["concept_id", "concept_name"]
        )  # usecols=vocabularies_tables_columns["concept"],
        # concept_dict = df_to_dict(df=df_concept, key="concept_name", value="concept_id")
    else:
        rprint("concept table is not found in the OMOP CDM v5.4!")
        raise ValueError
    fetures_not_shown_in_concept_table = []

    info_df = pd.DataFrame([])
    if isinstance(features, str):
        features = [features]
    # Get feature id for each input, and check if each feature occurs in the concept table
    for feature in features:
        # if the input is feature ID
        if isinstance(feature, numbers.Integral):
            feature_id = feature
            feature_id_1, feature_id_2 = map_concept_id(adata_dict=adata_dict, concept_id=feature_id, verbose=False)
            if feature_id_1 in df_concept["concept_id"].values:
                feature_name = df_concept[df_concept["concept_id"] == feature_id_1]["concept_name"].values[0]
            else:
                if ignore_not_shown_in_concept_table:
                    fetures_not_shown_in_concept_table.append(feature)
                    feature_name = feature_id_1
                else:
                    rprint(f"Feature ID - [red]{feature_id_1}[/] could not be found in concept table")
                    raise ValueError
            match_1_ratio = 100

        # if the input is feature name
        elif isinstance(feature, str):
            # return a list of (value, key, score)
            # result = get_close_matches_using_dict(feature, concept_dict, n=2, cutoff=0.2)
            from thefuzz import process

            # the thefuzz match returns a list of tuples of (matched string, match ratio)
            result = process.extract(feature, list(df_concept["concept_name"].values), limit=2)

            match_1 = result[0]
            match_1_name = match_1[0]
            match_1_ratio = match_1[1]
            # Most of the case: if find 2 best matches
            if len(result) == 2:
                match_2 = result[1]
                match_2_name = match_2[0]
                match_2_ratio = match_2[1]

                if match_1_ratio != 100:
                    if exact_match:
                        rprint(
                            f"Unable to find an exact match for [blue]{feature}[/] in the concept table.\nSimilar ones: 1) [blue]{match_1_name}[/] with match ratio [red]{match_1_ratio}[/] 2) [blue]{match_2_name}[/] with match ratio [red]{match_2_ratio}[/]"
                        )
                        raise ValueError
                else:
                    if match_2_ratio == 100:
                        match_1_id = df_concept[df_concept["concept_name"] == match_1_name]["concept_id"].values[0]
                        match_2_id = df_concept[df_concept["concept_name"] == match_2_name]["concept_id"].values[0]
                        rprint(
                            f"Found multiple exact matches for [blue]{feature}[/] in the concept table.\n1) concept id: [blue]{match_1_id}[/] 2) concept id: [blue]{match_2_id}[/]. Please specify concept_id directly."
                        )
                        raise ValueError

            # Very rare: if only find 1 match
            else:
                if exact_match and match_1_ratio != 1:
                    rprint(
                        f"Unable to find an exact match for [red]{feature}[/] in the concept table. Similiar one: [blue]{match_1_name}[/] with match ratio [red]{match_1_ratio}[/]"
                    )
                    raise ValueError

            feature_name = match_1_name
            feature_id = df_concept[df_concept["concept_name"] == feature_name]["concept_id"].values[0]
            feature_id_1, feature_id_2 = map_concept_id(adata_dict=adata_dict, concept_id=feature_id, verbose=False)

        else:
            rprint(
                "Please input either [red]feature name (string)[/] or [red]feature id (integer)[/] that you want to extarct"
            )
            raise TypeError

        info_df = pd.concat(
            [
                info_df,
                pd.DataFrame(
                    data=[[feature_name, feature_id_1, feature_id_2]],
                    columns=["feature_name", "feature_id_1", "feature_id_2"],
                ),
            ]
        )

        # feature_name_list.append(feature_name)
        # domain_id_list.append(df_concept.loc[df_concept["concept_id"] == feature_id, "domain_id"].reset_index(drop=True).compute()[0])
        # concept_class_id_list.append(df_concept.loc[df_concept["concept_id"] == feature_id, "concept_class_id"].reset_index(drop=True).compute()[0])
        # concept_code_list.append(df_concept.loc[df_concept["concept_id"] == feature_id, "concept_code"].reset_index(drop=True).compute()[0])

        if verbose:
            rprint(
                f"Detected: feature [green]{feature_name}[/], feature ID [green]{feature_id}[/] in concept table, match ratio = [green]{match_1_ratio}."
            )

    if info_df["feature_id_1"].equals(info_df["feature_id_2"]):
        info_df.drop("feature_id_2", axis=1, inplace=True)
        info_df = info_df.rename(columns={"feature_id_1": "feature_id"})
        info_df = info_df.reset_index(drop=True)
    else:
        info_df = info_df.reset_index(drop=True)
    return info_df
