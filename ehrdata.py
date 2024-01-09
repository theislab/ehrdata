import awkward as ak
import numpy as np
import pandas as pd
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ehrapy as ep
import scanpy as sc
from anndata import AnnData
import mudata as md
from mudata import MuData
from typing import List, Union, Literal
import os
import glob
import dask.dataframe as dd
from thefuzz import process
import sys
from rich import print as rprint
import missingno as msno
import warnings
import numbers
import os


clinical_tables_columns = {
    "person": ["person_id", "year_of_birth", "gender_source_value"],
    "observation_period": [],
    "death": ["person_id", "death_datetime"],
    "visit_occurrence": ["visit_occurrence_id", "person_id", "visit_start_datetime", "visit_end_datetime"],
    "visit_detail": [],
    "condition_occurrence": [],
    "drug_exposure": [
        "drug_exposure_id",
        "person_id",
        "visit_occurrence_id",
        "drug_concept_id",
    ],
    "procedure_occurrence": ["visit_occurrence_id", "person_id", "visit_start_datetime", "visit_end_datetime"],
    "device_exposure": [],
    "specimen": [],
    "measurement": [
        "measurement_id",
        "person_id",
        "visit_occurrence_id",
        "measurement_concept_id",
        "measurement_datetime",
        "value_as_number",
        "unit_source_value",
    ],
    "observation": [
        "observation_id",
        "person_id",
        "observation_concept_id",
        "observation_datetime",
        "value_as_number",
        "value_as_string",
    ],
    "note": [],
    "note_nlp": [],
    "fact_relationship": [],
    "procedure_occurrence": [],
}

health_system_tables_columns = {
    "location": [],
    "care_site": ["care_site_id", "care_site_name"],
    "provider": [],
}
vocabularies_tables_columns = {
    "concept": [
        "concept_id",
        "concept_name",
        "domain_id",
        "vocabulary_id",
        "concept_class_id",
        "standard_concept",
        "concept_code",
    ],
    "vocabulary": [],
    "domain": [],
    "concept_class": [],
    "concept_synonym": [],
    "concept_relationship": ["concept_id_1", "concept_id_2", "relationship_id"],
    "relationship": [],
    "concept_ancestor": [],
    "source_to_concept_map": [],
    "drug_strength": [],
}


from difflib import SequenceMatcher
from heapq import nlargest as _nlargest


def get_close_matches_using_dict(word, possibilities, n=2, cutoff=0.6):
    """Use SequenceMatcher to return a list of the indexes of the best
    "good enough" matches. word is a sequence for which close matches
    are desired (typically a string).
    possibilities is a dictionary of sequences.
    Optional arg n (default 2) is the maximum number of close matches to
    return.  n must be > 0.
    Optional arg cutoff (default 0.6) is a float in [0, 1].  Possibilities
    that don't score at least that similar to word are ignored.
    """

    if not n > 0:
        raise ValueError("n must be > 0: %r" % (n,))
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))
    result = []
    s = SequenceMatcher()
    s.set_seq2(word)
    for _, (key, value) in enumerate(possibilities.items()):
        s.set_seq1(value)
        if s.real_quick_ratio() >= cutoff and s.quick_ratio() >= cutoff and s.ratio() >= cutoff:
            result.append((s.ratio(), value, key))

    # Move the best scorers to head of list
    result = _nlargest(n, result)

    # Strip scores for the best n matches
    return [(value, key, score) for score, value, key in result]


def df_to_dict(df, key, value):
    if isinstance(df, dd.DataFrame):
        return pd.Series(df[value].compute().values, index=df[key].compute()).to_dict()
    else:
        return pd.Series(df[value].values, index=df[key]).to_dict()


def check_csv_has_only_header(file_path):
    if file_path.endswith('csv'):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            header = next(reader, None)  # Read the header
            if header is not None:
                second_row = next(reader, None)  # Try to read the next row
                return second_row is None  # If there's no second row, return True
            else:
                return False  # File is empty or not a valid CSV
    else:
        return False




class OMOP:
    def __init__(self, folder_path, delimiter=None):
        self.base = folder_path
        # TODO support also parquet and other formats
        file_list = glob.glob(os.path.join(folder_path, "*.csv")) + glob.glob(os.path.join(folder_path, "*.parquet"))
        
        self.loaded_tabel = None
        self.filepath = {}
        for file_path in file_list:
            file_name = file_path.split("/")[-1].split(".")[0]
            
            
            new_filepath = os.path.join(self.base, file_path.split("/")[-1].lower())
            if check_csv_has_only_header(file_path):
                pass
            else:
                # Rename the file
                os.rename(file_path, new_filepath)
                self.filepath[file_name] = new_filepath

        self.tables = list(self.filepath.keys())
        self.delimiter = delimiter
        """
        if "concept" in self.tables:
            df_concept = dd.read_csv(self.filepath["concept"], usecols=vocabularies_tables_columns["concept"])
            self.concept_id_to_name = dict(zip(df_concept['id'], df_concept['name']))
            self.concept_name_to_id = dict(zip(df_concept['name'], df_concept['id']))
        """

    def __repr__(self) -> str:
        # TODO this should be seperated by diff table categories
        def format_tables(tables, max_line_length=80):
            line = ""
            for table in tables:
                # Check if adding the next table would exceed the max line length
                if len(line) + len(table) > max_line_length:
                    # Yield the current line and start a new one
                    yield line
                    line = table
                else:
                    # Add the table to the current line
                    line += table if line == "" else ", " + table
            # Yield the last line
            yield line

        tables_str = "\n".join(format_tables(self.tables))
        return f'OMOP object ({os.path.basename(self.base)}) with {len(self.tables)} tables.\nTables: {tables_str}'

    def set_path(self, table_name, file_path):
        # TODO move to init
        self.tables.append(table_name)
        self.filepath[table_name] = file_path
            
    def _get_column_types(self, path=None, columns=None):
        column_types = {}
        parse_dates = []
        
        # If not a single file, read the first one
        if not os.path.isfile(path):
            folder_walk = os.walk(path)
            first_file_in_folder = next(folder_walk)[2][0]
            path = os.path.join(path, first_file_in_folder)
            
        if path.endswith('csv'):
            with open(path, "r") as f:
                dict_reader = csv.DictReader(f, delimiter=self.delimiter)
                columns = dict_reader.fieldnames
                columns = list(filter(None, columns))         
        elif path.endswith('parquet'):
            df = dd.read_parquet(path)
            columns = list(df.columns)
        else:
            raise TypeError("Only support CSV and Parquet file!")
        columns_lowercase = [column.lower() for column in columns]
        for i, column in enumerate(columns_lowercase):
            if hasattr(self, "additional_column"):
                if column in self.additional_column.keys():
                    column_types[columns[i]] = self.additional_column[column]
            
            elif column.endswith(
                (
                    "source_value",
                    "reason",
                    "measurement_time",
                    "as_string",
                    "title",
                    "text",
                    "name",
                    "concept",
                    "code",
                    "domain_id",
                    "vocabulary_id",
                    "concept_class_id",
                    "relationship_id",
                    "specimen_source_id",
                    "production_id",
                    "unique_device_id",
                    "sig",
                    "lot_number",
                )
            ):
                column_types[columns[i]] = str
            # TODO quantity in different tables have different types
            elif column.endswith(("as_number", "low", "high", "quantity")):
                column_types[columns[i]] = float
            elif column.endswith("date"):
                parse_dates.append(columns[i])
            elif column.endswith("datetime"):
                parse_dates.append(columns[i])
            elif column.endswith(("id", "birth", "id_1", "id_2", "refills", "days_supply")):
                column_types[columns[i]] = "Int64"
            else:
                raise KeyError(f"{columns[i]} is not defined in OMOP CDM")
        if len(parse_dates) == 0:
            parse_dates = None
        return column_types, parse_dates
    
    def _read_table(self, path, dtype=None, parse_dates=None, index=None, usecols=None, **kwargs):
    
        if not os.path.isfile(path):
            folder_walk = os.walk(path)
            filetype = next(folder_walk)[2][0].split(".")[-1]
        else:
            filetype = path.split(".")[-1]
        if filetype == 'csv':
            if not os.path.isfile(path):
                path = f"{path}/*.csv"
            if usecols:
                if parse_dates:
                    parse_dates = {key: parse_dates[key] for key in usecols if key in parse_dates}
                if usecols:
                    dtype = {key: dtype[key] for key in usecols if key in dtype}
            df = dd.read_csv(path, delimiter=self.delimiter, dtype=dtype, parse_dates=parse_dates, usecols=usecols)
        elif filetype == 'parquet':
            if not os.path.isfile(path):
                path = f"{path}/*.parquet"
            df = dd.read_parquet(path, dtype=dtype, parse_dates=parse_dates)
        else:
            raise TypeError("Only support CSV and Parquet file!")

        if index:
            df = df.set_index(index)
        return df
    
    @property
    def clinical_tables(self):
        """
        A dictionary containing all of the ``Clinical`` OMOP CDM tables in the connected database.
        """
        table_names = [
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
        return [table_name for table_name in self.tables if table_name in table_names]

    @property
    def vocabularies_tables(self):
        """
        A dictionary containing all of the ``Vocabularies`` OMOP CDM tables in the connected database.
        """
        table_names = [
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
        return [table_name for table_name in self.tables if table_name in table_names]

    @property
    def metadata_tables(self):
        """
        A dictionary containing all of the ``MetaData`` OMOP CDM tables in the connected database.
        """
        table_names = ["cdm_source", "metadata"]
        return [table_name for table_name in self.tables if table_name in table_names]

    @property
    def health_system_tables(self):
        """
        A dictionary containing all of the ``Health System`` OMOP CDM tables in the connected database.
        """
        table_names = ["location", "care_site", "provider"]
        return [table_name for table_name in self.tables if table_name in table_names]

    @property
    def derived_elements_tables(self):
        """
        A dictionary containing all of the ``Derived Elements`` OMOP CDM tables in the connected database.
        """
        table_names = ["cohort", "cohort_definition", "drug_era", "dose_era", "condition_era"]
        return [table_name for table_name in self.tables if table_name in table_names]

    @property
    def health_economics_tables(self):
        """
        A dictionary containing all of the ``Health Economics`` OMOP CDM tables in the connected database.
        """
        table_names = ["payer_plan_period", "cost"]
        return [table_name for table_name in self.tables if table_name in table_names]

    def load(self, level="stay_level", tables=["visit_occurrence", "person", "death"], remove_empty_column=True):
        # TODO patient level and hospital level
        if level == "stay_level":
            index = {"visit_occurrence": "visit_occurrence_id", "person": "person_id", "death": "person_id"}
            # TODO Only support clinical_tables_columns

            for table in tables:
                print(f"reading table [{table}]")
                column_types, parse_dates = self._get_column_types(self.filepath[table])
                df = self._read_table(self.filepath[table], dtype=column_types, parse_dates = parse_dates, index='person_id')
                if remove_empty_column:
                    columns = [column for column in df.columns if not df[column].compute().isna().all()]
                df = df.loc[:, columns]
                setattr(self, table, df)

            # concept_id_list = list(self.concept.concept_id)
            # concept_name_list = list(self.concept.concept_id)
            # concept_domain_id_list = list(set(self.concept.domain_id))

            # self.loaded_tabel = ['visit_occurrence', 'person', 'death', 'measurement', 'observation', 'drug_exposure']
            joined_table = dd.merge(self.visit_occurrence, self.person, left_index=True, right_index=True, how="left")
            joined_table = dd.merge(joined_table, self.death, left_index=True, right_index=True, how="left")

            joined_table = joined_table.compute()
            joined_table = joined_table.set_index("visit_occurrence_id")

            # obs_only_list = list(self.joined_table.columns)
            # obs_only_list.remove('visit_occurrence_id')
            columns_obs_only = list(set(joined_table.columns) - set(["year_of_birth", "gender_source_value"]))
            adata = ep.ad.df_to_anndata(
                joined_table, index_column="visit_occurrence_id", columns_obs_only=columns_obs_only
            )

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

        return adata

    def add_additional_column(self, column_name, type):
        if hasattr(self, "additional_column"):
            self.additional_column[column_name] = type
        else:
            self.additional_column = {column_name: type}
    
    def feature_statistics(
        self,
        source: Literal[
            "observation",
            "measurement",
            "procedure_occurrence",
            "specimen",
            "device_exposure",
            "drug_exposure",
            "condition_occurrence",
        ],
        number=20,
        key = None
    ):  
        column_types, parse_dates = self._get_column_types(self.filepath[source])
        df_source = self._read_table(self.filepath[source], dtype=column_types, parse_dates = parse_dates, usecols=[f"{source}_concept_id"])
        feature_counts = df_source[f"{source}_concept_id"].value_counts().compute()[0:number]
        feature_counts = feature_counts.to_frame().reset_index(drop=False)

        
        feature_counts[f"{source}_concept_id_1"], feature_counts[f"{source}_concept_id_2"] = self.map_concept_id(
            feature_counts[f"{source}_concept_id"], verbose=False
        )
        feature_counts["feature_name"] = self.get_concept_name(feature_counts[f"{source}_concept_id_1"])
        if feature_counts[f"{source}_concept_id_1"].equals(feature_counts[f"{source}_concept_id_2"]):
            feature_counts.drop(f"{source}_concept_id_2", axis=1, inplace=True)
            feature_counts.rename(columns={f"{source}_concept_id_1": f"{source}_concept_id"})
            feature_counts = feature_counts.reindex(columns=["feature_name", f"{source}_concept_id", "count"])
        else:
            feature_counts = feature_counts.reindex(
                columns=["feature_name", f"{source}_concept_id_1", f"{source}_concept_id_2", "count"]
            )

        ax = sns.barplot(feature_counts, x="feature_name", y="count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        return feature_counts

    def map_concept_id(self, concept_id: Union[str, List], verbose=True):
        if isinstance(concept_id, numbers.Integral):
            concept_id = [concept_id]
        concept_id_1 = []
        concept_id_2 = []
        concept_id_mapped_not_found = []
        
        if "concept_relationship" in self.tables:
            column_types, parse_dates = self._get_column_types(self.filepath["concept_relationship"])
            df_concept_relationship = self._read_csv(
                self.filepath["concept_relationship"], dtype=column_types, parse_dates=parse_dates
            )
            df_concept_relationship.compute().dropna(
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

    def get_concept_name(self, concept_id: Union[str, List], raise_error=False, verbose=True):
        if isinstance(concept_id, numbers.Integral):
            concept_id = [concept_id]

        column_types, parse_dates = self._get_column_types(self.filepath["concept"])
        df_concept = self._read_table(self.filepath["concept"], dtype=column_types, parse_dates=parse_dates)
        df_concept.compute().dropna(subset=["concept_id", "concept_name"], inplace=True, ignore_index=True)  # usecols=vocabularies_tables_columns["concept"]
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

    def extract_note(self, adata, source="note"):
        column_types, parse_dates = self._get_column_types(self.filepath[source])
        df_source = dd.read_csv(self.filepath[source], dtype=column_types, parse_dates=parse_dates)
        if columns is None:
            columns = df_source.columns
        obs_dict = [
            {
                column: list(df_source[df_source["visit_occurrence_id"] == int(visit_occurrence_id)][column])
                for column in columns
            }
            for visit_occurrence_id in adata.obs.index
        ]
        adata.obsm["note"] = ak.Array(obs_dict)
        return adata

    def note_nlp_map(
        self,
    ):
        # Got some inspirations from: https://github.com/aws-samples/amazon-comprehend-medical-omop-notes-mapping
        pass

    def extract_features(
        self,
        adata,
        source: Literal[
            "observation",
            "measurement",
            "procedure_occurrence",
            "specimen",
            "device_exposure",
            "drug_exposure",
            "condition_occurrence",
        ],
        features: str or int or List[Union[str, int]] = None,
        key: str = None,
        columns_in_source_table: str or List[str] = None,
        map_concept=True,
        add_aggregation_to_X: bool = True,
        aggregation_methods=None,
        add_all_data: bool = True,
        exact_match: bool = True,
        remove_empty_column: bool = True,
        ignore_not_shown_in_concept_table: bool = True,
        verbose: bool = False,
    ):
        if key is None:
            if source in ["measurement", "observation", "specimen"]:
                key = f"{source}_concept_id"
            elif source in ["device_exposure", "procedure_occurrence", "drug_exposure", "condition_occurrence"]:
                key = f"{source.split('_')[0]}_concept_id"
            else:
                raise KeyError(f"Extracting data from {source} is not supported yet")
        """
        if source == 'measurement':
            columns = ["value_as_number", "measurement_datetime"]
        elif source == 'observation':
            columns = ["value_as_number", "value_as_string", "measurement_datetime"]
        elif source == 'condition_occurrence':
            columns = None
        else:
            raise KeyError(f"Extracting data from {source} is not supported yet")
        """

        # TODO load using Dask or Dask-Awkward
        # Load source table using dask
        column_types, parse_dates = self._get_column_types(self.filepath[source])
        if parse_dates:
            if len(parse_dates) == 1:
                columns = list(column_types.keys()) + [parse_dates]
            else:
                columns = list(column_types.keys()) + parse_dates
        else:
            columns = list(column_types.keys())
        df_source = self._read_table(
            self.filepath[source], dtype=column_types, #parse_dates=parse_dates
        )  # , usecols=clinical_tables_columns[source]

        if not features:
            warnings.warn(
                "Please specify desired features you want to extract. Otherwise, it will try to extract all the features!"
            )
            features = list(df_source[key].compute().unique())
        else:
            if isinstance(features, str):
                features = [features]
            rprint(f"Trying to extarct the following features: {features}")

        # Input could be feature names/feature id (concept id)
        # First convert all input feaure names into feature id. Map concept using CONCEPT_RELATIONSHIP table if required.
        # Then try to extract feature data from source table using feature id.

        # TODO support features name

        if "concept" in self.tables:
            column_types, parse_dates = self._get_column_types(self.filepath["concept"])
            df_concept = self._read_table(self.filepath["concept"], dtype=column_types, parse_dates=parse_dates).dropna(
                subset=["concept_id", "concept_name"]
            )  # usecols=vocabularies_tables_columns["concept"],
            concept_dict = df_to_dict(df=df_concept, key="concept_id", value="concept_name")

        # TODO query this in the table

        feature_id_list = []
        feature_name_list = []
        domain_id_list = []
        concept_class_id_list = []
        concept_code_list = []

        fetures_not_shown_in_concept_table = []

        # Get feature id for each input, and check if each feature occurs in the concept table
        for feature in features:
            # if the input is feature ID
            if isinstance(feature, numbers.Integral):
                feature_id = feature
                feature_id_1, feature_id_2 = self.map_concept_id(feature_id, verbose=False)
                try:
                    feature_name = self.get_concept_name(feature_id_1, raise_error=True, verbose=False)
                except KeyError:
                    if ignore_not_shown_in_concept_table:
                        fetures_not_shown_in_concept_table.append(feature)
                        continue
                    else:
                        rprint(f"Feature ID - [red]{feature_id_1}[/] could not be found in concept table")
                        raise
                match_score = 1

            # if the input is feature name
            elif isinstance(feature, str):
                # return a list of (value, key, score)
                result = get_close_matches_using_dict(feature, concept_dict, n=2, cutoff=0.2)

                # if find 2 best matches
                if len(result) == 2:
                    match_score = result[0][2]

                    if match_score != 1:
                        if exact_match:
                            rprint(
                                f"Unable to find an exact match for [red]{feature}[/] in the concept table. Similar ones: 1) [red]{result[0][0]}[/] 2) [red]{result[1][0]}"
                            )
                            raise ValueError
                    else:
                        if result[1][1] == 1:
                            rprint(
                                f"Found multiple exact matches for [red]{feature}[/] in the concept table: 1) concept id: [red]{result[0][1]}[/] 2) concept id: [red]{result[1][1]}[/]. It is better to specify concept id directly."
                            )
                            raise ValueError
                    feature_name = feature
                    feature_id = result[0][1]
                # if only find 1 match
                else:
                    feature_name = result[0][0]
                    match_score = result[0][1]
                    feature_id = result[0][2]
                    if exact_match and match_score != 1:
                        rprint(
                            f"Unable to find an exact match for [red]{feature}[/] in the concept table Similar one is [red]{result[0][0]}"
                        )
                        raise ValueError
                feature_id_1, feature_id_2 = self.map_concept_id(feature_id)

            else:
                rprint(
                    "Please input either [red]feature name (string)[/] or [red]feature id (integer)[/] you want to extarct"
                )
                raise TypeError

            # feature_name_list.append(feature_name)
            # domain_id_list.append(df_concept.loc[df_concept["concept_id"] == feature_id, "domain_id"].reset_index(drop=True).compute()[0])
            # concept_class_id_list.append(df_concept.loc[df_concept["concept_id"] == feature_id, "concept_class_id"].reset_index(drop=True).compute()[0])
            # concept_code_list.append(df_concept.loc[df_concept["concept_id"] == feature_id, "concept_code"].reset_index(drop=True).compute()[0])

            if verbose:
                """
                if map_concept:
                    rprint(
                        f"Detected: feature [green]{feature_name}[/], feature ID [green]{feature_id}[/] in concept table, feature ID [green]{concept_id}[/] in concept relationship table, match socre = [green]{match_score}."
                    )
                else:
                """
                rprint(
                    f"Detected: feature [green]{feature_name}[/], feature ID [green]{feature_id}[/] in concept table, match socre = [green]{match_score}."
                )

            # for feature_id, feature_name, domain_id, concept_class_id, concept_code in zip(feature_id_list, feature_name_list, domain_id_list, concept_class_id_list, concept_code_list):
            try:
                feature_df = df_source[df_source[key] == feature_id_2].compute()
            except:
                print(f"Features ID could not be found in {source} table")
            # TODO add checks if all columns exist in source table
            if columns_in_source_table:
                columns = columns_in_source_table

            if remove_empty_column:
                columns = [column for column in columns if not feature_df[column].isna().all()]

            if len(feature_df) > 0:
                obs_dict = [
                    {
                        column: list(feature_df[feature_df["visit_occurrence_id"] == int(visit_occurrence_id)][column])
                        for column in columns
                    }
                    for visit_occurrence_id in adata.obs.index
                ]
                adata.obsm[feature_name] = ak.Array(obs_dict)

                if add_aggregation_to_X:
                    unit = feature_df["unit_source_value"].value_counts().index[0]
                    if aggregation_methods is None:
                        aggregation_methods = ["min", "max", "mean"]
                    var_name_list = [
                        f"{feature_name}_{aggregation_method}" for aggregation_method in aggregation_methods
                    ]
                    for aggregation_method in aggregation_methods:
                        func = getattr(ak, aggregation_method)
                        adata.obs[f"{feature_name}_{aggregation_method}"] = list(
                            func(adata.obsm[feature_name]["value_source_value"], axis=1)
                        )
                    adata = ep.ad.move_to_x(adata, var_name_list)

                    adata.var.loc[var_name_list, "Unit"] = unit
                    adata.var.loc[var_name_list, "domain_id"] = domain_id
                    adata.var.loc[var_name_list, "concept_class_id"] = concept_class_id
                    adata.var.loc[var_name_list, "concept_code"] = concept_code

        if len(fetures_not_shown_in_concept_table) > 0:
            rprint(f"Couldn't find concept {fetures_not_shown_in_concept_table} in concept table!")
        return adata

    # TODO add function to check feature and add concept
    
    # More IO functions
    def to_dataframe(
        self,
        adata,
        feature: str,  # TODO also support list of features
        # patient str or List,  # TODO also support subset of patients/visit
    ):
        # TODO
        # join index (visit_occurrence_id) to df
        # can be viewed as patient level - only select some patient

        df = ak.to_dataframe(adata.obsm[feature])

        df.reset_index(drop=False, inplace=True)
        df["entry"] = adata.obs.index[df["entry"]]
        df = df.rename(columns={"entry": "visit_occurrence_id"})
        del df["subentry"]
        return df

    # More Plot functions
    def plot_timeseries(
        self,
    ):
        # add one function from previous pipeline
        pass

    # More Pre-processing functions
    def sampling(
        self,
    ):
        # function from dask
        # need to check dask-awkward again
        pass
