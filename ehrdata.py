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
from typing import List, Union, Literal, Optional
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
from pandas.tseries.offsets import DateOffset as Offset

import anndata as ad
from collections.abc import Collection, Iterable, Mapping, Sequence
from enum import Enum
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Literal, Union

import scanpy as sc
from scanpy.plotting import DotPlot, MatrixPlot, StackedViolin
from matplotlib.axes import Axes



pth = 'auxillary_files/OMOP_CDMv5.4_Field_Level.csv'
field_level = pd.read_csv(pth)
dtype_mapping = {'integer': "Int64",
                'Integer': "Int64",
                'float': float,
                'bigint': "Int64",
                'varchar(MAX)': str,
                'varchar(2000)': str,
                'varchar(1000)': str,
                'varchar(255)': str,
                'varchar(250)': str,
                'varchar(80)': str,
                'varchar(60)': str,
                'varchar(50)': str,
                'varchar(25)': str,
                'varchar(20)': str,
                'varchar(10)': str,
                'varchar(9)': str,
                'varchar(3)': str,
                'varchar(2)': str,
                'varchar(1)': str,
                'datetime': object,
                'date': object}       
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
    def __init__(self, folder_path, delimiter=None, make_filename_lowercase=True, use_dask=False):
        self.base = folder_path
        self.delimiter = delimiter
        self.use_dask = use_dask
        # TODO support also parquet and other formats
        file_list = glob.glob(os.path.join(folder_path, "*.csv")) + glob.glob(os.path.join(folder_path, "*.parquet"))
        self.loaded_tabel = None
        self.filepath = {}
        for file_path in file_list:
            file_name = file_path.split("/")[-1].split(".")[0]
            if check_csv_has_only_header(file_path):
                pass
            else:
                # Rename the file
                if make_filename_lowercase:
                    new_filepath = os.path.join(self.base, file_path.split("/")[-1].lower())
                    if file_path != new_filepath:
                        warnings(f"Rename file [file_path] to [new_filepath]")
                        os.rename(file_path, new_filepath)
                    self.filepath[file_name] = new_filepath
                else:
                    self.filepath[file_name] = file_path
        self.check_with_omop_cdm()
        self.tables = list(self.filepath.keys())
        
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
    
    def check_with_omop_cdm(self):
        for file_name, path in self.filepath.items():
            if file_name not in set(field_level.cdmTableName):
                raise KeyError(f"Table [{file_name}] is not defined in OMOP CDM v5.4! Please change the table name manually!")
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
            
            invalid_column_name = []
            for _, column in enumerate(columns_lowercase):
                cdm_columns = set(field_level[field_level.cdmTableName == file_name]['cdmFieldName'])
                if column not in cdm_columns:
                    invalid_column_name.append(column)
            if len(invalid_column_name) > 0:
                print(f"Column {invalid_column_name} is not defined in Table [{file_name}] in OMOP CDM v5.4! Please change the column name manually!\nFor more information, please refer to: https://ohdsi.github.io/CommonDataModel/cdm54.html#{file_name.upper()}")
                raise KeyError
                        

    # TODO redo this using omop cdm csv file   
    def _get_column_types(self, 
                          path: str = None, 
                          filename: str = None):
        column_types = {}
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
        for _, column in enumerate(columns_lowercase):
            column_types[column] = dtype_mapping[field_level[(field_level.cdmTableName == filename) & (field_level.cdmFieldName == column)]['cdmDatatype'].values[0]]
        return column_types
    
    def _read_table(self, path, dtype=None, parse_dates=None, index=None, usecols=None, use_dask=False, **kwargs):
        
        if use_dask:
            if not os.path.isfile(path):
                folder_walk = os.walk(path)
                filetype = next(folder_walk)[2][0].split(".")[-1]
            else:
                filetype = path.split(".")[-1]
            if filetype == 'csv':
                if not os.path.isfile(path):
                    path = f"{path}/*.csv"
                if usecols:
                    dtype = {key: dtype[key] for key in usecols if key in dtype}
                    if parse_dates:
                        parse_dates = {key: parse_dates[key] for key in usecols if key in parse_dates}
                df = dd.read_csv(path, delimiter=self.delimiter, dtype=dtype, parse_dates=parse_dates, usecols=usecols)
            elif filetype == 'parquet':
                if not os.path.isfile(path):
                    path = f"{path}/*.parquet"
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
            if filetype == 'csv':
                if usecols:
                    dtype = {key: dtype[key] for key in usecols if key in dtype}
                    if parse_dates:
                        parse_dates = {key: parse_dates[key] for key in usecols if key in parse_dates}
                df = pd.read_csv(path, delimiter=self.delimiter, dtype=dtype, parse_dates=parse_dates, usecols=usecols)
            elif filetype == 'parquet':
                df = pd.read_parquet(path, columns=usecols)
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
                column_types = self._get_column_types(path = self.filepath[table], filename=table)
                df = self._read_table(self.filepath[table], dtype=column_types, index='person_id') # TODO parse_dates = parse_dates
                if remove_empty_column:
                    # TODO dask Support
                    #columns = [column for column in df.columns if not df[column].compute().isna().all()]
                    columns = [column for column in df.columns if not df[column].isna().all()]
                df = df.loc[:, columns]
                setattr(self, table, df)

            # concept_id_list = list(self.concept.concept_id)
            # concept_name_list = list(self.concept.concept_id)
            # concept_domain_id_list = list(set(self.concept.domain_id))

            # self.loaded_tabel = ['visit_occurrence', 'person', 'death', 'measurement', 'observation', 'drug_exposure']
            # TODO dask Support
            joined_table = pd.merge(self.visit_occurrence, self.person, left_index=True, right_index=True, how="left")
            
            joined_table = pd.merge(joined_table, self.death, left_index=True, right_index=True, how="left")
            
            # TODO dask Support
            #joined_table = joined_table.compute()
            
            # TODO check this earlier 
            joined_table = joined_table.drop_duplicates(subset='visit_occurrence_id')
            joined_table = joined_table.set_index("visit_occurrence_id")
            # obs_only_list = list(self.joined_table.columns)
            # obs_only_list.remove('visit_occurrence_id')
            columns_obs_only = list(set(joined_table.columns) - set(["year_of_birth", "gender_source_value"]))
            adata = ep.ad.df_to_anndata(
                joined_table, index_column="visit_occurrence_id", columns_obs_only=columns_obs_only
            )
            # TODO this needs to be fixed because anndata set obs index as string by default
            #adata.obs.index = adata.obs.index.astype(int)

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
    
    def feature_counts(
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
        
        if source == 'measurement':
            columns = ["value_as_number", "time", "visit_occurrence_id", "measurement_concept_id"]
        elif source == 'observation':
            columns = ["value_as_number", "value_as_string", "measurement_datetime"]
        elif source == 'condition_occurrence':
            columns = None
        else:
            raise KeyError(f"Extracting data from {source} is not supported yet")
        
        column_types = self._get_column_types(path = self.filepath[source], filename=source)
        df_source = self._read_table(self.filepath[source], dtype=column_types, usecols=[f"{source}_concept_id"], use_dask=True)
        # TODO dask Support
        #feature_counts = df_source[f"{source}_concept_id"].value_counts().compute()[0:number]
        feature_counts = df_source[f"{source}_concept_id"].value_counts().compute()
        feature_counts = feature_counts.to_frame().reset_index(drop=False)[0:number]

        
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
            column_types = self._get_column_types(path = self.filepath["concept_relationship"], filename="concept_relationship")
            df_concept_relationship = self._read_csv(
                self.filepath["concept_relationship"], dtype=column_types
            )
            # TODO dask Support
            #df_concept_relationship.compute().dropna(subset=["concept_id_1", "concept_id_2", "relationship_id"], inplace=True)  # , usecols=vocabularies_tables_columns["concept_relationship"],
            df_concept_relationship.dropna(subset=["concept_id_1", "concept_id_2", "relationship_id"], inplace=True)  # , usecols=vocabularies_tables_columns["concept_relationship"],
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

        column_types = self._get_column_types(path = self.filepath["concept"], filename="concept")
        df_concept = self._read_table(self.filepath["concept"], dtype=column_types)
        # TODO dask Support
        #df_concept.compute().dropna(subset=["concept_id", "concept_name"], inplace=True, ignore_index=True)  # usecols=vocabularies_tables_columns["concept"]
        df_concept.dropna(subset=["concept_id", "concept_name"], inplace=True, ignore_index=True)  # usecols=vocabularies_tables_columns["concept"]
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
        column_types = self._get_column_types(path = self.filepath[source], filename=source)
        df_source = dd.read_csv(self.filepath[source], dtype=column_types)
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


    def get_feature_info(
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
        ignore_not_shown_in_concept_table: bool = True,
        exact_match: bool = True,

        verbose: bool = False,
    ):
        if key is None:
            if source in ["measurement", "observation", "specimen"]:
                key = f"{source}_concept_id"
            elif source in ["device_exposure", "procedure_occurrence", "drug_exposure", "condition_occurrence"]:
                key = f"{source.split('_')[0]}_concept_id"
            else:
                raise KeyError(f"Extracting data from {source} is not supported yet")

        if isinstance(features, str):
            features = [features]
            rprint(f"Trying to extarct the following features: {features}")

        # Input could be feature names/feature id (concept id)
        # First convert all input feaure names into feature id. Map concept using CONCEPT_RELATIONSHIP table if required.
        # Then try to extract feature data from source table using feature id.

        # TODO support features name

        if "concept" in self.tables:
            column_types = self._get_column_types(path = self.filepath["concept"], filename="concept")
            df_concept = self._read_table(self.filepath["concept"], dtype=column_types).dropna(
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

        info_df = pd.DataFrame([])
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
            
            info_df = pd.concat([info_df, pd.DataFrame(data=[[feature_name, feature_id_1, feature_id_2]], columns=['feature_name', 'feature_id_1', 'feature_id_2'])])
            
        
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
        if info_df[f"feature_id_1"].equals(info_df[f"feature_id_2"]):
            info_df.drop(f"feature_id_2", axis=1, inplace=True)
            info_df = info_df.rename(columns={"feature_id_1": "feature_id"})
            info_df = info_df.reset_index(drop=True)
        else:
            info_df = info_df.reset_index(drop=True)
        return info_df

    def get_feature_statistics(
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
        level="stay_level",
        value_col: str = 'value_source_value',
        aggregation_methods: Union[Literal["min", "max", "mean", "std", "count"], List[Literal["min", "max", "mean", "std", "count"]]]=None,
        add_aggregation_to_X: bool = True,
        verbose: bool = False,
        use_dask: bool = None,
    ):
        if source in ["measurement", "observation", "specimen"]:
            key = f"{source}_concept_id"
        elif source in ["device_exposure", "procedure_occurrence", "drug_exposure", "condition_occurrence"]:
            key = f"{source.split('_')[0]}_concept_id"
        else:
            raise KeyError(f"Extracting data from {source} is not supported yet")
        
        if source == 'measurement':
            source_table_columns = ['visit_occurrence_id', 'measurement_datetime', key, value_col]
        elif source == 'observation':
            source_table_columns = ['visit_occurrence_id', "observation_datetime", key, value_col]
        elif source == 'condition_occurrence':
            source_table_columns = None
        else:
            raise KeyError(f"Extracting data from {source} is not supported yet")

        if use_dask is None:
            use_dask = self.use_dask
        source_column_types = self._get_column_types(path = self.filepath[source], filename=source)
        df_source = self._read_table(self.filepath[source], dtype=source_column_types, usecols=source_table_columns, use_dask=use_dask) 
        info_df = self.get_feature_info(adata, source=source, features=features, verbose=False)
        info_dict = info_df[['feature_id', 'feature_name']].set_index('feature_id').to_dict()['feature_name']
        
        # Select featrues
        df_source = df_source[df_source[key].isin(list(info_df.feature_id))]
        #TODO Select time
        #da_measurement = da_measurement[(da_measurement.time >= 0) & (da_measurement.time <= 48*60*60)]
        #df_source[f'{source}_name'] = df_source[key].map(info_dict)
        if aggregation_methods is None:
            aggregation_methods = ["min", "max", "mean", "std", "count"]
        if level == 'stay_level':
            result = df_source.groupby(['visit_occurrence_id', key]).agg({
                    value_col: aggregation_methods})
            
            if use_dask:
                result = result.compute()
            result = result.reset_index(drop=False)
            result.columns = ["_".join(a) for a in result.columns.to_flat_index()]
            result.columns  = result.columns.str.removesuffix('_')
            result.columns  = result.columns.str.removeprefix(f'{value_col}_')
            result[f'{source}_name'] = result[key].map(info_dict)

            df_statistics = result.pivot(index='visit_occurrence_id', 
                                columns=f'{source}_name', 
                                values=aggregation_methods)
            df_statistics.columns = df_statistics.columns.swaplevel()
            df_statistics.columns = ["_".join(a) for a in df_statistics.columns.to_flat_index()]

            
            # TODO
            sort_columns = True
            if sort_columns:
                new_column_order = []
                for feature in features:
                    for suffix in (f'_{aggregation_method}' for aggregation_method in aggregation_methods):
                        col_name = f'{feature}{suffix}'
                        if col_name in df_statistics.columns:
                            new_column_order.append(col_name)

                df_statistics.columns = new_column_order
            
        df_statistics.index = df_statistics.index.astype(str)
        
        adata.obs = adata.obs.join(df_statistics, how='left')
        
        if add_aggregation_to_X:
            adata = ep.ad.move_to_x(adata, list(df_statistics.columns))
        return adata

    
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
        source_table_columns: Union[str, List[str]] = None,
        dropna: Optional[bool] = True,
        verbose: Optional[bool] = True,
        use_dask: bool = None,
    ):
        
        if source in ["measurement", "observation", "specimen"]:
            key = f"{source}_concept_id"
        elif source in ["device_exposure", "procedure_occurrence", "drug_exposure", "condition_occurrence"]:
            key = f"{source.split('_')[0]}_concept_id"
        else:
            raise KeyError(f"Extracting data from {source} is not supported yet")
        
        if source_table_columns is None:
            if source == 'measurement':
                source_table_columns = ['visit_occurrence_id', 'measurement_datetime', 'value_as_number', key]
            elif source == 'observation':
                source_table_columns = ['visit_occurrence_id', "value_as_number", "value_as_string", "observation_datetime", key]
            elif source == 'condition_occurrence':
                source_table_columns = None
            else:
                raise KeyError(f"Extracting data from {source} is not supported yet")
        if use_dask is None:
            use_dask = self.use_dask                
        

        # TODO load using Dask or Dask-Awkward
        # Load source table using dask
        source_column_types = self._get_column_types(path = self.filepath[source], filename=source)
        df_source = self._read_table(self.filepath[source], dtype=source_column_types, usecols=source_table_columns, use_dask=use_dask) 
        info_df = self.get_feature_info(adata, source=source, features=features, verbose=False)
        info_dict = info_df[['feature_id', 'feature_name']].set_index('feature_id').to_dict()['feature_name']
        
        
        # Select featrues
        df_source = df_source[df_source[key].isin(list(info_df.feature_id))]
        
        # TODO select time period  
        #df_source = df_source[(df_source.time >= 0) & (df_source.time <= 48*60*60)]
        #da_measurement['measurement_name'] = da_measurement.measurement_concept_id.replace(info_dict)
        
        # TODO dask caching
        """ 
        from dask.cache import Cache
        cache = Cache(2e9)
        cache.register()
        """
        if use_dask:
            if dropna == True:
                df_source = df_source.compute().dropna()
            else:
                df_source = df_source.compute()
        else:
            if dropna == True:
                df_source = df_source.dropna()
        
        # Preprocess steps outside the loop
        unique_visit_occurrence_ids = set(adata.obs.index)#.astype(int))
        empty_entry = {source_table_column: [] for source_table_column in source_table_columns if source_table_column not in [key, 'visit_occurrence_id'] }
        
        # Filter data once, if possible
        filtered_data = {
            feature_id: df_source[df_source[key] == feature_id]
            for feature_id in set(info_dict.keys())
        }

        for feature_id in set(info_dict.keys()):
            df_feature = filtered_data[feature_id][list(set(source_table_columns) - set([key]))]
            grouped = df_feature.groupby("visit_occurrence_id")
            if verbose:
                print(f"Adding feature [{info_dict[feature_id]}] into adata.obsm")
            
            # Use set difference and intersection more efficiently
            feature_ids = unique_visit_occurrence_ids.intersection(grouped.groups.keys())

            # Creating the array more efficiently
            adata.obsm[info_dict[feature_id]] = ak.Array([
                grouped.get_group(visit_occurrence_id)[list(set(source_table_columns) - set([key, 'visit_occurrence_id']))].to_dict(orient='list') if visit_occurrence_id in feature_ids else empty_entry 
                for visit_occurrence_id in unique_visit_occurrence_ids
            ])

        return adata


    def drop_nan(self, 
                 adata, 
                 key: Union[str, List[str]],
                 slot: Union[str, None] = 'obsm', 
                 ):
        if isinstance(key, str):
            key_list = [key]
        else:
            key_list = key
        if slot == 'obsm':
            for key in key_list:
                ak_array = adata.obsm[key]
                
                # Update the combined mask based on the presence of None in each field
                for i, field in enumerate(ak_array.fields):
                    field_mask = ak.is_none(ak.nan_to_none(ak_array[field]), axis=1)
                    if i==0:
                        combined_mask = ak.full_like(field_mask, fill_value=False, dtype=bool)
                    combined_mask = combined_mask | field_mask
                ak_array = ak_array[~combined_mask]
                adata.obsm[key] = ak_array

        return adata

    # downsampling
    def aggregate_timeseries_in_bins(self,
                                     adata,
                                     features: Union[str, List[str]],
                                     slot: Union[str, None] = 'obsm',
                                     value_key: str = 'value_as_number',
                                     time_key: str = 'measurement_datetime',
                                     time_binning_method: Literal["floor", "ceil", "round"] = "floor",
                                     bin_size: Union[str, Offset] = 'h',
                                     aggregation_method: Literal['median', 'mean', 'min', 'max'] = 'median',
                                     time_upper_bound: int = 48# TODO
                                     ):

        if isinstance(features, str):
            features_list = [features]
        else:
            features_list = features

        # Ensure the time_binning_method provided is one of the expected methods
        if time_binning_method not in ["floor", "ceil", "round"]:
            raise ValueError(f"time_binning_method {time_binning_method} is not supported. Choose from 'floor', 'ceil', or 'round'.")

        if aggregation_method not in {'median', 'mean', 'min', 'max'}:
            raise ValueError(f"aggregation_method {aggregation_method} is not supported. Choose from 'median', 'mean', 'min', or 'max'.")

        if slot == 'obsm':
            for feature in features_list:
                print(f"processing feature [{feature}]")
                df = self.to_dataframe(adata, features)
                if pd.api.types.is_datetime64_any_dtype(df[time_key]):
                    func = getattr(df[time_key].dt, time_binning_method, None)
                    if func is not None:
                        df[time_key] = func(bin_size)
                else:
                    # TODO need to take care of this if it doesn't follow omop standard
                    if bin_size == 'h':
                        df[time_key] = df[time_key] / 3600
                        func = getattr(np, time_binning_method)
                        df[time_key] = func(df[time_key])
                
                df[time_key] = df[time_key].astype(str)
                # Adjust time values that are equal to the time_upper_bound
                #df.loc[df[time_key] == time_upper_bound, time_key] = time_upper_bound - 1
                
                # Group and aggregate data
                df = df.groupby(["visit_occurrence_id", time_key])[value_key].agg(aggregation_method).reset_index(drop=False)
                grouped = df.groupby("visit_occurrence_id")

                unique_visit_occurrence_ids = adata.obs.index
                empty_entry = {value_key: [], time_key: []}

                # Efficiently use set difference and intersection
                feature_ids = unique_visit_occurrence_ids.intersection(grouped.groups.keys())
                # Efficiently create the array
                ak_array = ak.Array([
                    grouped.get_group(visit_occurrence_id)[[value_key, time_key]].to_dict(orient='list') if visit_occurrence_id in feature_ids else empty_entry
                    for visit_occurrence_id in unique_visit_occurrence_ids
                ])
                adata.obsm[feature] = ak_array

        return adata
    
    def timeseries_discretizer(self,
                              adata,
                              key: Union[str, List[str]],
                              slot: Union[str, None] = 'obsm',
                              value_key: str = 'value_as_number',
                              time_key: str = 'measurement_datetime',
                              freq: str = 'hour', #TODO
                              time_limit: int = 48, #TODO
                              method: str = 'median' #TODO
                              ):
        
        pass
    
    
    
    def from_dataframe(
        self,
        adata,
        feature: str,
        df
    ):
        grouped = df.groupby("visit_occurrence_id")
        unique_visit_occurrence_ids = set(adata.obs.index)

        # Use set difference and intersection more efficiently
        feature_ids = unique_visit_occurrence_ids.intersection(grouped.groups.keys())
        empty_entry = {source_table_column: [] for source_table_column in set(df.columns) if source_table_column not in ['visit_occurrence_id'] }

        # Creating the array more efficiently
        ak_array = ak.Array([
            grouped.get_group(visit_occurrence_id)[list(set(df.columns) - set(['visit_occurrence_id']))].to_dict(orient='list') if visit_occurrence_id in feature_ids else empty_entry 
            for visit_occurrence_id in unique_visit_occurrence_ids])
        adata.obsm[feature] = ak_array
        
        return adata
        
    # TODO add function to check feature and add concept
    # More IO functions
    def to_dataframe(
        self,
        adata,
        features: Union[str, List[str]],  # TODO also support list of features
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
                if col.endswith('time'):
                    df[col] = pd.to_datetime(df[col])
            
            df['feature_name'] = feature
            df_concat = pd.concat([df_concat, df], axis= 0)
        
        
        return df_concat


    def plot_timeseries(self,
                        adata,
                        visit_occurrence_id: int,
                        key: Union[str, List[str]],
                        slot: Union[str, None] = 'obsm',
                        value_key: str = 'value_as_number',
                        time_key: str = 'measurement_datetime',
                        x_label: str = None
    ):
    
    
        if isinstance(key, str):
            key_list = [key]
        else:
            key_list = key

        # Initialize min_x and max_x
        min_x = None
        max_x = None

        if slot == 'obsm':
            fig, ax = plt.subplots(figsize=(20, 6))
            # Scatter plot
            for i, key in enumerate(key_list):
                df = self.to_dataframe(adata, key)
                x = df[df.visit_occurrence_id == visit_occurrence_id][time_key]
                y = df[df.visit_occurrence_id == visit_occurrence_id][value_key]

                # Check if x is empty
                if not x.empty:
                    ax.scatter(x=x, y=y, label=key)
                    ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=len(key_list), prop={"size": 14})
                    
                    ax.plot(x, y)
                    

                    if min_x is None or min_x > x.min():
                        min_x = x.min()
                    if max_x is None or max_x < x.max():
                        max_x = x.max()
                    
                    
                else:
                    # Skip this iteration if x is empty
                    continue
            
            if min_x is not None and max_x is not None:
                
                # Adapt this to input data
                # TODO step
                #plt.xticks(np.arange(min_x, max_x, step=1))
                # Adapt this to input data
                plt.xlabel(x_label if x_label else "Hours since ICU admission")
            
            plt.show()


    def violin(
        self,
        adata: AnnData,
        obsm_key: str = None,
        keys: Union[str, Sequence[str]] = None,
        groupby: Optional[str] = None,
        log: Optional[bool] = False,
        use_raw: Optional[bool] = None,
        stripplot: bool = True,
        jitter: Union[float, bool] = True,
        size: int = 1,
        layer: Optional[str] = None,
        scale: Literal["area", "count", "width"] = "width",
        order: Optional[Sequence[str]] = None,
        multi_panel: Optional[bool] = None,
        xlabel: str = "",
        ylabel: Union[str, Sequence[str]] = None,
        rotation: Optional[float] = None,
        show: Optional[bool] = None,
        save: Union[bool, str] = None,
        ax: Optional[Axes] = None,
        **kwds,
    ):  # pragma: no cover
        """Violin plot.

        Wraps :func:`seaborn.violinplot` for :class:`~anndata.AnnData`.

        Args:
            adata: :class:`~anndata.AnnData` object object containing all observations.
            keys: Keys for accessing variables of `.var_names` or fields of `.obs`.
            groupby: The key of the observation grouping to consider.
            log: Plot on logarithmic axis.
            use_raw: Whether to use `raw` attribute of `adata`. Defaults to `True` if `.raw` is present.
            stripplot: Add a stripplot on top of the violin plot. See :func:`~seaborn.stripplot`.
            jitter: Add jitter to the stripplot (only when stripplot is True) See :func:`~seaborn.stripplot`.
            size: Size of the jitter points.
            layer: Name of the AnnData object layer that wants to be plotted. By
                default adata.raw.X is plotted. If `use_raw=False` is set,
                then `adata.X` is plotted. If `layer` is set to a valid layer name,
                then the layer is plotted. `layer` takes precedence over `use_raw`.
            scale: The method used to scale the width of each violin.
                If 'width' (the default), each violin will have the same width.
                If 'area', each violin will have the same area.
                If 'count', a violin’s width corresponds to the number of observations.
            order: Order in which to show the categories.
            multi_panel: Display keys in multiple panels also when `groupby is not None`.
            xlabel: Label of the x axis. Defaults to `groupby` if `rotation` is `None`, otherwise, no label is shown.
            ylabel: Label of the y axis. If `None` and `groupby` is `None`, defaults to `'value'`.
                    If `None` and `groubpy` is not `None`, defaults to `keys`.
            rotation: Rotation of xtick labels.
            {show_save_ax}
            **kwds:
                Are passed to :func:`~seaborn.violinplot`.

        Returns:
            A :class:`~matplotlib.axes.Axes` object if `ax` is `None` else `None`.

        Example:
            .. code-block:: python

                import ehrapy as ep

                adata = ep.dt.mimic_2(encoded=True)
                ep.pp.knn_impute(adata)
                ep.pp.log_norm(adata, offset=1)
                ep.pp.neighbors(adata)
                ep.tl.leiden(adata, resolution=0.5, key_added="leiden_0_5")
                ep.pl.violin(adata, keys=["age"], groupby="leiden_0_5")

        Preview:
            .. image:: /_static/docstring_previews/violin.png
        """
        
        if obsm_key:
            df = self.to_dataframe(adata, features=obsm_key)
            df = df[["visit_occurrence_id", "value_as_number"]]
            df = df.rename(columns = {"value_as_number": obsm_key})
            
            if groupby:
                df = df.set_index('visit_occurrence_id').join(adata.obs[groupby].to_frame()).reset_index(drop=False)
                adata = ep.ad.df_to_anndata(df, columns_obs_only=['visit_occurrence_id', groupby])
            else:
                adata = ep.ad.df_to_anndata(df, columns_obs_only=['visit_occurrence_id'])
            keys=obsm_key
            
        violin_partial = partial(
            sc.pl.violin,
            keys=keys,
            log=log,
            use_raw=use_raw,
            stripplot=stripplot,
            jitter=jitter,
            size=size,
            layer=layer,
            scale=scale,
            order=order,
            multi_panel=multi_panel,
            xlabel=xlabel,
            ylabel=ylabel,
            rotation=rotation,
            show=show,
            save=save,
            ax=ax,
            **kwds,)
            
        return violin_partial(adata=adata, groupby=groupby)


    def qc_lab_measurements(
        self,
        adata: AnnData,
        reference_table: pd.DataFrame = None,
        measurements: list[str] = None,
        obsm_measurements: list[str] = None,
        action: Literal["remove"] = "remove",
        unit: Literal["traditional", "SI"] = None,
        layer: str = None,
        threshold: int = 20,
        age_col: str = None,
        age_range: str = None,
        sex_col: str = None,
        sex: str = None,
        ethnicity_col: str = None,
        ethnicity: str = None,
        copy: bool = False,
        verbose: bool = False,
    ) -> AnnData:

        if copy:
            adata = adata.copy()

        preprocessing_dir = '/Users/xinyuezhang/ehrapy/ehrapy/preprocessing'
        if reference_table is None:
            reference_table = pd.read_csv(
                f"{preprocessing_dir}/laboratory_reference_tables/laposata.tsv", sep="\t", index_col="Measurement"
            )
        if obsm_measurements:
            measurements = obsm_measurements
        for measurement in measurements:
            best_column_match, score = process.extractOne(
                query=measurement, choices=reference_table.index, score_cutoff=threshold
            )
            if best_column_match is None:
                rprint(f"[bold yellow]Unable to find a match for {measurement}")
                continue
            if verbose:
                rprint(
                    f"[bold blue]Detected [green]{best_column_match}[blue] for [green]{measurement}[blue] with score [green]{score}."
                )

            reference_column = "SI Reference Interval" if unit == "SI" else "Traditional Reference Interval"

            # Fetch all non None columns from the reference statistics
            not_none_columns = [col for col in [sex_col, age_col, ethnicity_col] if col is not None]
            not_none_columns.append(reference_column)
            reference_values = reference_table.loc[[best_column_match], not_none_columns]

            additional_columns = False
            if sex_col or age_col or ethnicity_col:  # check if additional columns were provided
                additional_columns = True

            # Check if multiple reference values occur and no additional information is available:
            if reference_values.shape[0] > 1 and additional_columns is False:
                raise ValueError(
                    f"Several options for {best_column_match} reference value are available. Please specify sex, age or "
                    f"ethnicity columns and their values."
                )

            try:
                if age_col:
                    min_age, max_age = age_range.split("-")
                    reference_values = reference_values[
                        (reference_values[age_col].str.split("-").str[0].astype(int) >= int(min_age))
                        and (reference_values[age_col].str.split("-").str[1].astype(int) <= int(max_age))
                    ]
                if sex_col:
                    sexes = "U|M" if sex is None else sex
                    reference_values = reference_values[reference_values[sex_col].str.contains(sexes)]
                if ethnicity_col:
                    reference_values = reference_values[reference_values[ethnicity_col].isin([ethnicity])]

                if layer is not None:
                    actual_measurements = adata[:, measurement].layers[layer]
                else:
                    if obsm_measurements:
                        actual_measurements = adata.obsm[measurement]['value_as_number']
                        ak_measurements = adata.obsm[measurement]
                    else:
                        actual_measurements = adata[:, measurement].X
            except TypeError:
                rprint(f"[bold yellow]Unable to find specified reference values for {measurement}.")

            check = reference_values[reference_column].values
            check_str: str = np.array2string(check)
            check_str = check_str.replace("[", "").replace("]", "").replace("'", "")
            if "<" in check_str:
                upperbound = float(check_str.replace("<", ""))
                if verbose:
                    rprint(f"[bold blue]Using upperbound [green]{upperbound}")
                upperbound_check_results = actual_measurements < upperbound
                if isinstance(actual_measurements, ak.Array):
                    if action == 'remove':
                        if verbose:
                            rprint(f"Removing {ak.count(actual_measurements) - ak.count(actual_measurements[upperbound_check_results])} outliers")
                        adata.obsm[measurement] = ak_measurements[upperbound_check_results]
                else:
                    upperbound_check_results_array: np.ndarray = upperbound_check_results.copy()       
                    adata.obs[f"{measurement} normal"] = upperbound_check_results_array
                    
            elif ">" in check_str:
                lower_bound = float(check_str.replace(">", ""))
                if verbose:
                    rprint(f"[bold blue]Using lowerbound [green]{lower_bound}")

                lower_bound_check_results = actual_measurements > lower_bound
                if isinstance(actual_measurements, ak.Array):
                    if action == 'remove':
                        adata.obsm[measurement] = ak_measurements[lower_bound_check_results]
                else:
                    adata.obs[f"{measurement} normal"] = lower_bound_check_results_array
                    lower_bound_check_results_array = lower_bound_check_results.copy()
            else:  # "-" range case
                min_value = float(check_str.split("-")[0])
                max_value = float(check_str.split("-")[1])
                if verbose:
                    rprint(f"[bold blue]Using minimum of [green]{min_value}[blue] and maximum of [green]{max_value}")

                range_check_results = (actual_measurements >= min_value) & (actual_measurements <= max_value)
                if isinstance(actual_measurements, ak.Array):
                    if action == 'remove':
                        adata.obsm[measurement] = ak_measurements[range_check_results]
                else:
                    adata.obs[f"{measurement} normal"] = range_check_results_array
                    range_check_results_array: np.ndarray = range_check_results.copy()

        if copy:
            return adata
