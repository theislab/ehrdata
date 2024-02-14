import os
import glob

import pandas as pd

import ehrapy as ep
from pathlib import Path
from .utils.omop_utils import *
from rich.console import Console
from rich.text import Text
import rich.repr
from rich import print as rprint
from typing import TYPE_CHECKING, Any, Callable, Literal, Union, List

@rich.repr.auto(angular=True)
class OMOP:
    def __init__(self, folder_path, delimiter=None, make_filename_lowercase=True, use_dask=False):
        self.base = folder_path
        self.delimiter = delimiter
        self.use_dask = use_dask
        filepath_list = glob.glob(os.path.join(folder_path, "*.csv")) + glob.glob(os.path.join(folder_path, "*.parquet"))
        self.loaded_tabel = None

        self.filepath_dict = check_with_omop_cdm(filepath_list, base=self.base, delimiter=self.delimiter, make_filename_lowercase=make_filename_lowercase)
        self.tables = list(self.filepath_dict.keys())
        
    '''
    def __repr__(self) -> str:        
        print_str = f'OMOP object ({os.path.basename(self.base)}) with {len(self.tables)} tables.\nTables:\n'
        table_catalog_dict = get_table_catalog_dict()
        for _, (key, value) in enumerate(table_catalog_dict.items()):
            table_list = [table_name for table_name in self.tables if table_name in value]
            if len(table_list) != 0:
                print_str = print_str + f"{key} tables: {', '.join(table_list)}\n"
        return print_str
    '''
    
    def __rich_repr__(self):
        console = Console()
        table_catalog_dict = get_table_catalog_dict()
        color_map = {
            'Clinical data': 'blue',
            'Health system data': 'green',
            'Health economics data': 'red',
            'Standardized derived elements': 'magenta',
            'Metadata': 'white',
            'Vocabulary': 'dark_orange'
        }
        # Object description
        print_str = f'OMOP object ([red]{os.path.basename(self.base)}[/]) with {len(self.tables)} tables.\n'
        
        # Tables information
        for key, value in table_catalog_dict.items():
            table_list = [table_name for table_name in self.tables if table_name in value]
            if len(table_list) != 0:
                print_str = print_str + f"[{color_map[key]}]{key} tables[/]: [black]{', '.join(table_list)}[/]\n"
                #table_list_str = ', '.join(table_list)
                
                #text = Text(f"{key} tables: ", style=color_map[key])
                #text.append(table_list_str)
                #yield None, f"{key} tables", "red"
        console.print(print_str)
        yield None
    
    
    #TODO
    def new_load(self,
             level: Literal["stay_level", "patient_level"] = "stay_level",
             tables: Union[str, List[str]] = None, 
             remove_empty_column=True):
        
        table_catalog_dict = get_table_catalog_dict()
        if not tables:
            tables = self.table
        
        for table in self.table:
            # Load Clinical data tables
            if table in table_catalog_dict['Clinical data']:
                # in patient level 
                if table in ["person", "death"]:
                    column_types = get_column_types(path = self.filepath_dict[table], delimiter=self.delimiter, filename=table)
                    df = read_table(self.filepath_dict[table], delimiter=self.delimiter, dtype=column_types, index='person_id')
                elif table in ["visit_occurrence_id"]:
                    column_types = get_column_types(path = self.filepath_dict[table], delimiter=self.delimiter, filename=table)
                    df = read_table(self.filepath_dict[table], delimiter=self.delimiter, dtype=column_types, index='person_id')
                else:
                    warnings(f"Please use extract_features function to extract features from table {table}")
                    continue
            elif table in table_catalog_dict["Health system data"]:
                column_types = get_column_types(path = self.filepath_dict[table], delimiter=self.delimiter, filename=table)
                df = read_table(self.filepath_dict[table], delimiter=self.delimiter, dtype=column_types, index='person_id')
                
                
        
        
        # Load Health system data tables
        
        # Load Health economics data tables
        
        # Load Standardized derived elements tables
        
        # Load Metadata tables
        
        # Load Vocabulary tables
        
        
        # TODO patient level and hospital level
        if level == "stay_level":
            index = {"visit_occurrence": "visit_occurrence_id", "person": "person_id", "death": "person_id"}
            # TODO Only support clinical_tables_columns

            for table in tables:
                print(f"reading table [{table}]")
                column_types = get_column_types(path = self.filepath_dict[table], delimiter=self.delimiter, filename=table)
                df = read_table(self.filepath_dict[table], delimiter=self.delimiter, dtype=column_types, index='person_id')
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
    
    def load(self,
             level: Literal["stay_level", "patient_level"] = "stay_level",
             tables: Union[str, List[str]] = None, 
             remove_empty_column=True):
        
        if not tables:
            tables = ['person', 'death', 'visit_occurrence']
        # TODO patient level and hospital level
        if level == "stay_level":
            index = {"visit_occurrence": "visit_occurrence_id", "person": "person_id", "death": "person_id"}
            # TODO Only support clinical_tables_columns

            for table in tables:
                print(f"reading table [{table}]")
                column_types = get_column_types(path = self.filepath_dict[table], delimiter=self.delimiter, table_name=table)
                df = read_table(self.filepath_dict[table], delimiter=self.delimiter, dtype=column_types, index='person_id')
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
    
    