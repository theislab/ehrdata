import awkward as ak
import numpy as np
import pandas as pd


import ehrapy as ep
import scanpy as sc
from anndata import AnnData
import mudata as md
from mudata import MuData
import os
import glob

class OMOP():
    
    def __init__(self, file_paths):
        self.base = file_paths
        file_list = glob.glob(os.path.join(file_paths, '*'))
        self.loaded_tabel = None
        self.tables = []
        for file_path in file_list:
            file_name = file_path.split('/')[-1].removesuffix('.csv')
            self.tables.append(file_name)
    

    @property
    def clinical_tables(self):
        """
        A dictionary containing all of the ``Clinical`` OMOP CDM tables in the connected database.
        """
        table_names = ['person','observation_period','specimen','death','visit_occurrence','visit_detail','procedure_occurrence','drug_exposure','device_exposure','condition_occurrence','measurement','note','note_nlp','observation','fact_relationship']
        return [table_name for table_name in self.tables if table_name in table_names]
    
    @property
    def vocabularies_tables(self):
        """
        A dictionary containing all of the ``Vocabularies`` OMOP CDM tables in the connected database.
        """
        table_names = ['concept','vocabulary','domain','concept_class','concept_relationship','relationship','concept_synonym','concept_ancestor','source_to_concept_map','drug_strength']
        return [table_name for table_name in self.tables if table_name in table_names]

    @property
    def metadata_tables(self):
        """
        A dictionary containing all of the ``MetaData`` OMOP CDM tables in the connected database.
        """
        table_names = ['cdm_source','metadata']
        return [table_name for table_name in self.tables if table_name in table_names]

    @property
    def health_system_tables(self):
        """
        A dictionary containing all of the ``Health System`` OMOP CDM tables in the connected database.
        """
        table_names = ['location','care_site','provider']
        return [table_name for table_name in self.tables if table_name in table_names]

    @property
    def derived_elements_tables(self):
        """
        A dictionary containing all of the ``Derived Elements`` OMOP CDM tables in the connected database.
        """
        table_names = ['cohort','cohort_definition','drug_era','dose_era','condition_era']
        return [table_name for table_name in self.tables if table_name in table_names]
    
    @property
    def health_economics_tables(self):
        """
        A dictionary containing all of the ``Health Economics`` OMOP CDM tables in the connected database.
        """
        table_names = ['payer_plan_period','cost']
        return [table_name for table_name in self.tables if table_name in table_names]


    def load(self, level='stay_level', add_to_X=None, features=None):
        if level == 'stay_level':
            self.visit_occurrence = pd.read_csv(f'{self.base}/visit_occurrence.csv')
            self.person = pd.read_csv(f'{self.base}/person.csv', index_col='person_id')
            self.death = pd.read_csv(f'{self.base}/death.csv', index_col='person_id')
            self.measurement = pd.read_csv(f'{self.base}/measurement.csv')
            self.observation = pd.read_csv(f'{self.base}/observation.csv')
            self.drug_exposure = pd.read_csv(f'{self.base}/drug_exposure.csv')

            self.loaded_tabel = ['visit_occurrence', 'person', 'death', 'measurement', 'observation', 'drug_exposure']
            self.joined_table = pd.merge(self.visit_occurrence, self.person, on='person_id', how='left')
            self.joined_table = pd.merge(self.joined_table, self.death, on='person_id', how='left')
            


            obs_only_list = list(self.joined_table.columns)
            obs_only_list.remove('visit_occurrence_id')
            adata = ep.ad.df_to_anndata(
                self.joined_table, index_column="visit_occurrence_id", columns_obs_only = obs_only_list)
            
       
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
            
        
        return adata
