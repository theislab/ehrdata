import awkward as ak
import numpy as np
import pandas as pd


import ehrapy as ep
import scanpy as sc
from anndata import AnnData
import mudata as md
from mudata import MuData
from typing import List, Union
import os
import glob
import dask.dataframe as dd
from thefuzz import process
import sys
from rich import print as rprint

clinical_tables_columns = {
    'person': ['person_id', 'year_of_birth', 'gender_source_value'],
    'observation_period': [],
    'death': ['person_id', 'death_datetime'], 
    'visit_occurrence': ['visit_occurrence_id', 'person_id', 'visit_start_datetime', 'visit_end_datetime'],
    'visit_detail': [],
    'condition_occurrence': [],
    'drug_exposure': ['drug_exposure_id', 'person_id', 'visit_occurrence_id', 'drug_concept_id', ],
    'procedure_occurrence': ['visit_occurrence_id', 'person_id', 'visit_start_datetime', 'visit_end_datetime'],
    'device_exposure': [],
    'specimen': [],
    'measurement': ['measurement_id', 'person_id', 'visit_occurrence_id', 'measurement_concept_id', 'measurement_datetime', 'value_as_number', 'unit_source_value'],
    'observation': ['observation_id', 'person_id', 'observation_concept_id', 'observation_datetime', "value_as_number", "value_as_string"],
    'note': [],
    'note_nlp': [],
    'fact_relationship': [],
    'procedure_occurrence': [],
}

health_system_tables_columns = {
    'location': [],
    'care_site': ['care_site_id', 'care_site_name'],
    'provider': [],
}
vocabularies_tables_columns ={
    'concept': ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id', 'concept_class_id', 'standard_concept', 'concept_code'],
    'vocabulary': [],
    'domain': [],
    'concept_class': [],
    'concept_synonym': [],
    'concept_relationship': ["concept_id_1", "concept_id_2", "relationship_id"],
    'relationship': [],
    'concept_ancestor': [],
    'source_to_concept_map': [],
    'drug_strength': []
}
dtypes_dict = {}
dtypes_dict['concept'] = {'concept_id': int, 'standard_concept': str}
dtypes_dict['measurement']={'measurement_source_concept_id': int,
       'measurement_source_value': str,
       'unit_concept_id': int,
       'value_as_number': 'float64',
       'value_source_value': str}


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

    if not n >  0:
        raise ValueError("n must be > 0: %r" % (n,))
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))
    result = []
    s = SequenceMatcher()
    s.set_seq2(word)
    for _, (key, x) in enumerate(possibilities.items()):
        s.set_seq1(x)
        if s.real_quick_ratio() >= cutoff and \
           s.quick_ratio() >= cutoff and \
           s.ratio() >= cutoff:
            result.append((s.ratio(), x, key))

    # Move the best scorers to head of list
    result = _nlargest(n, result)
    
    # Strip scores for the best n matches
    return [(x, score, key) for score, x, key in result]

def df_to_dict(df, key, value):
    if isinstance(df, dd.DataFrame):
        return pd.Series(df[value].compute().values, index=df[key].compute()).to_dict()
    else:
        return pd.Series(df[value].values, index=df[key]).to_dict()
     

class OMOP():
    
    def __init__(self, file_paths):
        self.base = file_paths
        file_list = glob.glob(os.path.join(file_paths, '*'))
        self.loaded_tabel = None
        self.filepath = {}
        for file_path in file_list:
            file_name = file_path.split('/')[-1].removesuffix('.csv')
            self.filepath[file_name] = file_path
            
        self.tables = list(self.filepath.keys())
        '''
        if "concept" in self.tables:
            df_concept = dd.read_csv(self.filepath["concept"], usecols=vocabularies_tables_columns["concept"])
            self.concept_id_to_name = dict(zip(df_concept['id'], df_concept['name']))
            self.concept_name_to_id = dict(zip(df_concept['name'], df_concept['id']))
        '''

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
    
    


    def load(self, level='stay_level', tables = ['visit_occurrence', 'person', 'death'], add_to_X=None, features=None):
        
        if level=='stay_level':
            index = {'visit_occurrence': 'visit_occurrence_id', 'person': 'person_id', 'death': "person_id"}
            # TODO Only support clinical_tables_columns
            for table in tables:
                setattr(self, table, dd.read_csv(self.filepath[table], usecols=clinical_tables_columns[table]).set_index("person_id"))
            

            #concept_id_list = list(self.concept.concept_id)
            #concept_name_list = list(self.concept.concept_id)
            #concept_domain_id_list = list(set(self.concept.domain_id))
            

            
            


            #self.loaded_tabel = ['visit_occurrence', 'person', 'death', 'measurement', 'observation', 'drug_exposure']
            joined_table = dd.merge(self.visit_occurrence, self.person, left_index=True, right_index=True, how='left')
            joined_table = dd.merge(joined_table, self.death, left_index=True, right_index=True, how='left')

            joined_table = joined_table.compute()
            joined_table = joined_table.set_index('visit_occurrence_id')
        
        


            #obs_only_list = list(self.joined_table.columns)
            #obs_only_list.remove('visit_occurrence_id')
            columns_obs_only = list(set(joined_table.columns) - set(['year_of_birth', 'gender_source_value']))
            adata = ep.ad.df_to_anndata(
                joined_table, index_column="visit_occurrence_id", columns_obs_only = columns_obs_only)
            
            '''
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
            '''
        
        return adata
    def extract_features(self, 
                         adata,
                         source: str, 
                         features: str or int or List[Union[str, int]], 
                         map_concept = True,
                         add_aggregation_to_X: bool=True, 
                         aggregation_methods = None,
                         add_all_data: bool = True,
                         exact_match: bool = True,
                         verbose: bool = False,):
        #source = 'measurement'
        #features = [3012501]
        #add_aggregation_to_X = True
        
        if source == 'measurement':
            columns = ["value_as_number", "measurement_datetime"]
        elif source == 'observation':
            columns = ["value_as_number", "value_as_string", "measurement_datetime"]
        else:
            raise KeyError(f"Extracting data from {source} is not supported yet")
    
        
        # TODO load using Dask or Dask-Awkward
        # Load source table using dask
        df_source = dd.read_csv(self.filepath[source], usecols=clinical_tables_columns[source], dtype=dtypes_dict["measurement"])
        
        if "concept" in self.tables:
            df_concept = dd.read_csv(self.filepath["concept"], usecols=vocabularies_tables_columns["concept"], dtype = dtypes_dict["concept"]).dropna(subset=['concept_id', 'concept_name'])  
            concept_dict = df_to_dict(df=df_concept, key = 'concept_id', value = 'concept_name')
        if map_concept:
            df_concept_relationship = dd.read_csv(self.filepath["concept_relationship"], usecols=vocabularies_tables_columns["concept_relationship"]).dropna(subset=['concept_id_1', 'concept_id_2', 'relationship_id'])  
            concept_relationship_dict = df_to_dict(df=df_concept_relationship[df_concept_relationship['relationship_id'] == 'Maps to'], key = 'concept_id_1', value = 'concept_id_2')
            map_concept_id_list = []
        # Input could be feature names/feature id (concept id)
        # TODO support features name
        if not features:
            raise KeyError(f"Please input the desired features you want to extarct")
        else:
            if isinstance(features, int) or isinstance(features, str):
                features = [features]

            # TODO query this in the table
                    
            #concept_name = 'Base Excess|Blood|Blood Gas'
            #unit = 'mEq/L'
            #domain_id = 'Measurement'
            feature_id_list = []
            feature_name_list = []
            domain_id_list = []
            concept_class_id_list = []
            concept_code_list = []
            # Get feature id for each input, and check if each feature occurs in the concept table 
            for feature in features:
                if isinstance(feature, int):
                    try:
                        feature_id = feature
                        feature_name = concept_dict[feature_id]
                        feature_id_list.append(feature_id)
                        match_score = 1
                    except KeyError:
                        rprint(f"Feature ID - [red]{feature}[/] could not be found in concept table")
                        raise
                elif isinstance(feature, str):
                    
                    result = get_close_matches_using_dict(feature, concept_dict, n=2, cutoff=0.2)
                    if len(result) == 2:
                        match_score = result[0][1]

                        if match_score != 1:
                            if exact_match:
                                rprint(f"Unable to find an exact match for [red]{feature}[/] in the concept table. Similar ones: 1) [red]{result[0][0]}[/] 2) [red]{result[1][0]}")
                                raise ValueError
                        else:
                            if result[1][1] == 1:
                                rprint(f"Found multiple exact matches for [red]{feature}[/] in the concept table: 1) concept id: [red]{result[0][2]}[/] 2) concept id: [red]{result[1][2]}[/]. It is better to specify concept id directly.")
                                raise ValueError
                        feature_name = feature
                        feature_id = result[0][2]
                    
                    else:
                        feature_name = result[0][0]
                        match_score = result[0][1]
                        feature_id = result[0][2]
                        if exact_match and match_score != 1:
                            rprint(f"Unable to find an exact match for [red]{feature}[/] in the concept table Similar one is [red]{result[0][0]}")
                            raise ValueError
                    feature_id_list.append(feature_id)
                else:
                    rprint("Please input either [red]feature name (string)[/] or [red]feature id (integer)[/] you want to extarct")
                    raise TypeError
                if map_concept:
                    concept_id = concept_relationship_dict[feature_id]
                    map_concept_id_list.append(concept_id)  
                    
                feature_name_list.append(feature_name)
                domain_id_list.append(df_concept.loc[df_concept["concept_id"] == feature_id, "domain_id"].reset_index(drop=True).compute()[0])
                concept_class_id_list.append(df_concept.loc[df_concept["concept_id"] == feature_id, "concept_class_id"].reset_index(drop=True).compute()[0])
                concept_code_list.append(df_concept.loc[df_concept["concept_id"] == feature_id, "concept_code"].reset_index(drop=True).compute()[0])
                if verbose:
                    if map_concept:
                        rprint(
                            f"Detected: feature [green]{feature_name}[/], feature ID [green]{feature_id}[/] in concept table, feature ID [green]{concept_id}[/] in concept relationship table, match socre = [green]{match_score}."
                        )
                    else:
                        rprint(
                            f"Detected: feature [green]{feature_name}[/], feature ID [green]{feature_id}[/] in concept table, match socre = [green]{match_score}."
                        )
            
            if map_concept:
                feature_id_list = map_concept_id_list
            for feature_id, feature_name, domain_id, concept_class_id, concept_code in zip(feature_id_list, feature_name_list, domain_id_list, concept_class_id_list, concept_code_list):
                try:
                    feature_df = df_source[df_source[f"{source}_concept_id"] == feature_id].compute()
                except:
                    print(f"Features ID could not be found in {source} table")

                if len(feature_df) > 0:
                    print("extracting features")
                    obs_dict = [{column: list(feature_df[feature_df['visit_occurrence_id'] == int(visit_occurrence_id)][column])  for column in columns} for visit_occurrence_id in adata.obs.index]
                    adata.obsm[feature_name] = ak.Array(obs_dict)
                    
                    if add_aggregation_to_X:
                        unit = feature_df['unit_source_value'].value_counts().index[0]
                        if aggregation_methods is None:
                            aggregation_methods = ['min', 'max', 'mean']
                        var_name_list = [f'{feature_name}_{aggregation_method}' for aggregation_method in aggregation_methods]
                        for aggregation_method in aggregation_methods:
                            func = getattr(ak, aggregation_method)
                            adata.obs[f'{feature_name}_{aggregation_method}'] = list(func(adata.obsm[feature_name]['value_as_number'], axis=1))
                        adata = ep.ad.move_to_x(adata, var_name_list)
                        adata.var.loc[var_name_list, 'Unit'] = unit
                        adata.var.loc[var_name_list,'domain_id'] = domain_id
                        adata.var.loc[var_name_list,'concept_class_id'] = concept_class_id
                        adata.var.loc[var_name_list,'concept_code'] = concept_code


        return adata

    # More IO functions           
    def to_dataframe(self, adata, feature, patient, visit):
        df = ak.to_dataframe(adata.obsm['Base Excess|Blood|Blood Gas'])

        # TODO
        # join index (visit_occurrence_id) to df
        # can be viewed as patient level - only select some patient
        

    # More Plot functions
    def plot_timeseries(self,):
        # add one function from previous pipeline
        pass
    
    # More Pre-processing functions
    def sampling(self,):
        # function from dask
        # need to check dask-awkward again 
        pass
