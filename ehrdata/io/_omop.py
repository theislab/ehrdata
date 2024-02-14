from typing import List, Union, Literal, Optional
import awkward as ak
import pandas as pd

def from_dataframe(
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

