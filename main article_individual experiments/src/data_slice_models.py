# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 17:21:38 2022

@author: John Meluso
"""

# Import libraries
import dask.dataframe as dd
import pandas as pd

# Import src files
from classes.Time import Time
import util.analysis as ua
import util.data as ud
import util.variables as uv


def merge_data(model, partition=1, test=False):
    
    # Start timer
    time = Time()
    time.begin('Model', 'to merge files', f'{model}')
    
    # Get execset and exec_list
    execset, exec_list = ua.get_model_execsets(model)
    
    # Get list of files
    file_list, paramset = ua.get_file_list(exec_list, partition)
    if test: file_list = file_list[300:315]
    
    # Make a unified points dataframe
    points = dd.read_csv(file_list, dtype=uv.get_dtypes(), blocksize=None)
    
    # Combine team function columns
    points['team_fn'] = points['team_fn_type'] + '_' \
        + points['team_fn_weight'] + '_' \
        + points['team_fn_frequency'] + '_' \
        + points['team_fn_exponent']
    
    # Combine neighborhood function columns
    points['nbhd_fn'] = points['nbhd_fn_type'] + '_' \
        + points['nbhd_fn_weight'] + '_' \
        + points['nbhd_fn_frequency'] + '_' \
        + points['nbhd_fn_exponent']
        
    # Cast variables to smaller types
    points = points.astype(dtype=ua.get_dtypes())
    points = points.categorize()
    
    # Build variables to drop (Keeps commented out variables)
    drop_vars = [
      'agent_fn_diff_integral',
      'agent_fn_exponent',
      'agent_fn_frequency',
      'agent_fn_type',
      'agent_fn_weight',
      'agent_steplim',
      'model_type',
      # 'nbhd_fn',
      'nbhd_fn_diff_integral',
      'nbhd_fn_exponent',
      'nbhd_fn_frequency',
      'nbhd_fn_type',
      'nbhd_fn_weight',
      'run_ind',
     # 'run_step',
     # 'team_fn',
      'team_fn_alignment',
      'team_fn_diff_integral',
      'team_fn_exponent',
      'team_fn_frequency',
      'team_fn_interdep',
      'team_fn_type',
      'team_fn_weight',
     # 'team_graph_assortativity',
     # 'team_graph_centrality_betweenness_mean',
     # 'team_graph_centrality_betweenness_stdev',
     # 'team_graph_centrality_degree_mean',
     # 'team_graph_centrality_degree_stdev',
     # 'team_graph_centrality_eigenvector_mean',
     # 'team_graph_centrality_eigenvector_stdev',
     # 'team_graph_clustering',
      'team_graph_density',
     # 'team_graph_diameter',
      'team_graph_k',
      'team_graph_m',
     # 'team_graph_nearest_neighbor_degree_mean',
     # 'team_graph_nearest_neighbor_degree_stdev',
      'team_graph_p',
     # 'team_graph_pathlength',
      'team_graph_type',
     # 'team_performance',
     # 'team_productivity',
     # 'team_size'
     ]

    # Drop variables & convert back to pandas dataframe
    points = points.drop(columns=drop_vars)
    
    # Normalize remaining float columns
    norm_cols = [
        col for col in points.columns \
        if (points[col].dtype == 'float' or points[col].dtype == 'single')
        and (col not in ['run_ind','team_performance','team_productivity'])
        ]
    for col in norm_cols:
        points[col] = normalize(points[col])

    # Compute dataframe
    points = points.compute()
    
    # Pickle stats
    out_loc = ud.get_directory(paramset, partition=partition, output=True)
    filename = out_loc +  f'model{model}.pickle'
    points.to_pickle(filename)
    time.end('Merging files', f'{model}')

def normalize(df_col):
    return (df_col - df_col.mean()) / (df_col.max() - df_col.min())
    
def merge_data_for(model, partition=1, test=False):
    fn = lambda: merge_data(model, partition, test)
    points = ua.run_dask_cluster(fn, 10)
    return points

if __name__ == '__main__':
    merge_data_for('3xx')
    merge_data_for('3xg')
