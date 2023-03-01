# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:48:56 2022

@author: John Meluso
"""

# Import libraries
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import numpy as np
import os
import pandas as pd
import scipy.stats as st

# Import model files
import util.data as ud
import util.variables as uv


def get_file_list(exec_list, partition=1):
    '''Builds list of files to run through dask.'''
    
    # Create list to store dictionaries in
    file_list = []
    
    # Cycle through executions in exec_list
    for paramset in exec_list:
        
        # Get directory prefix & suffix
        dir_prefix = ud.get_directory(paramset, partition)
        dir_suffix1 = '.csv'
        dir_suffix2 = '.xz'
        
        # Get directory without slash
        directory = dir_prefix[:-1]
    
        # Loop through files in directory
        for item in os.listdir(directory):
            
            if os.path.isfile(os.path.join(dir_prefix, item)) \
                and (item.endswith(dir_suffix1) or item.endswith(dir_suffix2)):
            
                # Append points dict to list
                file_list.append(dir_prefix + item)
                
    return file_list, paramset

def run_dask_cluster(fn, num_workers=4):
    
    # Create progress bar
    pbar = ProgressBar()
    pbar.register()
    
    # Create cluster
    with LocalCluster(n_workers=num_workers) as cluster, \
        Client(cluster) as client:
            
        # Display cluster information
        print(cluster)
        
        df = fn()
        
    # Deregister progress bar
    pbar.unregister()
            
    return df

def create_executions_dicts():
    
    execset2execs = {
        1: dict(
            execs = [1],
            models = ['3xx']
            )
        }
    
    model2simprop = {
        '3xx': dict(
            execset = 1,
            exec_list = [1],
            group_vars = uv.team_and_nbhd_indices(),
            drop_list = uv.not_team_or_nbhd_indices()
            )
        }
    
    execset2model2execlist = {
        1: {
            '3xx': [1]
            }
        }
    
    return execset2execs, model2simprop, execset2model2execlist

def get_dtypes():
    
    dtypes = {
        'agent_fn_diff_integral': 'single',
        'agent_fn_diff_peaks': 'float',
        # 'agent_fn_exponent': 'category',
        # 'agent_fn_frequency': 'category',
        # 'agent_fn_type': 'category',
        # 'agent_fn_weight': 'category',
        'agent_steplim': 'single',
        # 'model_type': 'category',
        # 'nbhd_fn': 'category',
        'nbhd_fn_diff_integral': 'single',
        'nbhd_fn_diff_peaks': 'float',
        # 'nbhd_fn_exponent': 'category',
        # 'nbhd_fn_frequency': 'category',
        # 'nbhd_fn_type': 'category',
        # 'nbhd_fn_weight': 'category',
        'run_ind': 'uint8',
        'run_step': 'uint8',
        # 'team_fn': 'category',
        'team_fn_alignment': 'single',
        'team_fn_diff_integral': 'single',
        'team_fn_diff_peaks': 'float',
        # 'team_fn_exponent': 'category',
        # 'team_fn_frequency': 'category',
        'team_fn_interdep': 'single',
        # 'team_fn_type': 'category',
        # 'team_fn_weight': 'category',
        'team_graph_assortativity': 'single',
        'team_graph_centrality_betweenness_mean': 'single',
        'team_graph_centrality_betweenness_stdev': 'single',
        'team_graph_centrality_degree_mean': 'single',
        'team_graph_centrality_degree_stdev': 'single',
        'team_graph_centrality_eigenvector_mean': 'single',
        'team_graph_centrality_eigenvector_stdev': 'single',
        'team_graph_clustering': 'single',
        'team_graph_density': 'single',
        'team_graph_diameter': 'single',
        # 'team_graph_k': 'category',
        # 'team_graph_m': 'category',
        'team_graph_nearest_neighbor_degree_mean': 'single',
        'team_graph_nearest_neighbor_degree_stdev': 'single',
        # 'team_graph_p': 'category',
        'team_graph_pathlength': 'single',
        # 'team_graph_type': 'category',
        'team_performance': 'single',
        'team_productivity': 'single',
        'team_size': 'uint8',
    }

    return dtypes

def get_graph_vars(connected=True):
    
    if connected:
        graph_vars = [
            'team_graph_centrality_degree_mean',
            'team_graph_centrality_degree_stdev',
            'team_graph_centrality_eigenvector_mean',
            'team_graph_centrality_eigenvector_stdev',
            'team_graph_centrality_betweenness_mean',
            'team_graph_centrality_betweenness_stdev',
            'team_graph_nearest_neighbor_degree_mean',
            'team_graph_nearest_neighbor_degree_stdev',
            'team_graph_clustering',
            'team_graph_assortativity',
            'team_graph_pathlength',
            'team_graph_diameter'
            ]
    else:
        graph_vars = [
            'team_graph_centrality_degree_mean',
            'team_graph_centrality_degree_stdev',
            'team_graph_centrality_eigenvector_mean',
            'team_graph_centrality_eigenvector_stdev',
            'team_graph_centrality_betweenness_mean',
            'team_graph_centrality_betweenness_stdev',
            'team_graph_nearest_neighbor_degree_mean',
            'team_graph_nearest_neighbor_degree_stdev',
            'team_graph_clustering',
            # 'team_graph_assortativity',  ##
            # 'team_graph_pathlength',      ## Not valid for unconnected graphs
            # 'team_graph_diameter'        ##
            ]
    return graph_vars

def get_execset_execs(execset):
    '''Get the executions corresponding to a complete execution set.'''
    return execset2execs[execset]['execs']

def get_execset_models(execset):
    '''Get the executions corresponding to a complete execution set.'''
    return execset2execs[execset]['models']

def get_model_execsets(model):
    '''Get the execset and exec_list that contain data on a model.'''
    return model2simprop[model]['execset'], model2simprop[model]['exec_list']

def get_execset_model_execs(execset, model):
    '''Get the execution list corresponding to an execution set and model.'''
    return execset2model2execlist[execset][model]

def get_group_vars(model):
    '''Returns variables to group by for the specified model.'''
    return model2simprop[model]['group_vars']

def get_drop_vars(model):
    '''Returns variables not relevant for the specified model.'''
    return model2simprop[model]['drop_list']

def load_execset(execset, dataset):
    file_in = f'../data/sets/execset{execset:03}_{dataset}.pickle'
    params2stats = ud.load_pickle(file_in)
    return params2stats

def build_aggregator_vars(df):
    var_prefixes = ['team_graph','team_fn','nbhd_fn','agent_fn']
    for prefix in var_prefixes: df = combine_columns(df, prefix)
    return df

def combine_columns(df, prefix):
    cols_with_prefix = df.columns[df.columns.str.startswith(prefix)]
    for col in cols_with_prefix:
        if prefix in df.columns:
            df[prefix] = df[prefix] + '_' + df[col].map(str)
        else:
            df[prefix] = df[col]
    return df

def conf_int(diff_means, s1, n1, s2, n2, alpha=0.05):
    '''Calculate lower and upper confidence interval limits.'''
    
    # degrees of freedom
    df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))  
    
    # t-critical value for CI
    t = st.t.ppf(1 - alpha/2, df)
    
    # Range of difference of means

    lower = diff_means - t * np.sqrt(s1**2 / n1 + s2**2 / n2)
    upper = diff_means + t * np.sqrt(s1**2 / n1 + s2**2 / n2)
    
    return lower, upper

execset2execs, model2simprop, execset2model2execlist \
    = create_executions_dicts()