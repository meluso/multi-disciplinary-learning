# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:16:46 2021

@author: John Meluso
"""

# Import libraries
import dask.dataframe as dd

# Import model functions
from classes.Time import Time
import util.analysis as ua
import util.variables as uv


All = slice(None)

def describe_execset_by_graph(execset):
    '''Calculates descriptive statistics for the provided dataframe.'''
    
    # Start timer
    time = Time()
    time.begin('Execution Set', 'Describe Points', f'{execset}')
    
    # Get exec_list
    exec_list = ua.get_execset_execs(execset)
    
    # Get list of files
    file_list, paramset = ua.get_file_list(exec_list)
            
    # Make a unified points dataframe and build aggregator fn & graph vars
    points = dd.read_csv(file_list, blocksize=None, dtype=uv.get_dtypes())
    
    # Concatenate graph and function fields
    points['team_graph'] = points['team_graph_type'] + '_' \
        + points['team_graph_k'] + '_' \
        + points['team_graph_m'] + '_' \
        + points['team_graph_p']
    
    points['team_fn'] = points['team_fn_type'] + '_' \
        + points['team_fn_weight'] + '_' \
        + points['team_fn_frequency'] + '_' \
        + points['team_fn_exponent']
        
    points['nbhd_fn'] = points['nbhd_fn_type'] + '_' \
        + points['nbhd_fn_weight'] + '_' \
        + points['nbhd_fn_frequency'] + '_' \
        + points['nbhd_fn_exponent']
    
    # Specify variables to keep for stats
    keep_cols = ['model_type', 'team_size', 'team_graph', 'agent_steplim',
                 'team_performance', 'team_productivity', 'team_graph_density',
                 'team_graph_centrality_eigenvector_mean']
    keep_rows = (points.team_fn == points.nbhd_fn)
        
    # Reduce to just model, size, graph, function, and outcomes
    points = points.loc[keep_rows, keep_cols]
    
    # Groupby index columns and calculate aggregations
    group_vars = ['model_type','team_size','team_graph', 'agent_steplim']
    ops = ['count','mean','std','var']
    groupagg = points.groupby(group_vars).aggregate(ops)
    
    # Convert from dask dataframe to pandas dataframe
    params2stats = groupagg.compute()
    
    # Return stat time
    time.end('Describe Points', f'{execset}')

    # Pickle stats
    if paramset == 'test': paramset = 0
    loc = f'../data/sets/execset{execset:03}_stats_by_graph.pickle'
    params2stats.to_pickle(loc)
    
    return params2stats

def describe_execset(execset, num_workers):
    fn = lambda: describe_execset_by_graph(execset)
    params2stats = ua.run_dask_cluster(fn, num_workers=num_workers)
    return params2stats


if __name__ == '__main__':
    describe_execset(8)
    