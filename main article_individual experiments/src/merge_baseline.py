# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 16:09:59 2021

@author: John Meluso
"""

# Import libraries
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import model files
from classes.Time import Time
import util.analysis as ua
import util.data as ud
import util.graphs as ug
import util.variables as uv


def get_baseline(group_type, group):
    '''Extracts the empty graph baseline from a MultiIndex DataFrame.'''
    
    # Check group_type
    if group_type == 'execset':
        filename = f'../data/sets/execset{group:03}_stats.pickle'
        empty_slice = (slice(None), slice(None), 'empty')
    elif group_type == 'model':
        filename = f'../data/sets/model{group}_stats.pickle'
        empty_slice = (slice(None), 'empty')
    else:
        raise RuntimeError(f'Group type {group_type} is not valid.')
        
    # Load data
    df = ud.load_pickle(filename)
    
    # Get subset with only empty stats
    
    baseline = df.loc[empty_slice,('team_productivity','mean')]
    baseline = pd.DataFrame(baseline.values, index=baseline.index,
                            columns=['team_productivity_base'])
    
    # Get empty columns and drop graph columns
    columns = [x for x in baseline.index.names if 'graph' not in x]
    
    return baseline.reset_index(), columns

def merge_baseline(group_type, group, test=False):
    
    # Get values specific to each type of subset
    if group_type == 'execset':
        exec_list = ua.get_execset_execs(group)
        group_name = 'Execution Set'
    elif group_type == 'model':
        execset, exec_list = ua.get_model_execsets(group)
        group_name = 'Model'
    else:
        raise RuntimeError(f'Group type {group_type} is not valid.')
    
    # Start timer
    time = Time()
    event = 'Merge Empty'
    time.begin(group_name, event, f'{group}')
    
    # Get the baseline values
    baseline, columns = get_baseline(group_type, group)
    
    # Get list of files
    file_list, paramset = ua.get_file_list(exec_list)
    
    # Reduce list for test
    if test: file_list = file_list[0:5]
    
    # Get graphs other than empty graph
    not_empty_graphs = [gr for gr in ug.set_all_graphs_default().keys()
                        if gr != 'empty']
            
    # Make a unified points dataframe and drop run index
    points = dd.read_csv(file_list, blocksize=None, dtype=uv.get_dtypes())
        
    # Subset data to exclude empty since incorporated into others
    not_empty = points.loc[points['team_graph_type'].isin(not_empty_graphs),:]
    
    # Merge with params2stats on non-graph columns
    midpoints = not_empty.merge(baseline, how='left', on=columns)
    
    # Calculate difference
    midpoints['team_productivity_diff'] \
        = midpoints['team_productivity'] - midpoints['team_productivity_base']
        
    # Correlate data
    output = midpoints.corr().compute()
    
    # Create correlation heatmap
    my_cmap = plt.get_cmap('inferno')
    mask = np.zeros_like(output)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(output, mask=mask, cmap=my_cmap, square=True)
    
    # Return stat time
    time.end(event, f'{group}')
    
    return output

def merge_baseline_3xx(test=False):
    fn = lambda: merge_baseline('model','3xx', test)
    output = ua.run_dask_cluster(fn, num_workers=6)
    return output

def merge_baseline_3xg(test=False):
    fn = lambda: merge_baseline('model','3xg', test)
    output = ua.run_dask_cluster(fn, num_workers=6)
    return output


if __name__ == '__main__':
    # baseline, columns = get_baseline('model', '3xx')
    output = merge_baseline_3xx()
    output = merge_baseline_3xg()
