# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 16:09:59 2021

@author: John Meluso
"""

# Import libraries
import pandas as pd

# Import model files
import util.analysis as ua
import util.data as ud
import util.variables as uv


def merge_baseline(params2stats, group_type, group):
    '''Extracts the complete graph baseline from a MultiIndex DataFrame.'''
    
    # Check group_type
    if group_type != 'execset' and group_type != 'model':
        raise RuntimeError(f'Group type {group_type} is not valid.')
    
    # Subset to performance data
    outcome_vars = list(uv.get_outcomes())
    
    # Get subset with only complete stats
    complete = params2stats[outcome_vars].xs('complete',level='team_graph_type')
    
    # Rename remaining ones
    complete = complete.rename(
        columns={
            'count': 'base_count',
            'mean': 'base_mean',
            'std': 'base_std',
            'var': 'base_var'
        })
    
    # Get complete columns and drop graph columns
    cols = [x for x in complete.index.names if 'graph' not in x]
    
    # Subset data to exclude complete since incorporated into others
    graphs = [gr for gr in pd.Categorical(
        params2stats.reset_index()['team_graph_type']
        ).categories if gr != 'complete']
    if group_type == 'execset':
        slice_graph_not_complete = (slice(None), slice(None), graphs)
    elif group_type == 'model':
        slice_graph_not_complete = (slice(None), graphs)
    not_complete = params2stats.loc(axis=0)[slice_graph_not_complete]
    
    # Merge with params2stats on non-graph columns
    params2stats = not_complete.reset_index().merge(
        complete,
        how='left',
        on=cols
        ).set_index(not_complete.index.names)
    
    
    for oc in outcome_vars:
        
        # Calculate difference
        params2stats[oc,'diff_mean'] \
            = params2stats[oc,'mean'] - params2stats[oc,'base_mean']
        
        # Calculate confidence interval
        params2stats[oc, 'diff_ci_lo'], params2stats[oc, 'diff_ci_hi'] \
            = ua.conf_int(
                diff_means=params2stats[oc, 'diff_mean'],
                s1=params2stats[oc, 'std'],
                n1=params2stats[oc, 'count'],
                s2=params2stats[oc, 'base_std'],
                n2=params2stats[oc, 'base_count']
                )
    
    # Pickle stats
    if group_type == 'execset':
        if group == 'test':
            group = 0
        grpstr = f'{group:03}'
    elif group_type == 'model':
        grpstr = group
    loc = f'../data/sets/{group_type}{grpstr}_vs_complete.pickle'
    params2stats.to_pickle(loc)
    
    return params2stats


def merge_baseline_execset(execset):
    '''Merges the graph results for the specified execution set.'''
    filename = f'../data/sets/execset{execset:03}_stats.pickle'
    params2stats = ud.load_pickle(filename)
    params2stats = merge_baseline(params2stats, 'execset', execset)
    return params2stats

def merge_baseline_model(model):
    '''Merges the graph results for the specified model.'''
    filename = f'../data/sets/model{model}_stats.pickle'
    params2stats = ud.load_pickle(filename)
    params2stats = merge_baseline(params2stats, 'model', model)
    return params2stats


# if __name__ == '__main__':
    
#     params2stats = merge_baseline_execset(8)
#     params2stats = merge_baseline_model('3xx')
#     params2stats = merge_baseline_model('3xg')
