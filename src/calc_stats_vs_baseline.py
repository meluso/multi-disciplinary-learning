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


def merge_baseline(params2stats, group_type, group, base_graph):
    '''Extracts graph baseline from a MultiIndex DataFrame.'''
    
    # Check group_type
    if group_type != 'execset' and group_type != 'model':
        raise RuntimeError(f'Group type {group_type} is not valid.')
    
    # Subset to performance data
    outcome_vars = list(uv.get_outcomes())
    
    # Get subset with only baseline graph's stats
    baseline = params2stats[outcome_vars].xs(
        base_graph,
        level='team_graph_type'
        )
    
    # Rename remaining ones
    baseline = baseline.rename(
        columns={
            'count': 'base_count',
            'mean': 'base_mean',
            'std': 'base_std',
            'var': 'base_var'
        })
    
    # Get baseline graph's columns and drop graph columns
    cols = [x for x in baseline.index.names if 'graph' not in x]
    
    # Subset data to exclude baseline graph's since incorporated into others
    graphs = [gr for gr in pd.Categorical(
        params2stats.reset_index()['team_graph_type']
        ).categories if gr != base_graph]
    if group_type == 'execset':
        slice_graph_not_baseline = (slice(None), slice(None), graphs)
    elif group_type == 'model':
        slice_graph_not_baseline = (slice(None), graphs)
    not_baseline = params2stats.loc(axis=0)[slice_graph_not_baseline]
    
    # Merge with params2stats on non-graph columns
    params2stats = not_baseline.reset_index().merge(
        baseline,
        how='left',
        on=cols
        ).set_index(not_baseline.index.names)
    
    
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
    loc = f'../data/sets/{group_type}{grpstr}_vs_{base_graph}.pickle'
    params2stats.to_pickle(loc)
    
    return params2stats


def merge_baseline_execset(execset, base_graph):
    '''Merges the graph results for the specified execution set.'''
    filename = f'../data/sets/execset{execset:03}_stats.pickle'
    params2stats = ud.load_pickle(filename)
    params2stats = merge_baseline(params2stats, 'execset', execset, base_graph)
    return params2stats

def merge_baseline_model(model, base_graph):
    '''Merges the graph results for the specified model.'''
    filename = f'../data/sets/model{model}_stats.pickle'
    params2stats = ud.load_pickle(filename)
    params2stats = merge_baseline(params2stats, 'model', model, base_graph)
    return params2stats


if __name__ == '__main__':
    
    params2stats = merge_baseline_execset(9, base_graph='empty')
#     params2stats = merge_baseline_model('3xx')
#     params2stats = merge_baseline_model('3xg')
