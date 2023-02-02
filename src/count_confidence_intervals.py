# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:20:51 2022

@author: John Meluso
"""

# Import libraries
import itertools as it
import pandas as pd

# Import source files
import fig_settings as fs
import util.analysis as ua
import util.data as ud
import util.variables as uv

fs.set_fonts()
All = slice(None)

def get_var2slice():
    
    # Build row indices
    var2slice = {key: All for key in uv.get_default_slices().keys()}
    var2slice['model_type'] = ['3xx','3xg']
    var2slice['team_size'] = 9
    del var2slice['run_ind']
    
    return var2slice

def mask_with_objectives(df):
    
    # Build objective conditions
    mask = (df.team_fn == df.nbhd_fn) \
        # & (df.run_step <= 12)
    
    # Get just rows with objectives to display
    df = df[mask]
    
    return df

def get_variable(execset, name, stat='mean', file_version='stats'):
    
    # Get execset
    file_in = f'../data/sets/execset{execset:03}_{file_version}.pickle'
    params2stats = ud.load_pickle(file_in)
    
    # Build column indices
    if stat is not None:
        cols = (name, stat)
    else:
        cols = (name)
    
    # Build row indices
    var2slice = get_var2slice()
    
    # Slice dataset down to the following fields:
    # Model | Graph | Team Fn | Nbhd Fn | name
    df = params2stats.loc[tuple(var2slice.values()),cols].reset_index()
    new_cols = []
    for (lvl1, lvl2) in df.columns:
        if lvl1 == 'team_performance':
            new_cols.append((f'{lvl1}_{lvl2}',''))
        else:
            new_cols.append((lvl1,lvl2))
    df.columns = pd.MultiIndex.from_tuples(new_cols)
    df = df.droplevel(1,axis=1)
        
    # Build cumulative variables for fns and graphs
    var_prefixes = ['team_graph','team_fn','nbhd_fn','agent_fn']
    for prefix in var_prefixes:
        df = ua.combine_columns(df, prefix)
    
    # Mask objectives
    df = mask_with_objectives(df)
    
    return df

def get_diff_means(execset, base_graph):
    
    # Get execset
    var_name = ['team_performance']
    stat = ['diff_mean','diff_std','diff_ci_lo','diff_ci_hi']
    file_version = f'vs_{base_graph}'
    df = get_variable(execset, var_name, stat, file_version)
    
    return df

def count_confidence_intervals(execset, model, base_graph, steplim,
                                outcome='team_performance_diff_ci_lo'):
    
    # Get data
    df = get_diff_means(execset, base_graph)
    df = df[(df.model_type == model) & (df.agent_steplim == steplim)]
    df = df.groupby(['team_graph','team_fn']).mean().reset_index()
    
    # Create pivot table for low and high confidence intervals
    pivot_lo = df.pivot(
        index='team_fn',
        columns='team_graph',
        values='team_performance_diff_ci_lo'
        )
    pivot_hi = df.pivot(
        index='team_fn',
        columns='team_graph',
        values='team_performance_diff_ci_hi'
        )
    
    # Count networks and tasks
    num_tasks = pivot_lo.index.size
    num_networks = pivot_lo.columns.size
    
    # Count networks with conf ints greater than and less than zero
    greater_than_zero = (pivot_lo > 0).sum(axis=1)
    count_above = (greater_than_zero == num_networks).sum()
    less_than_zero = (pivot_hi < 0).sum(axis=1)
    count_below = (less_than_zero == num_networks).sum()
    within_int = ((pivot_lo <= 0) & (pivot_hi >= 0)).sum(axis=1)
    count_in_ci = (within_int == num_networks).sum()
    
    # Return the number of tasks with ALL conf ints greater/less than zero
    print('')
    print(f'\tWithin Confidence Interval: {count_in_ci} ({count_in_ci/num_tasks})')
    print(f'\tAll Conf Ints Above: {count_above} ({count_above/num_tasks})')
    print(f'\tAll Conf Ints Below: {count_below} ({count_below/num_tasks})')
    print('')
    
    return pivot_lo, pivot_hi, greater_than_zero, less_than_zero
    

if __name__ == '__main__':
    
    execset = 10
    models = ['3xx']
    base_graph = 'empty'
    steplims = [0.1]
    for model, steplim in it.product(models, steplims):
        pivot_lo, pivot_hi, gtz, ltz = \
            count_confidence_intervals(execset, model, base_graph, steplim)
    
