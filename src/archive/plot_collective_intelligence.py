# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:20:51 2022

@author: John Meluso
"""

# Import libraries
from itertools import product
import seaborn as sns

# Import source files
import util.analysis as ua
import util.data as ud
import util.variables as uv

All = slice(None)

#%% Supporting Functions

def get_data(model, size, outcome, dataset):
    
    # Select stat
    if dataset == 'stats':
        stat = 'mean'
    elif dataset == 'vs_empty':
        stat = 'diff_mean'
        
    params2stats = load_execset_7(dataset)
    cols, var2slice = build_row_indices(outcome, stat, model, size)
    df = slice_to_core_props(params2stats, var2slice, cols)
    df = ua.build_aggregator_vars(df)
    df = mask_with_objectives(df)
    
    return df

def load_execset_7(dataset):
    execset = 7
    file_in = f'../data/sets/execset{execset:03}_{dataset}.pickle'
    params2stats = ud.load_pickle(file_in)
    return params2stats

def build_row_indices(outcome, stat, model, size):
    cols = (outcome, stat)
    var2slice = tuple(get_var2slice(model, size).values())
    return cols, var2slice

def slice_to_core_props(params2stats, var2slice, cols):
    
    # Slice dataset down to the following fields:
    # Model | Graph | Team Fn | Nbhd Fn | name
    df = params2stats.loc[var2slice,cols].reset_index()
    df = df.droplevel(1,axis=1)
    return df

def get_group_vars(for_combined_figure=False, lines=False):
    if for_combined_figure:
        group_vars = ['model_type','team_size','team_graph']
    else:
        group_vars = ['team_graph']
    if lines:
        group_vars.append('run_step')
    return group_vars

def get_lines(df, outcome, for_combined_figure=False):
    
    # Drop all the variables we don't need
    group_vars = get_group_vars(for_combined_figure, True)
    vars_to_get = group_vars
    vars_to_get.append('team_fn')
    vars_to_get.append(outcome)
    df = df[vars_to_get]
    
    # Group data by graph and step
    groups = df.groupby(by=group_vars)
    output = groups.mean().reset_index()
    
    return output

def get_bars(df, outcome, for_combined_figure=False):
    
    # Drop all the variables we don't need
    group_vars = get_group_vars(for_combined_figure)
    vars_to_get = group_vars
    vars_to_get.append('team_fn')
    vars_to_get.append(outcome)
    df = df[vars_to_get]
    
    # Group data by graph and step
    groups = df.groupby(by=group_vars)
    output = groups.mean().reset_index()
    
    return output

def get_var2slice(model, size):
    
    # Build row indices
    var2slice = {key: All for key in uv.get_default_slices().keys()}
    var2slice['team_size'] = size
    var2slice['model_type'] = model
    del var2slice['run_ind']
    
    return var2slice

def mask_with_objectives(df):
    
    # Build objective conditions
    mask = (df.team_fn == df.nbhd_fn)
    
    # Get just rows with objectives to display
    df = df[mask]
    
    return df

def combine_columns(df, prefix):
    cols_with_prefix = df.columns[df.columns.str.startswith(prefix)]
    for col in cols_with_prefix:
        if prefix in df.columns:
            df[prefix] = [x + '_' + str(y) for x, y in zip(df[prefix], df[col])]
        else:
            df[prefix] = df[col]
    return df

#%% Plotting Functions

def plot_collective_intelligence_lines(model='3xg', size=9, dataset='vs_empty',
           outcome='team_performance', for_combined_figure=False):
    
    # Get data and unique values
    df = get_data(model, size, outcome, dataset)
    df = get_lines(df, outcome, for_combined_figure)
    
    # Initial plot
    if for_combined_figure:
        sns.relplot(kind='line', data=df, x='run_step', y=outcome, ci=None,
                    hue='team_graph', row='model_type', col='team_size')
    else:
        sns.relplot(kind='line', data=df, x='run_step', y=outcome,
                    hue='team_graph', ci=None)
    
def plot_collective_intelligence_bars(model='3xg', size=9, dataset='vs_empty',
          outcome='team_productivity', for_combined_figure=False):
    
    # Get data and unique values
    df = get_data(model, size, outcome, dataset)
    df = get_bars(df, outcome, for_combined_figure)
    
    # Initial plot
    if for_combined_figure:
        sns.catplot(kind='bar', data=df, x='team_graph', y=outcome,
                    row='model_type', col='team_size', ci=None)
    else:
        sns.catplot(kind='bar', data=df, x='team_graph', y=outcome, ci=None)
    
def plot_ci_separate_figures():
    
    models = ['3xx','3xg']
    sizes = [4,9,16,25]
    
    for model, size in product(models, sizes):
        plot_collective_intelligence_lines(model, size)
        plot_collective_intelligence_bars(model, size)
        
def plot_ci_single_figure():
    
    # Get all models and sizes
    model = All
    size = All
    plot_collective_intelligence_lines(model, size, 'stats', for_combined_figure=True)
    plot_collective_intelligence_bars(model, size, for_combined_figure=True)
    

#%% Call Plotter

if __name__ == '__main__':
    
    # plot_ci_separate_figures()
    plot_ci_single_figure()
    
    