# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 18:26:08 2022

@author: John Meluso
"""

# Import libraries
from cycler import cycler
import itertools as it
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

# Import source files
import fig_settings as fs
import util.data as ud
import util.plots as up
import util.variables as uv

All = slice(None)
fs.set_fonts()

def get_data(execset):
    
    # Get execset data
    file_in = f'../data/sets/execset{execset:03}_stats.pickle'
    params2stats = ud.load_pickle(file_in)
    
    # Build row indices
    keeps = [
        'team_fn_diff_integral',
        'team_fn_interdep',
        'team_graph_centrality_eigenvector_mean',
        'team_graph_density'
        ]
    name = keeps
    name.append('team_performance')
    stat = 'mean'
    cols = (name, stat)
    var2slice = get_var2slice()
    
    # Slice dataset down to the following fields:
    # Model | Graph | Team Fn | Nbhd Fn | name
    df = params2stats.loc[tuple(var2slice.values()),cols].reset_index()
    if stat is not None:
        df = df.droplevel(1,axis=1)
        
    # Build cumulative variables for fns and graphs
    var_prefixes = ['team_graph','team_fn']
    for prefix in var_prefixes:
        df = combine_columns(df, prefix, keeps)
    
    # Drop all the variables we don't need
    for pref in var_prefixes: keeps.append(pref)
    df = df[keeps]
    
    # Group by everything but the run step
    groups = df.groupby(['team_graph','team_fn'])
    means = groups.mean().reset_index()
    
    return means

def get_var2slice():
    
    # Build row indices
    var2slice = {key: All for key in uv.get_default_slices().keys()}
    var2slice['model_type'] = ['3xx']
    var2slice['team_size'] = 9
    var2slice['agent_steplim'] = 0.1
    del var2slice['run_ind']
    
    return var2slice

def combine_columns(df, prefix, exclude=[]):
    mask = df.columns.str.startswith(prefix) & ~df.columns.isin(exclude)
    cols_with_prefix = df.columns[mask]
    for col in cols_with_prefix:
        if prefix in df.columns:
            df[prefix] = [x + '_' + str(y) for x, y in zip(df[prefix], df[col])]
        else:
            df[prefix] = df[col]
    return df

def plot_tasks_vs_network():
    
    # Get means of groups
    means = get_data(10)
    
    # Set dims
    xdims = {
        'team_fn_interdep': 'Task: Neighborhood interdependence',
        'team_fn_diff_integral': 'Task: 1 - Task integral'
        }
    ydims = {
        'team_graph_centrality_eigenvector_mean': 'Network: Eigenvector centrality',
        'team_graph_density': 'Network: Density'
        }
    perf = 'team_performance'
    
    for xdim, ydim in it.product(xdims.keys(), ydims.keys()):
    
        # Set dimensions for plot
        xorder = f'ordered by {xdim}'
        yorder = f'ordered by {ydim}'
        
        # Sort by integral and label it
        means = means.sort_values([xdim])
        means[xorder] = pd.Series(np.arange(len(means)), index=means.index)
            
        # Sort by eigencent and label it
        means = means.sort_values([ydim])
        means[yorder] = pd.Series(np.arange(len(means)), index=means.index)
        
        # Create figure
        fig, axs = plt.subplots(1, 2, dpi=1200)
        
        # Create scatterplot
        g1 = sns.scatterplot(
            data=means,
            x=xdim,
            y=perf,
            hue=ydim,
            size=ydim,
            palette='rocket',
            ax=axs[0]
        )
        g1.get_legend().remove()
        g1.set_title('With Values')
        
        # Create scatterplot
        g2 = sns.scatterplot(
            data=means,
            x=xorder,
            y=perf,
            hue=yorder,
            size=yorder,
            palette='rocket',
            ax=axs[1]
        )
        g2.set_title('With Ordering')
        
        # Add legend
        handles, labels = axs[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, title='team_performance')
        g2.get_legend().remove()
        
        # Add title
        xtitle = xdims[xdim]
        ytitle = ydims[ydim]
        fig.suptitle(f'{ytitle} vs {xtitle}')
        
        plt.tight_layout(rect=(0, 0.1, 1, 1))


if __name__ == '__main__':
    
    plot_tasks_vs_network()
    
    
    
    
