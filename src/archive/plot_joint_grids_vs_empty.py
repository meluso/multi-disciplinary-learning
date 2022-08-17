# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:20:51 2022

@author: John Meluso
"""

# Import libraries
import itertools as it
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import seaborn as sns

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
    var2slice['run_step'] = 5
    del var2slice['run_ind']
    
    return var2slice

def mask_with_objectives(df):
    
    # Build objective conditions
    mask = (df.team_fn == df.nbhd_fn) \
        # & (df.team_fn_type.str.startswith('kth') != True)
        # | (df.team_fn_type == 'average')
    
    # Get just rows with objectives to display
    df = df[mask]
    
    return df

def get_variable(name, stat='mean', file_version='stats'):
    
    # Get execset
    execset = 7
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
    if stat is not None:
        df = df.droplevel(1,axis=1)
        
    # Build cumulative variables for fns and graphs
    var_prefixes = ['team_graph','team_fn','nbhd_fn','agent_fn']
    for prefix in var_prefixes:
        df = ua.combine_columns(df, prefix)
    
    # Mask objectives
    df = mask_with_objectives(df)
    
    return df

def get_outcomes_and_descriptors():
    
    variables = []
    for out in uv.get_outcomes().keys(): variables.append(out)
    for desc in uv.get_descriptors().keys(): variables.append(desc)
    
    return variables

def get_all_vars():
    
    variables = get_outcomes_and_descriptors()
    stat = 'mean'
    file_version = 'vs_empty'
    df = get_variable(variables, stat, file_version)
    
    return df

def get_diff_means():
    
    # Get execset
    var_name = ['team_performance','team_productivity']
    stat = 'diff_mean'
    file_version = 'vs_empty'
    df = get_variable(var_name, stat, file_version)
    
    return df

def get_eigen_cent():
    
    var_name = 'team_graph_centrality_eigenvector_mean'
    df = get_variable(var_name)
    
    # Group by graphs and return means
    by = 'team_graph'
    eig = df.groupby(by, as_index=True).mean()[var_name].sort_values()
    
    # Order the entries
    eig_list = [ee for ee in eig.index
                if (ee != 'empty') and (ee != 'complete')]
    eig_list.append('complete')
    
    return eig_list

def get_fn_diff():
    
    var_name = 'nbhd_fn_difficulty'
    df = get_variable(var_name)
    
    # Group by graphs and return means
    by = 'nbhd_fn'
    diff = df.groupby(by, as_index=True).mean()[var_name].sort_values()
    
    return diff

def plot_means_on_heatmap1(df, eig, diff):
    
    # Create pivot table for heatmap
    piv = df.pivot(
        index=['model_type','team_fn','nbhd_fn'],
        columns=['team_graph'],
        values='team_performance'
        )
    
    # Creae list of models
    models = ['3xx','3xg']
    team_fn_indices = piv.index.get_level_values('team_fn')
    fns = {
        'Team Fn sams as Neighborhood Fn': team_fn_indices != 'average',
        'Team Fn = Average': team_fn_indices == 'average'
        }
    
    # Create figures and axes
    fig, axs = plt.subplots(len(models), len(fns),
                            sharex=True, sharey=True, figsize=(7,10))
    
    # Create a different heatmap for each model
    xlabels = [ee.replace('_',' ').title() for ee in eig]
    # ylabels = 
    for ii, model in enumerate(models):
        for jj, (title, fn) in enumerate(fns.items()):
            data = piv[fn]
            data = data[eig].xs(model).droplevel(axis=0,level=0)
            jd_col = 'nbhd_fn_difficulty'
            data = data.join(diff).sort_values(jd_col).drop(columns=jd_col)
            sns.heatmap(
                data,
                xticklabels=xlabels,
                ax=axs[ii,jj],
                cmap='RdGy_r',
                center=0,
                #vmin=-1, vmax=1,
                square=True,
                )
    
    return fig

def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[0], columns=args[1], values=args[2])
    sns.heatmap(d, **kwargs)

def plot_means_on_heatmap2(df):
    
    eig = get_eigen_cent()
    diff = get_fn_diff()
    
    # Creae list of models
    df['team_fn_average'] = df.team_fn_type == 'average'
    
    # Sort data by graph type eigenvector centrality and nbhd fn difficulty
    df['team_graph'] \
        = pd.Categorical(df['team_graph'], categories=eig, ordered=True)
    df['nbhd_fn'] \
        = pd.Categorical(df['nbhd_fn'], categories=diff.index, ordered=True)
    df = df.sort_values(['team_graph','nbhd_fn'])
    
    # Create a different heatmap for each model
    g = sns.FacetGrid(df, row='model_type', col='team_fn_average',
                      sharex=True, sharey=True, height=6)
    g.map_dataframe(
        draw_heatmap,
        'nbhd_fn',
        'team_graph_type',
        'team_performance',
        xticklabels=[ee.replace('_',' ').title() for ee in eig],
        yticklabels=list(diff.index),
        cmap='RdGy_r',
        center=0,
        square=True
        )
    g.tight_layout()
    
    return

def plot_means_on_heatmap3(df):
    
    # Create a different heatmap for each model
    g = sns.FacetGrid(df, col='model_type', sharex=True, sharey=True, height=6)
    g.map_dataframe(
        draw_heatmap,
        'nbhd_fn',
        'team_graph',
        'team_performance',
        cmap='RdGy_r',
        center=0,
        square=True
        )
    g.tight_layout()
    
    return

def plot_subset_heatmap():
    
    df = get_diff_means()
    fig = plot_means_on_heatmap3(df)
    return fig
    
def plot_linear_models(df):
    variables = [col for col in df.columns if col not in uv.get_descriptors()]
    df_melt = pd.melt(df, id_vars=variables)
    g = sns.lmplot(
        data=df_melt,
        x='value',
        y='team_performance',
        hue='model_type',
        col='variable',
        col_wrap=4)
    return 

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.grid(which="major", color="w", linestyle='-', linewidth=0)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def plot_joint_grid(model, outcome):
    
    # Get data
    df = get_diff_means()
    df = df[df.model_type == model]
    pivot = df.pivot(
        index='nbhd_fn',
        columns='team_graph',
        values=outcome
        )
    
    # Sort pivot data
    fn_means = pivot.mean(axis=1).sort_values()
    graph_means = pivot.mean(axis=0).sort_values()
    pivot = pivot.reindex(index=fn_means.index, columns=graph_means.index)
    
    # Build figure and axes
    fig, axs = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(16,16),
        gridspec_kw=dict(height_ratios=[1, 4],width_ratios=[3, 1],
                         hspace=0.02, wspace=0))
    axs[0, 1].set_visible(False)
    
    # Get max and min values for plotting
    images_min = [pivot.min(), fn_means.values, graph_means.values]
    images_max = [pivot.max(), fn_means.values, graph_means.values]
    vmin = min(im.min() for im in images_min)
    vmax = max(im.max() for im in images_max)
    my_cmap = plt.get_cmap('inferno')
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    
    # Plot data
    im, cbar = heatmap(pivot, pivot.index, pivot.columns, ax=axs[1,0],
                       vmin=vmin, vmax=vmax, cmap=my_cmap)
    plt.setp(axs[1,0].get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Plot bars
    axs[1, 1].barh(y=fn_means.index, width=fn_means.values,
                   # color=my_cmap(fn_means.values)
                   )
    axs[0, 0].bar(x=graph_means.index, height=graph_means.values,
                  # color=my_cmap(graph_means.values)
                  )
    
    # Move bars
    (x0m, y0m), (x1m, y1m) = axs[1, 0].get_position().get_points()  # main heatmap
    (x0h, y0h), (x1h, y1h) = axs[0, 0].get_position().get_points()  # horizontal histogram
    axs[0, 0].set_position(Bbox([[x0m, y0h], [x1m, y1h]]))
    (x0v, y0v), (x1v, y1v) = axs[1, 1].get_position().get_points()  # vertical histogram
    axs[1, 1].set_position(Bbox([[x0v, y0m], [x1v, y1m]]))
    
    plt.show()
    
    
    return fig

if __name__ == '__main__':
    
    # fig = plot_subset_heatmap()
    # g = plot_linear_models(get_all_vars())
    
    models = ['3xx','3xg']
    outcomes = ['team_performance','team_productivity']
    figs = []
    for model, outcome in it.product(models, outcomes):
        figs.append(plot_joint_grid(model, outcome))
    