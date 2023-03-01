# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:20:51 2022

@author: John Meluso
"""

# Import libraries
import itertools as it
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

# Import source files
import fig_settings as fs
import util.analysis as ua
import util.data as ud
import util.plots as up
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
    if stat is not None:
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
    var_name = ['team_performance','team_productivity']
    stat = 'diff_mean'
    file_version = f'vs_{base_graph}'
    df = get_variable(execset, var_name, stat, file_version)
    
    return df

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
    ax.grid(which="minor", color='0.8', linestyle='-', linewidth=1)
    ax.grid(which="major", color='0.8', linestyle='-', linewidth=0)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def plot_joint_grid(execset, model, base_graph, steplim,
                    outcome='team_performance'):
    
    # Get data
    df = get_diff_means(execset, base_graph)
    df = df[(df.model_type == model) & (df.agent_steplim == steplim)]
    df = df.groupby(['team_graph','team_fn']).mean().reset_index()
    pivot = df.pivot(
        index='team_fn',
        columns='team_graph',
        values=outcome
        )
    
    # Sort pivot data
    fn_stats = pivot.aggregate(['mean','std'],axis=1)
    fn_stats = fn_stats.sort_values(axis=0,by='mean',ascending=False)
    graph_stats = pivot.aggregate(['mean','std'],axis=0)
    graph_stats = graph_stats.sort_values(axis=1,by='mean').transpose()
    pivot = pivot.reindex(index=fn_stats.index, columns=graph_stats.index)
    
    # Build figure and axes
    fig, axs = plt.subplots(
        2, 2, sharex="col", sharey="row",
        figsize=(7.5,9),
        gridspec_kw=dict(
            height_ratios=[1, 4],width_ratios=[4, 1],
            hspace=0.02, wspace=0
            )
        )
    axs[0, 1].set_visible(False)
    
    # Get max and min values for plotting
    images_min = [pivot.min(), fn_stats['mean'], graph_stats['mean']]
    images_max = [pivot.max(), fn_stats['mean'], graph_stats['mean']]
    # vmin = min(im.min() for im in images_min)
    vmin = -0.41
    # vmax = max(im.max() for im in images_max)
    vmax = 0.41
    my_cmap = plt.get_cmap('RdBu')
    
    # Get graph and function labels
    graph_labels = up.get_graph_labels()
    fn_labels = up.get_fn_labels()
    
    # Plot data
    im, cbar = heatmap(
        data=pivot,
        row_labels=[fn_labels[fn] for fn in pivot.index],
        col_labels=[graph_labels[graph] for graph in pivot.columns],
        ax=axs[1,0],
        vmin=vmin,
        vmax=vmax,
        cmap=my_cmap
        )
    plt.setp(axs[1,0].get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.subplots_adjust(left=0.28, bottom=0.2)

    # Plot function bars
    
    # Mask data wrt 0
    fn_masks = ((fn_stats['mean'] > 0),(fn_stats['mean'] <= 0))
    gr_masks = ((graph_stats['mean'] <= 0),(graph_stats['mean'] > 0))
    fn_colors = ('#AA0000','#808080')
    gr_colors = ('#808080','#AA0000')
    
    for mask, color in zip(fn_masks, fn_colors):
        err = fn_stats['std'][mask]
        axs[1, 1].barh(
            y=fn_stats.index[mask],
            width=fn_stats['mean'][mask],
            xerr=err,
            color=color
            )
    axs[1, 1].set_xlabel('Ave. & St. Dev. of Performance Over\nAll Graphs' \
                         + 'vs. Avg. Performance')
    axs[1, 1].grid(True)
    axs[1, 1].axvline(linewidth=2, color='#0F0F0F')
    
    # Plot graph bars
    for mask, color in zip(gr_masks, gr_colors):
        axs[0, 0].bar(
            x=graph_stats.index[mask],
            height=graph_stats['mean'][mask],
            yerr=graph_stats['std'][mask],
            color=color
            )
    axs[0, 0].set_ylabel(
        'Ave. & St. Dev. of\nPerformance Over All Tasks\n' \
            + 'vs. Avg. Performance',
        rotation='horizontal', ha='right'
        )
    axs[0, 0].grid(True)
    axs[0, 0].axhline(linewidth=2, color='#0F0F0F')
    
    # Move bars
    (x0m, y0m), (x1m, y1m) = axs[1, 0].get_position().get_points()  # main heatmap
    (x0h, y0h), (x1h, y1h) = axs[0, 0].get_position().get_points()  # horizontal histogram
    axs[0, 0].set_position(Bbox([[x0m, y0h], [x1m, y1h]]))
    (x0v, y0v), (x1v, y1v) = axs[1, 1].get_position().get_points()  # vertical histogram
    axs[1, 1].set_position(Bbox([[x0v, y0m], [x1v, y1m]]))
    
    # name = f'joint_heatmap_vs_{base_graph}_model{model}_step{steplim}'
    name = f'heatmap_{int(1000*steplim):04}milliunits'
    fs.save_pub_fig(name, dpi=300)

if __name__ == '__main__':
    
    execset = 10
    models = ['3xx']
    base_graph = 'empty'
    steplims = [0.001, 0.01, 0.1, 1]
    # steplims = [0.1]
    for model, steplim in it.product(models, steplims):
        plot_joint_grid(execset, model, base_graph, steplim)
    