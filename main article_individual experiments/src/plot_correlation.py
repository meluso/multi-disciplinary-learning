# -*- coding: utf-8 -*-
"""
Created on Thu May 26 10:07:34 2022

@author: John Meluso
"""

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns

# Import source files
import fig_settings as fs
import util.plots as up


fs.set_fonts()

def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


def plot_correlations(execset=10, model='3xx'):

    # Load data
    file = f'../data/sets/execset{execset:03}_model{model}_team_is_nbhd.pickle'
    df = pd.read_pickle(file)
    
    # Drop rows that aren't step 0, & columns that don't contain graph or fn measures
    mask1 = df.columns.str.startswith('team_graph')
    mask2 = df.columns.str.startswith('team_fn_')
    df = df.loc[df['run_step']==0, (mask1 | mask2)]
    
    # Compute the correlation matrix
    corr = df.corr()
    
    # Cluster correlation matrix
    corr = cluster_corr(corr)
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(
        figsize=fs.fig_size(1,0.7),
        dpi=1200,
        constrained_layout=True
        )
    
    # Metric labels
    labels = up.get_metric_labels()
    xlabels = [labels[ii] for ii in corr.columns]
    ylabels = [labels[ii] for ii in corr.index]
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        ax=ax,
        vmin=-1,
        vmax=1,
        annot=True,
        cmap='RdBu',
        square=True,
        linewidths=.5,
        xticklabels=xlabels,
        yticklabels=ylabels,
        annot_kws={'size': 6},
        cbar_kws={"shrink": .5}
        )
    
    plt.xticks(
        ha='right',
        rotation=45,
        rotation_mode='anchor'
        )
    fs.save_pub_fig('correlation_matrix')
    
    return corr

if __name__ == '__main__':
    corr = plot_correlations()
