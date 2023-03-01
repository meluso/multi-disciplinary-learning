# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:31:07 2022

@author: John Meluso
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import pandas as pd
import seaborn as sns

import fig_settings as fs

fs.set_fonts()

def load_data(execset=1, model='3xx'):
    
    # Load data
    loc = '../data/sets/'
    filename = f'execset{execset:03}_model{model}_team_is_nbhd.pickle'
    df = pd.read_pickle(loc + filename)
    
    # Get only first 0th step, because all steps are the same
    df = df.loc[df['run_step']==0,:]
    
    # Create x and y
    x = df['team_fn_diff_integral']
    y = df['team_fn_diff_peaks']
    
    return x, y

def get_log_bins(min_val, max_val, bins):
    return np.logspace(np.log10(min_val),np.log10(max_val),bins)

def get_lin_bins(min_val, max_val, bins):
    return np.linspace(min_val,max_val,bins)

def joint_histograms(x, y, ax_integy, ax_integ, ax_peaks, bins):
    
    # Create adjusted colormap
    cmap = sns.color_palette('rocket', as_cmap=True)
    newcmp = mcolors.ListedColormap(cmap(np.linspace(0.05, 0.95, 256)))
    
    # no labels
    ax_integ.tick_params(axis="x", labelbottom=False)
    ax_peaks.tick_params(axis="y", labelleft=False)

    # the 2d histogram:
    hist, xbins, ybins, image = ax_integy.hist2d(
        x, y, bins=bins, cmap=newcmp, density=True, norm=mcolors.LogNorm())
    ax_integy.set_yscale('log')
    ax_integy.set_xlabel('Exploration Difficulty\n(1 - Task Integral)')
    ax_integy.set_ylabel('Exploit. Difficulty\n(Num. Task Peaks)')
    ax_integy.grid(True)
    ax_integy.set_axisbelow(True)

    # the x histogram
    nx, xbins, xpat = ax_integ.hist(x, bins=xbins, edgecolor="white", log=True,
        color='#AA0000', linewidth=0.5)
    ax_integ.set_ylim(bottom=10**4, top=10**6)
    ax_integ.set_ylabel('Num. Tasks')
    ax_integ.grid(True)
    ax_integ.set_axisbelow(True)
    
    ny, ybins, ypat = ax_peaks.hist(y, bins=ybins, edgecolor="white",
        orientation='horizontal', color='#AA0000', linewidth=0.5)
    ax_peaks.set_xlim(left=10**1, right=10**7)
    ax_peaks.set_xscale('log')
    ax_peaks.set_xlabel('Num. Tasks')
    ax_peaks.set_xticks(
        ticks=[10**1, 10**4, 10**7],
        labels=['$10$','$10^4$','$10^7$']
        )
    ax_peaks.grid(True)
    ax_peaks.set_axisbelow(True)
    
    return image
    
def plot_difficulties(subfig, label=True):
    
    # Load data
    x, y = load_data()
    
    # Create figure
    axs = subfig.subplots(
        nrows=2,
        ncols=2,
        sharex='col',
        sharey='row',
        gridspec_kw=dict(
            width_ratios=(7,2), height_ratios=(2,7),
            wspace=0.005, hspace=0.005
            )
        )
    
    if label:
        axs[0,1].text(10**3, 10**5, '(k)', ha='center', va='center', size=8)
        axs[0,1].axis('off')
    else:
        axs[0,1].set_visible(False)
    
    # Create bins
    bins=25
    xbins = get_lin_bins(x.min(), x.max(), bins)
    ybins = get_log_bins(y.min(), y.max(), bins)
    bin_set = (xbins, ybins)
    
    # Create 2d histogram
    image = joint_histograms(x, y, axs[1,0], axs[0,0], axs[1,1], bin_set)
    
    # Set colorbar
    cbar = subfig.colorbar(image, ax=axs, location='bottom',
                           orientation='horizontal')
    cbar.set_label('Fraction of tasks', loc='center', labelpad=0)
    
    return subfig

def test_plot_difficulties():
    
    fig = plt.figure(
        # figsize=fs.fig_size(0.5, 0.3),
        dpi=1200,
        layout='constrained',
        )
    subfigs = fig.subfigures()
    subfigs = plot_difficulties(subfigs)
    fig.show()
    
    
if __name__ == '__main__':
    import plot_tasks_networks_difficulties as ptnd
    ptnd.plot_tasks_networks_difficulties()