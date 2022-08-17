# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:35:35 2022

@author: jam
"""

# Import libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import source files
import fig_settings as fs
import plot_difficulties as pldiff
import plot_example_tasks_networks as pletn

def plot_tasks_networks_difficulties():
    
    fs.set_fonts(extra_params={
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'cm',
        'legend.fontsize': 8,
        'axes.labelsize': 6,
        'axes.titlesize': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'figure.titlesize': 6
        })
    
    # Create figure and subfigures
    fig = plt.figure(
        figsize=fs.fig_size(1, 0.25),
        dpi=1200,
        layout='constrained',
        )
    subfigs = fig.subfigures(nrows=1, ncols=2, wspace=0.03, width_ratios=[3,1])
    
    # Plot tasks and networks
    subfigs[0] = pletn.plot_examples(subfigs[0])
    
    # Plot difficulty distributions
    subfigs[1] = pldiff.plot_difficulties(subfigs[1])
    
    # Save figure
    fs.save_pub_fig('example_tasks_networks', bbox_inches='tight')
    
    # Show figure
    plt.show()

if __name__ == '__main__':
    plot_tasks_networks_difficulties()
