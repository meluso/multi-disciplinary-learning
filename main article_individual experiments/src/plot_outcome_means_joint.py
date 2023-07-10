# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 19:13:03 2022

@author: John Meluso
"""

# Import libraries
import matplotlib.pyplot as plt

# Import source files
import fig_settings as fs
import plot_outcome_means_networks as pomn
import plot_outcome_means_scatter as poms
import util.plots as up

fs.set_fonts()
All = slice(None)

def plot_outcome_means_joint(execset=10,team_size=9,plot_var='pct_',
                             base_graph='empty'):
    
    # Create figure and axes
    fig = plt.figure(
        figsize=fs.fig_size(1, 0.35),
        dpi=1200,
        layout='constrained',
        )
    subfigs = fig.subfigures(nrows=1, ncols=2, wspace=0.08, width_ratios=[1,1])
    
    # Plot networks on y, performance on x
    handles, labels = pomn.plot_outcome_means(
        subfig=subfigs[0],
        title_type='paper_main'
        )
    
    # Plot density on x, performance on y
    handles, labels = poms.plot_outcome_means(
        subfig=subfigs[1],
        title_type='paper_main'
        )
        
    # Add legend
    for hh, hand in enumerate(handles):
        handles[hh].has_xerr = False
        handles[hh].has_yerr = True
    fig.legend(handles, labels,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.08),
               ncol=3,
               handletextpad=0.5
               )
    
    # Save figure
    fs.save_pub_fig(
        f'relative_performance_{team_size}_agents_joint',
        bbox_inches='tight'
        )
    
    # Show figure
    plt.show()


if __name__ == '__main__':
    plot_outcome_means_joint()
