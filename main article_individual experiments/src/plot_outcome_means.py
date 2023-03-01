# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 19:13:03 2022

@author: John Meluso
"""

# Import libraries
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

# Import source files
import fig_settings as fs
import merge_by_graph as mg
import util.plots as up

fs.set_fonts()
All = slice(None)

def plot_outcome_means(execset=10,team_size=9,plot_var='pct_',
                       base_graph='empty',subfig=None,hyperlocal=False):
    
    # Set plot var names
    plot_lo = f'{plot_var}ci_lo'
    plot_hi = f'{plot_var}ci_hi'
    plot_var = f'{plot_var}mean'
    
    # Merge by graphs for specified team size
    df = mg.merge_by_graph(execset=execset, team_size=team_size,
                           base_graph=base_graph)
    df.index = df.index.droplevel()
    df.columns = df.columns.set_names(['variable','statistic'], level=[0,1])
    
    # Define groups
    outcomes = {
        'team_performance': {
            'title': f'Relative Average\nPerformance ($n={team_size}$)',
            'indep_label': 'Networks',
            'dep_label': 'Average Network Performance:\n' + \
                'Percent Diff. From Individual Learning'
            }
        }
    # Colors for reference:
    # Lightest ['#FFAAAA','#FF5555','#FF0000','#AA0000','#550000'] Darkest
    steplims = {
        'Global Search': (' ("exploration") ', 1.0, '#550000', '-'),
        'Intermediate Search': ('', 0.1, '#AA0000', (0, (5, 2))),
        'Local Search': (' ("exploitation") ', 0.01, '#FF5555', '-.')
        }
    if hyperlocal: steplims['Hyper-Local Search'] = ('', 0.001, '#FFAAAA', '-')
    graphs = up.get_graph_labels()
    
    # Make dataframe into pivot table
    df = df.reset_index().pivot(
        index='team_graph',
        columns='agent_steplim',
        )
    
    # Sort by metrics
    sort_list = [
        ('team_graph_density', 'mean', 1),
        ('team_graph_centrality_eigenvector_mean', 'mean', 1)
        ]
    df = df.sort_values(
        by=sort_list,
        ascending=(
            True,
            True
            )
        ).swaplevel(i=2,j=1,axis=1)
    indeces = list(df.index)

    # Create figure if subfigure isn't passed
    if not subfig:
        fig = plt.figure(figsize=fs.fig_size(0.5,0.5,2),dpi=1200)
        ax = fig.gca()
    else:
        ax = subfig.add_subplot(111)
    
    
    for ii, (sl_key, (sl_suffix, sl_value, color, ls)) \
        in enumerate(steplims.items()):
            
        # Slice data down to values
        data = df.loc[:,('team_performance',sl_value,All)] \
            .droplevel(['variable','agent_steplim'],axis=1)
        indep_vars = data.index
        dep_vals = data[plot_var]
        err_lo = dep_vals - data[plot_lo]
        err_hi = data[plot_hi] - dep_vals
        
        # Zero out base_graph error because it's the frame of reference
        err_lo[f'{base_graph}_na_na_na'] = None
        err_hi[f'{base_graph}_na_na_na'] = None
        
        ind = plot_horizontal_layout(
            ax, ii, indep_vars,
            dep_vals, err_lo, err_hi,
            sl_key, sl_suffix, sl_value, color, ls,
            graphs, outcomes, base_graph
            )
        
    # Set ticks
    ax.grid()
    indep_labels = [graphs[key] for key in indep_vars]
    ax.set_yticks(
        ticks=ind,
        labels=indep_labels,
        ha='right'
        )
        
    # Create labels
    ax.set_xlabel(outcomes['team_performance']['dep_label'])
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    base_loc = indep_vars.get_loc(f'{base_graph}_na_na_na')
    ax.axhline(y=base_loc, linewidth=2, color='#222222', ls='--')
    
    # Add grid and bold axis
    handles, labels = ax.get_legend_handles_labels()
    for hh, hand in enumerate(handles):
        handles[hh].has_xerr = False
        handles[hh].has_yerr = True
    if not subfig:
        fig.legend(handles, labels,
                  loc='lower center',
                  bbox_to_anchor=(0.5, 0),
                  # ncol=2,
                  handletextpad=0.5,
                  columnspacing=0.8,
                  )
    else:
        ax.set_title(f'Teams of $n={team_size}$')
    ax.grid(True)
    ax.axvline(linewidth=2,color='#222222')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top)
    ax.set_axisbelow(True)
    fs.set_border(ax)
    ax.tick_params(axis='both', bottom=False, left=False)
    
    
    # Show if fig, return if subfig
    if not subfig:
        plt.tight_layout(rect=(0,0.1,1,1))
        fs.save_pub_fig(f'relative_performance_{team_size}_agents')
        plt.show()
    else:
        return handles, labels

        
def plot_horizontal_layout(ax, ii, indep_vars, dep_vals, err_lo, err_hi,
    sl_key, sl_suffix, sl_value, color, ls, graphs, outcomes, base_graph):
    
    # Label locations and set bar width
    ind = np.arange(len(indep_vars))  # the label locations
        
    
    # Plot on axes
    leg_ent = f'Search radius $={sl_value}${sl_suffix}'
    ax.errorbar(
        y=ind,
        x=dep_vals,
        # height=height,
        xerr=(err_lo, err_hi),
        capsize=3,
        label=leg_ent,
        color=color,
        ls=ls
        )
    
    return ind

def plot_page():
    
    # Team sizes
    nn = [4,9,16,25]
    
    # Create figure
    fig = plt.figure(
        figsize=fs.fig_size(1, 0.9),
        dpi=1200,
        layout='constrained'
        )
    
    # Create subfigures for the plots and legend
    subfigs = fig.subfigures(
        nrows=2,
        ncols=1,
        height_ratios=[15,1]
        )
    plots = subfigs[0]
    legend = subfigs[1]
    
    # Create subfigures for each plot
    sub2figs = plots.subfigures(
        nrows=2,
        ncols=2,
        wspace=0.1,
        hspace=0.05
        )
    
    # Create each subfigure in turn
    for subfig, n in zip(sub2figs.flat, nn):
        
        handles, labels = plot_outcome_means(
            team_size=n,
            subfig=subfig,
            hyperlocal=True
            )
    
    # Add legend
    legend.legend(handles, labels,
                  loc='lower center',
                  bbox_to_anchor=(0.5, 0),
                  ncol=2
                  )
    
    fs.save_pub_fig('relative_performance_all')


if __name__ == '__main__':
    plot_outcome_means(team_size=9)
    plot_page()
