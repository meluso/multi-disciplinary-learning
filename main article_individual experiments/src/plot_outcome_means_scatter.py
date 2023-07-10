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

def plot_outcome_means(execset=10, team_size=9, plot_var='pct_',
                       base_graph='empty', subfig=None, hyperlocal=False,
                       title_type=None):
    
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
            'indep_label': 'Network Density',
            'dep_label': 'Average Network Performance:\n' + \
                'Percent Diff. From Individual Learning'                    
            }
        }
    # Colors for reference:
    # Lightest ['#FFAAAA','#FF5555','#FF0000','#AA0000','#550000'] Darkest
    steplims = {
        'Global Search': (' ("exploration") ', 1.0, '#550000', '#AA7F7F'),
        'Intermediate Search': ('', 0.1, '#AA0000', '#D47F7F'),
        'Local Search': (' ("exploitation") ', 0.01, '#FF5555', '#FFAAAA')
        }
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

    # Create figure if subfigure isn't passed
    if not subfig:
        fig = plt.figure(figsize=fs.fig_size(0.5,0.5,2),dpi=1200)
        ax = fig.gca()
    else:
        ax = subfig.add_subplot(111)
    
    # Set title
    if title_type == 'paper_main':
        title = '(b) Performance by Network Density'
    elif title_type == 'paper_supplement':
        title = f'Teams of $n={team_size}$'
    else: # No title provided
        title = None
    
    
    for sl_key, (sl_suffix, sl_value, color1, color2) in steplims.items():
            
        # Slice data down to values
        data = df.loc[:,('team_performance',sl_value,All)] \
            .droplevel(['variable','agent_steplim'],axis=1)
        indep_vars = df.loc[:,('team_graph_density',sl_value,'mean')]
        dep_vals = data[plot_var]
        err_lo = dep_vals - data[plot_lo]
        err_hi = data[plot_hi] - dep_vals
        
        # Zero out base_graph error because it's the frame of reference
        err_lo[f'{base_graph}_na_na_na'] = None
        err_hi[f'{base_graph}_na_na_na'] = None
        
        ind = plot_vertical_layout(
            ax, indep_vars,
            dep_vals, err_lo, err_hi,
            sl_key, sl_suffix, sl_value, color1, color2,
            graphs, outcomes, base_graph
            )
        
    # Set ticks
    ax.grid()
        
    # Create labels
    ax.set_xlabel(outcomes['team_performance']['indep_label'])
    ax.set_ylabel(outcomes['team_performance']['dep_label'])
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    base_loc = 0
    ax.axvline(x=base_loc, linewidth=2, color='#222222', ls='--')
    
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
        ax.set_title(title)
    ax.grid(True)
    ax.axhline(linewidth=2,color='#222222')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top)
    ax.set_axisbelow(True)
    fs.set_border(ax)
    ax.tick_params(axis='both', bottom=False, left=False)
    
    # Add 'performs better' arrow
    # xpos, ypos, length = -0.18, 0.955, 0.125
    xpos, ypos, length = 0.14, 0.92, 0.125
    arrow = up.arrow(ax, (xpos, ypos-length), (xpos, ypos))
    text = ax.text(x=xpos, y=ypos, s='Performs\nbetter',
            size=8, clip_on=False, va='bottom', ha='center', color='#666666',
            transform=ax.transAxes)
    arrow.set_in_layout(False)
    text.set_in_layout(False)
    
    # Add 'more dense' arrow
    up.arrow(ax, (0.7, 0.4), (0.85, 0.4))
    ax.text(x=0.775, y=0.42, s='More  dense',
            size=8, clip_on=False, va='bottom', ha='center', color='#666666',
            transform=ax.transAxes)
    
    
    # Show if fig, return if subfig
    if not subfig:
        plt.tight_layout(rect=(0,0.1,1,1))
        fs.save_pub_fig(f'relative_performance_{team_size}_agents_scatter')
        plt.show()
    else:
        return handles, labels

        
def plot_vertical_layout(ax, indep_vars, dep_vals, err_lo, err_hi, sl_key,
    sl_suffix, sl_value, color1, color2, graphs, outcomes, base_graph):
    
    # Label locations and set bar width
    ind = np.arange(len(indep_vars))  # the label locations

    # Calculate regression coefficients
    z = np.polyfit(x=indep_vars, y=dep_vals, deg=1)
    reg_x = np.linspace(0,1,100)
    reg_y = np.polyval(z, reg_x)
    
    # Plot regression line
    ax.plot(reg_x,reg_y,color=color2,zorder=2.4)        
    
    # Plot on axes
    leg_ent = f'Search radius $={sl_value}${sl_suffix}'
    ax.errorbar(
        x=indep_vars,
        y=dep_vals,
        yerr=(err_lo, err_hi),
        capsize=2,
        label=leg_ent,
        color=color1,
        ecolor=color2,
        ls='',
        marker='.',
        ms=5,
        elinewidth=1.5,
        zorder=2.5
        )
    
    return ind


if __name__ == '__main__':
    plot_outcome_means(team_size=9)
    # plot_page()
    # plot_outcomes_by_steplim()
