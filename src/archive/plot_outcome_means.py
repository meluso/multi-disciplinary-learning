# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 19:13:03 2022

@author: John Meluso
"""

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd

# Import source files
import fig_settings as fs
import merge_by_graph as mg

fs.set_fonts()

def plot_outcome_means(execset=8,team_size=9):
    
    # Merge by graphs for specified team size
    mg.merge_by_graph(execset=execset, team_size=team_size)

    # Load dataframe
    loc = f'../data/sets/execset{execset:03}_by_graph_vs_complete.pickle'
    df = pd.read_pickle(loc)
    
    # oc = 'team_productivity'
    # for mask in [(oc,'diff_mean'),(oc,'diff_ci_lo'),(oc,'diff_ci_hi')]:
    #     df[mask] = df[mask]*25*9
    
    # Define groups
    models = {'3xx': 'Model 1', '3xg': 'Model 2'}
    outcomes = {
        'team_performance': {
            'title': f'Relative Averaged Performance ($n={team_size}$)',
            'indep_label': 'Networks',
            'dep_label': 'Relative Averaged Performance:\n' + \
                'Shown Network - Complete Graph'
            },
        'team_productivity': {
            'title': f'Relative Avg. Agent Productivity ($n={team_size}$)',
            'indep_label': 'Networks',
            'dep_label': 'Relative Avg. Agent Productivity:\n' + \
                'Shown Network - Complete Graph'
            },
        }
    graphs = {
        'complete_na_na_na': 'Complete',
        'empty_na_na_na': 'Empty',
        'power_na_2_0.1': 'Power Law ($m=2$, $p=0.1$)',
        'power_na_2_0.5': 'Power Law ($m=2$, $p=0.5$)',
        'power_na_2_0.9': 'Power Law ($m=2$, $p=0.9$)',
        'random_na_na_0.1': 'Random ($p=0.1$)',
        'random_na_na_0.5': 'Random ($p=0.5$)',
        'random_na_na_0.9': 'Random ($p=0.9$)',
        'ring_cliques_na_na_na': 'Ring of Cliques',
        'rook_na_na_na': 'Rook\'s Graph',
        'small_world_2_na_0.0': 'Small World ($k=2$, $p=0$)',
        'small_world_2_na_0.1': 'Small World ($k=2$, $p=0.1$)',
        'small_world_2_na_0.5': 'Small World ($k=2$, $p=0.5$)',
        'small_world_2_na_0.9': 'Small World ($k=2$, $p=0.9$)',
        'star_na_na_na': 'Star',
        'tree_na_na_na': 'Tree',
        'wheel_na_na_na': 'Wheel',
        'windmill_na_na_na': 'Windmill'
        }

    # Create figure
    cm = 1/2.54  # centimeters in inches
    fig, axs = plt.subplots(nrows=len(models),ncols=len(outcomes),
                            figsize=(8,12), sharex='col', sharey='row')    
    
    for ii, md_key in enumerate(models.keys()):
        for dd, oc_key in enumerate(outcomes.keys()):
            
            # Slice data down to values
            data = df.loc[md_key, oc_key].sort_values(by='diff_mean')
            indep_vars = data.index
            indep_labels = [graphs[key] for key in data.index]
            dep_vals = data.diff_mean
            err_lo = dep_vals - data.diff_ci_lo
            err_hi = data.diff_ci_hi - dep_vals
            
            # Zero out Complete error because it's the frame of reference
            err_lo['complete_na_na_na'] = None
            err_hi['complete_na_na_na'] = None
            
            plot_horizontal_bars(ii, dd, axs, indep_vars, indep_labels, dep_vals,
                            err_lo, err_hi, models, md_key, outcomes, oc_key)
    
    # Show
    plt.tight_layout()
    fs.save_pub_fig(f'relative_ni_productivity_{team_size}_agents')
    plt.show()
    
def get_masks_and_colors(indep_vars, dep_vals, n_masks=2):
    
    if n_masks == 3:
    
        low_edges_mask = (
            (indep_vars == 'empty_na_na_na')
            | (indep_vars == 'random_na_na_0.1')
            )
        
        high_edges_mask = ~low_edges_mask
        
        masks = ((dep_vals <= 0),
            ((dep_vals > 0) & high_edges_mask),
            ((dep_vals > 0) & low_edges_mask)
            )
        colors = ('#808080','#AA0000','#AA8800')
        
    else: # n_masks == 2
        
        masks = ((dep_vals <= 0),(dep_vals > 0))
        colors = ('#808080','#AA0000')
    
    return masks, colors
    
def plot_vertical_bars(ii, dd, axs, indep_vars, indep_labels, dep_vals,
                  err_lo, err_hi, models, md_key, outcomes, oc_key):
    
    # Get column colors
    masks, colors = get_masks_and_colors(indep_vars, dep_vals)
    
    for mask, color in zip(masks, colors):
    
        # Plot on axes
        axs[dd,ii].bar(
            x=indep_vars[mask],
            height=dep_vals[mask],
            yerr=(err_lo[mask], err_hi[mask]),
            capsize=5,
            color=color
            )
        
    
    # Add tick labels
    axs[dd,ii].set_xticks(
        ticks=indep_vars,
        labels=indep_labels,
        rotation=45,
        ha='right'
        )
    
    # Add axis labels
    if ii==0: axs[dd,ii].set_ylabel(outcomes[oc_key]['title'])
    if dd==1: axs[dd,ii].set_xlabel(outcomes[oc_key]['dep_label'])
    if dd==0:
        axs[dd,ii].set_ylabel(outcomes[oc_key]['indep_label'])
        complete_loc = indep_vars.get_loc('complete_na_na_na')
        axs[0,ii].axvline(x=complete_loc, linewidth=2,color='#0F0F0F',ls='--')
        axs[1,ii].axvline(x=complete_loc, linewidth=2,color='#0F0F0F',ls='--')
        
    # Add grid and bold axis
    axs[dd,ii].grid(True)
    axs[dd,ii].axhline(linewidth=2,color='#0F0F0F')
    axs[dd,ii].set_axisbelow(True)
        
def plot_horizontal_bars(ii, dd, axs, indep_vars, indep_labels, dep_vals,
                    err_lo, err_hi, models, md_key, outcomes, oc_key):
    
    # Get column colors
    masks, colors = get_masks_and_colors(indep_vars, dep_vals)
    
    for mask, color in zip(masks, colors):
    
        # Plot on axes
        axs[ii,dd].barh(
            y=indep_vars[mask],
            width=dep_vals[mask],
            xerr=(err_lo[mask], err_hi[mask]),
            capsize=5,
            color=color
            )
        
    axs[ii,dd].grid()
    axs[ii,dd].set_yticks(
        ticks=indep_vars,
        labels=indep_labels,
        ha='right'
        )
    
    if ii==0: axs[ii,dd].set_title(outcomes[oc_key]['title'])
    if ii==1: axs[ii,dd].set_xlabel(outcomes[oc_key]['dep_label'])
    if dd==0:
        axs[ii,dd].set_ylabel(outcomes[oc_key]['indep_label'])
        axs[ii,dd].text(-0.2, 1, models[md_key], fontsize=12, ha='right',
                        va='center', transform=axs[ii,dd].transAxes,
                        weight='bold')
        complete_loc = indep_vars.get_loc('complete_na_na_na')
        axs[ii,0].axhline(y=complete_loc, linewidth=2,color='#0F0F0F',ls='--')
        axs[ii,1].axhline(y=complete_loc, linewidth=2,color='#0F0F0F',ls='--')
    
    
    # Add grid and bold axis
    axs[ii,dd].grid(True)
    axs[ii,dd].axvline(linewidth=2,color='#0F0F0F')
    axs[ii,dd].set_axisbelow(True)



if __name__ == '__main__':
    for n in [4,9,16,25]:
    # for n in [9]:
        plot_outcome_means(team_size=n)