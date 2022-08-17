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

def plot_outcome_means(team_size=9):
    
    # Merge by graphs for specified team size
    mg.merge_by_graph(team_size=team_size)

    # Load dataframe
    loc = f'../data/sets/execset007_by_graph_vs_complete.pickle'
    df = pd.read_pickle(loc)
    
    # oc = 'team_productivity'
    # for mask in [(oc,'diff_mean'),(oc,'diff_ci_lo'),(oc,'diff_ci_hi')]:
    #     df[mask] = df[mask]*25*9
    
    # Define groups
    models = {'3xx': 'Model 1', '3xg': 'Model 2'}
    outcomes = {
        'team_performance': {
            'title': f'Relative Network Intelligence ($n={team_size}$)',
            'indep_label': 'Networks',
            'dep_label': 'Relative Network Intelligence:\n' + \
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
        'empty_na_na_na': 'Only Self-Edges',
        'power_na_2_0.1': 'Holme-Kim ($m=2$, $p=0.1$)',
        'power_na_2_0.5': 'Holme-Kim ($m=2$, $p=0.5$)',
        'power_na_2_0.9': 'Holme-Kim ($m=2$, $p=0.9$)',
        'random_na_na_0.1': 'Random ($p=0.1$)',
        'random_na_na_0.5': 'Random ($p=0.5$)',
        'random_na_na_0.9': 'Random ($p=0.9$)',
        'ring_cliques_na_na_na': 'Ring of Cliques',
        'rook_na_na_na': 'Rook\'s Graph',
        'small_world_2_na_0.1': 'Watts-Strogatz ($k=2$, $p=0.1$)',
        'small_world_2_na_0.5': 'Watts-Strogatz ($k=2$, $p=0.5$)',
        'small_world_2_na_0.9': 'Watts-Strogatz ($k=2$, $p=0.9$)',
        'star_na_na_na': 'Star',
        'tree_na_na_na': 'Tree',
        'wheel_na_na_na': 'Wheel',
        'windmill_na_na_na': 'Windmill'
        }

    # Create figure
    cm = 1/2.54  # centimeters in inches
    fig, axs = plt.subplots(figsize=(5,6))
    
    md_key = '3xx'
    oc_key = 'team_performance'
            
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
    
    plot_horizontal_bars(axs, indep_vars, indep_labels, dep_vals,
                    err_lo, err_hi, models, md_key, outcomes, oc_key)

    # Show
    plt.tight_layout()
    fs.save_pub_fig(f'Relative NI - Model 1 - {team_size} agents')
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

        
def plot_horizontal_bars(axs, indep_vars, indep_labels, dep_vals,
                    err_lo, err_hi, models, md_key, outcomes, oc_key):
    
    # Get column colors
    masks, colors = get_masks_and_colors(indep_vars, dep_vals)
    
    for mask, color in zip(masks, colors):
    
        # Plot on axes
        axs.barh(
            y=indep_vars[mask],
            width=dep_vals[mask],
            xerr=(err_lo[mask], err_hi[mask]),
            capsize=5,
            color=color
            )
        
    axs.grid()
    axs.set_yticks(
        ticks=indep_vars,
        labels=indep_labels,
        ha='right'
        )
    
    axs.set_title(outcomes[oc_key]['title'])
    axs.set_xlabel(outcomes[oc_key]['dep_label'])
    axs.set_ylabel(outcomes[oc_key]['indep_label'])
    complete_loc = indep_vars.get_loc('complete_na_na_na')
    axs.axhline(y=complete_loc, linewidth=2,color='#0F0F0F',ls='--')
    axs.axhline(y=complete_loc, linewidth=2,color='#0F0F0F',ls='--')
    
    
    # Add grid and bold axis
    axs.grid(True)
    axs.axvline(linewidth=2,color='#0F0F0F')
    axs.set_axisbelow(True)



if __name__ == '__main__':
    plot_outcome_means(team_size=9)