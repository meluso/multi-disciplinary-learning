# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:51:31 2022

@author: jam
"""

# Import libraries
import itertools as it
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
import numpy as np
import pandas as pd

# Import source files
import fig_settings as fs
import util.plots as up

fs.set_fonts()

def load_one_regression(reg):
        
    # Load slim results for each regression
    fname = f'../data/analysis/reg_slim_{reg:02}.pickle'
    results = pd.read_pickle(fname)
    
    # Create list of stats
    stats = ['params','HC2_se','pvalues']
    
    # Combine stats into a dataframe
    results['df'] = pd.concat(
        [results['params'], results['HC2_se'], results['pvalues']],
        axis=1
        )
    results['df'].columns = ['params','HC2_se','pvalues']
    
    for stat in stats:
        
        # Remove dataframed stats
        results.pop(stat)
        
        # Remove insignificant values
        results['df'][stat].loc[results['df']['pvalues']>0.001] = 0
    
    # Build masks for data selection
    mask = {}
    mask['ntwk'] = results['df'].index.str.contains(
        'team_graph',regex=False)
    mask['task'] = results['df'].index.str.contains(
        'team_fn_',regex=False)
    mask['main'] = ~results['df'].index.str.contains(
        'log10(agent_steplim)',regex=False)
    mask['intx'] = results['df'].index.str.contains(
        'log10(agent_steplim):',regex=False)
    
    # Get main and interaction effects for each group
    for group in ['ntwk','task']:
        
        # Define variable names
        main = f'{group}_main'
        intx = f'{group}_intx'
    
        # Get main and interaction data for networks
        results[main] = results['df'].loc[(mask[group] & mask['main']), :]
        results[intx] = results['df'].loc[(mask[group] & mask['intx']), :]
        
        # Update the interaction effect indeces
        results[intx].set_index(results[intx].index.str.replace(
            'log10(agent_steplim):','',regex=False),
            inplace=True
            )
        
        # Sort effects
        effect_sum = results[main]['params'] + results[intx]['params']
        effect_sum = effect_sum.sort_values(ascending=False)
        sort_order = effect_sum.sort_values(ascending=False).index
        results[main] = results[main].reindex(sort_order)
        results[intx] = results[intx].reindex(sort_order)
        
        # Drop effects smaller than 0.005
        mask_size = abs(effect_sum) > 0.005
        results[main] = results[main].loc[mask_size, :]
        results[intx] = results[intx].loc[mask_size, :]
        
        # Build interaction anchor
        results[intx]['anchor'] = results[main]['params']
        prod = results[main]['params']*results[intx]['params']
        results[intx]['anchor'][prod < 0] = 0
    
    return results

def load_regressions(regs_to_load=range(12), networks=False, tasks=False):
    
    regs = {}
    for index in regs_to_load:
        
        # Load slim results for each regression
        fname = f'../data/analysis/reg_slim_{index:02}.pickle'
        regs[index] = pd.read_pickle(fname)
        
        # Remove any coefficients and errors that aren't significant
        regs[index]['params'].loc[regs[index]['pvalues']>0.001] = 0
        regs[index]['HC2_se'].loc[regs[index]['pvalues']>0.001] = 0
    
    # Create lists
    df_stats2vals = {'params':[], 'HC2_se':[], 'pvalues':[]}
    
    # Get parameters, standard errors of the means, and p-values for regressions
    for (key, reg), stat in it.product(regs.items(), df_stats2vals.keys()):
        name = reg['name'].replace('\n',' ')
        df_stats2vals[stat].append(reg[stat].rename(name))
    
    # Build dataframes
    for ii, stat in enumerate(df_stats2vals.keys()):
        df_stats2vals[stat] = pd.concat(df_stats2vals[stat], axis=1)
        
        # Build masks
        mask_ntwk = df_stats2vals[stat].index.str.contains('team_graph')
        mask_task = df_stats2vals[stat].index.str.contains('team_fn_')
        mask_intx = df_stats2vals[stat].index.str.contains('agent_steplim')
        
        # Get requested data
        if networks and tasks:
            mask = mask_ntwk | mask_task | mask_intx
        elif networks:
            mask = mask_ntwk | mask_intx
        elif tasks:
            mask = mask_task | mask_intx
        else:
            raise RuntimeError('Function inputs invalid.')
        
        df_stats2vals[stat] = df_stats2vals[stat][(mask)]
        
    # Sort parameters
    df_stats2vals['params']['mean'] = df_stats2vals['params'].mean(axis=1)
    df_stats2vals['params'] = df_stats2vals['params'].sort_values(
        'mean', ascending=False)
    
    # Get sorting order
    order = pd.Categorical(df_stats2vals['params'].index)
    
    # reset indeces and sort by order
    for stat in df_stats2vals.keys():
        df_stats2vals[stat] = df_stats2vals[stat].reset_index()
        df_stats2vals[stat]['index'] \
            = pd.Categorical(df_stats2vals[stat]['index'], categories=order)
        df_stats2vals[stat].sort_values('index')
        df_stats2vals[stat] = df_stats2vals[stat].set_index('index')
        
    # drop the mean column
    df_stats2vals['params'].pop('mean')
    
    # Add the r-squared and f p-values
    pt_stats2vals = {'rsquared_adj': [], 'f_pvalue': [], 'num_obs': []}
    for (key, reg), stat in it.product(regs.items(), pt_stats2vals.keys()):
        pt_stats2vals[stat].append(reg[stat])
    
    return df_stats2vals, pt_stats2vals

def plot_all_regression_coefficients():
    
    # Load data for all regressions with interaction terms
    df_stats2vals, pt_stats2vals = load_regressions(
        [3,4,5,9,10,11],
        networks=True, tasks=True
        )
        
    # Plot prep
    fig = plt.figure(figsize=(5,10), dpi=1200)
    ax = fig.gca()
        
    # Plot the data with pandas
    df_stats2vals['params'].plot.barh(
        ax=ax,
        xerr=df_stats2vals['HC2_se'],
        width=0.8
        )
    handles, labels = ax.get_legend_handles_labels()
    labels = [
        'Connected, Fn Metrics',
        'Connected, Fn Fixed Effects',
        'Connected, Metrics & FEs',
        'All Networks, Fn Metrics',
        'All Networks, Fn Fixed Effects',
        'All Networks, Metrics & FEs',
        'Mean',
        ]
    
    # Labels
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title('Regression coefficients')
    ax.set_xlabel('Variable term')
    ax.set_ylabel('Coefficient magnitude')
    ax.set_xscale('symlog', linthresh=0.01, linscale=1)
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))   
    
def plot_regression_coefficients(reg='connected', save=True):
    
    # Load data for all regressions with interaction terms
    if reg == 'connected':
        results = load_one_regression(5)
    elif reg == 'all':
        results = load_one_regression(11)
    else:
        raise RuntimeError('Not a valid regression selection')
    
    # Construct axis specifications
    ax2label2prop2val = {
        'Network measure effects': {
            'main effects': {
                'color': '#AA0000',
                'anchor': None,
                'height': results['ntwk_main']['params'],
                'error': results['ntwk_main']['HC2_se']
                },
            'log$_{10}$(search radius) effects': {
                'color': '#FFAAAA',
                'anchor': results['ntwk_intx']['anchor'],
                'height': results['ntwk_intx']['params'],
                'error': results['ntwk_intx']['HC2_se']
                }
            },
        'Task effects': {
            'main effects': {
                'color': '#AA0000',
                'anchor': None,
                'height': results['task_main']['params'],
                'error': results['task_main']['HC2_se']
                },
            'log$_{10}$(search radius) effects': {
                'color': '#FFAAAA',
                'anchor': results['task_intx']['anchor'],
                'height': results['task_intx']['params'],
                'error': results['task_intx']['HC2_se']
                }
            }
        }
    
    # Plot prep
    ntwk_bars = len(results['ntwk_main']['params'])
    task_bars = len(results['task_main']['params'])
    fig, axs = plt.subplots(1, 2, sharey=True,
        figsize=fs.fig_size(0.5, 0.33, 2), dpi=1200,
        gridspec_kw=dict(width_ratios=(ntwk_bars,task_bars))
        )
    
    # Iteratively build axes in figure
    for ii, (title, label2prop2val) in enumerate(ax2label2prop2val.items()):
        for ll, (label, prop2val) in enumerate(label2prop2val.items()):
            
            # Set constants
            width = 0.55
            
            # Extract direct values from dict
            height = prop2val['height']
            yerr = prop2val['error']
            color = prop2val['color']
            
            # Calculate locations
            x = pd.Series(
                np.arange(len(height)),
                index=prop2val['height'].index
                )
            
            # Calculate anchor
            if prop2val['anchor'] is not None:
                bottom = prop2val['anchor']
            else:
                bottom = pd.Series(
                    np.zeros(len(prop2val['height'])),
                    index=height.index
                    )
            
            # Plot the effects in turn
            axs[ii].bar(
                x=x,
                height=height,
                width=width,
                bottom=bottom,
                yerr=yerr,
                color=color,
                label=label
                )
        
        # Other plots specs
        axs[ii].set_title(title)
        axs[ii].axhline(linewidth=1.5,color='#222222')
        axs[ii].grid(True)
        axs[ii].set_axisbelow(True)
    
        # Set limits the same
        metric_labels = up.get_metric_labels()
        axs[ii].set_xticks(
            ticks=x,
            labels=[metric_labels[key] for key in height.index],
            ha='right',
            rotation=45,
            rotation_mode='anchor'
            )
        
    # Set scale and label
    axs[0].set_yscale('symlog', linthresh=0.01, linscale=1)
    axs[0].set_ylim(bottom=-2, top=2)
    axs[0].set_ylabel('log$_{10}$(Effect size)')
    
    # Update tick labels
    handles, labels = axs[0].get_legend_handles_labels()
    
    # Get connected handles
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, 0),
        loc='lower center',
        ncol=2
        )
    
    # Set figure spacing
    fig.tight_layout(rect=(0,0.05,1,1), w_pad=-6)
    
    # Save figure
    if save:
        fs.save_pub_fig(
            f'regression_effects_{reg}_graphs',
            bbox_inches='tight'
            )
    
    
if __name__ == '__main__':
    
    # Plot all models
    plot_all_regression_coefficients()
    
    # # With interaction terms
    # for mode in ['connected','all']:
    #     plot_regression_coefficients(mode)
